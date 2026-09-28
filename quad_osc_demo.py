#!/usr/bin/env python3
"""Demo: OSC people tracking, sequential MP3 playback, and quad panning.

OSC messages:
    /people/xy x1 y1 x2 y2 x3 y3 x4 y4 x5 y5
        Coordinates must be normalized to [-1, 1]. x=-1 is left, x=1 right;
        y=-1 is front, y=1 rear.
    /sequence/start
        Play pajaros.mp3, then autos.mp3 when the first file ends.

Speaker channel order used here: FL, FR, RR, RL (clockwise around the room).
Place pajaros.mp3 and autos.mp3 next to this script.
"""

from __future__ import annotations

import math
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer


SAMPLE_RATE = 48_000
BLOCK_SIZE = 256
OSC_BIND_ADDRESS = "0.0.0.0"
OSC_PORT = 9000

# Set this to an integer from sd.query_devices() if the default device is not
# the four-output interface. The interface must expose at least four outputs.
OUTPUT_DEVICE = None

CAR_ORBIT_SECONDS = 8.0
EDGE_FADE_SECONDS = 0.01

HERE = Path(__file__).resolve().parent


def decode_mp3_mono(path: Path) -> np.ndarray:
    """Decode once at startup to mono float32 PCM in memory using ffmpeg."""
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    if samples.size == 0:
        raise ValueError(f"No audio decoded from {path}")
    return samples


def point_gains(x: float, y: float) -> np.ndarray:
    """Equal-power gains for a point inside a square quad speaker layout."""
    x = float(np.clip(x, -1.0, 1.0))
    y = float(np.clip(y, -1.0, 1.0))
    u = (x + 1.0) * 0.5
    v = (y + 1.0) * 0.5

    # Channel order: front-left, front-right, rear-right, rear-left.
    gains = np.array(
        [(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v],
        dtype=np.float32,
    )
    power = float(np.sqrt(np.dot(gains, gains)))
    if power > 0:
        gains /= power
    return gains


def orbit_gains(angle: float) -> np.ndarray:
    """Equal-power crossfade between adjacent speakers, clockwise."""
    quarter_turns = (angle % (2 * math.pi)) / (math.pi / 2)
    speaker = int(quarter_turns) % 4
    fraction = quarter_turns - int(quarter_turns)

    gains = np.zeros(4, dtype=np.float32)
    gains[speaker] = math.cos(fraction * math.pi / 2)
    gains[(speaker + 1) % 4] = math.sin(fraction * math.pi / 2)
    return gains


class QuadEngine:
    def __init__(self, bird: np.ndarray, car: np.ndarray, events: queue.SimpleQueue):
        self.samples = {"birds": bird, "cars": car}
        self.events = events
        self.people = np.zeros((5, 2), dtype=np.float32)
        self.active: str | None = None
        self.cursor = 0
        self.car_angle = 0.0
        self.previous_gains = np.zeros(4, dtype=np.float32)
        self.fade_frames = max(1, int(EDGE_FADE_SECONDS * SAMPLE_RATE))

    def _apply_pending_events(self) -> None:
        # Drain a bounded number so a burst of OSC traffic can't monopolize
        # the audio callback.
        for _ in range(16):
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break

            if kind == "people":
                self.people = value
            elif kind == "start":
                self.active = "birds"
                self.cursor = 0
                self.car_angle = 0.0
                self.previous_gains.fill(0.0)

    def _advance_after_clip(self) -> None:
        if self.active == "birds":
            self.active = "cars"
            self.cursor = 0
            self.car_angle = 0.0
            self.previous_gains = orbit_gains(0.0)
        else:
            self.active = None
            self.cursor = 0

    def callback(self, outdata, frames, time_info, status) -> None:
        # The callback always fills the complete four-channel output block.
        outdata.fill(0.0)
        self._apply_pending_events()

        offset = 0
        while offset < frames and self.active is not None:
            source = self.samples[self.active]
            available = len(source) - self.cursor
            if available <= 0:
                self._advance_after_clip()
                continue

            count = min(frames - offset, available)

            if self.active == "birds":
                center = self.people.mean(axis=0)
                target_gains = point_gains(center[0], center[1])
            else:
                next_angle = self.car_angle + (
                    2 * math.pi * count / (CAR_ORBIT_SECONDS * SAMPLE_RATE)
                )
                target_gains = orbit_gains(next_angle)
                self.car_angle = next_angle % (2 * math.pi)

            # Ramp gains across the block to soften changes from OSC tracking
            # updates and to keep the car's orbit continuous between blocks.
            ramp = np.linspace(
                self.previous_gains,
                target_gains,
                num=count,
                endpoint=True,
                dtype=np.float32,
            )
            self.previous_gains = target_gains

            indices = np.arange(self.cursor, self.cursor + count, dtype=np.float32)
            fade_in = np.minimum(1.0, indices / self.fade_frames)
            fade_out = np.minimum(
                1.0, (len(source) - 1 - indices) / self.fade_frames
            )
            envelope = np.clip(np.minimum(fade_in, fade_out), 0.0, 1.0)
            mono = source[self.cursor : self.cursor + count] * envelope

            outdata[offset : offset + count, :] += mono[:, None] * ramp

            self.cursor += count
            offset += count
            if self.cursor >= len(source):
                self._advance_after_clip()


def main() -> None:
    bird_path = HERE / "pajaros.mp3"
    car_path = HERE / "autos.mp3"
    for path in (bird_path, car_path):
        if not path.exists():
            raise SystemExit(f"Falta el archivo de audio: {path}")

    try:
        bird = decode_mp3_mono(bird_path)
        car = decode_mp3_mono(car_path)
    except FileNotFoundError as exc:
        raise SystemExit("No encuentro ffmpeg. Instálalo y vuelve a ejecutar.") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ffmpeg no pudo decodificar un MP3: {exc}") from exc

    print(f"Pájaros: {len(bird) / SAMPLE_RATE:.1f} s; autos: {len(car) / SAMPLE_RATE:.1f} s")
    print("Canales físicos esperados: 1=FL, 2=FR, 3=RR, 4=RL")
    try:
        sd.check_output_settings(
            device=OUTPUT_DEVICE,
            channels=4,
            samplerate=SAMPLE_RATE,
            dtype="float32",
        )
    except Exception as exc:
        print("Dispositivos disponibles:\n", sd.query_devices())
        raise SystemExit(
            "El dispositivo seleccionado no ofrece cuatro salidas a 48 kHz. "
            "Configura OUTPUT_DEVICE en el script."
        ) from exc

    events: queue.SimpleQueue = queue.SimpleQueue()

    def receive_people(address, *values):
        if len(values) != 10:
            return
        try:
            coords = np.asarray(values, dtype=np.float32).reshape(5, 2)
        except (TypeError, ValueError):
            return
        if np.isfinite(coords).all():
            events.put(("people", coords))

    def start_sequence(address, *args):
        events.put(("start", None))

    dispatcher = Dispatcher()
    dispatcher.map("/people/xy", receive_people)
    dispatcher.map("/sequence/start", start_sequence)
    osc_server = BlockingOSCUDPServer(
        (OSC_BIND_ADDRESS, OSC_PORT), dispatcher
    )
    osc_thread = threading.Thread(target=osc_server.serve_forever, daemon=True)
    osc_thread.start()

    engine = QuadEngine(bird, car, events)
    print(f"Escuchando OSC en UDP {OSC_PORT}.")
    print("Envía /people/xy con 10 floats y /sequence/start para disparar ambos sonidos.")

    try:
        with sd.OutputStream(
            device=OUTPUT_DEVICE,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=4,
            dtype="float32",
            callback=engine.callback,
        ):
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        osc_server.shutdown()
        osc_server.server_close()


if __name__ == "__main__":
    main()
