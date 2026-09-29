#!/usr/bin/env python3
"""Reproduce un sonido una vez, sin OSC ni efectos."""

from __future__ import annotations

import argparse
from pathlib import Path

import sounddevice as sd

from manipulacion_sonido import FORMATOS_AUDIO, reproducir_archivo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archivo", nargs="?", type=Path, help="MP3 o WAV que se reproducirá.")
    parser.add_argument("--modo", choices=("stereo", "cuadrafonico"), default="stereo")
    parser.add_argument("--dispositivo", type=int, default=None)
    parser.add_argument("--listar-dispositivos", action="store_true")
    args = parser.parse_args()
    if args.listar_dispositivos:
        print(sd.query_devices())
        return

    archivo = args.archivo
    if archivo is None:
        carpeta = Path(__file__).resolve().parent / "sonidos"
        archivos = sorted(
            path for path in carpeta.iterdir()
            if path.is_file() and path.suffix.lower() in FORMATOS_AUDIO
        ) if carpeta.is_dir() else []
        if not archivos:
            parser.error("Indica un archivo o copia primero un MP3/WAV en la carpeta sonidos/.")
        archivo = archivos[0]
    if not archivo.is_file():
        parser.error(f"No existe el archivo: {archivo}")
    try:
        reproducir_archivo(archivo, modo=args.modo, dispositivo=args.dispositivo)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
