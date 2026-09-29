"""Funciones compartidas por los ejemplos numerados."""

from __future__ import annotations

import argparse
from pathlib import Path

import sounddevice as sd

from manipulacion_sonido import BancoSonidos


CARPETA_EJEMPLOS = Path(__file__).resolve().parent


def agregar_argumentos_audio(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--modo", choices=("stereo", "cuadrafonico"), default="stereo",
        help="Salidas de audio; cuadrafónico requiere una interfaz de 4 canales.",
    )
    parser.add_argument(
        "--sonidos", type=Path, default=CARPETA_EJEMPLOS / "sonidos",
        help="Carpeta con archivos MP3 o WAV.",
    )
    parser.add_argument("--puerto", type=int, default=12000, help="Puerto OSC de entrada.")
    parser.add_argument("--dispositivo", type=int, default=None, help="Índice de salida de audio.")
    parser.add_argument("--listar-dispositivos", action="store_true")
    parser.add_argument("--semilla", type=int, default=None, help="Semilla opcional para el azar.")


def listar_dispositivos_si_corresponde(args: argparse.Namespace) -> bool:
    if args.listar_dispositivos:
        print(sd.query_devices())
        return True
    return False


def crear_banco(args: argparse.Namespace) -> BancoSonidos:
    try:
        return BancoSonidos(args.sonidos)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

