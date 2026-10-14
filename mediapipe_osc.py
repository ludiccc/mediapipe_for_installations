#!/usr/bin/env python3
"""Analiza una cámara en vivo con MediaPipe y envía los resultados por OSC.

Analyze a live camera with MediaPipe and send the results as OSC messages.

Ejemplos / examples:
    python mediapipe_osc.py                       # pose, cámara/webcam 0 → 127.0.0.1:5005
    python mediapipe_osc.py --pose --hands --face --mirror
    python mediapipe_osc.py --gesture --ip 192.168.1.20 --port 9000
    python mediapipe_osc.py --list-cameras        # cámaras disponibles / available cameras
    python mediapipe_osc.py --source 2 --template minimal                  # cámara 2 / camera 2
    python mediapipe_osc.py --blobs --format kinect --port 12000           # cámara cenital / ceiling camera → Sonido_Cuadrafonico
    python mediapipe_osc.py --source kinect --blobs --format kinect --port 12000   # Kinect v2/v1 cenital / ceiling (depth)
    python mediapipe_osc.py --format kinect --port 12000 --kinect-y size   # cámara frontal / frontal camera
    python mediapipe_osc.py --format legacy       # sketches originales / original Processing sketches
    python mediapipe_osc.py --source video.mp4 --no-preview --verbose      # no necesita cámara / no camera needed

Ejecutá con --help para ver todas las opciones / Run with --help for all options.

Manuales / manuals:
* README_mediapipe_osc.md (English) / README_mediapipe_osc_es.md (castellano):
  esquema OSC y panel / OSC schema and panel.
* GUIDE_mediapipe_osc.md (English) / GUIDE_mediapipe_osc_es.md (castellano):
  funcionamiento interno / how the program works inside.

El código está en el paquete mp_osc; este archivo solo llama a main().
The code lives in the mp_osc package; this file only calls main().
"""
import sys

from mp_osc.app import main

if __name__ == "__main__":
    sys.exit(main())
