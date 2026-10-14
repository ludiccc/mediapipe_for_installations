#!/usr/bin/env python3
"""Analyze a live webcam with MediaPipe and send the results as OSC messages.

Examples:
    python mediapipe_osc.py                       # pose, webcam 0, OSC to 127.0.0.1:5005
    python mediapipe_osc.py --pose --hands --face --mirror
    python mediapipe_osc.py --gesture --ip 192.168.1.20 --port 9000
    python mediapipe_osc.py --list-cameras        # which cameras are there (also listed in the window)
    python mediapipe_osc.py --source 2 --template minimal                  # camera 2, another look
    python mediapipe_osc.py --blobs --format kinect --port 12000           # ceiling webcam -> Sonido_Cuadrafonico
    python mediapipe_osc.py --source kinect --blobs --format kinect --port 12000   # ceiling Kinect v2/v1 (depth)
    python mediapipe_osc.py --format kinect --port 12000 --kinect-y size   # frontal webcam, body poses
    python mediapipe_osc.py --format legacy       # original Processing sketches of this repo
    python mediapipe_osc.py --source video.mp4 --no-preview --verbose      # no webcam needed

Run with --help for all options. See README_mediapipe_osc.md for the OSC schema
and the panel, and GUIDE_mediapipe_osc.md for how the program works inside
(start there if you want to modify it). The code lives in the mp_osc/ package;
this file only calls its main().
"""
import sys

from mp_osc.app import main

if __name__ == "__main__":
    sys.exit(main())
