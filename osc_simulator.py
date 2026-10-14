#!/usr/bin/env python3
"""Fake mediapipe_osc.py: sends made-up but well-formed OSC data for every detector.

Use it to try a receiver (the examples in python-examples/, processing-examples/,
openFrameworks-examples/ and touchdesigner-examples/, or your own installation)
when there is no camera at hand, or to see how a receiver behaves when people
enter and leave. It sends exactly the same addresses and argument types as
``mediapipe_osc.py --pose --hands --gesture --face --face-landmarks --blobs``
(the default ``mp`` format, ids as first argument).

    python osc_simulator.py                       # everything -> 127.0.0.1:5005, until Ctrl+C
    python osc_simulator.py --only pose,blobs     # just body landmarks and blobs
    python osc_simulator.py --port 9000 --fps 30 --seconds 10
    python osc_simulator.py --no-face-landmarks   # skip the big 478-point mesh message
    python osc_simulator.py --quiet

What it simulates (all coordinates normalized 0..1, origin top-left, like the app):

* **pose**   two people walking around, one of them leaves for a while and comes
             back with a new id (stable ids + enter/leave events, like the tracker)
* **hands**  two hands (Left / Right) with a pinch that opens and closes; with
             ``gesture`` on, the gesture name changes every couple of seconds
* **face**   one face per person (nose tip, bounding box, 52 blendshapes that
             blink / smile / open the jaw, and the 478-point mesh)
* **blobs**  three shapes seen from the ceiling, with contours, areas, angles,
             and one of them leaving and re-entering

The landmark shapes are real detections (captured once from the app on
test_portrait_synthetic.png), only moved and scaled.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time

try:
    from pythonosc import osc_message_builder, udp_client
except ImportError:  # pragma: no cover
    sys.exit("python-osc is missing:  pip install python-osc")

VERSION = "sim-1.0"

BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft",
    "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]
GESTURES = ["None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up",
            "Victory", "ILoveYou"]

# --- shapes: (x, y, z) centred on 0, height 1 (largest side), captured from the real app ----
POSE_SHAPE = [
    (0.019,-0.474,-0.181), (0.03,-0.499,-0.156), (0.037,-0.5,-0.156), (0.042,-0.5,-0.156),
    (0.007,-0.496,-0.152), (0,-0.494,-0.152), (-0.006,-0.493,-0.152), (0.049,-0.491,0.01),
    (-0.012,-0.487,0.027), (0.033,-0.451,-0.119), (0.006,-0.452,-0.115), (0.102,-0.358,0.134),
    (-0.058,-0.37,0.147), (0.123,-0.177,0.131), (-0.124,-0.202,-0.032), (0.124,-0.035,-0.076),
    (-0.125,-0.346,-0.393), (0.129,0.012,-0.105), (-0.129,-0.395,-0.463), (0.113,0.005,-0.139),
    (-0.123,-0.411,-0.451), (0.107,-0.011,-0.095), (-0.117,-0.395,-0.4), (0.045,-0.056,0.275),
    (-0.05,-0.064,0.28), (0.043,0.207,0.161), (-0.054,0.199,0.191), (0.04,0.43,0.415),
    (-0.056,0.418,0.379), (0.041,0.467,0.438), (-0.057,0.451,0.391), (0.024,0.5,0.289),
    (-0.048,0.496,0.226),
]

POSE_VISIBILITY = [
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.93, 0.92, 0.90, 0.97, 0.90, 0.95, 0.90, 0.95, 0.90, 0.94, 0.90, 0.90, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80
]

HAND_SHAPE = [
    (0.012,0.5,0.077), (0.136,0.373,0.021), (0.209,0.185,0), (0.254,0.022,-0.017),
    (0.303,-0.099,-0.033), (0.069,-0.053,0.043), (0.068,-0.251,0.014), (0.057,-0.377,-0.014),
    (0.041,-0.487,-0.037), (-0.02,-0.045,0.044), (-0.048,-0.258,0.019), (-0.07,-0.391,-0.008),
    (-0.091,-0.5,-0.03), (-0.098,0.009,0.036), (-0.136,-0.185,0.006), (-0.16,-0.306,-0.02),
    (-0.181,-0.414,-0.04), (-0.169,0.097,0.022), (-0.227,-0.034,-0.011), (-0.267,-0.123,-0.03),
    (-0.303,-0.212,-0.042),
]

FACE_SHAPE = [
    (0.014,0.161,-0.085), (-0.009,0.04,-0.143), (0.001,0.08,-0.082), (-0.046,-0.075,-0.097),
    (-0.015,0,-0.15), (-0.021,-0.048,-0.136), (-0.034,-0.163,-0.053), (-0.206,-0.123,0.062),
    (-0.048,-0.264,-0.025), (-0.055,-0.315,-0.025), (-0.08,-0.471,0.02), (0.016,0.175,-0.082),
    (0.019,0.189,-0.075), (0.02,0.199,-0.065), (0.021,0.2,-0.069), (0.023,0.214,-0.075),
    (0.026,0.234,-0.084), (0.029,0.255,-0.085), (0.039,0.309,-0.068), (-0.004,0.062,-0.131),
    (-0.027,0.07,-0.095), (-0.3,-0.213,0.204), (-0.136,-0.117,0.03), (-0.159,-0.107,0.033),
    (-0.182,-0.099,0.042), (-0.215,-0.103,0.066), (-0.117,-0.129,0.032), (-0.188,-0.212,0.029),
    (-0.159,-0.215,0.028), (-0.213,-0.196,0.038), (-0.227,-0.172,0.049), (-0.233,-0.071,0.078),
    (-0.046,0.412,-0.03), (-0.216,-0.127,0.072), (-0.294,-0.051,0.199), (-0.262,-0.083,0.108),
    (-0.139,0.048,-0.011), (-0.023,0.164,-0.079), (-0.015,0.198,-0.069), (-0.056,0.183,-0.062),
    (-0.079,0.199,-0.038), (-0.043,0.205,-0.055), (-0.065,0.213,-0.034), (-0.1,0.269,-0.003),
    (-0.03,0.047,-0.14), (-0.039,0.008,-0.145), (-0.263,-0.193,0.063), (-0.096,-0.06,-0.008),
    (-0.089,0.053,-0.072), (-0.093,0.033,-0.063), (-0.215,0.066,0.03), (-0.043,-0.037,-0.126),
    (-0.217,-0.26,0.018), (-0.247,-0.233,0.039), (-0.288,-0.29,0.158), (-0.103,-0.257,-0.013),
    (-0.133,-0.205,0.034), (-0.128,0.236,0.008), (-0.22,0.316,0.23), (-0.066,0.07,-0.065),
    (-0.044,0.076,-0.071), (-0.105,0.224,0.007), (-0.098,0.223,0), (-0.261,-0.249,0.055),
    (-0.088,0.069,-0.059), (-0.17,-0.271,0), (-0.179,-0.296,-0.003), (-0.212,-0.407,0.06),
    (-0.274,-0.268,0.101), (-0.195,-0.353,0.028), (-0.277,-0.202,0.087), (-0.288,-0.204,0.144),
    (-0.02,0.182,-0.077), (-0.05,0.195,-0.059), (-0.072,0.206,-0.038), (-0.059,0.075,-0.061),
    (-0.102,0.223,0.004), (-0.085,0.232,-0.016), (-0.094,0.222,-0.001), (-0.056,0.056,-0.1),
    (-0.06,0.217,-0.03), (-0.037,0.212,-0.046), (-0.011,0.206,-0.059), (0.002,0.319,-0.063),
    (-0.006,0.263,-0.079), (-0.009,0.241,-0.078), (-0.01,0.223,-0.068), (-0.009,0.208,-0.062),
    (-0.059,0.219,-0.032), (-0.062,0.227,-0.038), (-0.066,0.239,-0.043), (-0.068,0.252,-0.038),
    (-0.106,0.17,-0.028), (-0.269,0.129,0.286), (-0.001,0.073,-0.099), (-0.075,0.222,-0.013),
    (-0.08,0.227,-0.017), (-0.035,0.089,-0.074), (-0.076,0.088,-0.044), (-0.039,0.083,-0.073),
    (-0.12,-0.031,0.001), (-0.16,0.004,0.009), (-0.095,0.052,-0.049), (-0.261,-0.354,0.107),
    (-0.245,-0.318,0.063), (-0.228,-0.282,0.026), (-0.072,0.293,-0.023), (-0.119,-0.303,-0.019),
    (-0.136,-0.373,0.006), (-0.153,-0.443,0.03), (-0.202,-0.097,0.054), (-0.254,-0.035,0.092),
    (-0.105,-0.142,0.036), (-0.246,-0.143,0.073), (-0.081,-0.089,-0.017), (-0.076,0.035,-0.097),
    (-0.274,0.005,0.119), (-0.236,-0.019,0.065), (-0.205,-0.019,0.039), (-0.159,-0.04,0.024),
    (-0.125,-0.065,0.017), (-0.101,-0.089,0.01), (-0.056,-0.147,-0.042), (-0.265,0.079,0.106),
    (-0.264,-0.14,0.087), (-0.018,0.065,-0.129), (-0.09,-0.025,-0.023), (-0.295,-0.043,0.284),
    (-0.084,-0.115,0.007), (-0.097,0.055,-0.023), (-0.228,-0.12,0.078), (-0.08,0.011,-0.086),
    (-0.249,0.222,0.265), (-0.107,-0.154,0.045), (-0.063,-0.018,-0.11), (-0.166,0.369,0.085),
    (-0.151,0.426,0.125), (-0.276,0.11,0.197), (-0.204,0.333,0.123), (-0.292,-0.132,0.179),
    (-0.042,0.459,-0.02), (-0.012,0.075,-0.096), (-0.11,0.01,-0.015), (-0.278,-0.066,0.134),
    (-0.182,-0.124,0.045), (-0.163,-0.129,0.038), (-0.088,0.24,-0.012), (-0.252,0.158,0.106),
    (0.015,0.5,-0.024), (-0.068,0.48,0.043), (-0.106,0.459,0.08), (-0.069,-0.394,-0.004),
    (0.069,0.488,-0.036), (-0.144,-0.137,0.036), (-0.125,-0.145,0.039), (-0.113,-0.15,0.044),
    (-0.279,-0.138,0.115), (-0.133,-0.172,0.038), (-0.155,-0.176,0.035), (-0.175,-0.173,0.037),
    (-0.195,-0.161,0.044), (-0.206,-0.147,0.052), (-0.302,-0.135,0.251), (-0.196,-0.122,0.054),
    (0.005,0.108,-0.078), (-0.081,0.144,-0.045), (-0.067,0.064,-0.074), (-0.033,0.119,-0.074),
    (-0.04,-0.21,-0.029), (-0.123,0.407,0.052), (-0.084,0.434,0.017), (0.007,0.468,-0.053),
    (-0.186,0.383,0.18), (-0.116,-0.161,0.042), (-0.064,-0.084,-0.055), (0.063,0.456,-0.064),
    (-0.029,0.494,0.005), (-0.258,0.195,0.184), (-0.036,0.215,-0.048), (-0.039,0.226,-0.054),
    (-0.04,0.242,-0.061), (-0.039,0.262,-0.059), (-0.038,0.313,-0.045), (-0.083,0.218,-0.015),
    (-0.089,0.216,-0.014), (-0.094,0.213,-0.013), (-0.124,0.2,-0.007), (-0.223,0.16,0.05),
    (-0.068,-0.12,-0.027), (-0.092,-0.189,0.026), (-0.107,-0.179,0.038), (-0.076,0.221,-0.013),
    (-0.209,0.262,0.08), (-0.069,-0.195,-0.01), (-0.042,0.362,-0.037), (-0.025,-0.087,-0.107),
    (-0.05,-0.108,-0.072), (-0.03,-0.123,-0.079), (-0.075,-0.022,-0.06), (0.055,0.411,-0.074),
    (0.047,0.358,-0.071), (0.002,0.368,-0.062), (-0.114,0.301,0.009), (-0.114,0.091,-0.017),
    (-0.08,0.339,-0.013), (-0.176,0.104,-0.003), (-0.135,0.134,-0.011), (-0.189,0.173,0.016),
    (0.003,0.423,-0.063), (-0.088,0.004,-0.042), (-0.126,0.349,0.029), (-0.085,0.384,0.002),
    (-0.146,0.257,0.018), (-0.236,0.226,0.108), (-0.173,0.285,0.042), (-0.235,0.272,0.165),
    (-0.151,0.184,0.001), (-0.077,-0.055,-0.038), (-0.065,0.051,-0.104), (-0.079,0.064,-0.077),
    (-0.059,0.021,-0.122), (-0.118,-0.221,0.023), (-0.163,-0.236,0.022), (-0.199,-0.23,0.025),
    (-0.227,-0.21,0.037), (-0.243,-0.181,0.053), (-0.242,-0.103,0.089), (-0.287,0.03,0.199),
    (-0.218,-0.063,0.063), (-0.192,-0.063,0.045), (-0.159,-0.075,0.032), (-0.13,-0.092,0.026),
    (-0.107,-0.111,0.025), (-0.092,-0.127,0.023), (-0.283,0.042,0.294), (-0.077,0.073,-0.065),
    (-0.061,-0.052,-0.079), (-0.048,0.048,-0.126), (-0.03,0.064,-0.118), (-0.046,0.055,-0.114),
    (-0.069,0.082,-0.058), (-0.026,0.065,-0.125), (-0.02,0.073,-0.096), (-0.098,-0.156,0.039),
    (-0.083,-0.151,0.021), (-0.073,-0.147,0.001), (-0.212,-0.136,0.061), (-0.233,-0.147,0.063),
    (-0.003,-0.089,-0.1), (0.143,-0.233,0.04), (0.022,0.053,-0.098), (0.215,-0.383,0.171),
    (0.08,-0.187,0.015), (0.104,-0.191,0.016), (0.127,-0.198,0.022), (0.157,-0.221,0.042),
    (0.059,-0.186,0.021), (0.1,-0.306,0.011), (0.073,-0.291,0.014), (0.128,-0.305,0.018),
    (0.148,-0.292,0.027), (0.183,-0.203,0.053), (0.151,0.349,-0.041), (0.151,-0.243,0.049),
    (0.256,-0.229,0.165), (0.209,-0.233,0.08), (0.126,-0.037,-0.026), (0.05,0.14,-0.084),
    (0.054,0.175,-0.073), (0.087,0.138,-0.07), (0.114,0.138,-0.05), (0.083,0.166,-0.062),
    (0.107,0.157,-0.044), (0.158,0.185,-0.017), (0.013,0.033,-0.142), (0.01,-0.007,-0.148),
    (0.177,-0.332,0.036), (0.055,-0.11,-0.018), (0.076,-0.001,-0.081), (0.074,-0.021,-0.072),
    (0.205,-0.069,0.005), (0.002,-0.053,-0.129), (0.11,-0.367,-0.001), (0.147,-0.359,0.016),
    (0.177,-0.444,0.128), (0.006,-0.296,-0.02), (0.053,-0.266,0.022), (0.175,0.137,-0.009),
    (0.301,0.148,0.198), (0.06,0.028,-0.072), (0.041,0.048,-0.074), (0.149,0.142,-0.008),
    (0.143,0.145,-0.013), (0.156,-0.381,0.03), (0.08,0.014,-0.068), (0.063,-0.35,-0.013),
    (0.062,-0.378,-0.016), (0.062,-0.5,0.043), (0.165,-0.412,0.074), (0.06,-0.439,0.013),
    (0.189,-0.349,0.059), (0.201,-0.363,0.113), (0.053,0.159,-0.081), (0.085,0.152,-0.067),
    (0.111,0.148,-0.049), (0.056,0.037,-0.066), (0.146,0.143,-0.01), (0.132,0.16,-0.028),
    (0.14,0.147,-0.014), (0.044,0.022,-0.106), (0.103,0.164,-0.04), (0.081,0.174,-0.053),
    (0.053,0.186,-0.063), (0.076,0.293,-0.067), (0.066,0.238,-0.083), (0.062,0.217,-0.082),
    (0.057,0.199,-0.073), (0.053,0.188,-0.067), (0.104,0.165,-0.041), (0.109,0.169,-0.047),
    (0.115,0.178,-0.052), (0.12,0.188,-0.049), (0.132,0.094,-0.04), (0.295,-0.052,0.251),
    (0.121,0.157,-0.024), (0.126,0.159,-0.028), (0.036,0.065,-0.078), (0.077,0.04,-0.051),
    (0.039,0.058,-0.076), (0.087,-0.099,-0.012), (0.134,-0.091,-0.009), (0.083,-0.007,-0.059),
    (0.128,-0.485,0.082), (0.12,-0.439,0.041), (0.113,-0.394,0.006), (0.139,0.224,-0.035),
    (0.005,-0.348,-0.026), (-0.001,-0.42,-0.002), (-0.005,-0.494,0.021), (0.147,-0.208,0.032),
    (0.215,-0.184,0.064), (0.046,-0.189,0.026), (0.176,-0.276,0.047), (0.033,-0.127,-0.024),
    (0.056,-0.008,-0.105), (0.248,-0.164,0.088), (0.2,-0.159,0.039), (0.17,-0.14,0.016),
    (0.121,-0.13,0.007), (0.083,-0.132,0.004), (0.053,-0.14,0), (-0.009,-0.162,-0.045),
    (0.26,-0.089,0.074), (0.194,-0.285,0.059), (0.009,0.056,-0.13), (0.059,-0.074,-0.033),
    (0.267,-0.226,0.247), (0.031,-0.153,-0.001), (0.087,-0.004,-0.033), (0.165,-0.244,0.053),
    (0.054,-0.033,-0.093), (0.302,0.046,0.231), (0.044,-0.202,0.034), (0.027,-0.047,-0.116),
    (0.258,0.238,0.061), (0.265,0.296,0.101), (0.287,-0.071,0.163), (0.283,0.179,0.095),
    (0.229,-0.301,0.147), (0.163,0.395,-0.032), (0.008,0.068,-0.097), (0.088,-0.055,-0.027),
    (0.232,-0.229,0.104), (0.119,-0.221,0.026), (0.101,-0.214,0.02), (0.137,0.164,-0.024),
    (0.273,-0.009,0.075), (0.122,0.466,-0.03), (0.198,0.398,0.027), (0.23,0.355,0.06),
    (0.081,-0.21,0.021), (0.063,-0.205,0.026), (0.051,-0.202,0.033), (0.212,-0.294,0.085),
    (0.063,-0.233,0.025), (0.082,-0.251,0.02), (0.101,-0.26,0.02), (0.121,-0.261,0.024),
    (0.136,-0.254,0.031), (0.243,-0.315,0.216), (0.133,-0.227,0.033), (0.099,0.086,-0.055),
    (0.059,0.023,-0.08), (0.044,0.095,-0.078), (0.226,0.297,0.033), (0.197,0.347,0.002),
    (0.117,0.433,-0.059), (0.287,0.232,0.151), (0.05,-0.214,0.03), (0.015,-0.111,-0.06),
    (0.164,0.435,-0.008), (0.295,0.019,0.151), (0.081,0.175,-0.055), (0.086,0.184,-0.062),
    (0.092,0.198,-0.069), (0.096,0.214,-0.067), (0.113,0.262,-0.053), (0.127,0.151,-0.026),
    (0.131,0.145,-0.026), (0.134,0.14,-0.027), (0.16,0.109,-0.023), (0.244,0.012,0.024),
    (0.011,-0.147,-0.033), (0.019,-0.226,0.019), (0.037,-0.226,0.028), (0.121,0.157,-0.024),
    (0.264,0.113,0.053), (-0.007,-0.217,-0.014), (0.133,0.305,-0.046), (-0.006,-0.123,-0.075),
    (0.043,-0.061,-0.066), (0.091,0.34,-0.067), (0.183,0.205,-0.007), (0.115,0.017,-0.029),
    (0.161,0.26,-0.026), (0.178,-0.009,-0.022), (0.149,0.044,-0.027), (0.213,0.045,-0.006),
    (0.107,0.39,-0.069), (0.063,-0.046,-0.051), (0.209,0.242,0.01), (0.182,0.299,-0.013),
    (0.199,0.145,-0.002), (0.279,0.062,0.079), (0.235,0.157,0.02), (0.296,0.103,0.133),
    (0.181,0.079,-0.018), (0.036,-0.092,-0.045), (0.05,0.012,-0.111), (0.069,0.015,-0.085),
    (0.034,-0.01,-0.127), (0.034,-0.271,0.013), (0.069,-0.313,0.008), (0.104,-0.329,0.007),
    (0.135,-0.326,0.015), (0.16,-0.309,0.029), (0.184,-0.238,0.063), (0.274,-0.152,0.165),
    (0.171,-0.186,0.039), (0.145,-0.171,0.024), (0.112,-0.162,0.015), (0.08,-0.16,0.012),
    (0.055,-0.163,0.014), (0.037,-0.169,0.014), (0.282,-0.141,0.258), (0.071,0.024,-0.072),
    (0.019,-0.079,-0.084), (0.031,0.021,-0.13), (0.022,0.046,-0.12), (0.033,0.028,-0.118),
    (0.067,0.038,-0.064), (0.017,0.05,-0.127), (0.017,0.062,-0.097), (0.035,-0.198,0.03),
    (0.021,-0.185,0.014), (0.012,-0.176,-0.005), (0.145,-0.248,0.039), (0.163,-0.272,0.04),
    (-0.165,-0.152,0.002), (-0.14,-0.16,0.002), (-0.17,-0.184,0.002), (-0.19,-0.144,0.002),
    (-0.16,-0.12,0.002), (0.103,-0.237,-0.001), (0.127,-0.244,-0.001), (0.098,-0.269,-0.001),
    (0.078,-0.23,-0.001), (0.107,-0.205,-0.001),
]


# ---------------------------------------------------------------------------
# OSC encoding (same typing rules as mp_osc/osc_out.py: int -> i, float -> f, str -> s)
# ---------------------------------------------------------------------------
def encode(address, args):
    mb = osc_message_builder.OscMessageBuilder(address=address)
    for a in args:
        if isinstance(a, bool):
            mb.add_arg(a, arg_type="T" if a else "F")
        elif isinstance(a, int):
            mb.add_arg(a, arg_type="i")
        elif isinstance(a, float):
            mb.add_arg(a, arg_type="f")
        else:
            mb.add_arg(str(a), arg_type="s")
    return mb.build()


class Sender:
    def __init__(self, ip, port, verbose=False):
        self.client = udp_client.SimpleUDPClient(ip, port)
        self.verbose = verbose
        self.count = 0

    def send(self, address, args):
        self.client.send(encode(address, args))
        self.count += 1
        if self.verbose:
            shown = " ".join(f"{a:.3f}" if isinstance(a, float) else str(a) for a in args[:12])
            if len(args) > 12:
                shown += f" ... ({len(args)} args)"
            print(f"{address:28s} {shown}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def place(shape, cx, cy, height, width=None):
    """Scale a centred shape to ``height`` (normalized) and move it to (cx, cy)."""
    w = height if width is None else width
    return [(cx + x * w, cy + y * height, z * height) for (x, y, z) in shape]


def bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


class Track:
    """A simulated thing with a stable id and a velocity (finite differences)."""

    def __init__(self, track_id):
        self.id = track_id
        self.prev = None
        self.vx = self.vy = self.speed = 0.0

    def move(self, x, y, dt):
        if self.prev is not None and dt > 0:
            vx, vy = (x - self.prev[0]) / dt, (y - self.prev[1]) / dt
            # light smoothing, like the app
            self.vx = 0.7 * self.vx + 0.3 * vx
            self.vy = 0.7 * self.vy + 0.3 * vy
            self.speed = math.hypot(self.vx, self.vy)
        self.prev = (x, y)


class IdCounter:
    def __init__(self):
        self.next = 0

    def new(self):
        self.next += 1
        return self.next - 1


# ---------------------------------------------------------------------------
# the simulated scene
# ---------------------------------------------------------------------------
class Scene:
    """Two people (with a face and a hand each), three blobs.  Person 1 and blob 2
    periodically leave the picture and come back with a new id."""

    def __init__(self, detectors, face_landmarks=True):
        self.det = detectors
        self.face_landmarks = face_landmarks
        self.ids = {k: IdCounter() for k in ("pose", "hand", "face", "blob")}
        self.tracks = {k: {} for k in ("pose", "hand", "face", "blob")}   # slot -> Track
        self.frame = 0
        self.last_t = None

    # -- who is visible at time t -------------------------------------------
    @staticmethod
    def person_visible(slot, t):
        if slot == 0:
            return True
        return (t % 12.0) < 8.0            # person 1: 8 s in, 4 s out

    @staticmethod
    def blob_visible(slot, t):
        if slot < 2:
            return True
        return (t % 9.0) < 6.0

    @staticmethod
    def person_pos(slot, t):
        """Center of the body (x, y) for a person slot."""
        if slot == 0:
            return 0.5 + 0.28 * math.sin(t * 0.5), 0.6 + 0.03 * math.sin(t * 1.7)
        return 0.5 + 0.3 * math.cos(t * 0.35 + 1.0), 0.58 + 0.05 * math.cos(t * 1.1)

    @staticmethod
    def blob_pos(slot, t):
        if slot == 0:
            return 0.3 + 0.2 * math.sin(t * 0.4), 0.5 + 0.3 * math.cos(t * 0.4)
        if slot == 1:
            return 0.7 + 0.15 * math.cos(t * 0.6), 0.4 + 0.2 * math.sin(t * 0.9)
        return 0.5 + 0.4 * math.sin(t * 0.25), 0.75 + 0.1 * math.sin(t * 2.0)

    # -- enter / leave bookkeeping --------------------------------------------
    def _sync(self, kind, slots_visible):
        """Return (enter_ids, leave_ids) after updating self.tracks[kind]."""
        tracks = self.tracks[kind]
        enter, leave = [], []
        for slot, visible in slots_visible.items():
            if visible and slot not in tracks:
                tr = Track(self.ids[kind].new())
                tracks[slot] = tr
                enter.append(tr.id)
            elif not visible and slot in tracks:
                leave.append(tracks.pop(slot).id)
        return enter, leave

    # -- one frame ------------------------------------------------------------
    def step(self, send, t, fps):
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        det = self.det
        out = []   # (address, args) in the app's order: frame, then per kind enter/data/leave

        persons = {s: self.person_visible(s, t) for s in (0, 1)}
        pose_e, pose_l = self._sync("pose", persons if "pose" in det else {})
        hand_e, hand_l = self._sync("hand", persons if "hands" in det else {})
        face_e, face_l = self._sync("face", persons if "face" in det else {})
        blobs_vis = {s: self.blob_visible(s, t) for s in (0, 1, 2)}
        blob_e, blob_l = self._sync("blob", blobs_vis if "blobs" in det else {})

        n_pose, n_hand = len(self.tracks["pose"]), len(self.tracks["hand"])
        n_face, n_blob = len(self.tracks["face"]), len(self.tracks["blob"])
        out.append(("/mp/frame", [self.frame, float(t), float(fps), n_pose, n_hand, n_face, n_blob]))

        # ---- poses ---------------------------------------------------------
        out += [("/mp/pose/enter", [i]) for i in pose_e]
        for slot, tr in self.tracks["pose"].items():
            cx, cy = self.person_pos(slot, t)
            height = 0.8 + 0.05 * math.sin(t * 0.3 + slot)
            pts = place(POSE_SHAPE, cx, cy, height)
            tr.move(cx, cy, dt)
            out.append(("/mp/pose/center", [tr.id, cx, cy, float(pts[0][2])]))
            out.append(("/mp/pose/bbox", [tr.id] + bbox(pts)))
            out.append(("/mp/pose/velocity", [tr.id, tr.vx, tr.vy, tr.speed]))
            lm = []
            for (x, y, z), v in zip(pts, POSE_VISIBILITY):
                lm += [x, y, z, float(v)]
            out.append(("/mp/pose/landmarks", [tr.id] + lm))
        out += [("/mp/pose/leave", [i]) for i in pose_l]

        # ---- hands ---------------------------------------------------------
        out += [("/mp/hand/enter", [i]) for i in hand_e]
        for slot, tr in self.tracks["hand"].items():
            px, py = self.person_pos(slot, t)
            # the hand waves next to the head
            cx = px + (0.18 if slot == 0 else -0.18) + 0.03 * math.sin(t * 3.0)
            cy = py - 0.15 + 0.04 * math.cos(t * 2.0 + slot)
            pts = place(HAND_SHAPE, cx, cy, 0.22)
            tr.move(cx, cy, dt)
            handed = "Right" if slot == 0 else "Left"
            out.append(("/mp/hand/info", [tr.id, handed, 0.95]))
            out.append(("/mp/hand/center", [tr.id, cx, cy, 0.0]))
            out.append(("/mp/hand/bbox", [tr.id] + bbox(pts)))
            out.append(("/mp/hand/velocity", [tr.id, tr.vx, tr.vy, tr.speed]))
            pinch = 0.09 + 0.08 * math.sin(t * 1.5 + slot * 2)
            out.append(("/mp/hand/pinch", [tr.id, max(0.005, pinch)]))
            if "gesture" in det:
                g = GESTURES[int(t / 2.0 + slot * 3) % len(GESTURES)]
                out.append(("/mp/hand/gesture", [tr.id, g, 0.6 + 0.3 * abs(math.sin(t))]))
            lm = []
            for x, y, z in pts:
                lm += [x, y, z]
            out.append(("/mp/hand/landmarks", [tr.id] + lm))
        out += [("/mp/hand/leave", [i]) for i in hand_l]

        # ---- faces ---------------------------------------------------------
        out += [("/mp/face/enter", [i]) for i in face_e]
        for slot, tr in self.tracks["face"].items():
            px, py = self.person_pos(slot, t)
            cx, cy = px, py - 0.36            # the head of the body shape
            pts = place(FACE_SHAPE, cx, cy, 0.17)
            tr.move(cx, cy, dt)
            nose = pts[1]
            out.append(("/mp/face/center", [tr.id, float(nose[0]), float(nose[1]), float(nose[2])]))
            out.append(("/mp/face/bbox", [tr.id] + bbox(pts)))
            out.append(("/mp/face/velocity", [tr.id, tr.vx, tr.vy, tr.speed]))
            bs = [0.0] * len(BLENDSHAPE_NAMES)
            blink = 1.0 if (t + slot) % 4.0 < 0.25 else 0.0
            bs[BLENDSHAPE_NAMES.index("eyeBlinkLeft")] = blink
            bs[BLENDSHAPE_NAMES.index("eyeBlinkRight")] = blink
            bs[BLENDSHAPE_NAMES.index("jawOpen")] = 0.5 + 0.5 * math.sin(t * 1.3 + slot)
            smile = 0.5 + 0.5 * math.sin(t * 0.7 + slot * 2)
            bs[BLENDSHAPE_NAMES.index("mouthSmileLeft")] = smile
            bs[BLENDSHAPE_NAMES.index("mouthSmileRight")] = smile
            bs[BLENDSHAPE_NAMES.index("browInnerUp")] = 0.5 + 0.5 * math.cos(t * 0.9)
            out.append(("/mp/face/blendshapes", [tr.id] + bs))
            if self.face_landmarks:
                lm = []
                for x, y, z in pts:
                    lm += [x, y, z]
                out.append(("/mp/face/landmarks", [tr.id] + lm))
        out += [("/mp/face/leave", [i]) for i in face_l]

        # ---- blobs ---------------------------------------------------------
        out += [("/mp/blob/enter", [i]) for i in blob_e]
        if "blobs" in det:
            infos = []
            for slot, tr in self.tracks["blob"].items():
                cx, cy = self.blob_pos(slot, t)
                tr.move(cx, cy, dt)
                length = 0.18 + 0.04 * slot + 0.02 * math.sin(t + slot)
                width = length * (0.5 + 0.1 * slot)
                angle = ((t * (20 + 10 * slot) + slot * 50) % 180.0) - 90.0
                area = math.pi * (length / 2) * (width / 2)
                infos.append((tr, cx, cy, length, width, angle, area))
            by_size = sorted(infos, key=lambda i: -i[6])
            out.append(("/mp/blobs", [len(infos)] + [i[0].id for i in by_size]))
            for tr, cx, cy, length, width, angle, area in infos:
                a = math.radians(angle)
                contour = []
                for k in range(16):
                    th = -math.pi / 2 + 2 * math.pi * k / 16   # start at the topmost point
                    ex, ey = (length / 2) * math.cos(th), (width / 2) * math.sin(th)
                    contour.append((cx + ex * math.cos(a) - ey * math.sin(a),
                                    cy + ex * math.sin(a) + ey * math.cos(a)))
                out.append(("/mp/blob/center", [tr.id, cx, cy, 0.0]))
                out.append(("/mp/blob/bbox", [tr.id] + bbox(contour)))
                out.append(("/mp/blob/velocity", [tr.id, tr.vx, tr.vy, tr.speed]))
                out.append(("/mp/blob/info", [tr.id, area, angle, length, width]))
                flat = []
                for x, y in contour:
                    flat += [x, y]
                out.append(("/mp/blob/contour", [tr.id] + flat))
        out += [("/mp/blob/leave", [i]) for i in blob_l]

        for addr, args in out:
            send(addr, args)
        self.frame += 1

    def all_ids(self, kind):
        return [tr.id for tr in self.tracks[kind].values()]


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="127.0.0.1", help="receiver host (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=5005, help="receiver port (default 5005)")
    ap.add_argument("--fps", type=float, default=30.0, help="frames per second (default 30)")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this time (0 = until Ctrl+C)")
    ap.add_argument("--only", default="pose,hands,gesture,face,blobs",
                    help="comma list of detectors: pose, hands, gesture, face, blobs (default: all)")
    ap.add_argument("--no-face-landmarks", action="store_true", help="do not send /mp/face/landmarks (478 points)")
    ap.add_argument("--verbose", "-v", action="store_true", help="print every message")
    ap.add_argument("--quiet", "-q", action="store_true", help="print nothing but errors")
    args = ap.parse_args(argv)

    detectors = {d.strip() for d in args.only.split(",") if d.strip()}
    unknown = detectors - {"pose", "hands", "gesture", "face", "blobs"}
    if unknown:
        ap.error(f"unknown detector(s): {', '.join(sorted(unknown))}")
    if "gesture" in detectors:
        detectors.add("hands")

    sender = Sender(args.ip, args.port, verbose=args.verbose)
    scene = Scene(detectors, face_landmarks=not args.no_face_landmarks)

    if not args.quiet:
        print(f"OSC simulator -> {args.ip}:{args.port}  detectors: {', '.join(sorted(detectors))}  "
              f"{args.fps:g} fps  (Ctrl+C to stop)")

    sender.send("/mp/status", ["started", VERSION])
    if "face" in detectors:
        sender.send("/mp/face/blendshape_names", list(BLENDSHAPE_NAMES))
    sender.send("/mp/z_source", ["mediapipe"])

    stop = {"flag": False}

    def on_signal(*_):
        stop["flag"] = True

    signal.signal(signal.SIGINT, on_signal)
    try:
        signal.signal(signal.SIGTERM, on_signal)
    except (AttributeError, ValueError):  # pragma: no cover - not on every platform
        pass

    period = 1.0 / max(args.fps, 0.1)
    t0 = time.monotonic()
    next_frame = t0
    last_report = t0
    try:
        while not stop["flag"]:
            now = time.monotonic()
            if now < next_frame:
                time.sleep(min(period, next_frame - now))
                continue
            next_frame += period
            t = now - t0
            if args.seconds and t >= args.seconds:
                break
            scene.step(sender.send, t, args.fps)
            if not args.quiet and not args.verbose and now - last_report >= 2.0:
                last_report = now
                print(f"  t={t:5.1f}s  frame {scene.frame}  messages sent: {sender.count}   "
                      f"poses {scene.all_ids('pose')}  hands {scene.all_ids('hand')}  "
                      f"faces {scene.all_ids('face')}  blobs {scene.all_ids('blob')}")
    finally:
        # like the app: everybody leaves, then status stopped
        for kind in ("pose", "hand", "face", "blob"):
            for i in scene.all_ids(kind):
                sender.send(f"/mp/{kind}/leave", [i])
        sender.send("/mp/status", ["stopped", VERSION])
        if not args.quiet:
            print(f"stopped after {scene.frame} frames, {sender.count} messages")


if __name__ == "__main__":
    main()
