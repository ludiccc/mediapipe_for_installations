"""MediaPipe -> OSC: analyze a live webcam with MediaPipe and broadcast the
results as OSC messages for installations (Processing, TouchDesigner, Pure
Data, Max, SuperCollider, Python, ...).

Modules, in the order a frame goes through them (see GUIDE_mediapipe_osc.md):

    config     - the options (Config dataclass + command line parser)
    capture    - frame sources: webcams, Kinects, several side by side, video, image
    cameras    - finds the cameras of the computer for the panel's list
    detectors  - MediaPipe Tasks (pose, hands, gestures, face) -> numpy arrays, + Blobs
    blobs      - the OpenCV blob detector (shapes, e.g. a ceiling camera)
    mapping    - crop & mapping of every camera (input/output quads) + the editor
    depth      - "Z from Kinect": landmark z <- measured distance
    dialogs    - native open/save file dialogs for the configuration files
    tracking   - stable ids, enter/leave events, velocity, smoothing
    osc_out    - OSC message schema (mp / legacy / kinect formats) and UDP sending
    themes     - look & feel: the Theme base class and the template loader
    preview    - default drawing of landmarks / blobs / ids on the image
    ui         - the control panel widgets drawn inside the OpenCV window
    app        - App: everything wired together, the per-frame step, the threads
"""

__version__ = "1.2.0"
