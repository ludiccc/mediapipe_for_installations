#!/usr/bin/env python3
"""Run a touchdesigner-examples build script outside TouchDesigner, with td_stub.

    python run_example.py ../../../touchdesigner-examples/blobs/blobs.py [seconds]

Builds the network in a fake /project1, feeds it the messages of osc_simulator.py's
scene for `seconds` of simulated time (no network involved), cooks every Script
CHOP / SOP after each frame, and prints what the tables and channels contain.
Exits with 1 if anything raises or if no data arrived in the tables.
"""

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import td_stub  # noqa: E402
import osc_simulator  # noqa: E402

DETECTORS = {"body_landmarks": {"pose"}, "hand_landmarks": {"hands"}, "hand_gestures": {"hands", "gesture"},
             "face_landmarks": {"face"}, "blobs": {"blobs"}}


def walk(comp):
    for child in comp.children:
        yield child
        yield from walk(child)


def main():
    script = sys.argv[1]
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    name = os.path.splitext(os.path.basename(script))[0]
    # everything (like a student who runs all detectors at once): the example must ignore the rest
    detectors = {"pose", "hands", "gesture", "face", "blobs"}

    project1 = td_stub.new_project()
    me = project1.create(td_stub.DAT, name)                # the dragged-in Text DAT
    ns = td_stub.globals_for(me)
    exec(compile(open(script).read(), script, "exec"), ns)
    comps = [c for c in project1.children if isinstance(c, td_stub.COMP)]
    if not comps:
        print("### nothing was built"); return 1
    c = comps[0]
    osc = [o for o in walk(c) if isinstance(o, td_stub.OscInDAT)][0]
    print(f"### built {c.path} with {len(list(walk(c)))} operators; OSC In DAT {osc.path} port {osc.par.port.eval()}")

    scene = osc_simulator.Scene(detectors)
    failures = 0
    messages = 0
    fps = 30.0

    def send(address, args):
        nonlocal messages, failures
        messages += 1
        try:
            osc.receive(address, args)
        except Exception:
            failures += 1
            if failures <= 3:
                print(f"### callback failed on {address} {args[:6]}:")
                traceback.print_exc()

    frames = int(seconds * fps)
    for f in range(frames):
        td_stub.absTime.frame, td_stub.absTime.seconds = f, f / fps
        scene.step(send, f / fps, fps)
    # like the app on exit
    for kind in ("pose", "hand", "face", "blob"):
        for i in scene.all_ids(kind):
            send(f"/mp/{kind}/leave", [i])
    send("/mp/status", ["stopped", "sim"])

    # rerun a few frames so the tables are full again for the summary, then cook everything
    scene2 = osc_simulator.Scene(detectors)
    for f in range(10):
        scene2.step(send, f / fps, fps)
    rows = 0
    for o in walk(c):
        try:
            if isinstance(o, td_stub.ScriptCHOP):
                o.cook(force=True)
                print(f"### {o.name}: {o.numChans} channels x {o.numSamples} samples  {[ch.name for ch in o.chans][:12]}")
            elif isinstance(o, td_stub.ScriptSOP):
                o.cook(force=True)
                print(f"### {o.name}: {o.numPrims} polygons")
            elif isinstance(o, td_stub.TableDAT) and not isinstance(o, td_stub.OscInDAT):
                print(f"### table {o.name}: {o.numRows - 1} rows x {o.numCols} cols  header {o.rows[0][:8]}")
                rows += o.numRows - 1
        except Exception:
            failures += 1
            print(f"### cook failed for {o.path}:")
            traceback.print_exc()
    print(f"### messages {messages}, callback failures {failures}, data rows {rows}")
    return 0 if failures == 0 and rows > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
