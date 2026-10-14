// Face landmarks receiver — Processing + oscP5.
//
// Listens to the OSC messages of mediapipe_osc.py for the **Face** detector and,
// for every face: draws the bounding box and the nose tip, the 478-point mesh
// (when the app sends it), the head roll computed from the eyes, a "mask" placed
// over the face (where an installation would put its own image — a bird's head,
// for instance), and bars for the most expressive blendshapes (jaw open, smile,
// blinks, brows).  Opening the mouth is turned into an event.
//
//   python mediapipe_osc.py --face --face-landmarks --mirror   (terminal: the camera side)
//   open this sketch in Processing and press Run  (or: processing-java --sketch=<this folder> --run)
//
// Library: oscP5 (Sketch > Import Library... > Manage Libraries > "oscP5").
// IMPORTANT: the 478-point mesh is a ~7 KB message; oscP5's default datagram buffer
// is 1536 bytes, so setup() enlarges it with OscProperties.setDatagramSize().
//
// Messages used (first argument = stable id of the face):
//   /mp/face/enter            id
//   /mp/face/leave            id
//   /mp/face/center           id x y z                 the nose tip
//   /mp/face/bbox             id xmin ymin xmax ymax
//   /mp/face/velocity         id vx vy speed
//   /mp/face/blendshapes      id 52 x value            0..1; order = /mp/face/blendshape_names
//   /mp/face/blendshape_names 52 x name                sent once when the app starts
//   /mp/face/landmarks        id 478 x (x y z)         only with --face-landmarks
// Landmark indices: https://developers.google.com/mediapipe/solutions/vision/face_landmarker

// --- libraries -----------------------------------------------------------------
// oscP5 / netP5 : the OSC library (Sketch > Import Library... > Manage Libraries).
// ConcurrentHashMap : a map that can be read by draw() while oscEvent() (which
// runs on oscP5's own thread) adds and removes entries - a plain HashMap would
// throw ConcurrentModificationException sooner or later.
import oscP5.*;
import netP5.*;
import java.util.concurrent.ConcurrentHashMap;

// --- settings ------------------------------------------------------------------
// The app sends normalized coordinates (0..1): multiply by width / height to draw.
int PORT = 5005;
float STALE_SECONDS = 1.0;
// Blendshapes are 52 numbers between 0 (neutral) and 1 (maximum) that describe
// the expression (jawOpen, eyeBlinkLeft, mouthSmileRight ...).  A threshold on
// one of them turns an expression into an event.
float MOUTH_OPEN = 0.4;             // jawOpen above this = "mouth open" event

// A few useful landmark indices of the 478-point mesh.
// The mesh has 478 points; these are the handful we use by name.  The eye corners
// give the head roll; FACE_OVAL is drawn as an outline; KEY_POINTS as white dots.
final int NOSE_TIP = 1, CHIN = 152, FOREHEAD = 10;
final int LEFT_EYE_OUTER = 263, RIGHT_EYE_OUTER = 33;   // "left" = the person's left
final int MOUTH_LEFT = 291, MOUTH_RIGHT = 61, LIP_TOP = 13, LIP_BOTTOM = 14;
final int[] FACE_OVAL = {10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
  152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109};
final int[] KEY_POINTS = {LEFT_EYE_OUTER, RIGHT_EYE_OUTER, MOUTH_LEFT, MOUTH_RIGHT, LIP_TOP, LIP_BOTTOM, CHIN, FOREHEAD};

// Default blendshape order (the app sends the list at start-up too; we take whichever arrives).
// The 52 blendshape names in the order the values arrive.  The app also sends
// the list at start-up (/mp/face/blendshape_names) and we take that one if it
// comes.  SHOWN_BLENDSHAPES are the few drawn as bars (pick any of the 52).
String[] blendshapeNames = {
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
  "noseSneerLeft", "noseSneerRight"
};
final String[] SHOWN_BLENDSHAPES = {"jawOpen", "mouthSmileLeft", "mouthSmileRight", "eyeBlinkLeft", "eyeBlinkRight",
  "browInnerUp", "mouthPucker", "cheekPuff"};

OscP5 oscP5;
// --- shared state ----------------------------------------------------------------
// faces: id -> Face, filled by oscEvent() and read by draw().  meshReceived
// remembers whether the 478-point message ever arrived (needs --face-landmarks).
ConcurrentHashMap<Integer, Face> faces = new ConcurrentHashMap<Integer, Face>();
int messages = 0, frameNumber = 0;
float analysisFps = 0;
boolean meshReceived = false;
String status = "waiting for mediapipe_osc.py ...";

// --- the data model --------------------------------------------------------------
// One object per face, by stable id.  `blendshapes` keeps the 52 values in the
// order of blendshapeNames; value("jawOpen") looks one up by name.
class Face {
  int id;
  PVector center = new PVector(0.5, 0.5, 0);
  float[] bbox = {0, 0, 0, 0};
  PVector velocity = new PVector();
  float speed = 0;
  float[] blendshapes = new float[52];
  // Empty (null) without --face-landmarks; everything that needs the mesh checks for that.
  PVector[] landmarks = null;          // 478 points or null
  boolean mouthOpen = false;
  float lastSeen;

  Face(int id) {
    this.id = id;
    lastSeen = millis() / 1000.0;
  }

  // atan2 of the line between the eye corners: rotate() takes radians, the label shows degrees().
  // Head roll (tilt) from the line between the outer eye corners, 0 = upright, in radians.
  float roll() {
    if (landmarks == null) return 0;
    PVector l = landmarks[LEFT_EYE_OUTER], r = landmarks[RIGHT_EYE_OUTER];
    return atan2(l.y - r.y, l.x - r.x);
  }

  // A blendshape by name: find its position in blendshapeNames, return the value there.
  float value(String name) {
    for (int i = 0; i < blendshapeNames.length && i < blendshapes.length; i++) {
      if (blendshapeNames[i].equals(name)) return blendshapes[i];
    }
    return 0;
  }
}

// Fetch the object for an id, creating it if needed: a sketch started after the
// app, or one that missed an enter message, still works.  Also refreshes lastSeen.
Face getFace(int id) {
  Face f = faces.get(id);
  if (f == null) {
    f = new Face(id);
    faces.put(id, f);
  }
  f.lastSeen = millis() / 1000.0;
  return f;
}

// setup() runs once: open the window and start listening for OSC.  The
// OscProperties object is the way to change oscP5's options before creating it.
void setup() {
  size(800, 600);
  frameRate(60);
  // The 478-point mesh message is ~7 KB: oscP5's default buffer (1536 bytes) would truncate it.
  OscProperties props = new OscProperties();
  props.setListeningPort(PORT);
  props.setDatagramSize(65536);
  oscP5 = new OscP5(this, props);
  textSize(13);
  println("listening for OSC on port " + PORT + " - run mediapipe_osc.py --face --face-landmarks");
}

// draw() runs 60 times per second on Processing's animation thread.  It never
// parses OSC: it only draws what oscEvent() stored in the objects.
void draw() {
  background(20);
  float now = millis() / 1000.0;
  for (Face f : faces.values()) {
    // A leave message can be lost (UDP gives no guarantee), so anything that stopped
    // sending for STALE_SECONDS is dropped as a safety net.
    if (now - f.lastSeen > STALE_SECONDS) faces.remove(f.id);
  }

  for (Face f : faces.values()) {
    int c = colorFor(f.id);
    float x0 = f.bbox[0] * width, y0 = f.bbox[1] * height, x1 = f.bbox[2] * width, y1 = f.bbox[3] * height;
    noFill();
    stroke(c);
    strokeWeight(1);
    rect(x0, y0, x1 - x0, y1 - y0);

    // pushMatrix / translate / rotate / popMatrix: move the origin to the centre of
    // the face and turn the axes by the roll, draw at (0, 0), then restore.  Anything
    // drawn between translate and popMatrix follows the head.
    // the "mask": an ellipse the size of the face, rotated with the head.
    // An installation would draw its own image here instead (image() inside the same push/pop).
    float roll = f.roll();
    pushMatrix();
    translate((x0 + x1) / 2, (y0 + y1) / 2);
    rotate(roll);
    fill(60);
    stroke(c);
    strokeWeight(2);
    ellipse(0, 0, x1 - x0, y1 - y0);
    popMatrix();

    // With the mesh: every point as a tiny dot, the face oval as a closed shape, the key points and the eye line.
    if (f.landmarks != null) {
      noStroke();
      fill(c);
      for (PVector p : f.landmarks) ellipse(p.x * width, p.y * height, 2, 2);    // the mesh
      noFill();
      stroke(c);
      strokeWeight(1);
      beginShape();                                                             // the face oval
      for (int i : FACE_OVAL) vertex(f.landmarks[i].x * width, f.landmarks[i].y * height);
      endShape(CLOSE);
      fill(255);
      noStroke();
      for (int i : KEY_POINTS) ellipse(f.landmarks[i].x * width, f.landmarks[i].y * height, 8, 8);
      stroke(255);
      PVector l = f.landmarks[LEFT_EYE_OUTER], r = f.landmarks[RIGHT_EYE_OUTER];
      line(l.x * width, l.y * height, r.x * width, r.y * height);
    }

    float nx = f.center.x * width, ny = f.center.y * height;                    // nose tip
    fill(255, 0, 0);
    noStroke();
    ellipse(nx, ny, 10, 10);
    stroke(c);
    strokeWeight(2);
    line(nx, ny, nx + f.velocity.x * width * 0.25, ny + f.velocity.y * height * 0.25);

    fill(c);
    text("F" + f.id + "  roll " + nf(degrees(roll), 1, 0) + " deg" + (f.mouthOpen ? "  MOUTH OPEN" : ""), x0, max(14, y0 - 6));

    // Each bar is 80 px long at value 1; the label is the blendshape name.
    // blendshape bars, to the right of the face (or to the left near the border)
    float bx = x1 + 150 < width ? x1 + 10 : x0 - 150;
    for (int k = 0; k < SHOWN_BLENDSHAPES.length; k++) {
      float v = f.value(SHOWN_BLENDSHAPES[k]);
      float y = y0 + k * 14;
      noStroke();
      fill(c);
      rect(bx, y, 80 * v, 8);
      noFill();
      stroke(90);
      rect(bx, y, 80, 8);
      fill(200);
      textSize(10);
      text(SHOWN_BLENDSHAPES[k], bx + 86, y + 8);
      textSize(13);
    }
  }

  fill(255);
  text("faces: " + faces.size() + "   " + (meshReceived ? "mesh: yes" : "mesh: no (start the app with --face-landmarks)"), 10, height - 12);
  fill(200);
  text(status + "   frame " + frameNumber + "   analysis " + nf(analysisFps, 1, 1) + " fps   messages " + messages, 10, 20);
}

// A colour per id: hue from the id in HSB mode, then back to RGB so the rest
// of the sketch is not affected.
int colorFor(int id) {
  colorMode(HSB, 255);
  int c = color((id * 47) % 255, 160, 255);
  colorMode(RGB, 255);
  return c;
}

// ---------------------------------------------------------------- OSC
// --- OSC ----------------------------------------------------------------------------
// oscP5 calls oscEvent() for every message that arrives, on its own thread.
// m.addrPattern() is the address; m.get(i) is the i-th argument, read with
// intValue() / floatValue() / stringValue().  The first argument of every
// /mp/<kind>/... message is the stable id.
void oscEvent(OscMessage m) {
  messages++;
  if (messages == 1) status = "receiving";
  String addr = m.addrPattern();

  // /mp/frame: a heartbeat once per analysed camera frame (frame number, time, fps, counts).
  if (addr.equals("/mp/frame")) {
    frameNumber = m.get(0).intValue();
    analysisFps = m.get(2).floatValue();
    return;
  }
  // /mp/status "started" | "stopped": on "stopped" we forget everything, because
  // no more leave messages will come.
  if (addr.equals("/mp/status")) {
    status = "app " + m.get(0).stringValue();
    println("mediapipe_osc.py " + m.get(0).stringValue());
    if (m.get(0).stringValue().equals("stopped")) faces.clear();
    return;
  }
  // /mp/face/blendshape_names: 52 strings sent once at start-up; they replace the built-in list.
  if (addr.equals("/mp/face/blendshape_names")) {
    int n = m.arguments().length;
    String[] names = new String[n];
    for (int i = 0; i < n; i++) names[i] = m.get(i).stringValue();
    blendshapeNames = names;
    return;
  }
  if (!addr.startsWith("/mp/face/")) return;      // poses, hands, blobs ... are ignored here

  int id = m.get(0).intValue();
  // enter / leave: faces appearing and disappearing.  Ids are never reused.
  if (addr.equals("/mp/face/enter")) {
    faces.put(id, new Face(id));
    println("face " + id + " entered");
    return;
  }
  if (addr.equals("/mp/face/leave")) {
    faces.remove(id);
    println("face " + id + " left");
    return;
  }

  // Data messages.  center: the nose tip; bbox: the rectangle around the face;
  // velocity: movement of the nose in picture sizes per second; then the two below.
  Face f = getFace(id);
  if (addr.equals("/mp/face/center")) {
    f.center.set(m.get(1).floatValue(), m.get(2).floatValue(), m.get(3).floatValue());
  } else if (addr.equals("/mp/face/bbox")) {
    for (int i = 0; i < 4; i++) f.bbox[i] = m.get(1 + i).floatValue();
  } else if (addr.equals("/mp/face/velocity")) {
    f.velocity.set(m.get(1).floatValue(), m.get(2).floatValue());
    f.speed = m.get(3).floatValue();
  // blendshapes: argument 0 is the id, then the 52 values.  After storing them the
  // jawOpen threshold is compared with the previous state: the "mouth open /
  // closed" event.  Replace the println by your own reaction.
  } else if (addr.equals("/mp/face/blendshapes")) {
    int n = m.arguments().length - 1;
    float[] values = new float[n];
    for (int i = 0; i < n; i++) values[i] = m.get(1 + i).floatValue();
    f.blendshapes = values;
    boolean open = f.value("jawOpen") > MOUTH_OPEN;
    if (open != f.mouthOpen) {                      // an event
      f.mouthOpen = open;
      println("face " + id + " mouth " + (open ? "open" : "closed"));
    }
  // landmarks: the mesh, 3 numbers per point after the id (only with
  // --face-landmarks).  A new array is built and swapped in at the end.
  } else if (addr.equals("/mp/face/landmarks")) {
    int n = (m.arguments().length - 1) / 3;
    if (n < 468) return;
    PVector[] lm = new PVector[n];
    for (int i = 0; i < n; i++) {
      lm[i] = new PVector(m.get(1 + i * 3).floatValue(), m.get(2 + i * 3).floatValue(), m.get(3 + i * 3).floatValue());
    }
    f.landmarks = lm;
    meshReceived = true;
  }
}
