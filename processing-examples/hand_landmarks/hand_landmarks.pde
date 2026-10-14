// Hand landmarks receiver — Processing + oscP5.
//
// Listens to the OSC messages of mediapipe_osc.py for the **Hands** detector and
// draws every hand: the 21 landmarks with their connections, left/right, the
// pinch (distance thumb tip - index tip) as a circle, and the fingertips.
//
//   python mediapipe_osc.py --hands --mirror      (terminal: the camera side)
//   open this sketch in Processing and press Run  (or: processing-java --sketch=<this folder> --run)
//
// Library: oscP5 (Sketch > Import Library... > Manage Libraries > "oscP5").
//
// Messages used (first argument = stable id of the hand):
//   /mp/hand/enter      id
//   /mp/hand/leave      id
//   /mp/hand/info       id handedness score       handedness = "Left" | "Right"
//   /mp/hand/center     id x y z
//   /mp/hand/bbox       id xmin ymin xmax ymax
//   /mp/hand/velocity   id vx vy speed
//   /mp/hand/pinch      id distance               thumb tip <-> index tip, roughly metres
//   /mp/hand/landmarks  id 21 x (x y z)
// Landmark indices: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

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
// The pinch is the distance between the thumb tip and the index tip, roughly in
// metres.  Below PINCH_CLOSED we call it "closed": a free "click" for an installation.
float PINCH_CLOSED = 0.04;          // pinch distance below this = "pinching" (tune for your camera)

// Names for the landmark indices: wrist (0), then four points per finger (thumb
// 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20); the last of each group
// is the fingertip.  CONNECTIONS are the pairs joined by a line.
final int WRIST = 0, THUMB_TIP = 4, INDEX_TIP = 8, MIDDLE_TIP = 12, RING_TIP = 16, PINKY_TIP = 20;
final int[] FINGERTIPS = {THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP};
final int[][] CONNECTIONS = {        // MediaPipe's HAND_CONNECTIONS
  {0, 1}, {1, 2}, {2, 3}, {3, 4},              // thumb
  {0, 5}, {5, 6}, {6, 7}, {7, 8},              // index
  {5, 9}, {9, 10}, {10, 11}, {11, 12},         // middle
  {9, 13}, {13, 14}, {14, 15}, {15, 16},       // ring
  {13, 17}, {0, 17}, {17, 18}, {18, 19}, {19, 20}   // pinky + palm
};

OscP5 oscP5;
// --- shared state ----------------------------------------------------------------
// The map id -> Hand is filled by oscEvent() and read by draw().
ConcurrentHashMap<Integer, Hand> hands = new ConcurrentHashMap<Integer, Hand>();   // safe across threads
int messages = 0, frameNumber = 0;
float analysisFps = 0;
String status = "waiting for mediapipe_osc.py ...";

// --- the data model --------------------------------------------------------------
// One object per hand, by the stable id the app gives it.  `pinching` remembers
// the last state so that a change becomes an event.
class Hand {
  int id;
  String handedness = "?";
  float score = 0;
  PVector center = new PVector(0.5, 0.5, 0);
  float[] bbox = {0, 0, 0, 0};
  PVector velocity = new PVector();
  float speed = 0;
  float pinch = 1;
  boolean pinching = false;
  // Empty (null) until the first landmarks message; draw() checks for that.
  PVector[] landmarks = null;        // 21 points
  float lastSeen;

  Hand(int id) {
    this.id = id;
    lastSeen = millis() / 1000.0;
  }
}

// Fetch the object for an id, creating it if needed: a sketch started after the
// app, or one that missed an enter message, still works.  Also refreshes lastSeen.
Hand getHand(int id) {
  Hand h = hands.get(id);
  if (h == null) {
    h = new Hand(id);
    hands.put(id, h);
  }
  h.lastSeen = millis() / 1000.0;
  return h;
}

// setup() runs once: open the window and start listening for OSC.  The
// OscProperties object is the way to change oscP5's options before creating it.
void setup() {
  size(800, 600);
  frameRate(60);
  OscProperties props = new OscProperties();   // bigger datagram buffer than the default 1536 bytes
  props.setListeningPort(PORT);
  props.setDatagramSize(65536);
  oscP5 = new OscP5(this, props);
  textSize(13);
  println("listening for OSC on port " + PORT + " - run mediapipe_osc.py --hands");
}

// draw() runs 60 times per second on Processing's animation thread.  It never
// parses OSC: it only draws what oscEvent() stored in the objects.
void draw() {
  background(20);
  float now = millis() / 1000.0;
  for (Hand h : hands.values()) {
    // A leave message can be lost (UDP gives no guarantee), so anything that stopped
    // sending for STALE_SECONDS is dropped as a safety net.
    if (now - h.lastSeen > STALE_SECONDS) hands.remove(h.id);
  }

  for (Hand h : hands.values()) {
    // One colour per handedness.  Then: box, skeleton lines, dots (fingertips bigger), pinch circle, arrow, label.
    int c = h.handedness.equals("Right") ? color(255, 200, 80) : color(60, 160, 255);

    noFill();
    stroke(c);
    strokeWeight(1);
    rect(h.bbox[0] * width, h.bbox[1] * height, (h.bbox[2] - h.bbox[0]) * width, (h.bbox[3] - h.bbox[1]) * height);

    if (h.landmarks != null) {
      strokeWeight(2);
      for (int[] conn : CONNECTIONS) {
        PVector a = h.landmarks[conn[0]], b = h.landmarks[conn[1]];
        line(a.x * width, a.y * height, b.x * width, b.y * height);
      }
      noStroke();
      fill(c);
      for (int i = 0; i < 21; i++) {
        float d = isFingertip(i) ? 12 : 6;
        ellipse(h.landmarks[i].x * width, h.landmarks[i].y * height, d, d);
      }
      // Its diameter follows the pinch distance (x 800 px, at least 8).
      // the pinch: a circle between thumb tip and index tip, filled when closed
      PVector t = h.landmarks[THUMB_TIP], ix = h.landmarks[INDEX_TIP];
      float mx = (t.x + ix.x) / 2 * width, my = (t.y + ix.y) / 2 * height;
      stroke(255);
      strokeWeight(1);
      if (h.pinching) fill(255); else noFill();
      float d = max(8, h.pinch * 800);
      ellipse(mx, my, d, d);
    }

    float cx = h.center.x * width, cy = h.center.y * height;
    stroke(c);
    strokeWeight(2);
    line(cx, cy, cx + h.velocity.x * width * 0.25, cy + h.velocity.y * height * 0.25);
    fill(c);
    String label = "H" + h.id + " " + h.handedness + " (" + nf(h.score, 1, 2) + ")  pinch " + nf(h.pinch, 1, 3) + (h.pinching ? "  PINCH" : "");
    text(label, h.bbox[0] * width, max(14, h.bbox[1] * height - 6));
  }

  fill(255);
  text("hands: " + hands.size(), 10, height - 12);
  fill(200);
  text(status + "   frame " + frameNumber + "   analysis " + nf(analysisFps, 1, 1) + " fps   messages " + messages, 10, 20);
}

// True for the five fingertip indices (used to draw them bigger).
boolean isFingertip(int i) {
  for (int f : FINGERTIPS) if (f == i) return true;
  return false;
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
    if (m.get(0).stringValue().equals("stopped")) hands.clear();
    return;
  }
  if (!addr.startsWith("/mp/hand/")) return;      // poses, faces, blobs ... are ignored here

  int id = m.get(0).intValue();
  // enter / leave: hands appearing and disappearing.  Ids are never reused.
  if (addr.equals("/mp/hand/enter")) {
    hands.put(id, new Hand(id));
    println("hand " + id + " entered");
    return;
  }
  if (addr.equals("/mp/hand/leave")) {
    hands.remove(id);
    println("hand " + id + " left");
    return;
  }

  // Data messages.  info: "Left" / "Right" and the confidence; center, bbox and
  // velocity like the other detectors; pinch and landmarks below.
  Hand h = getHand(id);
  if (addr.equals("/mp/hand/info")) {
    h.handedness = m.get(1).stringValue();
    h.score = m.get(2).floatValue();
  } else if (addr.equals("/mp/hand/center")) {
    h.center.set(m.get(1).floatValue(), m.get(2).floatValue(), m.get(3).floatValue());
  } else if (addr.equals("/mp/hand/bbox")) {
    for (int i = 0; i < 4; i++) h.bbox[i] = m.get(1 + i).floatValue();
  } else if (addr.equals("/mp/hand/velocity")) {
    h.velocity.set(m.get(1).floatValue(), m.get(2).floatValue());
    h.speed = m.get(3).floatValue();
  // pinch arrives every frame: compare with the threshold and only print when the
  // state changes - a continuous value becomes an event (pinch! / released).
  } else if (addr.equals("/mp/hand/pinch")) {
    h.pinch = m.get(1).floatValue();
    boolean pinching = h.pinch < PINCH_CLOSED;
    if (pinching != h.pinching) {                 // an event you could trigger something with
      h.pinching = pinching;
      println("hand " + id + " (" + h.handedness + ") " + (pinching ? "pinch!" : "released"));
    }
  // landmarks: argument 0 is the id, then 3 numbers per point (the i-th point's x
  // is argument 1 + i * 3).  A new array is built and swapped in at the end, so
  // draw() never sees a half-filled hand.
  } else if (addr.equals("/mp/hand/landmarks")) {
    if (m.arguments().length < 1 + 21 * 3) return;
    PVector[] lm = new PVector[21];
    for (int i = 0; i < 21; i++) {
      lm[i] = new PVector(m.get(1 + i * 3).floatValue(), m.get(2 + i * 3).floatValue(), m.get(3 + i * 3).floatValue());
    }
    h.landmarks = lm;
  }
}
