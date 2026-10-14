// Body landmarks (pose) receiver — Processing + oscP5.
//
// Listens to the OSC messages of mediapipe_osc.py for the **Body** detector and
// draws every person it knows about: skeleton, bounding box, id, velocity arrow,
// which third of the picture they stand in, and whether a hand is raised.
//
//   python mediapipe_osc.py --pose --mirror       (terminal: the camera side)
//   open this sketch in Processing and press Run  (or: processing-java --sketch=<this folder> --run)
//
// Library: oscP5 (Sketch > Import Library... > Manage Libraries > "oscP5").
//
// Messages used (first argument = stable id of the person):
//   /mp/pose/enter      id
//   /mp/pose/leave      id
//   /mp/pose/center     id x y z                  visibility-weighted centre of the body
//   /mp/pose/bbox       id xmin ymin xmax ymax
//   /mp/pose/velocity   id vx vy speed            normalized units per second
//   /mp/pose/landmarks  id 33 x (x y z visibility)
// x, y are normalized (0..1, origin top-left).  Landmark indices:
// https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

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
float STALE_SECONDS = 1.0;          // forget a person if nothing arrived for this long

// Landmark indices and the lines between them (MediaPipe's POSE_CONNECTIONS).
// Names for the landmark indices we use (0 nose ... 32 right foot index, see the
// MediaPipe link above), and the pairs joined by a line to draw the skeleton.
final int NOSE = 0, L_SHOULDER = 11, R_SHOULDER = 12, L_WRIST = 15, R_WRIST = 16;
final int[][] CONNECTIONS = {
  {0, 1}, {0, 4}, {1, 2}, {2, 3}, {3, 7}, {4, 5}, {5, 6}, {6, 8}, {9, 10},
  {11, 12}, {11, 13}, {11, 23}, {12, 14}, {12, 24}, {13, 15}, {14, 16},
  {15, 17}, {15, 19}, {15, 21}, {16, 18}, {16, 20}, {16, 22}, {17, 19}, {18, 20},
  {23, 24}, {23, 25}, {24, 26}, {25, 27}, {26, 28}, {27, 29}, {27, 31}, {28, 30}, {28, 32},
  {29, 31}, {30, 32}
};

OscP5 oscP5;

// The OSC events arrive on another thread than draw(): a ConcurrentHashMap can be
// iterated while it is modified (a plain HashMap would throw an exception).
// --- shared state ----------------------------------------------------------------
// The map id -> object is filled by oscEvent() and read by draw().
ConcurrentHashMap<Integer, Person> people = new ConcurrentHashMap<Integer, Person>();
int messages = 0, frameNumber = 0;
float analysisFps = 0;
String status = "waiting for mediapipe_osc.py ...";

// --- the data model --------------------------------------------------------------
// One object per person, created on /mp/pose/enter (or on the first message
// with an unknown id) and removed on /mp/pose/leave.  The methods are examples
// of "reading" the data: a raised hand, the zone of the picture.
class Person {
  int id;
  PVector center = new PVector(0.5, 0.5, 0);
  float[] bbox = {0, 0, 0, 0};
  PVector velocity = new PVector();
  float speed = 0;
  // Empty (null) until the first landmarks message; draw() checks for that.
  PVector[] landmarks = null;        // 33 points (x, y, z)
  float[] visibility = null;         // 33 values 0..1
  // When the last message for this person arrived, in seconds (to drop stale people).
  float lastSeen;

  Person(int id) {
    this.id = id;
    lastSeen = millis() / 1000.0;
  }

  // True when a wrist is above the shoulders (y grows downwards).
  boolean handRaised() {
    if (landmarks == null) return false;
    // In image coordinates y grows downwards, so "above" means a smaller y.  The
    // visibility (0..1) says how sure MediaPipe is that the wrist is in the picture.
    float shoulders = min(landmarks[L_SHOULDER].y, landmarks[R_SHOULDER].y);
    return (landmarks[L_WRIST].y < shoulders && visibility[L_WRIST] > 0.5)
        || (landmarks[R_WRIST].y < shoulders && visibility[R_WRIST] > 0.5);
  }

  // Which third of the picture: 0 = left, 1 = centre, 2 = right.
  int zone() {
    // (int)(x * 3) is 0, 1 or 2 for x in 0..1; constrain() clamps values outside the picture.
    return constrain((int) (center.x * 3), 0, 2);
  }
}

// Fetch the Person for an id, creating it if needed: a sketch started after the
// app, or one that missed an enter message, still works.  Also refreshes lastSeen.
Person getPerson(int id) {
  Person p = people.get(id);
  if (p == null) {                   // data for somebody we did not see enter (started late)
    p = new Person(id);
    people.put(id, p);
  }
  p.lastSeen = millis() / 1000.0;
  return p;
}

// setup() runs once: open the window and start listening for OSC.  The
// OscProperties object is the way to change oscP5's options before creating it.
void setup() {
  size(800, 600);
  frameRate(60);
  // A bigger datagram buffer than oscP5's default (1536 bytes): the app can send
  // big messages (the 478-point face mesh is ~7 KB) and a truncated packet is lost.
  OscProperties props = new OscProperties();
  props.setListeningPort(PORT);
  props.setDatagramSize(65536);
  oscP5 = new OscP5(this, props);
  textSize(13);
  println("listening for OSC on port " + PORT + " - run mediapipe_osc.py --pose");
}

// draw() runs 60 times per second on Processing's animation thread.  It never
// parses OSC: it only draws what oscEvent() stored in the objects.
void draw() {
  background(20);
  float now = millis() / 1000.0;

  // A leave message can be lost (UDP gives no guarantee), so anything that stopped
  // sending for STALE_SECONDS is dropped as a safety net.
  // forget people whose data stopped arriving (e.g. the app was killed)
  for (Person p : people.values()) {
    if (now - p.lastSeen > STALE_SECONDS) people.remove(p.id);
  }

  // Two vertical lines divide the window into the three zones of Person.zone().
  // the three zones
  stroke(50);
  line(width / 3, 0, width / 3, height);
  line(2 * width / 3, 0, 2 * width / 3, height);
  int[] zoneCount = {0, 0, 0};

  for (Person p : people.values()) {
    int c = colorFor(p.id);
    zoneCount[p.zone()]++;

    // bounding box
    noFill();
    stroke(c);
    strokeWeight(1);
    rect(p.bbox[0] * width, p.bbox[1] * height, (p.bbox[2] - p.bbox[0]) * width, (p.bbox[3] - p.bbox[1]) * height);

    // A line for every pair in CONNECTIONS (skipped when one end is barely visible),
    // then a dot per landmark whose size follows the visibility.
    // skeleton: lines, then points (bigger when more visible)
    if (p.landmarks != null) {
      strokeWeight(2);
      for (int[] conn : CONNECTIONS) {
        PVector a = p.landmarks[conn[0]], b = p.landmarks[conn[1]];
        if (p.visibility[conn[0]] > 0.3 && p.visibility[conn[1]] > 0.3) {
          line(a.x * width, a.y * height, b.x * width, b.y * height);
        }
      }
      noStroke();
      fill(c);
      for (int i = 0; i < 33; i++) {
        float d = 4 + 6 * p.visibility[i];
        ellipse(p.landmarks[i].x * width, p.landmarks[i].y * height, d, d);
      }
    }

    // centre, velocity arrow, label
    float cx = p.center.x * width, cy = p.center.y * height;
    fill(c);
    noStroke();
    ellipse(cx, cy, 16, 16);
    stroke(c);
    strokeWeight(2);
    line(cx, cy, cx + p.velocity.x * width * 0.25, cy + p.velocity.y * height * 0.25);
    // nf(value, 1, 2) formats with two decimals; "LCR".charAt(zone) is a one-letter zone name.
    String label = "P" + p.id + "  speed " + nf(p.speed, 1, 2) + "  zone " + "LCR".charAt(p.zone());
    if (p.handRaised()) label += "  HAND UP!";
    fill(c);
    text(label, p.bbox[0] * width, max(14, p.bbox[1] * height - 6));
  }

  fill(255);
  text("people: " + people.size() + "   left " + zoneCount[0] + "  centre " + zoneCount[1] + "  right " + zoneCount[2], 10, height - 12);
  fill(200);
  text(status + "   frame " + frameNumber + "   analysis " + nf(analysisFps, 1, 1) + " fps   messages " + messages, 10, 20);
}

// A different colour per id.
// A colour per id: hue from the id in HSB mode, then back to RGB so the rest
// of the sketch is not affected.
int colorFor(int id) {
  colorMode(HSB, 255);
  int c = color((id * 47) % 255, 200, 255);
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
    if (m.get(0).stringValue().equals("stopped")) people.clear();
    return;
  }
  // From here on every message is /mp/pose/<something> and starts with the id.
  if (!addr.startsWith("/mp/pose/")) return;      // hands, faces, blobs ... are ignored here

  int id = m.get(0).intValue();
  // enter / leave: where an installation would start and stop something per
  // person (a sound, a particle system, a colour ...).  Ids are never reused.
  if (addr.equals("/mp/pose/enter")) {
    people.put(id, new Person(id));
    println("person " + id + " entered");
    return;
  }
  if (addr.equals("/mp/pose/leave")) {
    people.remove(id);
    println("person " + id + " left");
    return;
  }

  // Data messages.  center: the visibility-weighted centre of the body;
  // bbox: the rectangle around the person; velocity: movement of the centre in
  // picture sizes per second; landmarks: the 33 points (the big one).
  Person p = getPerson(id);
  if (addr.equals("/mp/pose/center")) {
    p.center.set(m.get(1).floatValue(), m.get(2).floatValue(), m.get(3).floatValue());
  } else if (addr.equals("/mp/pose/bbox")) {
    for (int i = 0; i < 4; i++) p.bbox[i] = m.get(1 + i).floatValue();
  } else if (addr.equals("/mp/pose/velocity")) {
    p.velocity.set(m.get(1).floatValue(), m.get(2).floatValue());
    p.speed = m.get(3).floatValue();
  } else if (addr.equals("/mp/pose/landmarks")) {
    // Argument 0 is the id, then 4 numbers per landmark: the i-th landmark's x is
    // argument 1 + i * 4, its visibility argument 4 + i * 4.  We build new arrays
    // and swap them in at the end, so draw() never sees a half-filled skeleton.
    // 33 x (x y z visibility) after the id
    if (m.arguments().length < 1 + 33 * 4) return;
    PVector[] lm = new PVector[33];
    float[] vis = new float[33];
    for (int i = 0; i < 33; i++) {
      lm[i] = new PVector(m.get(1 + i * 4).floatValue(), m.get(2 + i * 4).floatValue(), m.get(3 + i * 4).floatValue());
      vis[i] = m.get(4 + i * 4).floatValue();
    }
    p.landmarks = lm;
    p.visibility = vis;
  }
}
