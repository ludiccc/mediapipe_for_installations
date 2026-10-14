// Hand gestures receiver — Processing + oscP5.
//
// Listens to the OSC messages of mediapipe_osc.py for the **Gestures** detector
// (hands + gesture classification) and shows, for every hand, the recognised
// gesture with its score, plus a log of gesture *changes* — the events an
// installation usually reacts to.  As a demo, each gesture tints the background
// with its own colour and a Thumb_Up makes the screen flash.
//
//   python mediapipe_osc.py --gesture --mirror    (terminal: the camera side; --gesture implies --hands)
//   open this sketch in Processing and press Run  (or: processing-java --sketch=<this folder> --run)
//
// Library: oscP5 (Sketch > Import Library... > Manage Libraries > "oscP5").
//
// Messages used (first argument = stable id of the hand):
//   /mp/hand/enter      id
//   /mp/hand/leave      id
//   /mp/hand/info       id handedness score
//   /mp/hand/center     id x y z
//   /mp/hand/bbox       id xmin ymin xmax ymax
//   /mp/hand/gesture    id name score    name: None, Closed_Fist, Open_Palm, Pointing_Up,
//                                              Thumb_Down, Thumb_Up, Victory, ILoveYou
//   /mp/hand/landmarks  id 21 x (x y z)

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
// The recogniser sends a score (0..1) with every gesture.  Below MIN_SCORE we
// treat the gesture as "None" to avoid flickering between doubtful guesses.
float MIN_SCORE = 0.5;              // ignore gestures the classifier is not sure about

// The eight gesture names MediaPipe's recogniser knows - exactly the strings that
// arrive in /mp/hand/gesture -, with a small icon and a background colour each
// (same index in the three arrays).  CONNECTIONS: the hand skeleton lines.
final String[] GESTURE_NAMES = {"None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"};
final String[] GESTURE_ICONS = {"", "(o)", "\\o/", "^", "-1", "+1", "V", "\\m/"};
final int[][] GESTURE_RGB = {{40, 40, 40}, {180, 60, 60}, {60, 160, 60}, {40, 120, 180}, {120, 40, 120}, {220, 200, 40}, {40, 160, 200}, {200, 60, 180}};
final int[][] CONNECTIONS = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {0, 5}, {5, 6}, {6, 7}, {7, 8}, {5, 9}, {9, 10}, {10, 11},
  {11, 12}, {9, 13}, {13, 14}, {14, 15}, {15, 16}, {13, 17}, {0, 17}, {17, 18}, {18, 19}, {19, 20}};

OscP5 oscP5;
// --- shared state ----------------------------------------------------------------
// hands: id -> Hand, filled by oscEvent() and read by draw().  events / eventTimes:
// a short log of what happened (both threads use them inside synchronized blocks).
// flashUntil: the time until which the screen stays white after a Thumb_Up.
ConcurrentHashMap<Integer, Hand> hands = new ConcurrentHashMap<Integer, Hand>();
ArrayList<String> events = new ArrayList<String>();     // last gesture changes, newest last
ArrayList<Float> eventTimes = new ArrayList<Float>();
int messages = 0, frameNumber = 0;
float analysisFps = 0, flashUntil = 0;
String status = "waiting for mediapipe_osc.py ...";

// --- the data model --------------------------------------------------------------
// One object per hand, by stable id.  `gesture` holds the *last* gesture, so a
// new name in the next message means "the gesture changed" = an event.
class Hand {
  int id;
  String handedness = "?";
  PVector center = new PVector(0.5, 0.5, 0);
  float[] bbox = {0, 0, 0, 0};
  String gesture = "None";
  float score = 0;
  PVector[] landmarks = null;
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

// Position of a gesture name in GESTURE_NAMES (0 = "None" if unknown), for the icon and colour.
int gestureIndex(String name) {
  for (int i = 0; i < GESTURE_NAMES.length; i++) if (GESTURE_NAMES[i].equals(name)) return i;
  return 0;
}

// Print an event and keep the last eight in the on-screen log.  synchronized:
// oscEvent() adds lines while draw() reads them, so only one at a time.
void logEvent(String text) {
  println(text);
  synchronized (events) {
    events.add(text);
    eventTimes.add(millis() / 1000.0);
    while (events.size() > 8) {
      events.remove(0);
      eventTimes.remove(0);
    }
  }
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
  println("listening for OSC on port " + PORT + " - run mediapipe_osc.py --gesture");
}

// draw() runs 60 times per second on Processing's animation thread.  It never
// parses OSC: it only draws what oscEvent() stored in the objects.
void draw() {
  float now = millis() / 1000.0;
  for (Hand h : hands.values()) {
    // A leave message can be lost (UDP gives no guarantee), so anything that stopped
    // sending for STALE_SECONDS is dropped as a safety net.
    if (now - h.lastSeen > STALE_SECONDS) hands.remove(h.id);
  }

  // The hand with the best score decides the background colour (an example of reacting to a gesture).
  // background: the colour of the strongest current gesture, white on a flash
  Hand best = null;
  for (Hand h : hands.values()) if (best == null || h.score > best.score) best = h;
  int[] rgb = best == null ? GESTURE_RGB[0] : GESTURE_RGB[gestureIndex(best.gesture)];
  if (now < flashUntil) background(255); else background(rgb[0], rgb[1], rgb[2]);

  for (Hand h : hands.values()) {
    float x0 = h.bbox[0] * width, y0 = h.bbox[1] * height, x1 = h.bbox[2] * width, y1 = h.bbox[3] * height;
    noFill();
    stroke(255);
    strokeWeight(1);
    rect(x0, y0, x1 - x0, y1 - y0);
    if (h.landmarks != null) {
      for (int[] conn : CONNECTIONS) {
        PVector a = h.landmarks[conn[0]], b = h.landmarks[conn[1]];
        line(a.x * width, a.y * height, b.x * width, b.y * height);
      }
    }
    // The bar length is the score (0..1) times the width of the box.
    // gesture name, icon and a score bar under the hand
    fill(255);
    textSize(18);
    text("H" + h.id + " " + h.handedness + ": " + h.gesture + "  " + GESTURE_ICONS[gestureIndex(h.gesture)], x0, max(20, y0 - 8));
    textSize(13);
    noStroke();
    rect(x0, y1 + 4, (x1 - x0) * h.score, 6);
    noFill();
    stroke(255);
    rect(x0, y1 + 4, x1 - x0, 6);
  }

  // Older events are drawn darker (the grey level drops 20 per second, never below 80).
  // event log
  synchronized (events) {
    for (int i = 0; i < events.size(); i++) {
      float age = now - eventTimes.get(i);
      fill(max(80, 255 - age * 20));
      text(events.get(i), 10, 50 + i * 20);
    }
  }

  fill(230);
  text("gestures: None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou", 10, height - 12);
  text(status + "   frame " + frameNumber + "   analysis " + nf(analysisFps, 1, 1) + " fps   messages " + messages, 10, 20);
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
    logEvent("mediapipe_osc.py " + m.get(0).stringValue());
    if (m.get(0).stringValue().equals("stopped")) hands.clear();
    return;
  }
  if (!addr.startsWith("/mp/hand/")) return;

  int id = m.get(0).intValue();
  // enter / leave: hands appearing and disappearing (logged as events too).
  if (addr.equals("/mp/hand/enter")) {
    hands.put(id, new Hand(id));
    logEvent("hand " + id + " entered");
    return;
  }
  if (addr.equals("/mp/hand/leave")) {
    hands.remove(id);
    logEvent("hand " + id + " left");
    return;
  }

  Hand h = getHand(id);
  if (addr.equals("/mp/hand/gesture")) {
    // Replace the logEvent / flashUntil lines by your own reaction: play a sound,
    // change a scene ... The gesture comes with the hand's position (center, bbox).
    // The interesting one: a gesture per hand, every frame.  We turn it into events.
    String name = m.get(1).stringValue();
    h.score = m.get(2).floatValue();
    if (h.score < MIN_SCORE) name = "None";
    if (!name.equals(h.gesture)) {                      // the gesture changed -> an event
      h.gesture = name;
      logEvent("hand " + id + " (" + h.handedness + "): " + name + "  (" + nf(h.score, 1, 2) + ")");
      if (name.equals("Thumb_Up")) flashUntil = millis() / 1000.0 + 0.3;   // react: flash the screen
    }
  // info / center / bbox / landmarks: where the hand is and how it looks (for the drawing).
  } else if (addr.equals("/mp/hand/info")) {
    h.handedness = m.get(1).stringValue();
  } else if (addr.equals("/mp/hand/center")) {
    h.center.set(m.get(1).floatValue(), m.get(2).floatValue(), m.get(3).floatValue());
  } else if (addr.equals("/mp/hand/bbox")) {
    for (int i = 0; i < 4; i++) h.bbox[i] = m.get(1 + i).floatValue();
  } else if (addr.equals("/mp/hand/landmarks")) {
    if (m.arguments().length < 1 + 21 * 3) return;
    PVector[] lm = new PVector[21];
    for (int i = 0; i < 21; i++) {
      lm[i] = new PVector(m.get(1 + i * 3).floatValue(), m.get(2 + i * 3).floatValue(), m.get(3 + i * 3).floatValue());
    }
    h.landmarks = lm;
  }
}
