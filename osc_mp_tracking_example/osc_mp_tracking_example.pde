// Receiver for mediapipe_osc.py (default "mp" format, --id-style arg).
//
//   python mediapipe_osc.py --pose --hands --mirror
//
// Each person/hand/face has a stable integer id. The first argument of every
// /mp/pose/..., /mp/hand/..., /mp/face/... message is that id, the rest are
// floats (normalized 0..1 coordinates). /mp/pose/enter and /mp/pose/leave tell
// you when somebody appears or disappears.
//
// NOTE: this sketch was written alongside the Python app but could not be run
// in the environment where the app was developed (no Processing there). It
// uses only the oscP5 API already used by the other sketches of this repo.

import oscP5.*;
import netP5.*;

OscP5 oscP5;

HashMap<Integer, Person> people = new HashMap<Integer, Person>();
HashMap<Integer, PVector[]> hands = new HashMap<Integer, PVector[]>();
int frameCounter = 0;
float appFps = 0;

class Person {
  PVector center = new PVector();
  PVector velocity = new PVector();
  float[] bbox = new float[4];
  PVector[] landmarks = new PVector[33];
  float lastUpdate = 0;
}

void setup() {
  size(800, 600);
  frameRate(60);
  oscP5 = new OscP5(this, 5005);
  textSize(14);
}

void draw() {
  background(20);

  for (Integer id : people.keySet()) {
    Person p = people.get(id);
    colorMode(HSB, 255);
    stroke((id * 60) % 255, 200, 255);
    fill((id * 60) % 255, 200, 255);
    colorMode(RGB, 255);

    // bounding box
    noFill();
    rect(p.bbox[0] * width, p.bbox[1] * height,
         (p.bbox[2] - p.bbox[0]) * width, (p.bbox[3] - p.bbox[1]) * height);

    // landmarks
    for (PVector lm : p.landmarks) {
      if (lm != null) ellipse(lm.x * width, lm.y * height, 5, 5);
    }

    // center + velocity arrow + id
    float cx = p.center.x * width, cy = p.center.y * height;
    ellipse(cx, cy, 14, 14);
    line(cx, cy, cx + p.velocity.x * width * 0.25, cy + p.velocity.y * height * 0.25);
    text("P" + id + "  v=" + nf(p.velocity.mag(), 1, 2), cx + 10, cy - 10);
  }

  for (Integer id : hands.keySet()) {
    PVector[] lms = hands.get(id);
    stroke(255); fill(255);
    for (PVector lm : lms) {
      if (lm != null) ellipse(lm.x * width, lm.y * height, 4, 4);
    }
  }

  fill(255);
  text("people: " + people.size() + "   hands: " + hands.size() +
       "   frame: " + frameCounter + "   analysis fps: " + nf(appFps, 1, 1), 10, 20);
}

void oscEvent(OscMessage m) {
  String addr = m.addrPattern();

  if (addr.equals("/mp/frame")) {
    frameCounter = m.get(0).intValue();
    appFps = m.get(2).floatValue();
    return;
  }
  if (addr.equals("/mp/status")) {
    println("app status: " + m.get(0).stringValue());
    if (m.get(0).stringValue().equals("stopped")) { people.clear(); hands.clear(); }
    return;
  }

  // ---- people -----------------------------------------------------------
  if (addr.equals("/mp/pose/enter")) {
    int id = m.get(0).intValue();
    people.put(id, new Person());
    println("person " + id + " entered");
    return;
  }
  if (addr.equals("/mp/pose/leave")) {
    int id = m.get(0).intValue();
    people.remove(id);
    println("person " + id + " left");
    return;
  }
  if (addr.startsWith("/mp/pose/")) {
    int id = m.get(0).intValue();
    Person p = people.get(id);
    if (p == null) { p = new Person(); people.put(id, p); }

    if (addr.equals("/mp/pose/center")) {
      p.center.set(m.get(1).floatValue(), m.get(2).floatValue(), m.get(3).floatValue());
    } else if (addr.equals("/mp/pose/velocity")) {
      p.velocity.set(m.get(1).floatValue(), m.get(2).floatValue());
    } else if (addr.equals("/mp/pose/bbox")) {
      for (int i = 0; i < 4; i++) p.bbox[i] = m.get(1 + i).floatValue();
    } else if (addr.equals("/mp/pose/landmarks")) {
      // 33 x (x y z visibility) after the id
      for (int i = 0; i < 33; i++) {
        p.landmarks[i] = new PVector(m.get(1 + i * 4).floatValue(),
                                     m.get(2 + i * 4).floatValue(),
                                     m.get(3 + i * 4).floatValue());
      }
    }
    return;
  }

  // ---- hands ------------------------------------------------------------
  if (addr.equals("/mp/hand/leave")) {
    hands.remove(m.get(0).intValue());
    return;
  }
  if (addr.equals("/mp/hand/landmarks")) {
    int id = m.get(0).intValue();
    PVector[] lms = new PVector[21];
    for (int i = 0; i < 21; i++) {
      lms[i] = new PVector(m.get(1 + i * 3).floatValue(),
                           m.get(2 + i * 3).floatValue(),
                           m.get(3 + i * 3).floatValue());
    }
    hands.put(id, lms);
    return;
  }
  if (addr.equals("/mp/hand/gesture")) {
    println("hand " + m.get(0).intValue() + " gesture: " + m.get(1).stringValue());
  }
}
