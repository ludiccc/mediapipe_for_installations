// Receiver for the Blobs detector of mediapipe_osc.py ("mp" format, --id-style arg).
//
//   python mediapipe_osc.py --blobs                 # ceiling webcam, background learned at start
//   python mediapipe_osc.py --blobs --blob-mode dark
//
// Every blob has a stable integer id. Messages (first argument = id):
//   /mp/blobs        count, id, id, ...          blobs on screen, biggest first
//   /mp/blob/enter   id
//   /mp/blob/leave   id
//   /mp/blob/center  id x y z                    normalized 0..1
//   /mp/blob/bbox    id xmin ymin xmax ymax
//   /mp/blob/velocity id vx vy speed
//   /mp/blob/info    id area angle length width  area = fraction of the image, angle in degrees
//   /mp/blob/contour id x0 y0 x1 y1 ... (16 points)
//
// NOTE: written alongside the Python app but not run in the environment where
// the app was developed (no Processing there). It uses only the oscP5 API
// already used by the other sketches of this repository.

import oscP5.*;
import netP5.*;

OscP5 oscP5;
HashMap<Integer, Blob> blobs = new HashMap<Integer, Blob>();
int[] order = new int[0];   // ids, biggest first (from /mp/blobs)

class Blob {
  PVector center = new PVector();
  PVector velocity = new PVector();
  float[] bbox = new float[4];
  float area = 0, angle = 0;
  PVector[] contour = new PVector[0];
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
  stroke(60);
  for (int i = 1; i < 4; i++) {           // a grid, so positions are readable
    line(i * width / 4, 0, i * width / 4, height);
    line(0, i * height / 4, width, i * height / 4);
  }
  for (Integer id : blobs.keySet()) {
    Blob b = blobs.get(id);
    if (millis() - b.lastUpdate > 1000) continue;   // stale (missed a leave)
    colorMode(HSB, 255);
    color c = color((id * 47) % 255, 200, 255);
    colorMode(RGB, 255);

    // contour
    if (b.contour.length > 1) {
      noFill();
      stroke(c);
      strokeWeight(2);
      beginShape();
      for (PVector p : b.contour) vertex(p.x * width, p.y * height);
      endShape(CLOSE);
    }
    // bounding box
    noFill();
    stroke(c, 120);
    strokeWeight(1);
    rect(b.bbox[0] * width, b.bbox[1] * height,
         (b.bbox[2] - b.bbox[0]) * width, (b.bbox[3] - b.bbox[1]) * height);
    // center, velocity, id
    float x = b.center.x * width, y = b.center.y * height;
    fill(c);
    noStroke();
    ellipse(x, y, 12, 12);
    stroke(c);
    strokeWeight(2);
    line(x, y, x + b.velocity.x * width * 0.25, y + b.velocity.y * height * 0.25);
    fill(255);
    text("B" + id + "  " + nf(b.area * 100, 1, 1) + "%  " + nf(b.angle, 1, 0) + "°", x + 10, y - 10);
  }
  fill(255);
  String ids = "";
  for (int id : order) ids += id + " ";
  text("blobs: " + order.length + "   (biggest first: " + ids + ")", 10, 20);
}

void oscEvent(OscMessage m) {
  String addr = m.addrPattern();
  if (addr.equals("/mp/blobs")) {
    int n = m.get(0).intValue();
    order = new int[n];
    for (int i = 0; i < n; i++) order[i] = m.get(1 + i).intValue();
    return;
  }
  if (addr.equals("/mp/status")) {
    if (m.get(0).stringValue().equals("stopped")) blobs.clear();
    return;
  }
  if (!addr.startsWith("/mp/blob/")) return;

  int id = m.get(0).intValue();
  if (addr.equals("/mp/blob/enter")) { blobs.put(id, new Blob()); println("blob " + id + " entered"); return; }
  if (addr.equals("/mp/blob/leave")) { blobs.remove(id); println("blob " + id + " left"); return; }

  Blob b = blobs.get(id);
  if (b == null) { b = new Blob(); blobs.put(id, b); }
  b.lastUpdate = millis();

  if (addr.equals("/mp/blob/center")) {
    b.center.set(m.get(1).floatValue(), m.get(2).floatValue());
  } else if (addr.equals("/mp/blob/bbox")) {
    for (int i = 0; i < 4; i++) b.bbox[i] = m.get(1 + i).floatValue();
  } else if (addr.equals("/mp/blob/velocity")) {
    b.velocity.set(m.get(1).floatValue(), m.get(2).floatValue());
  } else if (addr.equals("/mp/blob/info")) {
    b.area = m.get(1).floatValue();
    b.angle = m.get(2).floatValue();
  } else if (addr.equals("/mp/blob/contour")) {
    int n = (m.arguments().length - 1) / 2;
    b.contour = new PVector[n];
    for (int i = 0; i < n; i++) {
      b.contour[i] = new PVector(m.get(1 + 2 * i).floatValue(), m.get(2 + 2 * i).floatValue());
    }
  }
}
