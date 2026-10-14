// Blobs receiver — Processing + oscP5.
//
// Listens to the OSC messages of mediapipe_osc.py for the **Blobs** detector
// (OpenCV shapes that stand out from the background — people seen from a
// ceiling camera, objects on a table, silhouettes) and draws every blob: its
// outline, bounding box, centroid, velocity arrow, area, orientation and id.
// The floor is divided into a 3 x 3 grid of zones and the sketch tells which
// zone every blob is in — the typical "something happens where the visitor
// stands" logic of an installation.
//
//   python mediapipe_osc.py --blobs               (terminal: ceiling camera; empty scene at start)
//   open this sketch in Processing and press Run  (or: processing-java --sketch=<this folder> --run)
//
// Library: oscP5 (Sketch > Import Library... > Manage Libraries > "oscP5").
//
// Messages used (first argument = stable id of the blob):
//   /mp/blobs           count id id ...          blobs on screen, biggest first (every frame)
//   /mp/blob/enter      id
//   /mp/blob/leave      id
//   /mp/blob/center     id x y 0                 centroid
//   /mp/blob/bbox       id xmin ymin xmax ymax
//   /mp/blob/velocity   id vx vy speed
//   /mp/blob/info       id area angle length width    area = fraction of the image (0..1),
//                                                     angle = main axis in degrees (-90..90)
//   /mp/blob/contour    id 16 x (x y)            outline, from the topmost point

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
// The picture is divided into GRID x GRID cells ("zones").  With a ceiling camera
// a zone is a part of the floor: the installation can react to *where* a visitor
// stands without any geometry beyond a division.
int GRID = 3;                       // zones per side

OscP5 oscP5;
// --- shared state ----------------------------------------------------------------
// blobs: id -> Blob, filled by oscEvent() and read by draw().  order: the ids
// sorted by size that arrive every frame in /mp/blobs.
ConcurrentHashMap<Integer, Blob> blobs = new ConcurrentHashMap<Integer, Blob>();
int[] order = new int[0];           // ids biggest first, from /mp/blobs
int messages = 0, frameNumber = 0;
float analysisFps = 0;
String status = "waiting for mediapipe_osc.py ...";

// --- the data model --------------------------------------------------------------
// One object per blob, by stable id.  A blob is a shape that stands out from the
// learned background - with a ceiling camera, usually one person.  zoneCol /
// zoneRow remember the last cell so that a change becomes an event.
class Blob {
  int id;
  PVector center = new PVector(0.5, 0.5);
  float[] bbox = {0, 0, 0, 0};
  PVector velocity = new PVector();
  float speed = 0, area = 0, angle = 0, blobLength = 0, blobWidth = 0;
  // Empty (null) until the first contour message; draw() checks for that.
  PVector[] contour = null;           // 16 points
  int zoneCol = -1, zoneRow = -1;
  float lastSeen;

  Blob(int id) {
    this.id = id;
    lastSeen = millis() / 1000.0;
  }

  // The cell that contains the centroid: (int)(x * GRID), clamped to the grid.
  int col() { return constrain((int) (center.x * GRID), 0, GRID - 1); }
  int row() { return constrain((int) (center.y * GRID), 0, GRID - 1); }
}

// Fetch the object for an id, creating it if needed: a sketch started after the
// app, or one that missed an enter message, still works.  Also refreshes lastSeen.
Blob getBlob(int id) {
  Blob b = blobs.get(id);
  if (b == null) {
    b = new Blob(id);
    blobs.put(id, b);
  }
  b.lastSeen = millis() / 1000.0;
  return b;
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
  println("listening for OSC on port " + PORT + " - run mediapipe_osc.py --blobs");
}

// draw() runs 60 times per second on Processing's animation thread.  It never
// parses OSC: it only draws what oscEvent() stored in the objects.
void draw() {
  background(20);
  float now = millis() / 1000.0;
  for (Blob b : blobs.values()) {
    // A leave message can be lost (UDP gives no guarantee), so anything that stopped
    // sending for STALE_SECONDS is dropped as a safety net.
    if (now - b.lastSeen > STALE_SECONDS) blobs.remove(b.id);
  }

  // For every cell: is there a blob whose col()/row() is this cell?  Fill it if so, always draw its border.
  // zones: light up the cells that contain a blob
  float cw = width / (float) GRID, ch = height / (float) GRID;
  for (int col = 0; col < GRID; col++) {
    for (int row = 0; row < GRID; row++) {
      boolean occupied = false;
      for (Blob b : blobs.values()) if (b.col() == col && b.row() == row) occupied = true;
      stroke(60);
      strokeWeight(1);
      if (occupied) fill(70, 45, 45); else noFill();
      rect(col * cw, row * ch, cw, ch);
    }
  }

  for (Blob b : blobs.values()) {
    int c = colorFor(b.id);
    // The contour as a closed shape, filled with a transparent colour (alpha 90) and outlined.
    if (b.contour != null) {
      fill(c, 90);
      stroke(c);
      strokeWeight(2);
      beginShape();
      for (PVector p : b.contour) vertex(p.x * width, p.y * height);
      endShape(CLOSE);
    }
    noFill();
    stroke(c);
    strokeWeight(1);
    rect(b.bbox[0] * width, b.bbox[1] * height, (b.bbox[2] - b.bbox[0]) * width, (b.bbox[3] - b.bbox[1]) * height);

    float cx = b.center.x * width, cy = b.center.y * height;
    fill(c);
    noStroke();
    ellipse(cx, cy, 12, 12);
    // A white line through the centroid in the direction of the main axis, as long as the blob.
    // main axis (angle is clockwise on screen, length normalized)
    float dx = cos(radians(b.angle)) * b.blobLength / 2 * width, dy = sin(radians(b.angle)) * b.blobLength / 2 * height;
    stroke(255);
    strokeWeight(1);
    line(cx - dx, cy - dy, cx + dx, cy + dy);
    stroke(c);
    strokeWeight(2);
    line(cx, cy, cx + b.velocity.x * width * 0.25, cy + b.velocity.y * height * 0.25);

    // Position in the biggest-first list (1 = the biggest blob on screen, 0 = not listed yet).
    int rank = 0;
    for (int i = 0; i < order.length; i++) if (order[i] == b.id) rank = i + 1;
    fill(c);
    text("B" + b.id + "  #" + rank + "  " + nf(b.area * 100, 1, 1) + "%  " + nf(b.angle, 1, 0) + " deg  zone (" + b.col() + ", " + b.row() + ")",
         b.bbox[0] * width, max(14, b.bbox[1] * height - 6));
  }

  String ids = "";
  for (int id : order) ids += id + " ";
  fill(255);
  text("blobs: " + blobs.size() + "   biggest first: " + ids, 10, height - 12);
  fill(200);
  text(status + "   frame " + frameNumber + "   analysis " + nf(analysisFps, 1, 1) + " fps   messages " + messages, 10, 20);
}

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
    if (m.get(0).stringValue().equals("stopped")) {
      blobs.clear();
      order = new int[0];
    }
    return;
  }
  // /mp/blobs count id id ...: every frame, the ids of the blobs on screen, biggest
  // first.  Argument 0 is the count, then the ids.
  if (addr.equals("/mp/blobs")) {
    int n = m.get(0).intValue();
    int[] ids = new int[n];
    for (int i = 0; i < n; i++) ids[i] = m.get(1 + i).intValue();
    order = ids;
    return;
  }
  if (!addr.startsWith("/mp/blob/")) return;      // poses, hands, faces ... are ignored here

  int id = m.get(0).intValue();
  // enter / leave: blobs appearing and disappearing.  Ids are never reused.
  if (addr.equals("/mp/blob/enter")) {
    blobs.put(id, new Blob(id));
    println("blob " + id + " entered");
    return;
  }
  if (addr.equals("/mp/blob/leave")) {
    blobs.remove(id);
    println("blob " + id + " left");
    return;
  }

  // Data messages.  center: the centroid - we work out its zone and print only
  // when it differs from the previous one, so a position stream becomes a
  // "changed zone" event; bbox, velocity: like the other detectors; info: the
  // area as a fraction of the picture (0.01 = 1 %), the angle of the main axis in
  // degrees and the length / width of the equivalent ellipse; contour: 16 points,
  // 2 numbers each after the id.
  Blob b = getBlob(id);
  if (addr.equals("/mp/blob/center")) {
    b.center.set(m.get(1).floatValue(), m.get(2).floatValue());
    if (b.col() != b.zoneCol || b.row() != b.zoneRow) {     // moved to another zone -> an event
      b.zoneCol = b.col();
      b.zoneRow = b.row();
      println("blob " + id + " is now in zone column " + b.zoneCol + ", row " + b.zoneRow);
    }
  } else if (addr.equals("/mp/blob/bbox")) {
    for (int i = 0; i < 4; i++) b.bbox[i] = m.get(1 + i).floatValue();
  } else if (addr.equals("/mp/blob/velocity")) {
    b.velocity.set(m.get(1).floatValue(), m.get(2).floatValue());
    b.speed = m.get(3).floatValue();
  } else if (addr.equals("/mp/blob/info")) {
    b.area = m.get(1).floatValue();
    b.angle = m.get(2).floatValue();
    b.blobLength = m.get(3).floatValue();
    b.blobWidth = m.get(4).floatValue();
  } else if (addr.equals("/mp/blob/contour")) {
    int n = (m.arguments().length - 1) / 2;
    if (n < 3) return;
    PVector[] pts = new PVector[n];
    for (int i = 0; i < n; i++) pts[i] = new PVector(m.get(1 + 2 * i).floatValue(), m.get(2 + 2 * i).floatValue());
    b.contour = pts;
  }
}
