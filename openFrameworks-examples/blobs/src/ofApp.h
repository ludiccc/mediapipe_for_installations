// Blobs receiver — openFrameworks + ofxOsc.
//
// Listens to the OSC messages of mediapipe_osc.py for the **Blobs** detector
// (OpenCV shapes that stand out from the background — people seen from a
// ceiling camera, objects on a table, silhouettes) and draws every blob: its
// outline, bounding box, centroid, velocity arrow, area, orientation and id.
// The floor is divided into a 3 x 3 grid of zones and the app tells which
// zone every blob is in — the typical "something happens where the visitor
// stands" logic of an installation.
//
//   python mediapipe_osc.py --blobs              (terminal: ceiling camera; empty scene at start)
//   make && make RunRelease                      (this app; or open the IDE project)
//
// Addon: ofxOsc (ships with openFrameworks, listed in addons.make).
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

#pragma once

// --- includes ------------------------------------------------------------------
// ofMain.h : all of openFrameworks.  ofxOsc.h : the OSC addon (ofxOscReceiver,
// ofxOscMessage), listed in addons.make so the project generator / Makefile
// compile it in.  std::map / std::vector: the C++ containers used below.
#include "ofMain.h"
#include "ofxOsc.h"
#include <map>
#include <vector>

// --- settings ------------------------------------------------------------------
// The app sends normalized coordinates (0..1): multiply by ofGetWidth() /
// ofGetHeight() to draw, whatever the window size.
static const int PORT = 5005;
static const float STALE_SECONDS = 1.0f;
// The picture is divided into GRID x GRID cells ("zones").  With a ceiling camera
// a zone is a part of the floor: the installation can react to *where* a visitor
// stands without any geometry beyond a division.
static const int GRID = 3;                  // zones per side

// --- the data model --------------------------------------------------------------
// One struct per blob, by stable id.  A blob is a shape that stands out from the
// learned background - with a ceiling camera, usually one person.  zoneCol /
// zoneRow remember the last cell so that a change becomes an event.
struct Blob {
	int id = 0;
	glm::vec2 center{0.5f, 0.5f};
	float bbox[4] = {0, 0, 0, 0};
	glm::vec2 velocity{0, 0};
	float speed = 0, area = 0, angle = 0, length = 0, width = 0;
	std::vector<glm::vec2> contour;     // 16 points
	int zoneCol = -1, zoneRow = -1;
	float lastSeen = 0;

	// The cell that contains the centroid: (int)(x * GRID), clamped to the grid.
	int col() const { return std::max(0, std::min(GRID - 1, (int)(center.x * GRID))); }
	int row() const { return std::max(0, std::min(GRID - 1, (int)(center.y * GRID))); }
};

// --- the app -----------------------------------------------------------------------
// openFrameworks calls setup() once, then update() and draw() every frame
// (60 per second here), exit() when the window closes and keyPressed() on keys.
// OSC messages are read in update(), so - unlike Processing or Python - there
// is only one thread and no locking.
class ofApp : public ofBaseApp {
public:
	void setup();
	void update();
	void draw();
	void exit();
	void keyPressed(int key);

	void handleMessage(const ofxOscMessage& m);
	Blob& getBlob(int id);
	ofColor colorFor(int id);

	// receiver: listens on the UDP port.  The map id -> object is the state the
	// drawing is made from; the other members feed the status line.
	// order: the ids sorted by size that arrive every frame in /mp/blobs.
	ofxOscReceiver receiver;
	std::map<int, Blob> blobs;
	std::vector<int> order;             // ids biggest first, from /mp/blobs
	int messages = 0, frameNumber = 0;
	float analysisFps = 0;
	std::string status = "waiting for mediapipe_osc.py ...";
};
