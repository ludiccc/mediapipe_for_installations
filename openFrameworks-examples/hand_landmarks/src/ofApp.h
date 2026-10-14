// Hand landmarks receiver — openFrameworks + ofxOsc.
//
// Listens to the OSC messages of mediapipe_osc.py for the **Hands** detector and
// draws every hand: the 21 landmarks with their connections, left/right, the
// pinch (distance thumb tip - index tip) as a circle, and the fingertips.
//
//   python mediapipe_osc.py --hands --mirror     (terminal: the camera side)
//   make && make RunRelease                      (this app; or open the IDE project)
//
// Addon: ofxOsc (ships with openFrameworks, listed in addons.make).
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
// The pinch is the distance between the thumb tip and the index tip, roughly in
// metres.  Below PINCH_CLOSED we call it "closed": a free "click" for an installation.
static const float PINCH_CLOSED = 0.04f;         // pinch distance below this = "pinching" (tune for your camera)

// --- the data model --------------------------------------------------------------
// One struct per hand, kept in a std::map by the stable id the app gives it.
// `pinching` remembers the last state so that a change becomes an event.
struct Hand {
	int id = 0;
	std::string handedness = "?";
	float score = 0;
	glm::vec3 center{0.5f, 0.5f, 0.0f};
	float bbox[4] = {0, 0, 0, 0};
	glm::vec2 velocity{0, 0};
	float speed = 0;
	float pinch = 1;
	bool pinching = false;
	std::vector<glm::vec3> landmarks;   // 21 points, empty until the first landmarks message
	float lastSeen = 0;
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
	Hand& getHand(int id);

	// receiver: listens on the UDP port.  The map id -> object is the state the
	// drawing is made from; the other members feed the status line.
	ofxOscReceiver receiver;
	std::map<int, Hand> hands;           // id -> Hand
	int messages = 0, frameNumber = 0;
	float analysisFps = 0;
	std::string status = "waiting for mediapipe_osc.py ...";
};
