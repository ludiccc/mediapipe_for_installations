// Face landmarks receiver — openFrameworks + ofxOsc.
//
// Listens to the OSC messages of mediapipe_osc.py for the **Face** detector and,
// for every face: draws the bounding box and the nose tip, the 478-point mesh
// (when the app sends it), the head roll computed from the eyes, a "mask" placed
// over the face (where an installation would put its own image — a bird's head,
// for instance), and bars for the most expressive blendshapes (jaw open, smile,
// blinks, brows).  Opening the mouth is turned into an event.
//
//   python mediapipe_osc.py --face --face-landmarks --mirror   (terminal: the camera side)
//   make && make RunRelease                                    (this app; or open the IDE project)
//
// Addon: ofxOsc (ships with openFrameworks, listed in addons.make).
// NOTE: the 478-point mesh is a ~7 KB message.  The oscpack library inside ofxOsc
// receives datagrams into a fixed buffer (IP_MTU_SIZE, 1536 bytes in stock oscpack);
// if the window says "mesh: no" although the app runs with --face-landmarks, see
// README.md (one #define to change).  Everything else works without the mesh.
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
// Blendshapes are 52 numbers between 0 (neutral) and 1 (maximum) that describe
// the expression (jawOpen, eyeBlinkLeft, mouthSmileRight ...).  A threshold on
// one of them turns an expression into an event.
static const float MOUTH_OPEN = 0.4f;            // jawOpen above this = "mouth open" event

// --- the data model --------------------------------------------------------------
// One struct per face, by stable id.  `blendshapes` keeps the 52 values in the
// order of the names; value(names, "jawOpen") looks one up by name.
struct Face {
	int id = 0;
	glm::vec3 center{0.5f, 0.5f, 0.0f};       // nose tip
	float bbox[4] = {0, 0, 0, 0};
	glm::vec2 velocity{0, 0};
	float speed = 0;
	std::vector<float> blendshapes;           // 52 values, same order as blendshapeNames
	std::vector<glm::vec3> landmarks;         // 478 points, empty without --face-landmarks
	bool mouthOpen = false;
	float lastSeen = 0;

	float rollDegrees() const;                // head tilt from the outer eye corners
	float value(const std::vector<std::string>& names, const std::string& name) const;
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
	Face& getFace(int id);
	ofColor colorFor(int id);

	// receiver: listens on the UDP port.  The map id -> object is the state the
	// drawing is made from; the other members feed the status line.
	// meshReceived remembers whether the 478-point message ever arrived (needs --face-landmarks).
	ofxOscReceiver receiver;
	std::map<int, Face> faces;
	std::vector<std::string> blendshapeNames;  // default order, replaced by /mp/face/blendshape_names
	bool meshReceived = false;
	int messages = 0, frameNumber = 0;
	float analysisFps = 0;
	std::string status = "waiting for mediapipe_osc.py ...";
};
