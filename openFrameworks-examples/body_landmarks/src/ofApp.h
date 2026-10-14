// Body landmarks (pose) receiver — openFrameworks + ofxOsc.
//
// Listens to the OSC messages of mediapipe_osc.py for the **Body** detector and
// draws every person it knows about: skeleton, bounding box, id, velocity arrow,
// which third of the picture they stand in, and whether a hand is raised.
//
//   python mediapipe_osc.py --pose --mirror      (terminal: the camera side)
//   make && make RunRelease                      (this app; or open the IDE project)
//
// Addon: ofxOsc (ships with openFrameworks, listed in addons.make).
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
static const float STALE_SECONDS = 1.0f;          // forget a person if nothing arrived for this long

// --- the data model --------------------------------------------------------------
// One struct per person, kept in a std::map by the stable id the app gives it:
// created on /mp/pose/enter (or on the first message with an unknown id),
// erased on /mp/pose/leave.  handRaised() and zone() are examples of "reading"
// the data for an installation.
struct Person {
	int id = 0;
	glm::vec3 center{0.5f, 0.5f, 0.0f};
	float bbox[4] = {0, 0, 0, 0};   // xmin ymin xmax ymax
	glm::vec2 velocity{0, 0};
	float speed = 0;
	std::vector<glm::vec3> landmarks;   // 33 points, empty until the first landmarks message
	std::vector<float> visibility;      // 33 values 0..1
	// When the last message for this person arrived (seconds), to drop stale people.
	float lastSeen = 0;

	bool handRaised() const;            // a wrist above the shoulders
	int zone() const;                   // 0 = left, 1 = centre, 2 = right third
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
	Person& getPerson(int id);
	ofColor colorFor(int id);

	// receiver: listens on the UDP port.  The map id -> object is the state the
	// drawing is made from; the other members feed the status line.
	ofxOscReceiver receiver;
	std::map<int, Person> people;        // id -> Person
	int messages = 0, frameNumber = 0;
	float analysisFps = 0;
	std::string status = "waiting for mediapipe_osc.py ...";
};
