// Hand gestures receiver — openFrameworks + ofxOsc.
//
// Listens to the OSC messages of mediapipe_osc.py for the **Gestures** detector
// (hands + gesture classification) and shows, for every hand, the recognised
// gesture with its score, plus a log of gesture *changes* — the events an
// installation usually reacts to.  As a demo, each gesture tints the background
// with its own colour and a Thumb_Up makes the screen flash.
//
//   python mediapipe_osc.py --gesture --mirror   (terminal: the camera side; --gesture implies --hands)
//   make && make RunRelease                      (this app; or open the IDE project)
//
// Addon: ofxOsc (ships with openFrameworks, listed in addons.make).
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

#pragma once

// --- includes ------------------------------------------------------------------
// ofMain.h : all of openFrameworks.  ofxOsc.h : the OSC addon (ofxOscReceiver,
// ofxOscMessage), listed in addons.make so the project generator / Makefile
// compile it in.  std::map / std::vector: the C++ containers used below.
// std::deque: a list that is cheap to shorten at the front (the event log).
#include "ofMain.h"
#include "ofxOsc.h"
#include <deque>
#include <map>
#include <vector>

// --- settings ------------------------------------------------------------------
// The app sends normalized coordinates (0..1): multiply by ofGetWidth() /
// ofGetHeight() to draw, whatever the window size.
static const int PORT = 5005;
static const float STALE_SECONDS = 1.0f;
// The recogniser sends a score (0..1) with every gesture.  Below MIN_SCORE we
// treat the gesture as "None" to avoid flickering between doubtful guesses.
static const float MIN_SCORE = 0.5f;            // ignore gestures the classifier is not sure about

// --- the data model --------------------------------------------------------------
// One struct per hand, by stable id.  `gesture` holds the *last* gesture, so a
// new name in the next message means "the gesture changed" = an event.
struct Hand {
	int id = 0;
	std::string handedness = "?";
	glm::vec3 center{0.5f, 0.5f, 0.0f};
	float bbox[4] = {0, 0, 0, 0};
	std::string gesture = "None";
	float score = 0;
	std::vector<glm::vec3> landmarks;
	float lastSeen = 0;
};

// One line of the on-screen log: when it happened and what.
struct Event {
	float time;
	std::string text;
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
	void logEvent(const std::string& text);
	int gestureIndex(const std::string& name) const;

	// receiver: listens on the UDP port.  The map id -> object is the state the
	// drawing is made from; the other members feed the status line.
	// flashUntil: the time until which the screen stays white after a Thumb_Up.
	ofxOscReceiver receiver;
	std::map<int, Hand> hands;
	std::deque<Event> events;            // last gesture changes, newest last
	int messages = 0, frameNumber = 0;
	float analysisFps = 0, flashUntil = 0;
	std::string status = "waiting for mediapipe_osc.py ...";
};
