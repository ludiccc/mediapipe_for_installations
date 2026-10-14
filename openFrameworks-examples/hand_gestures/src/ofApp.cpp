#include "ofApp.h"

// The eight gesture names MediaPipe's recogniser knows - exactly the strings that
// arrive in /mp/hand/gesture -, with a small icon and a background colour each
// (same index in the three arrays).  CONNECTIONS: the hand skeleton lines.
static const char* GESTURE_NAMES[] = {"None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"};
static const char* GESTURE_ICONS[] = {"", "(o)", "\\o/", "^", "-1", "+1", "V", "\\m/"};
static const ofColor GESTURE_COLORS[] = {
	ofColor(40, 40, 40), ofColor(180, 60, 60), ofColor(60, 160, 60), ofColor(40, 120, 180),
	ofColor(120, 40, 120), ofColor(220, 200, 40), ofColor(40, 160, 200), ofColor(200, 60, 180)
};
static const int N_GESTURES = 8;
static const int CONNECTIONS[][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {0, 5}, {5, 6}, {6, 7}, {7, 8}, {5, 9}, {9, 10}, {10, 11},
	{11, 12}, {9, 13}, {13, 14}, {14, 15}, {15, 16}, {13, 17}, {0, 17}, {17, 18}, {18, 19}, {19, 20}};
static const int N_CONNECTIONS = sizeof(CONNECTIONS) / sizeof(CONNECTIONS[0]);

//--------------------------------------------------------------
// setup(): window title, frame rate, background colour, and start listening.
// ofxOscReceiver::setup(port) opens the UDP socket; messages queue up until
// update() reads them.
void ofApp::setup() {
	ofSetWindowTitle("hand gestures (OSC)");
	ofSetFrameRate(60);
	receiver.setup(PORT);
	ofLogNotice() << "listening for OSC on port " << PORT << " - run mediapipe_osc.py --gesture";
}

// Position of a gesture name in GESTURE_NAMES (0 = "None" if unknown), for the icon and colour.
int ofApp::gestureIndex(const std::string& name) const {
	for (int i = 0; i < N_GESTURES; i++) if (name == GESTURE_NAMES[i]) return i;
	return 0;
}

// Print an event and keep the last eight in the on-screen log.
void ofApp::logEvent(const std::string& text) {
	ofLogNotice() << text;
	events.push_back({ofGetElapsedTimef(), text});
	while (events.size() > 8) events.pop_front();
}

//--------------------------------------------------------------
// update(): first empty the queue of messages that arrived since the last
// frame (hasWaitingMessages / getNextMessage), then drop stale objects.  A leave
// message can be lost (UDP gives no guarantee), so the timeout is a safety net.
void ofApp::update() {
	while (receiver.hasWaitingMessages()) {
		ofxOscMessage m;
		receiver.getNextMessage(m);
		messages++;
		if (messages == 1) status = "receiving";
		handleMessage(m);
	}
	float now = ofGetElapsedTimef();
	// Erasing while iterating a std::map: erase() returns the next iterator, so
	// only advance with ++it when nothing was erased.
	for (auto it = hands.begin(); it != hands.end();) {
		if (now - it->second.lastSeen > STALE_SECONDS) it = hands.erase(it);
		else ++it;
	}
}

//--------------------------------------------------------------
// hands[id] creates a default Hand when the id is new.  Returns a reference.
Hand& ofApp::getHand(int id) {
	Hand& h = hands[id];
	h.id = id;
	h.lastSeen = ofGetElapsedTimef();
	return h;
}

// handleMessage(): one `if` per address.  getAddress() is the address;
// getArgAsInt32(i) / getArgAsFloat(i) / getArgAsString(i) read the i-th
// argument.  The first argument of every /mp/<kind>/... message is the stable id.
void ofApp::handleMessage(const ofxOscMessage& m) {
	const std::string& addr = m.getAddress();

	// /mp/frame: a heartbeat once per analysed camera frame (frame number, time, fps, counts).
	if (addr == "/mp/frame") {
		frameNumber = m.getArgAsInt32(0);
		analysisFps = m.getArgAsFloat(2);
		return;
	}
	// /mp/status "started" | "stopped": on "stopped" we forget everything, because
	// no more leave messages will come.
	if (addr == "/mp/status") {
		status = "app " + m.getArgAsString(0);
		logEvent("mediapipe_osc.py " + m.getArgAsString(0));
		if (m.getArgAsString(0) == "stopped") hands.clear();
		return;
	}
	if (addr.compare(0, 9, "/mp/hand/") != 0) return;

	int id = m.getArgAsInt32(0);
	// enter / leave: hands appearing and disappearing (logged as events too).
	if (addr == "/mp/hand/enter") {
		hands[id] = Hand();
		getHand(id);
		logEvent("hand " + ofToString(id) + " entered");
		return;
	}
	if (addr == "/mp/hand/leave") {
		hands.erase(id);
		logEvent("hand " + ofToString(id) + " left");
		return;
	}

	Hand& h = getHand(id);
	if (addr == "/mp/hand/gesture") {
		// Replace the logEvent / flashUntil lines by your own reaction: play a sound,
		// change a scene ... The gesture comes with the hand's position (center, bbox).
		// The interesting one: a gesture per hand, every frame.  We turn it into events.
		std::string name = m.getArgAsString(1);
		h.score = m.getArgAsFloat(2);
		if (h.score < MIN_SCORE) name = "None";
		if (name != h.gesture) {                            // the gesture changed -> an event
			h.gesture = name;
			logEvent("hand " + ofToString(id) + " (" + h.handedness + "): " + name + "  (" + ofToString(h.score, 2) + ")");
			if (name == "Thumb_Up") flashUntil = ofGetElapsedTimef() + 0.3f;   // react: flash the screen
		}
	// info / center / bbox / landmarks: where the hand is and how it looks (for the drawing).
	} else if (addr == "/mp/hand/info") {
		h.handedness = m.getArgAsString(1);
	} else if (addr == "/mp/hand/center") {
		h.center = glm::vec3(m.getArgAsFloat(1), m.getArgAsFloat(2), m.getArgAsFloat(3));
	} else if (addr == "/mp/hand/bbox") {
		for (int i = 0; i < 4; i++) h.bbox[i] = m.getArgAsFloat(1 + i);
	} else if (addr == "/mp/hand/landmarks") {
		if (m.getNumArgs() < 1 + 21 * 3) return;
		h.landmarks.resize(21);
		for (int i = 0; i < 21; i++) {
			h.landmarks[i] = glm::vec3(m.getArgAsFloat(1 + i * 3), m.getArgAsFloat(2 + i * 3), m.getArgAsFloat(3 + i * 3));
		}
	}
}

//--------------------------------------------------------------
// draw(): only reads the state filled by update().  Every coordinate is
// normalized (0..1) and multiplied by the window size here.
void ofApp::draw() {
	float w = ofGetWidth(), h = ofGetHeight();
	float now = ofGetElapsedTimef();

	// The hand with the best score decides the background colour (an example of reacting to a gesture).
	// background: the colour of the strongest current gesture, white on a flash
	const Hand* best = nullptr;
	for (auto& kv : hands) if (!best || kv.second.score > best->score) best = &kv.second;
	if (now < flashUntil) ofBackground(255);
	else ofBackground(best ? GESTURE_COLORS[gestureIndex(best->gesture)] : GESTURE_COLORS[0]);

	for (auto& kv : hands) {
		const Hand& hd = kv.second;
		float x0 = hd.bbox[0] * w, y0 = hd.bbox[1] * h, x1 = hd.bbox[2] * w, y1 = hd.bbox[3] * h;
		ofSetColor(255);
		ofNoFill();
		ofSetLineWidth(1);
		ofDrawRectangle(x0, y0, x1 - x0, y1 - y0);
		if (hd.landmarks.size() == 21) {
			for (int k = 0; k < N_CONNECTIONS; k++) {
				const glm::vec3& a = hd.landmarks[CONNECTIONS[k][0]];
				const glm::vec3& b = hd.landmarks[CONNECTIONS[k][1]];
				ofDrawLine(a.x * w, a.y * h, b.x * w, b.y * h);
			}
		}
		// The bar length is the score (0..1) times the width of the box.
		// gesture name, icon and a score bar under the hand
		ofDrawBitmapStringHighlight("H" + ofToString(hd.id) + " " + hd.handedness + ": " + hd.gesture + "  "
		                            + GESTURE_ICONS[gestureIndex(hd.gesture)], x0, std::max(16.0f, y0 - 8));
		ofFill();
		ofDrawRectangle(x0, y1 + 4, (x1 - x0) * hd.score, 6);
		ofNoFill();
		ofDrawRectangle(x0, y1 + 4, x1 - x0, 6);
	}

	// Older events are drawn darker (the grey level drops 20 per second, never below 80).
	// event log
	int i = 0;
	for (const Event& e : events) {
		float age = now - e.time;
		ofSetColor((int)std::max(80.0f, 255 - age * 20));
		ofDrawBitmapString(e.text, 10, 50 + i++ * 16);
	}

	ofSetColor(230);
	ofDrawBitmapString("gestures: None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou", 10, h - 12);
	ofDrawBitmapString(status + "   frame " + ofToString(frameNumber) + "   analysis " + ofToString(analysisFps, 1)
	                   + " fps   messages " + ofToString(messages), 10, 20);
}

//--------------------------------------------------------------
// exit() runs when the window closes: a summary in the console (also used by `make test`).
void ofApp::exit() {
	ofLogNotice() << "done: " << messages << " messages received, " << hands.size() << " hands on screen";
}

void ofApp::keyPressed(int key) {
	if (key == 'q' || key == OF_KEY_ESC) ofExit();
}
