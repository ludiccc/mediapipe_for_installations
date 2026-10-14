#include "ofApp.h"

// Names for the landmark indices: wrist (0), then four points per finger (thumb
// 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20); the last of each group
// is the fingertip.  CONNECTIONS are the pairs joined by a line.
static const int WRIST = 0, THUMB_TIP = 4, INDEX_TIP = 8, MIDDLE_TIP = 12, RING_TIP = 16, PINKY_TIP = 20;
static const int FINGERTIPS[] = {THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP};
static const int CONNECTIONS[][2] = {        // MediaPipe's HAND_CONNECTIONS
	{0, 1}, {1, 2}, {2, 3}, {3, 4},              // thumb
	{0, 5}, {5, 6}, {6, 7}, {7, 8},              // index
	{5, 9}, {9, 10}, {10, 11}, {11, 12},         // middle
	{9, 13}, {13, 14}, {14, 15}, {15, 16},       // ring
	{13, 17}, {0, 17}, {17, 18}, {18, 19}, {19, 20}   // pinky + palm
};
static const int N_CONNECTIONS = sizeof(CONNECTIONS) / sizeof(CONNECTIONS[0]);

// True for the five fingertip indices (drawn bigger).
static bool isFingertip(int i) {
	for (int f : FINGERTIPS) if (f == i) return true;
	return false;
}

//--------------------------------------------------------------
// setup(): window title, frame rate, background colour, and start listening.
// ofxOscReceiver::setup(port) opens the UDP socket; messages queue up until
// update() reads them.
void ofApp::setup() {
	ofSetWindowTitle("hand landmarks (OSC)");
	ofSetFrameRate(60);
	ofBackground(20);
	receiver.setup(PORT);
	ofLogNotice() << "listening for OSC on port " << PORT << " - run mediapipe_osc.py --hands";
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
// hands[id] on a std::map creates a default Hand when the id is new, so a
// receiver started after the app still works.  Returns a reference.
Hand& ofApp::getHand(int id) {
	Hand& h = hands[id];                // creates it if we did not see it enter
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
		ofLogNotice() << "mediapipe_osc.py " << m.getArgAsString(0);
		if (m.getArgAsString(0) == "stopped") hands.clear();
		return;
	}
	// addr.compare(0, 9, "/mp/hand/") compares the first 9 characters: the prefix test.
	if (addr.compare(0, 9, "/mp/hand/") != 0) return;   // poses, faces, blobs ... are ignored here

	int id = m.getArgAsInt32(0);
	// enter / leave: hands appearing and disappearing.  Ids are never reused.
	if (addr == "/mp/hand/enter") {
		hands[id] = Hand();
		getHand(id);
		ofLogNotice() << "hand " << id << " entered";
		return;
	}
	if (addr == "/mp/hand/leave") {
		hands.erase(id);
		ofLogNotice() << "hand " << id << " left";
		return;
	}

	// Data messages.  info: "Left" / "Right" and the confidence; center, bbox,
	// velocity like the other detectors; pinch and landmarks below.
	Hand& h = getHand(id);
	if (addr == "/mp/hand/info") {
		h.handedness = m.getArgAsString(1);
		h.score = m.getArgAsFloat(2);
	} else if (addr == "/mp/hand/center") {
		h.center = glm::vec3(m.getArgAsFloat(1), m.getArgAsFloat(2), m.getArgAsFloat(3));
	} else if (addr == "/mp/hand/bbox") {
		for (int i = 0; i < 4; i++) h.bbox[i] = m.getArgAsFloat(1 + i);
	} else if (addr == "/mp/hand/velocity") {
		h.velocity = glm::vec2(m.getArgAsFloat(1), m.getArgAsFloat(2));
		h.speed = m.getArgAsFloat(3);
	// pinch arrives every frame: compare with the threshold and only log when the
	// state changes - a continuous value becomes an event (pinch! / released).
	} else if (addr == "/mp/hand/pinch") {
		h.pinch = m.getArgAsFloat(1);
		bool pinching = h.pinch < PINCH_CLOSED;
		if (pinching != h.pinching) {                  // an event you could trigger something with
			h.pinching = pinching;
			ofLogNotice() << "hand " << id << " (" << h.handedness << ") " << (pinching ? "pinch!" : "released");
		}
	// landmarks: argument 0 is the id, then 3 numbers per point (the i-th point's x is argument 1 + i * 3).
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

	for (auto& kv : hands) {
		const Hand& hd = kv.second;
		// One colour per handedness.  Then: box, skeleton lines, dots (fingertips bigger), pinch circle, arrow, label.
		ofColor c = hd.handedness == "Right" ? ofColor(255, 200, 80) : ofColor(60, 160, 255);

		ofNoFill();
		ofSetLineWidth(1);
		ofSetColor(c);
		ofDrawRectangle(hd.bbox[0] * w, hd.bbox[1] * h, (hd.bbox[2] - hd.bbox[0]) * w, (hd.bbox[3] - hd.bbox[1]) * h);

		if (hd.landmarks.size() == 21) {
			ofSetLineWidth(2);
			for (int k = 0; k < N_CONNECTIONS; k++) {
				const glm::vec3& a = hd.landmarks[CONNECTIONS[k][0]];
				const glm::vec3& b = hd.landmarks[CONNECTIONS[k][1]];
				ofDrawLine(a.x * w, a.y * h, b.x * w, b.y * h);
			}
			ofFill();
			for (int i = 0; i < 21; i++) {
				ofDrawCircle(hd.landmarks[i].x * w, hd.landmarks[i].y * h, isFingertip(i) ? 6 : 3);
			}
			// Its radius follows the pinch distance (x 400 px, at least 4).
			// the pinch: a circle between thumb tip and index tip, filled when closed
			const glm::vec3& t = hd.landmarks[THUMB_TIP];
			const glm::vec3& ix = hd.landmarks[INDEX_TIP];
			float mx = (t.x + ix.x) / 2 * w, my = (t.y + ix.y) / 2 * h;
			ofSetColor(255);
			ofSetLineWidth(1);
			if (hd.pinching) ofFill(); else ofNoFill();
			ofDrawCircle(mx, my, std::max(4.0f, hd.pinch * 400));
		}

		float cx = hd.center.x * w, cy = hd.center.y * h;
		ofSetColor(c);
		ofSetLineWidth(2);
		ofDrawLine(cx, cy, cx + hd.velocity.x * w * 0.25f, cy + hd.velocity.y * h * 0.25f);
		std::string label = "H" + ofToString(hd.id) + " " + hd.handedness + " (" + ofToString(hd.score, 2) + ")  pinch "
		                    + ofToString(hd.pinch, 3) + (hd.pinching ? "  PINCH" : "");
		ofDrawBitmapString(label, hd.bbox[0] * w, std::max(14.0f, hd.bbox[1] * h - 6));
	}

	ofSetColor(255);
	ofDrawBitmapString("hands: " + ofToString(hands.size()), 10, h - 12);
	ofSetColor(200);
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
