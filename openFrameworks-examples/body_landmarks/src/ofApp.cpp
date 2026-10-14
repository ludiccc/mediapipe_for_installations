#include "ofApp.h"

// Landmark indices and the lines between them (MediaPipe's POSE_CONNECTIONS).
// Names for the landmark indices we use (0 nose ... 32 right foot index, see the
// MediaPipe link in ofApp.h), and the pairs joined by a line to draw the skeleton.
// `static` keeps them private to this file.
static const int NOSE = 0, L_SHOULDER = 11, R_SHOULDER = 12, L_WRIST = 15, R_WRIST = 16;
static const int CONNECTIONS[][2] = {
	{0, 1}, {0, 4}, {1, 2}, {2, 3}, {3, 7}, {4, 5}, {5, 6}, {6, 8}, {9, 10},
	{11, 12}, {11, 13}, {11, 23}, {12, 14}, {12, 24}, {13, 15}, {14, 16},
	{15, 17}, {15, 19}, {15, 21}, {16, 18}, {16, 20}, {16, 22}, {17, 19}, {18, 20},
	{23, 24}, {23, 25}, {24, 26}, {25, 27}, {26, 28}, {27, 29}, {27, 31}, {28, 30}, {28, 32},
	{29, 31}, {30, 32}
};
static const int N_CONNECTIONS = sizeof(CONNECTIONS) / sizeof(CONNECTIONS[0]);

// The visibility (0..1) says how sure MediaPipe is that the wrist is in the picture.
// True when a wrist is above the shoulders (y grows downwards).
bool Person::handRaised() const {
	if (landmarks.size() < 33) return false;
	float shoulders = std::min(landmarks[L_SHOULDER].y, landmarks[R_SHOULDER].y);
	return (landmarks[L_WRIST].y < shoulders && visibility[L_WRIST] > 0.5f)
	    || (landmarks[R_WRIST].y < shoulders && visibility[R_WRIST] > 0.5f);
}

// (int)(x * 3) is 0, 1 or 2 for x in 0..1; min / max clamp values outside the picture.
int Person::zone() const {
	return std::max(0, std::min(2, (int)(center.x * 3)));
}

//--------------------------------------------------------------
// setup(): window title, frame rate, background colour, and start listening.
// ofxOscReceiver::setup(port) opens the UDP socket; messages queue up until
// update() reads them.
void ofApp::setup() {
	ofSetWindowTitle("body landmarks (OSC)");
	ofSetFrameRate(60);
	ofBackground(20);
	receiver.setup(PORT);
	ofLogNotice() << "listening for OSC on port " << PORT << " - run mediapipe_osc.py --pose";
}

//--------------------------------------------------------------
// update(): first empty the queue of messages that arrived since the last
// frame (hasWaitingMessages / getNextMessage), then drop stale objects.  A leave
// message can be lost (UDP gives no guarantee), so the timeout is a safety net.
void ofApp::update() {
	// read every message that arrived since the last frame
	while (receiver.hasWaitingMessages()) {
		ofxOscMessage m;
		receiver.getNextMessage(m);
		messages++;
		if (messages == 1) status = "receiving";
		handleMessage(m);
	}
	// forget people whose data stopped arriving (e.g. the app was killed)
	float now = ofGetElapsedTimef();
	// Erasing while iterating a std::map: erase() returns the next iterator, so
	// only advance with ++it when nothing was erased.
	for (auto it = people.begin(); it != people.end();) {
		if (now - it->second.lastSeen > STALE_SECONDS) it = people.erase(it);
		else ++it;
	}
}

//--------------------------------------------------------------
// people[id] on a std::map creates a default Person when the id is new: a
// receiver started after the app, or one that missed an enter message, still
// works.  Returns a reference, so the caller modifies the stored object.
Person& ofApp::getPerson(int id) {
	Person& p = people[id];             // creates it if we did not see it enter (started late)
	p.id = id;
	p.lastSeen = ofGetElapsedTimef();
	return p;
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
		if (m.getArgAsString(0) == "stopped") people.clear();
		return;
	}
	// addr.compare(0, 9, "/mp/pose/") compares the first 9 characters: the prefix test.
	if (addr.compare(0, 9, "/mp/pose/") != 0) return;   // hands, faces, blobs ... are ignored here

	int id = m.getArgAsInt32(0);
	// enter / leave: where an installation would start and stop something per
	// person (a sound, a particle system, a colour ...).  Ids are never reused.
	if (addr == "/mp/pose/enter") {
		people[id] = Person();
		getPerson(id);
		ofLogNotice() << "person " << id << " entered";
		return;
	}
	if (addr == "/mp/pose/leave") {
		people.erase(id);
		ofLogNotice() << "person " << id << " left";
		return;
	}

	// Data messages.  center: the visibility-weighted centre of the body; bbox:
	// the rectangle around the person; velocity: movement of the centre in picture
	// sizes per second; landmarks: the 33 points (the big one).
	Person& p = getPerson(id);
	if (addr == "/mp/pose/center") {
		p.center = glm::vec3(m.getArgAsFloat(1), m.getArgAsFloat(2), m.getArgAsFloat(3));
	} else if (addr == "/mp/pose/bbox") {
		for (int i = 0; i < 4; i++) p.bbox[i] = m.getArgAsFloat(1 + i);
	} else if (addr == "/mp/pose/velocity") {
		p.velocity = glm::vec2(m.getArgAsFloat(1), m.getArgAsFloat(2));
		p.speed = m.getArgAsFloat(3);
	} else if (addr == "/mp/pose/landmarks") {
		// Argument 0 is the id, then 4 numbers per landmark: the i-th landmark's x is
		// argument 1 + i * 4, its visibility argument 4 + i * 4.
		// 33 x (x y z visibility) after the id
		if (m.getNumArgs() < 1 + 33 * 4) return;
		p.landmarks.resize(33);
		p.visibility.resize(33);
		for (int i = 0; i < 33; i++) {
			p.landmarks[i] = glm::vec3(m.getArgAsFloat(1 + i * 4), m.getArgAsFloat(2 + i * 4), m.getArgAsFloat(3 + i * 4));
			p.visibility[i] = m.getArgAsFloat(4 + i * 4);
		}
	}
}

//--------------------------------------------------------------
// A colour per id: the hue comes from the id (ofColor::fromHsb takes 0..255).
ofColor ofApp::colorFor(int id) {
	return ofColor::fromHsb((id * 47) % 255, 200, 255);   // a different colour per id
}

// draw(): only reads the state filled by update().  Every coordinate is
// normalized (0..1) and multiplied by the window size here.
void ofApp::draw() {
	float w = ofGetWidth(), h = ofGetHeight();

	// Two vertical lines divide the window into the three zones of Person::zone().
	// the three zones
	ofSetColor(50);
	ofDrawLine(w / 3, 0, w / 3, h);
	ofDrawLine(2 * w / 3, 0, 2 * w / 3, h);
	int zoneCount[3] = {0, 0, 0};

	for (auto& kv : people) {
		const Person& p = kv.second;
		ofColor c = colorFor(p.id);
		zoneCount[p.zone()]++;

		// bounding box
		ofNoFill();
		ofSetLineWidth(1);
		ofSetColor(c);
		ofDrawRectangle(p.bbox[0] * w, p.bbox[1] * h, (p.bbox[2] - p.bbox[0]) * w, (p.bbox[3] - p.bbox[1]) * h);

		// A line for every pair in CONNECTIONS (skipped when one end is barely visible),
		// then a dot per landmark whose radius follows the visibility.
		// skeleton: lines, then points (bigger when more visible)
		if (p.landmarks.size() == 33) {
			ofSetLineWidth(2);
			for (int k = 0; k < N_CONNECTIONS; k++) {
				int a = CONNECTIONS[k][0], b = CONNECTIONS[k][1];
				if (p.visibility[a] > 0.3f && p.visibility[b] > 0.3f) {
					ofDrawLine(p.landmarks[a].x * w, p.landmarks[a].y * h, p.landmarks[b].x * w, p.landmarks[b].y * h);
				}
			}
			ofFill();
			for (int i = 0; i < 33; i++) {
				ofDrawCircle(p.landmarks[i].x * w, p.landmarks[i].y * h, 2 + 3 * p.visibility[i]);
			}
		}

		// centre, velocity arrow, label
		float cx = p.center.x * w, cy = p.center.y * h;
		ofFill();
		ofDrawCircle(cx, cy, 8);
		ofSetLineWidth(2);
		ofDrawLine(cx, cy, cx + p.velocity.x * w * 0.25f, cy + p.velocity.y * h * 0.25f);
		// ofToString(value, 2) formats with two decimals; "LCR"[zone] is a one-letter zone name.
		std::string label = "P" + ofToString(p.id) + "  speed " + ofToString(p.speed, 2) + "  zone " + "LCR"[p.zone()];
		if (p.handRaised()) label += "  HAND UP!";
		ofDrawBitmapString(label, p.bbox[0] * w, std::max(14.0f, p.bbox[1] * h - 6));
	}

	ofSetColor(255);
	ofDrawBitmapString("people: " + ofToString(people.size()) + "   left " + ofToString(zoneCount[0])
	                   + "  centre " + ofToString(zoneCount[1]) + "  right " + ofToString(zoneCount[2]), 10, h - 12);
	ofSetColor(200);
	ofDrawBitmapString(status + "   frame " + ofToString(frameNumber) + "   analysis " + ofToString(analysisFps, 1)
	                   + " fps   messages " + ofToString(messages), 10, 20);
}

//--------------------------------------------------------------
// exit() runs when the window closes: a summary in the console (also used by `make test`).
void ofApp::exit() {
	ofLogNotice() << "done: " << messages << " messages received, " << people.size() << " people on screen";
}

void ofApp::keyPressed(int key) {
	if (key == 'q' || key == OF_KEY_ESC) ofExit();
}
