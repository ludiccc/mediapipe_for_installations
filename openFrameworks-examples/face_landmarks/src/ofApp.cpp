#include "ofApp.h"

// A few useful landmark indices of the 478-point mesh.
// The mesh has 478 points; these are the handful we use by name.  The eye corners
// give the head roll; FACE_OVAL is drawn as an outline; KEY_POINTS as white dots.
static const int NOSE_TIP = 1, CHIN = 152, FOREHEAD = 10;
static const int LEFT_EYE_OUTER = 263, RIGHT_EYE_OUTER = 33;   // "left" = the person's left
static const int MOUTH_LEFT = 291, MOUTH_RIGHT = 61, LIP_TOP = 13, LIP_BOTTOM = 14;
static const int FACE_OVAL[] = {10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
	152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109};
static const int KEY_POINTS[] = {LEFT_EYE_OUTER, RIGHT_EYE_OUTER, MOUTH_LEFT, MOUTH_RIGHT, LIP_TOP, LIP_BOTTOM, CHIN, FOREHEAD};
// The 52 blendshape names in the order the values arrive.  The app also sends
// the list at start-up (/mp/face/blendshape_names) and we take that one if it
// comes.  SHOWN_BLENDSHAPES are the few drawn as bars (pick any of the 52).
static const char* DEFAULT_BLENDSHAPE_NAMES[] = {
	"_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
	"browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft",
	"eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
	"eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
	"eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
	"jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
	"mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
	"mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
	"mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
	"mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
	"noseSneerLeft", "noseSneerRight"
};
static const char* SHOWN_BLENDSHAPES[] = {"jawOpen", "mouthSmileLeft", "mouthSmileRight", "eyeBlinkLeft", "eyeBlinkRight",
	"browInnerUp", "mouthPucker", "cheekPuff"};

// Head roll (tilt) from the line between the outer eye corners, 0 = upright.
// atan2 of the line between the two outer eye corners: 0 when the eyes are
// level, positive / negative when the head tilts.  An installation uses it to
// rotate an image placed over the face.
float Face::rollDegrees() const {
	if (landmarks.size() <= (size_t)LEFT_EYE_OUTER) return 0;
	const glm::vec3& l = landmarks[LEFT_EYE_OUTER];
	const glm::vec3& r = landmarks[RIGHT_EYE_OUTER];
	return ofRadToDeg(std::atan2(l.y - r.y, l.x - r.x));
}

// A blendshape by name: find its position in the names, return the value there (0 if unknown).
float Face::value(const std::vector<std::string>& names, const std::string& name) const {
	for (size_t i = 0; i < names.size() && i < blendshapes.size(); i++) {
		if (names[i] == name) return blendshapes[i];
	}
	return 0;
}

//--------------------------------------------------------------
// setup(): window title, frame rate, background colour, and start listening.
// ofxOscReceiver::setup(port) opens the UDP socket; messages queue up until
// update() reads them.
// The default blendshape names are copied into the vector that the messages may later replace.
void ofApp::setup() {
	ofSetWindowTitle("face landmarks (OSC)");
	ofSetFrameRate(60);
	ofBackground(20);
	for (const char* n : DEFAULT_BLENDSHAPE_NAMES) blendshapeNames.push_back(n);
	receiver.setup(PORT);
	ofLogNotice() << "listening for OSC on port " << PORT << " - run mediapipe_osc.py --face --face-landmarks";
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
	for (auto it = faces.begin(); it != faces.end();) {
		if (now - it->second.lastSeen > STALE_SECONDS) it = faces.erase(it);
		else ++it;
	}
}

//--------------------------------------------------------------
// faces[id] creates a default Face when the id is new.  Returns a reference.
Face& ofApp::getFace(int id) {
	Face& f = faces[id];
	f.id = id;
	f.lastSeen = ofGetElapsedTimef();
	return f;
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
		if (m.getArgAsString(0) == "stopped") faces.clear();
		return;
	}
	// /mp/face/blendshape_names: 52 strings sent once at start-up; they replace the built-in list.
	if (addr == "/mp/face/blendshape_names") {
		blendshapeNames.clear();
		for (size_t i = 0; i < m.getNumArgs(); i++) blendshapeNames.push_back(m.getArgAsString(i));
		return;
	}
	if (addr.compare(0, 9, "/mp/face/") != 0) return;   // poses, hands, blobs ... are ignored here

	int id = m.getArgAsInt32(0);
	// enter / leave: faces appearing and disappearing.  Ids are never reused.
	if (addr == "/mp/face/enter") {
		faces[id] = Face();
		getFace(id);
		ofLogNotice() << "face " << id << " entered";
		return;
	}
	if (addr == "/mp/face/leave") {
		faces.erase(id);
		ofLogNotice() << "face " << id << " left";
		return;
	}

	// Data messages.  center: the nose tip; bbox: the rectangle around the face;
	// velocity: movement of the nose in picture sizes per second; then the two below.
	Face& f = getFace(id);
	if (addr == "/mp/face/center") {
		f.center = glm::vec3(m.getArgAsFloat(1), m.getArgAsFloat(2), m.getArgAsFloat(3));
	} else if (addr == "/mp/face/bbox") {
		for (int i = 0; i < 4; i++) f.bbox[i] = m.getArgAsFloat(1 + i);
	} else if (addr == "/mp/face/velocity") {
		f.velocity = glm::vec2(m.getArgAsFloat(1), m.getArgAsFloat(2));
		f.speed = m.getArgAsFloat(3);
	// blendshapes: argument 0 is the id, then the 52 values.  After storing them the
	// jawOpen threshold is compared with the previous state: the "mouth open /
	// closed" event.  Replace the log line by your own reaction.
	} else if (addr == "/mp/face/blendshapes") {
		f.blendshapes.resize(m.getNumArgs() - 1);
		for (size_t i = 1; i < m.getNumArgs(); i++) f.blendshapes[i - 1] = m.getArgAsFloat(i);
		bool open = f.value(blendshapeNames, "jawOpen") > MOUTH_OPEN;
		if (open != f.mouthOpen) {                       // an event
			f.mouthOpen = open;
			ofLogNotice() << "face " << id << " mouth " << (open ? "open" : "closed");
		}
	// landmarks: the mesh, 3 numbers per point after the id (only with --face-landmarks).
	} else if (addr == "/mp/face/landmarks") {
		size_t n = (m.getNumArgs() - 1) / 3;
		if (n < 468) return;
		f.landmarks.resize(n);
		for (size_t i = 0; i < n; i++) {
			f.landmarks[i] = glm::vec3(m.getArgAsFloat(1 + i * 3), m.getArgAsFloat(2 + i * 3), m.getArgAsFloat(3 + i * 3));
		}
		meshReceived = true;
	}
}

//--------------------------------------------------------------
// A colour per id: the hue comes from the id (ofColor::fromHsb takes 0..255).
ofColor ofApp::colorFor(int id) {
	return ofColor::fromHsb((id * 47) % 255, 160, 255);
}

// draw(): only reads the state filled by update().  Every coordinate is
// normalized (0..1) and multiplied by the window size here.
void ofApp::draw() {
	float w = ofGetWidth(), h = ofGetHeight();

	for (auto& kv : faces) {
		const Face& f = kv.second;
		ofColor c = colorFor(f.id);
		float x0 = f.bbox[0] * w, y0 = f.bbox[1] * h, x1 = f.bbox[2] * w, y1 = f.bbox[3] * h;
		ofNoFill();
		ofSetLineWidth(1);
		ofSetColor(c);
		ofDrawRectangle(x0, y0, x1 - x0, y1 - y0);

		// ofPushMatrix / ofTranslate / ofRotateDeg / ofPopMatrix: move the origin to
		// the centre of the face and turn the axes by the roll, draw at (0, 0), then
		// restore.  Anything drawn between ofTranslate and ofPopMatrix follows the head.
		// the "mask": an ellipse the size of the face, rotated with the head.
		// An installation would draw its own image here instead (ofImage::draw inside the same push/pop).
		float roll = f.rollDegrees();
		ofPushMatrix();
		ofTranslate((x0 + x1) / 2, (y0 + y1) / 2);
		ofRotateDeg(roll);
		ofFill();
		ofSetColor(60);
		ofDrawEllipse(0, 0, x1 - x0, y1 - y0);
		ofNoFill();
		ofSetLineWidth(2);
		ofSetColor(c);
		ofDrawEllipse(0, 0, x1 - x0, y1 - y0);
		ofPopMatrix();

		// With the mesh: every point as a tiny dot, the face oval as an ofPolyline, the key points and the eye line.
		if (!f.landmarks.empty()) {
			ofFill();
			for (const glm::vec3& p : f.landmarks) ofDrawCircle(p.x * w, p.y * h, 1);    // the mesh
			ofPolyline oval;                                                             // the face oval
			for (int i : FACE_OVAL) oval.addVertex(f.landmarks[i].x * w, f.landmarks[i].y * h);
			oval.close();
			ofSetLineWidth(1);
			oval.draw();
			ofSetColor(255);
			for (int i : KEY_POINTS) ofDrawCircle(f.landmarks[i].x * w, f.landmarks[i].y * h, 4);
			const glm::vec3& l = f.landmarks[LEFT_EYE_OUTER];
			const glm::vec3& r = f.landmarks[RIGHT_EYE_OUTER];
			ofDrawLine(l.x * w, l.y * h, r.x * w, r.y * h);
		}

		float nx = f.center.x * w, ny = f.center.y * h;                                  // nose tip
		ofFill();
		ofSetColor(255, 0, 0);
		ofDrawCircle(nx, ny, 5);
		ofSetColor(c);
		ofSetLineWidth(2);
		ofDrawLine(nx, ny, nx + f.velocity.x * w * 0.25f, ny + f.velocity.y * h * 0.25f);

		std::string label = "F" + ofToString(f.id) + "  roll " + ofToString(roll, 0) + " deg" + (f.mouthOpen ? "  MOUTH OPEN" : "");
		ofDrawBitmapString(label, x0, std::max(14.0f, y0 - 6));

		// Each bar is 80 px long at value 1; the label is the blendshape name.
		// blendshape bars, to the right of the face (or to the left near the border)
		float bx = x1 + 150 < w ? x1 + 10 : x0 - 150;
		int k = 0;
		for (const char* name : SHOWN_BLENDSHAPES) {
			float v = f.value(blendshapeNames, name);
			float y = y0 + k++ * 14;
			ofFill();
			ofSetColor(c);
			ofDrawRectangle(bx, y, 80 * v, 8);
			ofNoFill();
			ofSetColor(90);
			ofDrawRectangle(bx, y, 80, 8);
			ofSetColor(200);
			ofDrawBitmapString(name, bx + 86, y + 8);
		}
	}

	ofSetColor(255);
	ofDrawBitmapString("faces: " + ofToString(faces.size()) + "   "
	                   + (meshReceived ? "mesh: yes" : "mesh: no (start the app with --face-landmarks; see README about the buffer size)"), 10, h - 12);
	ofSetColor(200);
	ofDrawBitmapString(status + "   frame " + ofToString(frameNumber) + "   analysis " + ofToString(analysisFps, 1)
	                   + " fps   messages " + ofToString(messages), 10, 20);
}

//--------------------------------------------------------------
// exit() runs when the window closes: a summary in the console (also used by `make test`).
void ofApp::exit() {
	ofLogNotice() << "done: " << messages << " messages received, " << faces.size() << " faces on screen, mesh received: " << (meshReceived ? "yes" : "no");
}

void ofApp::keyPressed(int key) {
	if (key == 'q' || key == OF_KEY_ESC) ofExit();
}
