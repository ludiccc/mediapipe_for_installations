#include "ofApp.h"

//--------------------------------------------------------------
// setup(): window title, frame rate, background colour, and start listening.
// ofxOscReceiver::setup(port) opens the UDP socket; messages queue up until
// update() reads them.
void ofApp::setup() {
	ofSetWindowTitle("blobs (OSC)");
	ofSetFrameRate(60);
	ofBackground(20);
	receiver.setup(PORT);
	ofLogNotice() << "listening for OSC on port " << PORT << " - run mediapipe_osc.py --blobs";
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
	for (auto it = blobs.begin(); it != blobs.end();) {
		if (now - it->second.lastSeen > STALE_SECONDS) it = blobs.erase(it);
		else ++it;
	}
}

//--------------------------------------------------------------
// blobs[id] creates a default Blob when the id is new.  Returns a reference.
Blob& ofApp::getBlob(int id) {
	Blob& b = blobs[id];
	b.id = id;
	b.lastSeen = ofGetElapsedTimef();
	return b;
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
		if (m.getArgAsString(0) == "stopped") { blobs.clear(); order.clear(); }
		return;
	}
	// /mp/blobs count id id ...: every frame, the ids of the blobs on screen, biggest
	// first.  Argument 0 is the count, then the ids.
	if (addr == "/mp/blobs") {
		int n = m.getArgAsInt32(0);
		order.clear();
		for (int i = 0; i < n && (size_t)(1 + i) < m.getNumArgs(); i++) order.push_back(m.getArgAsInt32(1 + i));
		return;
	}
	if (addr.compare(0, 9, "/mp/blob/") != 0) return;   // poses, hands, faces ... are ignored here

	int id = m.getArgAsInt32(0);
	// enter / leave: blobs appearing and disappearing.  Ids are never reused.
	if (addr == "/mp/blob/enter") {
		blobs[id] = Blob();
		getBlob(id);
		ofLogNotice() << "blob " << id << " entered";
		return;
	}
	if (addr == "/mp/blob/leave") {
		blobs.erase(id);
		ofLogNotice() << "blob " << id << " left";
		return;
	}

	// Data messages.  center: the centroid - we work out its zone and log only
	// when it differs from the previous one, so a position stream becomes a
	// "changed zone" event; bbox, velocity: like the other detectors; info: the
	// area as a fraction of the picture (0.01 = 1 %), the angle of the main axis in
	// degrees and the length / width of the equivalent ellipse; contour: 16 points,
	// 2 numbers each after the id.
	Blob& b = getBlob(id);
	if (addr == "/mp/blob/center") {
		b.center = glm::vec2(m.getArgAsFloat(1), m.getArgAsFloat(2));
		if (b.col() != b.zoneCol || b.row() != b.zoneRow) {      // moved to another zone -> an event
			b.zoneCol = b.col();
			b.zoneRow = b.row();
			ofLogNotice() << "blob " << id << " is now in zone column " << b.zoneCol << ", row " << b.zoneRow;
		}
	} else if (addr == "/mp/blob/bbox") {
		for (int i = 0; i < 4; i++) b.bbox[i] = m.getArgAsFloat(1 + i);
	} else if (addr == "/mp/blob/velocity") {
		b.velocity = glm::vec2(m.getArgAsFloat(1), m.getArgAsFloat(2));
		b.speed = m.getArgAsFloat(3);
	} else if (addr == "/mp/blob/info") {
		b.area = m.getArgAsFloat(1);
		b.angle = m.getArgAsFloat(2);
		b.length = m.getArgAsFloat(3);
		b.width = m.getArgAsFloat(4);
	} else if (addr == "/mp/blob/contour") {
		size_t n = (m.getNumArgs() - 1) / 2;
		if (n < 3) return;
		b.contour.resize(n);
		for (size_t i = 0; i < n; i++) b.contour[i] = glm::vec2(m.getArgAsFloat(1 + 2 * i), m.getArgAsFloat(2 + 2 * i));
	}
}

//--------------------------------------------------------------
// A colour per id: the hue comes from the id (ofColor::fromHsb takes 0..255).
ofColor ofApp::colorFor(int id) {
	return ofColor::fromHsb((id * 47) % 255, 200, 255);
}

// draw(): only reads the state filled by update().  Every coordinate is
// normalized (0..1) and multiplied by the window size here.
void ofApp::draw() {
	float w = ofGetWidth(), h = ofGetHeight();

	// For every cell: is there a blob whose col()/row() is this cell?  Fill it if so, always draw its border.
	// zones: light up the cells that contain a blob
	float cw = w / GRID, ch = h / GRID;
	for (int col = 0; col < GRID; col++) {
		for (int row = 0; row < GRID; row++) {
			bool occupied = false;
			for (auto& kv : blobs) if (kv.second.col() == col && kv.second.row() == row) occupied = true;
			if (occupied) {
				ofFill();
				ofSetColor(70, 45, 45);
				ofDrawRectangle(col * cw, row * ch, cw, ch);
			}
			ofNoFill();
			ofSetLineWidth(1);
			ofSetColor(60);
			ofDrawRectangle(col * cw, row * ch, cw, ch);
		}
	}

	for (auto& kv : blobs) {
		const Blob& b = kv.second;
		ofColor c = colorFor(b.id);
		// The contour twice: filled with a transparent colour (ofBeginShape / ofVertex /
		// ofEndShape(true) closes the shape) and outlined with an ofPolyline.
		if (!b.contour.empty()) {
			ofPolyline outline;
			for (const glm::vec2& p : b.contour) outline.addVertex(p.x * w, p.y * h);
			outline.close();
			ofSetColor(c, 90);
			ofFill();
			ofBeginShape();
			for (const glm::vec2& p : b.contour) ofVertex(p.x * w, p.y * h);
			ofEndShape(true);
			ofSetColor(c);
			ofSetLineWidth(2);
			outline.draw();
		}
		ofNoFill();
		ofSetLineWidth(1);
		ofSetColor(c);
		ofDrawRectangle(b.bbox[0] * w, b.bbox[1] * h, (b.bbox[2] - b.bbox[0]) * w, (b.bbox[3] - b.bbox[1]) * h);

		float cx = b.center.x * w, cy = b.center.y * h;
		ofFill();
		ofDrawCircle(cx, cy, 6);
		// A white line through the centroid in the direction of the main axis, as long as the blob.
		// main axis (angle is clockwise on screen, length normalized)
		float dx = std::cos(ofDegToRad(b.angle)) * b.length / 2 * w, dy = std::sin(ofDegToRad(b.angle)) * b.length / 2 * h;
		ofSetColor(255);
		ofSetLineWidth(1);
		ofDrawLine(cx - dx, cy - dy, cx + dx, cy + dy);
		ofSetColor(c);
		ofSetLineWidth(2);
		ofDrawLine(cx, cy, cx + b.velocity.x * w * 0.25f, cy + b.velocity.y * h * 0.25f);

		// Position in the biggest-first list (1 = the biggest blob on screen, 0 = not listed yet).
		int rank = 0;
		for (size_t i = 0; i < order.size(); i++) if (order[i] == b.id) rank = (int)i + 1;
		std::string label = "B" + ofToString(b.id) + "  #" + ofToString(rank) + "  " + ofToString(b.area * 100, 1) + "%  "
		                    + ofToString(b.angle, 0) + " deg  zone (" + ofToString(b.col()) + ", " + ofToString(b.row()) + ")";
		ofDrawBitmapString(label, b.bbox[0] * w, std::max(14.0f, b.bbox[1] * h - 6));
	}

	std::string ids;
	for (int id : order) ids += ofToString(id) + " ";
	ofSetColor(255);
	ofDrawBitmapString("blobs: " + ofToString(blobs.size()) + "   biggest first: " + ids, 10, h - 12);
	ofSetColor(200);
	ofDrawBitmapString(status + "   frame " + ofToString(frameNumber) + "   analysis " + ofToString(analysisFps, 1)
	                   + " fps   messages " + ofToString(messages), 10, 20);
}

//--------------------------------------------------------------
// exit() runs when the window closes: a summary in the console (also used by `make test`).
void ofApp::exit() {
	ofLogNotice() << "done: " << messages << " messages received, " << blobs.size() << " blobs on screen";
}

void ofApp::keyPressed(int key) {
	if (key == 'q' || key == OF_KEY_ESC) ofExit();
}
