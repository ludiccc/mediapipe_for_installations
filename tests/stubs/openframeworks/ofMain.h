// Stand-in for openFrameworks' ofMain.h, used only by tests/test_examples.py to
// compile and run the apps of openFrameworks-examples/ without openFrameworks.
// Drawing calls do nothing; ofRunApp() calls setup(), then update()/draw() at
// ~30 fps for OF_STUB_SECONDS seconds (default 3), then exit().
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace glm {
struct vec2 {
	float x = 0, y = 0;
	vec2() {}
	vec2(float x_, float y_) : x(x_), y(y_) {}
};
struct vec3 {
	float x = 0, y = 0, z = 0;
	vec3() {}
	vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
	vec3(const vec2& v, float z_ = 0) : x(v.x), y(v.y), z(z_) {}
};
inline vec2 operator-(const vec2& a, const vec2& b) { return vec2(a.x - b.x, a.y - b.y); }
inline vec2 operator+(const vec2& a, const vec2& b) { return vec2(a.x + b.x, a.y + b.y); }
inline vec2 operator*(const vec2& a, float s) { return vec2(a.x * s, a.y * s); }
inline vec3 operator-(const vec3& a, const vec3& b) { return vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline float length(const vec2& v) { return std::sqrt(v.x * v.x + v.y * v.y); }
inline float length(const vec3& v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
inline float distance(const vec2& a, const vec2& b) { return length(a - b); }
inline float distance(const vec3& a, const vec3& b) { return length(a - b); }
}  // namespace glm

enum { OF_WINDOW = 0, OF_FULLSCREEN = 1, OF_KEY_ESC = 27, OF_KEY_LEFT = 356, OF_KEY_RIGHT = 358 };

struct ofColor {
	float r = 255, g = 255, b = 255, a = 255;
	ofColor() {}
	ofColor(float gray) : r(gray), g(gray), b(gray) {}
	ofColor(float r_, float g_, float b_, float a_ = 255) : r(r_), g(g_), b(b_), a(a_) {}
	static ofColor fromHsb(float h, float s, float b, float a = 255) { return ofColor(h, s, b, a); }
	ofColor& setBrightness(float v) { return *this; }
	ofColor lerp(const ofColor& o, float t) const { return *this; }
	static const ofColor white, black, red;
};
inline const ofColor ofColor::white(255), ofColor::black(0), ofColor::red(255, 0, 0);

struct ofLogStream {
	std::ostringstream s;
	bool print;
	ofLogStream(bool p = true) : print(p) {}
	template <class T> ofLogStream& operator<<(const T& v) { s << v; return *this; }
	~ofLogStream() { if (print) std::cout << s.str() << std::endl; }
};
inline ofLogStream ofLogNotice() { return ofLogStream(true); }
inline ofLogStream ofLogNotice(const std::string&) { return ofLogStream(true); }
inline ofLogStream ofLogWarning() { return ofLogStream(true); }
inline ofLogStream ofLogError() { return ofLogStream(true); }
inline ofLogStream ofLogVerbose() { return ofLogStream(false); }

namespace of_stub {
inline int width = 800, height = 600;
inline std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
inline bool exitRequested = false;
inline int drawCalls = 0;
}

inline float ofGetElapsedTimef() {
	return std::chrono::duration<float>(std::chrono::steady_clock::now() - of_stub::start).count();
}
inline unsigned long long ofGetElapsedTimeMillis() { return (unsigned long long)(ofGetElapsedTimef() * 1000); }
inline int ofGetWidth() { return of_stub::width; }
inline int ofGetHeight() { return of_stub::height; }
inline float ofGetFrameRate() { return 30; }
inline int ofGetFrameNum() { return of_stub::drawCalls; }
inline void ofSetFrameRate(int) {}
inline void ofSetWindowTitle(const std::string&) {}
inline void ofSetVerticalSync(bool) {}
inline void ofEnableAlphaBlending() {}
inline void ofEnableAntiAliasing() {}
inline void ofEnableSmoothing() {}
inline void ofBackground(int) {}
inline void ofBackground(int, int, int) {}
inline void ofBackground(const ofColor&) {}
inline void ofSetColor(int) {}
inline void ofSetColor(int, int) {}
inline void ofSetColor(int, int, int) {}
inline void ofSetColor(int, int, int, int) {}
inline void ofSetColor(const ofColor&) {}
inline void ofSetColor(const ofColor&, int) {}
inline void ofFill() {}
inline void ofNoFill() {}
inline void ofSetLineWidth(float) {}
inline void ofSetCircleResolution(int) {}
inline void ofDrawRectangle(float, float, float, float) {}
inline void ofDrawRectRounded(float, float, float, float, float) {}
inline void ofDrawCircle(float, float, float) {}
inline void ofDrawCircle(const glm::vec2&, float) {}
inline void ofDrawCircle(const glm::vec3&, float) {}
inline void ofDrawEllipse(float, float, float, float) {}
inline void ofDrawLine(float, float, float, float) {}
inline void ofDrawLine(const glm::vec2&, const glm::vec2&) {}
inline void ofDrawLine(const glm::vec3&, const glm::vec3&) {}
inline void ofDrawTriangle(float, float, float, float, float, float) {}
inline void ofDrawBitmapString(const std::string&, float, float) {}
inline void ofDrawBitmapStringHighlight(const std::string&, float, float) {}
inline void ofDrawBitmapStringHighlight(const std::string&, float, float, const ofColor&, const ofColor&) {}
inline void ofBeginShape() {}
inline void ofVertex(float, float) {}
inline void ofVertex(const glm::vec2&) {}
inline void ofEndShape(bool = false) {}
inline void ofPushMatrix() {}
inline void ofPopMatrix() {}
inline void ofPushStyle() {}
inline void ofPopStyle() {}
inline void ofTranslate(float, float, float = 0) {}
inline void ofRotateDeg(float) {}
inline void ofRotateZDeg(float) {}
inline void ofScale(float, float, float = 1) {}
inline float ofClamp(float v, float lo, float hi) { return std::max(lo, std::min(hi, v)); }
inline float ofMap(float v, float a, float b, float c, float d, bool clamp = false) {
	float r = c + (d - c) * ((v - a) / (b - a));
	return clamp ? ofClamp(r, std::min(c, d), std::max(c, d)) : r;
}
inline float ofLerp(float a, float b, float t) { return a + (b - a) * t; }
inline float ofRandom(float hi) { return hi * (float)rand() / RAND_MAX; }
inline float ofRandom(float lo, float hi) { return lo + (hi - lo) * (float)rand() / RAND_MAX; }
inline float ofDist(float x1, float y1, float x2, float y2) { return std::hypot(x2 - x1, y2 - y1); }
inline float ofRadToDeg(float r) { return r * 180.0f / 3.14159265f; }
inline float ofDegToRad(float d) { return d * 3.14159265f / 180.0f; }
inline void ofExit(int = 0) { of_stub::exitRequested = true; }

template <class T> std::string ofToString(const T& v) { std::ostringstream s; s << v; return s.str(); }
inline std::string ofToString(float v, int precision) { char b[64]; snprintf(b, sizeof b, "%.*f", precision, v); return b; }
inline std::string ofToString(double v, int precision) { char b[64]; snprintf(b, sizeof b, "%.*f", precision, v); return b; }
inline float ofToFloat(const std::string& s) { return std::stof(s); }
inline int ofToInt(const std::string& s) { return std::stoi(s); }

struct ofPolyline {
	std::vector<glm::vec3> pts;
	bool closed = false;
	void addVertex(float x, float y, float z = 0) { pts.push_back(glm::vec3(x, y, z)); }
	void addVertex(const glm::vec3& p) { pts.push_back(p); }
	void addVertex(const glm::vec2& p) { pts.push_back(glm::vec3(p.x, p.y, 0)); }
	void close() { closed = true; }
	void clear() { pts.clear(); }
	size_t size() const { return pts.size(); }
	void draw() const {}
};

struct ofPath {
	void moveTo(float, float) {}
	void lineTo(float, float) {}
	void close() {}
	void setFillColor(const ofColor&) {}
	void setFilled(bool) {}
	void draw() {}
};

struct ofImage {
	bool load(const std::string&) { return false; }
	void draw(float, float, float, float) {}
	bool isAllocated() const { return false; }
};

struct ofTrueTypeFont {
	bool load(const std::string&, int) { return false; }
	void drawString(const std::string&, float, float) {}
};

class ofBaseApp {
public:
	virtual ~ofBaseApp() {}
	virtual void setup() {}
	virtual void update() {}
	virtual void draw() {}
	virtual void exit() {}
	virtual void keyPressed(int) {}
	virtual void keyReleased(int) {}
	virtual void mouseMoved(int, int) {}
	virtual void mouseDragged(int, int, int) {}
	virtual void mousePressed(int, int, int) {}
	virtual void mouseReleased(int, int, int) {}
	virtual void windowResized(int, int) {}
};

inline void ofSetupOpenGL(int w, int h, int) { of_stub::width = w; of_stub::height = h; }

void of_stub_summary(ofBaseApp* app);   // defined in ofxOsc.h (needs the receiver)

inline void ofRunApp(ofBaseApp* app) {
	float seconds = 3;
	if (const char* s = std::getenv("OF_STUB_SECONDS")) seconds = std::atof(s);
	app->setup();
	auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds((int)(seconds * 1000));
	while (!of_stub::exitRequested && std::chrono::steady_clock::now() < end) {
		app->update();
		app->draw();
		of_stub::drawCalls++;
		std::this_thread::sleep_for(std::chrono::milliseconds(33));
	}
	app->exit();
	std::cout << "### draw() called " << of_stub::drawCalls << " times" << std::endl;
	of_stub_summary(app);
	delete app;
}
