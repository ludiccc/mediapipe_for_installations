// The entry point of every openFrameworks app: open a window of the given size
// (OF_WINDOW = windowed, OF_FULLSCREEN = full screen) and hand control to ofApp,
// whose setup() / update() / draw() do the work (see ofApp.h / ofApp.cpp).
#include "ofMain.h"
#include "ofApp.h"

int main() {
	ofSetupOpenGL(800, 600, OF_WINDOW);   // 4:3 like the default 640x480 camera
	ofRunApp(new ofApp());
}
