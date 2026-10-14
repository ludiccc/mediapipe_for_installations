package processing.core;

// Stand-in for Processing's PApplet, used only by tests/test_examples.py to
// compile and run the sketches of processing-examples/ without Processing.
// Drawing calls are no-ops; setup() and draw() are called like Processing does
// (draw() at ~30 fps for HEADLESS_SECONDS seconds, then a summary is printed).

import java.util.Locale;

public class PApplet {
    public static final float PI = (float) Math.PI, TWO_PI = (float) (2 * Math.PI), HALF_PI = (float) (Math.PI / 2);
    public static final int RGB = 1, HSB = 3, CENTER = 3, LEFT = 37, RIGHT = 39, TOP = 101, BOTTOM = 102, CLOSE = 2;
    public static final int P2D = 0, P3D = 0;
    public int width = 100, height = 100, frameCount = 0;
    public char key = 0;
    public boolean keyPressed = false;
    public int mouseX = 0, mouseY = 0;
    private long start = System.currentTimeMillis();
    private int colorMode = RGB;
    private float colorMax = 255;
    public int drawCalls = 0;

    // --- lifecycle --------------------------------------------------------
    public void settings() {}
    public void setup() {}
    public void draw() {}
    public void exit() { stopped = true; }
    private volatile boolean stopped = false;

    public void runHeadless(float seconds) throws Exception {
        settings();
        setup();
        long end = System.currentTimeMillis() + (long) (seconds * 1000);
        while (!stopped && System.currentTimeMillis() < end) {
            frameCount++;
            draw();
            drawCalls++;
            Thread.sleep(33);
        }
    }

    // --- window -----------------------------------------------------------
    public void size(int w, int h) { width = w; height = h; }
    public void size(int w, int h, String renderer) { width = w; height = h; }
    public void size(int w, int h, int renderer) { width = w; height = h; }
    public void frameRate(float f) {}
    public int millis() { return (int) (System.currentTimeMillis() - start); }
    public void noLoop() {}
    public void loop() {}

    // --- colour -----------------------------------------------------------
    public void colorMode(int mode) { colorMode = mode; }
    public void colorMode(int mode, float max) { colorMode = mode; colorMax = max; }
    public int color(float g) { return color(g, g, g, colorMax); }
    public int color(float g, float a) { return color(g, g, g, a); }
    public int color(float r, float g, float b) { return color(r, g, b, colorMax); }
    public int color(float r, float g, float b, float a) {
        if (colorMode == HSB) {
            int rgb = java.awt.Color.HSBtoRGB(r / colorMax, g / colorMax, b / colorMax);
            return (((int) (a / colorMax * 255)) << 24) | (rgb & 0xFFFFFF);
        }
        return (((int) (a / colorMax * 255)) << 24) | (((int) (r / colorMax * 255)) << 16)
             | (((int) (g / colorMax * 255)) << 8) | ((int) (b / colorMax * 255));
    }
    public float red(int c) { return (c >> 16) & 0xFF; }
    public float green(int c) { return (c >> 8) & 0xFF; }
    public float blue(int c) { return c & 0xFF; }
    public float hue(int c) { return 0; }
    public int lerpColor(int a, int b, float t) { return a; }
    public void background(float g) {}
    public void background(float r, float g, float b) {}
    public void background(int c, float a) {}
    public void fill(float g) {}
    public void fill(float g, float a) {}
    public void fill(float r, float g, float b) {}
    public void fill(float r, float g, float b, float a) {}
    public void noFill() {}
    public void stroke(float g) {}
    public void stroke(float g, float a) {}
    public void stroke(float r, float g, float b) {}
    public void stroke(float r, float g, float b, float a) {}
    public void noStroke() {}
    public void strokeWeight(float w) {}

    // --- shapes -----------------------------------------------------------
    public void ellipse(float a, float b, float c, float d) {}
    public void circle(float a, float b, float c) {}
    public void rect(float a, float b, float c, float d) {}
    public void rect(float a, float b, float c, float d, float r) {}
    public void line(float a, float b, float c, float d) {}
    public void line(float a, float b, float c, float d, float e, float f) {}
    public void point(float a, float b) {}
    public void arc(float a, float b, float c, float d, float e, float f) {}
    public void triangle(float a, float b, float c, float d, float e, float f) {}
    public void beginShape() {}
    public void beginShape(int kind) {}
    public void vertex(float x, float y) {}
    public void vertex(float x, float y, float z) {}
    public void endShape() {}
    public void endShape(int mode) {}
    public void rectMode(int m) {}
    public void ellipseMode(int m) {}

    // --- transforms -------------------------------------------------------
    public void pushMatrix() {}
    public void popMatrix() {}
    public void translate(float x, float y) {}
    public void translate(float x, float y, float z) {}
    public void rotate(float a) {}
    public void scale(float s) {}
    public void scale(float x, float y) {}
    public void pushStyle() {}
    public void popStyle() {}

    // --- text -------------------------------------------------------------
    public void textSize(float s) {}
    public void textAlign(int a) {}
    public void textAlign(int a, int b) {}
    public void text(String s, float x, float y) {}
    public void text(int v, float x, float y) {}
    public void text(float v, float x, float y) {}
    public void text(char c, float x, float y) {}
    public float textWidth(String s) { return s.length() * 7; }

    // --- maths ------------------------------------------------------------
    public static float constrain(float v, float lo, float hi) { return Math.max(lo, Math.min(hi, v)); }
    public static int constrain(int v, int lo, int hi) { return Math.max(lo, Math.min(hi, v)); }
    public static float map(float v, float a, float b, float c, float d) { return c + (d - c) * ((v - a) / (b - a)); }
    public static float lerp(float a, float b, float t) { return a + (b - a) * t; }
    public static float dist(float a, float b, float c, float d) { return (float) Math.hypot(c - a, d - b); }
    public static float mag(float a, float b) { return (float) Math.hypot(a, b); }
    public static float max(float a, float b) { return Math.max(a, b); }
    public static int max(int a, int b) { return Math.max(a, b); }
    public static float min(float a, float b) { return Math.min(a, b); }
    public static int min(int a, int b) { return Math.min(a, b); }
    public static float abs(float a) { return Math.abs(a); }
    public static int abs(int a) { return Math.abs(a); }
    public static float sqrt(float a) { return (float) Math.sqrt(a); }
    public static float pow(float a, float b) { return (float) Math.pow(a, b); }
    public static float sin(float a) { return (float) Math.sin(a); }
    public static float cos(float a) { return (float) Math.cos(a); }
    public static float atan2(float a, float b) { return (float) Math.atan2(a, b); }
    public static float degrees(float a) { return (float) Math.toDegrees(a); }
    public static float radians(float a) { return (float) Math.toRadians(a); }
    public static int floor(float a) { return (int) Math.floor(a); }
    public static int ceil(float a) { return (int) Math.ceil(a); }
    public static int round(float a) { return Math.round(a); }
    public float random(float hi) { return (float) (Math.random() * hi); }
    public float random(float lo, float hi) { return lo + (float) (Math.random() * (hi - lo)); }
    public static String nf(float v, int left, int right) { return String.format(Locale.US, "%" + (left + right + 1) + "." + right + "f", v); }
    public static String nf(int v, int digits) { return String.format(Locale.US, "%0" + digits + "d", v); }
    public static String str(int v) { return Integer.toString(v); }
    public static String str(float v) { return Float.toString(v); }
    public static float parseFloat(String s) { return Float.parseFloat(s); }
    public static int parseInt(String s) { return Integer.parseInt(s); }
    public static void println(Object o) { System.out.println(o); }
    public static void println() { System.out.println(); }
    public static void print(Object o) { System.out.print(o); }
    public static String join(String[] list, String sep) { return String.join(sep, list); }
}
