package processing.core;

public class PVector {
    public float x, y, z;
    public PVector() {}
    public PVector(float x, float y) { this.x = x; this.y = y; }
    public PVector(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }
    public PVector set(float x, float y) { this.x = x; this.y = y; return this; }
    public PVector set(float x, float y, float z) { this.x = x; this.y = y; this.z = z; return this; }
    public PVector set(PVector v) { x = v.x; y = v.y; z = v.z; return this; }
    public PVector copy() { return new PVector(x, y, z); }
    public float mag() { return (float) Math.sqrt(x * x + y * y + z * z); }
    public float dist(PVector v) { return (float) Math.sqrt((x - v.x) * (x - v.x) + (y - v.y) * (y - v.y) + (z - v.z) * (z - v.z)); }
    public static float dist(PVector a, PVector b) { return a.dist(b); }
    public PVector add(PVector v) { x += v.x; y += v.y; z += v.z; return this; }
    public PVector sub(PVector v) { x -= v.x; y -= v.y; z -= v.z; return this; }
    public static PVector sub(PVector a, PVector b) { return new PVector(a.x - b.x, a.y - b.y, a.z - b.z); }
    public static PVector add(PVector a, PVector b) { return new PVector(a.x + b.x, a.y + b.y, a.z + b.z); }
    public PVector mult(float s) { x *= s; y *= s; z *= s; return this; }
    public PVector div(float s) { x /= s; y /= s; z /= s; return this; }
    public PVector normalize() { float m = mag(); if (m > 0) div(m); return this; }
    public float heading() { return (float) Math.atan2(y, x); }
    public static PVector lerp(PVector a, PVector b, float t) { return new PVector(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t); }
    public PVector lerp(PVector v, float t) { x += (v.x - x) * t; y += (v.y - y) * t; z += (v.z - z) * t; return this; }
    public String toString() { return "[ " + x + ", " + y + ", " + z + " ]"; }
}
