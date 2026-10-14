// Runs a converted sketch headlessly for N seconds: java Runner <SketchClass> <seconds>
public class Runner {
    public static void main(String[] a) throws Exception {
        Class<?> cls = Class.forName(a[0]);
        processing.core.PApplet sketch = (processing.core.PApplet) cls.getDeclaredConstructor().newInstance();
        sketch.runHeadless(Float.parseFloat(a[1]));
        System.out.println("### draw() called " + sketch.drawCalls + " times");
        oscP5.OscP5 o = oscP5.OscP5.last;
        if (o != null) System.out.println("### osc: received " + o.received + ", dropped " + o.dropped
                                          + " (truncated " + o.truncated + "), oscEvent failures " + o.failures);
        // what the sketch knows: every Map field (people, hands, faces, blobs ...) and its keys
        for (java.lang.reflect.Field f : cls.getDeclaredFields()) {
            f.setAccessible(true);
            Object v = f.get(sketch);
            if (v instanceof java.util.Map) {
                System.out.println("### " + f.getName() + ": " + new java.util.TreeSet<Object>(((java.util.Map<?, ?>) v).keySet()));
            }
        }
        System.exit(0);
    }
}
