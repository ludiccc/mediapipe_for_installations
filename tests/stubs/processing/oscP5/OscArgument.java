package oscP5;

public class OscArgument {
    private final Object value;
    public OscArgument(Object v) { value = v; }
    public int intValue() {
        if (value instanceof Integer) return (Integer) value;
        if (value instanceof Float) return ((Float) value).intValue();
        if (value instanceof Long) return ((Long) value).intValue();
        return 0;
    }
    public float floatValue() {
        if (value instanceof Float) return (Float) value;
        if (value instanceof Integer) return ((Integer) value).floatValue();
        if (value instanceof Double) return ((Double) value).floatValue();
        return 0;
    }
    public String stringValue() { return value == null ? "" : value.toString(); }
    public Object getValue() { return value; }
}
