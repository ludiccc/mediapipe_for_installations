package oscP5;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;

// Minimal OSC 1.0 message parser (types i f s d T F) for the test harness.
public class OscMessage {
    private String address;
    private String typetag;
    private Object[] args;

    public OscMessage(String address, Object[] args, String typetag) {
        this.address = address; this.args = args; this.typetag = typetag;
    }

    public String addrPattern() { return address; }
    public String typetag() { return typetag; }
    public Object[] arguments() { return args; }
    public OscArgument get(int i) { return new OscArgument(args[i]); }
    public boolean checkAddrPattern(String p) { return address.equals(p); }
    public boolean checkTypetag(String t) { return typetag.equals(t); }
    public String toString() { return address + " " + typetag; }

    private static int pad4(int n) { return (n + 4) & ~3; }

    private static String readString(ByteBuffer b) {
        int start = b.position();
        while (b.get() != 0) {}
        int end = b.position();                       // position after the NUL
        String s = new String(b.array(), start, end - 1 - start, StandardCharsets.UTF_8);
        b.position(start + pad4(end - 1 - start));
        return s;
    }

    public static OscMessage parse(byte[] data, int length) {
        ByteBuffer b = ByteBuffer.wrap(data, 0, length);
        String address = readString(b);
        if (address.startsWith("#bundle")) return null;   // bundles are not needed by the examples
        String tags = b.hasRemaining() ? readString(b) : ",";
        ArrayList<Object> out = new ArrayList<Object>();
        for (int i = 1; i < tags.length(); i++) {
            switch (tags.charAt(i)) {
                case 'i': out.add(b.getInt()); break;
                case 'f': out.add(b.getFloat()); break;
                case 'd': out.add(b.getDouble()); break;
                case 's': out.add(readString(b)); break;
                case 'T': out.add(Boolean.TRUE); break;
                case 'F': out.add(Boolean.FALSE); break;
                default: return new OscMessage(address, out.toArray(), tags.substring(1));
            }
        }
        return new OscMessage(address, out.toArray(), tags.substring(1));
    }
}
