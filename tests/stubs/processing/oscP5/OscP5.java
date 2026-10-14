package oscP5;

import java.lang.reflect.Method;
import java.net.DatagramPacket;
import java.net.DatagramSocket;

// Stand-in for oscP5's OscP5: really listens on UDP and calls parent.oscEvent(OscMessage)
// from a background thread, like the real library does.  Only used by the tests.
public class OscP5 {
    private final Object parent;
    private final int port, datagramSize;
    private volatile boolean running = true;
    private DatagramSocket socket;
    public int received = 0, dropped = 0, truncated = 0, failures = 0;
    public static OscP5 last;

    public OscP5(Object parent, int port) { this(parent, port, 1536); }

    public OscP5(Object parent, OscProperties props) { this(parent, props.port, props.datagramSize); }

    private OscP5(Object parent, int port, int datagramSize) {
        if (System.getProperty("osc.port") != null) port = Integer.parseInt(System.getProperty("osc.port"));
        this.parent = parent; this.port = port; this.datagramSize = datagramSize; last = this;
        try {
            socket = new DatagramSocket(port);
            socket.setReceiveBufferSize(1 << 20);
        } catch (Exception e) {
            throw new RuntimeException("cannot bind UDP port " + port, e);
        }
        final Method handler;
        try {
            handler = parent.getClass().getMethod("oscEvent", OscMessage.class);
        } catch (NoSuchMethodException e) {
            throw new RuntimeException("the sketch has no oscEvent(OscMessage) method", e);
        }
        Thread t = new Thread(new Runnable() {
            public void run() {
                byte[] buf = new byte[datagramSize];          // like oscP5: bigger datagrams are truncated
                while (running) {
                    DatagramPacket p = new DatagramPacket(buf, buf.length);
                    OscMessage m;
                    try {
                        socket.receive(p);
                        if (p.getLength() >= datagramSize) truncated++;
                        m = OscMessage.parse(p.getData(), p.getLength());
                        if (m == null) continue;
                    } catch (Exception e) {
                        dropped++;                                    // truncated or malformed packet
                        continue;
                    }
                    received++;
                    try {
                        handler.invoke(parent, m);
                    } catch (Exception e) {
                        System.err.println("### oscEvent failed on " + m + ": " + e.getCause());
                        e.getCause().printStackTrace();
                        failures++;
                    }
                }
            }
        });
        t.setDaemon(true);
        t.start();
        System.out.println("### [stub oscP5] listening on UDP port " + port + " (datagram size " + datagramSize + ")");
    }

    public void stop() { running = false; socket.close(); }
    public void dispose() { stop(); }
}
