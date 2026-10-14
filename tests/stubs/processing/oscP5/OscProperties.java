package oscP5;

public class OscProperties {
    public static final int UDP = 0, TCP = 1, MULTICAST = 2;
    int port = 0, datagramSize = 1536;
    public void setListeningPort(int p) { port = p; }
    public int listeningPort() { return port; }
    public void setDatagramSize(int s) { datagramSize = s; }
    public int datagramSize() { return datagramSize; }
    public void setNetworkProtocol(int p) {}
}
