// Stand-in for ofxOsc: a real UDP receiver with a minimal OSC 1.0 parser
// (types i f s d T F).  Only used by the tests.
#pragma once

#include "ofMain.h"
#include <arpa/inet.h>
#include <cstring>
#include <deque>
#include <mutex>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

enum ofxOscArgType { OFXOSC_TYPE_NONE, OFXOSC_TYPE_INT32, OFXOSC_TYPE_FLOAT, OFXOSC_TYPE_STRING, OFXOSC_TYPE_TRUE, OFXOSC_TYPE_FALSE };

class ofxOscMessage {
public:
	struct Arg { ofxOscArgType type; int i; float f; std::string s; };
	std::string address;
	std::vector<Arg> args;

	const std::string& getAddress() const { return address; }
	std::size_t getNumArgs() const { return args.size(); }
	ofxOscArgType getArgType(std::size_t i) const { return args[i].type; }
	int getArgAsInt32(std::size_t i) const { return args[i].type == OFXOSC_TYPE_FLOAT ? (int)args[i].f : args[i].i; }
	int getArgAsInt(std::size_t i) const { return getArgAsInt32(i); }
	float getArgAsFloat(std::size_t i) const { return args[i].type == OFXOSC_TYPE_INT32 ? (float)args[i].i : args[i].f; }
	std::string getArgAsString(std::size_t i) const { return args[i].s; }
	bool getArgAsBool(std::size_t i) const { return args[i].type == OFXOSC_TYPE_TRUE; }
	void setAddress(const std::string& a) { address = a; }
	void addIntArg(int v) { args.push_back({OFXOSC_TYPE_INT32, v, 0, ""}); }
	void addFloatArg(float v) { args.push_back({OFXOSC_TYPE_FLOAT, 0, v, ""}); }
	void addStringArg(const std::string& v) { args.push_back({OFXOSC_TYPE_STRING, 0, 0, v}); }

	static bool parse(const char* data, size_t len, ofxOscMessage& out) {
		size_t pos = 0;
		auto readString = [&](std::string& s) {
			size_t start = pos;
			while (pos < len && data[pos] != 0) pos++;
			if (pos >= len) return false;
			s.assign(data + start, pos - start);
			pos = start + ((pos - start + 4) & ~3u);
			return true;
		};
		if (!readString(out.address)) return false;
		if (out.address.rfind("#bundle", 0) == 0) return false;
		std::string tags;
		if (pos < len && !readString(tags)) return false;
		for (size_t k = 1; k < tags.size(); k++) {
			Arg a{OFXOSC_TYPE_NONE, 0, 0, ""};
			switch (tags[k]) {
				case 'i': { if (pos + 4 > len) return false; uint32_t v; memcpy(&v, data + pos, 4); a.i = (int)ntohl(v); a.type = OFXOSC_TYPE_INT32; pos += 4; break; }
				case 'f': { if (pos + 4 > len) return false; uint32_t v; memcpy(&v, data + pos, 4); v = ntohl(v); memcpy(&a.f, &v, 4); a.type = OFXOSC_TYPE_FLOAT; pos += 4; break; }
				case 's': { if (!readString(a.s)) return false; a.type = OFXOSC_TYPE_STRING; break; }
				case 'T': a.type = OFXOSC_TYPE_TRUE; break;
				case 'F': a.type = OFXOSC_TYPE_FALSE; break;
				default: return true;
			}
			out.args.push_back(a);
		}
		return true;
	}
};

class ofxOscReceiver {
public:
	int received = 0, dropped = 0;
	static ofxOscReceiver* last;

	bool setup(int port) {
		if (const char* p = std::getenv("OF_STUB_PORT")) port = std::atoi(p);
		last = this;
		sock = socket(AF_INET, SOCK_DGRAM, 0);
		int big = 1 << 20;
		setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &big, sizeof big);
		sockaddr_in addr{};
		addr.sin_family = AF_INET;
		addr.sin_addr.s_addr = INADDR_ANY;
		addr.sin_port = htons(port);
		if (bind(sock, (sockaddr*)&addr, sizeof addr) != 0) { perror("bind"); return false; }
		running = true;
		thread = std::thread([this] { run(); });
		std::cout << "### [stub ofxOsc] listening on UDP port " << port << std::endl;
		return true;
	}
	~ofxOscReceiver() { stop(); }
	void stop() {
		if (!running) return;
		running = false;
		shutdown(sock, SHUT_RDWR);
		close(sock);
		if (thread.joinable()) thread.join();
	}
	bool hasWaitingMessages() { std::lock_guard<std::mutex> g(mx); return !queue.empty(); }
	bool getNextMessage(ofxOscMessage& m) {
		std::lock_guard<std::mutex> g(mx);
		if (queue.empty()) return false;
		m = queue.front();
		queue.pop_front();
		return true;
	}

private:
	void run() {
		std::vector<char> buf(65536);                // the real oscpack buffer may be smaller: see the README
		while (running) {
			ssize_t n = recv(sock, buf.data(), buf.size(), 0);
			if (n <= 0) continue;
			ofxOscMessage m;
			if (!ofxOscMessage::parse(buf.data(), (size_t)n, m)) { dropped++; continue; }
			std::lock_guard<std::mutex> g(mx);
			queue.push_back(m);
			received++;
		}
	}
	int sock = -1;
	bool running = false;
	std::thread thread;
	std::mutex mx;
	std::deque<ofxOscMessage> queue;
};
inline ofxOscReceiver* ofxOscReceiver::last = nullptr;

class ofxOscSender {
public:
	void setup(const std::string&, int) {}
	void sendMessage(const ofxOscMessage&, bool = false) {}
};

inline void of_stub_summary(ofBaseApp*) {
	if (ofxOscReceiver::last)
		std::cout << "### osc: received " << ofxOscReceiver::last->received << ", dropped " << ofxOscReceiver::last->dropped << std::endl;
}
