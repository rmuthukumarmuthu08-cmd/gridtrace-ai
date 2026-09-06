"""
Minimal local MQTT 3.1.1 broker (QoS 0).
=======================================

Why this exists: the prototype has to run on a laptop with no cloud and no
install ceremony. Mosquitto is the "proper" answer but it is another install and
a Windows service; this is ~180 lines of stdlib that speaks real MQTT on the wire,
so `publisher.py` and `infer.py` use ordinary paho-mqtt clients and nothing about
the pipeline is faked.

Scope, deliberately: CONNECT / PUBLISH / SUBSCRIBE / UNSUBSCRIBE / PINGREQ /
DISCONNECT at QoS 0, no retained messages, no persistent sessions, no auth.
That is exactly what this telemetry pipeline uses. If you already run Mosquitto,
skip this module entirely - the clients don't know the difference.
"""
from __future__ import annotations

import logging
import socket
import socketserver
import threading

log = logging.getLogger("gridtrace.broker")

CONNECT, CONNACK, PUBLISH, SUBSCRIBE, SUBACK = 1, 2, 3, 8, 9
UNSUBSCRIBE, UNSUBACK, PINGREQ, PINGRESP, DISCONNECT = 10, 11, 12, 13, 14

_subs: dict[socket.socket, set[str]] = {}
_subs_lock = threading.Lock()


def _topic_matches(filt: str, topic: str) -> bool:
    """MQTT wildcard match: '+' one level, '#' the rest."""
    f, t = filt.split("/"), topic.split("/")
    for i, part in enumerate(f):
        if part == "#":
            return True
        if i >= len(t):
            return False
        if part != "+" and part != t[i]:
            return False
    return len(f) == len(t)


def _encode_len(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n % 128
        n //= 128
        if n:
            b |= 0x80
        out.append(b)
        if not n:
            return bytes(out)


def _read_len(rfile) -> int | None:
    mult, value = 1, 0
    for _ in range(4):
        ch = rfile.read(1)
        if not ch:
            return None
        b = ch[0]
        value += (b & 0x7F) * mult
        if not b & 0x80:
            return value
        mult *= 128
    return value


def _publish_packet(topic: str, payload: bytes) -> bytes:
    tb = topic.encode()
    body = len(tb).to_bytes(2, "big") + tb + payload
    return bytes([PUBLISH << 4]) + _encode_len(len(body)) + body


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        sock = self.request
        sock.settimeout(None)
        with _subs_lock:
            _subs[sock] = set()
        try:
            while True:
                head = self.rfile.read(1)
                if not head:
                    break
                ptype, flags = head[0] >> 4, head[0] & 0x0F
                rem = _read_len(self.rfile)
                if rem is None:
                    break
                body = self.rfile.read(rem) if rem else b""
                if len(body) != rem:
                    break

                if ptype == CONNECT:
                    sock.sendall(bytes([CONNACK << 4, 2, 0, 0]))

                elif ptype == PUBLISH:
                    tlen = int.from_bytes(body[0:2], "big")
                    topic = body[2:2 + tlen].decode(errors="replace")
                    off = 2 + tlen
                    if (flags >> 1) & 0x03:          # QoS > 0 carries a packet id
                        off += 2
                    payload = body[off:]
                    pkt = _publish_packet(topic, payload)
                    with _subs_lock:
                        targets = [s for s, fs in _subs.items()
                                   if any(_topic_matches(f, topic) for f in fs)]
                    for s in targets:
                        try:
                            s.sendall(pkt)
                        except OSError:
                            pass                      # dropped subscriber; reaped on its own thread

                elif ptype == SUBSCRIBE:
                    pid, i, granted = body[0:2], 2, []
                    while i < len(body):
                        tlen = int.from_bytes(body[i:i + 2], "big")
                        filt = body[i + 2:i + 2 + tlen].decode(errors="replace")
                        i += 2 + tlen + 1             # +1 for the requested QoS byte
                        with _subs_lock:
                            _subs[sock].add(filt)
                        granted.append(0)             # we only grant QoS 0
                    resp = pid + bytes(granted)
                    sock.sendall(bytes([SUBACK << 4]) + _encode_len(len(resp)) + resp)

                elif ptype == UNSUBSCRIBE:
                    pid, i = body[0:2], 2
                    while i < len(body):
                        tlen = int.from_bytes(body[i:i + 2], "big")
                        filt = body[i + 2:i + 2 + tlen].decode(errors="replace")
                        i += 2 + tlen
                        with _subs_lock:
                            _subs[sock].discard(filt)
                    sock.sendall(bytes([UNSUBACK << 4, 2]) + pid)

                elif ptype == PINGREQ:
                    sock.sendall(bytes([PINGRESP << 4, 0]))

                elif ptype == DISCONNECT:
                    break
        except (OSError, ValueError, IndexError):
            pass                                      # a malformed client must not kill the broker
        finally:
            with _subs_lock:
                _subs.pop(sock, None)


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(host: str = "127.0.0.1", port: int = 1883, background: bool = False):
    """Start the broker. Returns the server object (call .shutdown() to stop)."""
    srv = _Server((host, port), _Handler)
    if background:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        log.info("broker listening on %s:%d (background)", host, port)
        return srv
    log.info("broker listening on %s:%d", host, port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
    return srv


def port_is_open(host: str, port: int, timeout: float = 0.4) -> bool:
    """True if something already accepts TCP here (e.g. Mosquitto is running)."""
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


if __name__ == "__main__":
    from gridtrace import config
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)-20s %(message)s")
    serve(config.MQTT_HOST, config.MQTT_PORT)
