"""Endpoint reachability: a dead or wrong port must fail in seconds, not after every probe times out."""
import socket
import threading
import time

from conftest import completion

from wsr.pipeline import TriageInput, triage
from wsr.probes import Endpoint


def _silent_port():
    """A port that accepts TCP and never answers, like Windows' discard service on :9."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(5)
    held = []
    threading.Thread(target=lambda: held.append(s.accept()), daemon=True).start()
    return s, s.getsockname()[1]


def test_live_server_is_reachable(fake_server):
    assert Endpoint(fake_server(lambda b: completion("hi")).url, "m").reachable() is None


def test_silent_port_fails_fast():
    s, port = _silent_port()
    t = time.monotonic()
    why = Endpoint(f"http://127.0.0.1:{port}", "m").reachable(timeout=1)
    assert why and "does not answer HTTP" in why and time.monotonic() - t < 5
    s.close()


def test_triage_on_dead_endpoint_is_a_runtime_verdict():
    s, port = _silent_port()
    e = Endpoint(f"http://127.0.0.1:{port}", "m", timeout=1)
    e.reachable(timeout=1)
    rep = triage(TriageInput(endpoint=e, messages=[{"role": "user", "content": "hi"}], params={}))
    assert rep.verdict["verdict"] == "runtime"
    assert rep.verdict["primary"]["code"] == "ENDPOINT_UNREACHABLE"
    s.close()
