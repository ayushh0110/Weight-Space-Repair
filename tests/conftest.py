import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path

import pytest


def write_safetensors(path: Path, tensors: dict, *, truncate: int = 0, header_override=None) -> Path:
    """tensors: name -> (dtype, shape). Data is zeros except a 1.0 per tensor."""
    sizes = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8}
    header, offset, blobs = {}, 0, []
    for name, (dt, shape) in tensors.items():
        n = 1
        for d in shape:
            n *= d
        nb = n * sizes[dt]
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [offset, offset + nb]}
        blob = bytearray(nb)
        if dt == "F32" and nb >= 4:
            blob[0:4] = struct.pack("<f", 1.0)
        blobs.append(bytes(blob))
        offset += nb
    if header_override is not None:
        header = header_override
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    data = struct.pack("<Q", len(raw)) + raw + b"".join(blobs)
    if truncate:
        data = data[:-truncate]
    path.write_bytes(data)
    return path


def _gguf_str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path: Path, *, metadata=None, tensors=None, truncate: int = 0, magic=b"GGUF") -> Path:
    """metadata: key -> (type_id, value); strings type 8, u32 type 4, string-array ('arr8', [..])."""
    metadata = metadata or {}
    tensors = tensors or [("token_embd.weight", [8, 16], 0)]  # (name, ggml dims, ggml type)
    out = bytearray(magic + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(metadata)))
    for k, (t, v) in metadata.items():
        out += _gguf_str(k)
        if t == "arr8":
            out += struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(v))
            for s in v:
                out += _gguf_str(s)
        elif t == 8:
            out += struct.pack("<I", 8) + _gguf_str(v)
        elif t == 4:
            out += struct.pack("<I", 4) + struct.pack("<I", v)
        elif t == 7:
            out += struct.pack("<I", 7) + struct.pack("<?", v)
        elif t == "arr7":
            out += struct.pack("<I", 9) + struct.pack("<I", 7) + struct.pack("<Q", len(v))
            out += b"".join(struct.pack("<?", x) for x in v)
    offset, sizes = 0, []
    for name, dims, typ in tensors:
        n = 1
        for d in dims:
            n *= d
        nb = n * {0: 4, 1: 2}[typ]
        out += _gguf_str(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<I", typ) + struct.pack("<Q", offset)
        sizes.append(nb)
        offset += nb + (-nb % 32)
    out += b"\0" * (-len(out) % 32)
    for nb in sizes:
        out += b"\0" * (nb + (-nb % 32))
    if truncate:
        out = out[:-truncate]
    path.write_bytes(bytes(out))
    return path


CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def make_hf_dir(d: Path, *, template=CHATML, eos_ids=(2,), vocab_n=10, embed_rows=16, bos_in_template=False,
                add_bos=False) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({
        "architectures": ["Qwen2ForCausalLM"], "max_position_embeddings": 4096, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "hidden_size": 8, "vocab_size": embed_rows,
        "eos_token_id": list(eos_ids)}))
    (d / "generation_config.json").write_text(json.dumps({"eos_token_id": list(eos_ids)}))
    tc = {"bos_token": "<s>", "eos_token": "<|endoftext|>", "add_bos_token": add_bos}
    if template is not None:
        tc["chat_template"] = ("{{ bos_token }}" if bos_in_template else "") + template
    (d / "tokenizer_config.json").write_text(json.dumps(tc))
    vocab = {f"t{i}": i for i in range(vocab_n)}
    (d / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": vocab},
        "added_tokens": [
            {"id": 0, "content": "<s>", "special": True},
            {"id": 1, "content": "<|im_start|>", "special": True},
            {"id": 2, "content": "<|endoftext|>", "special": True},
            {"id": 3, "content": "<|im_end|>", "special": True},
        ]}))
    write_safetensors(d / "model.safetensors", {"model.embed_tokens.weight": ("F32", [embed_rows, 8]),
                                                "model.norm.weight": ("F32", [8])})
    return d


class FakeServer:
    """OpenAI-compatible server whose behaviour is a function of the request body."""

    def __init__(self, handler, threaded=False):
        self.handler = handler
        self.calls = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path == "/v1/models":
                    return self._send(200, {"data": [{"id": "m", "owned_by": "test"}]})
                self._send(404, {"error": "nope"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.calls.append(body)
                res = outer.handler(body)
                if isinstance(res, str):  # a raw stream (SSE / NDJSON)
                    raw = res.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                self._send(200, res)

        self.httpd = (ThreadingHTTPServer if threaded else HTTPServer)(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        self.t = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.t.start()

    def close(self):
        self.httpd.shutdown()


def completion(text, finish="stop", prompt_tokens=10, completion_tokens=5, **msg):
    return {"choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": text, **msg}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}


@pytest.fixture
def fake_server():
    servers = []

    def make(handler, threaded=False):
        s = FakeServer(handler, threaded)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


@pytest.fixture(autouse=True)
def _isolated_wsr_cache(tmp_path_factory, monkeypatch):
    """Remembered consent answers and cached files stay inside each test."""
    monkeypatch.setenv("WSR_CACHE_DIR", str(tmp_path_factory.mktemp("wsr_cache")))
    for k in ("WSR_ASSUME", "WSR_ALLOW", "WSR_DENY"):
        monkeypatch.delenv(k, raising=False)
