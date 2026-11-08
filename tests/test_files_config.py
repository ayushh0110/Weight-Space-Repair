import json

from conftest import CHATML, make_hf_dir, write_gguf, write_safetensors

from wsr.config_checks import check_config, load_gguf
from wsr.file_integrity import check_files


def fails(findings):
    return {f.code for f in findings if f.failed}


# -- layer 1 ---------------------------------------------------------------------

def test_valid_safetensors(tmp_path):
    p = write_safetensors(tmp_path / "m.safetensors", {"a": ("F32", [4, 4]), "b": ("BF16", [3])})
    assert fails(check_files(p)) == set()


def test_truncated_safetensors(tmp_path):
    p = write_safetensors(tmp_path / "m.safetensors", {"a": ("F32", [64, 64])}, truncate=100)
    assert "TRUNCATED_FILE" in fails(check_files(p))


def test_safetensors_bad_header(tmp_path):
    p = tmp_path / "m.safetensors"
    p.write_bytes(b"\xff" * 8 + b"{}")
    assert "ST_HEADER_INVALID" in fails(check_files(p))


def test_safetensors_overlap_and_size(tmp_path):
    hdr = {"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
           "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]}}
    p = write_safetensors(tmp_path / "m.safetensors", {"a": ("F32", [3])}, header_override=hdr)
    assert "ST_OFFSETS_NOT_TILED" in fails(check_files(p))
    hdr = {"a": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}
    p = write_safetensors(tmp_path / "n.safetensors", {"a": ("F32", [2])}, header_override=hdr)
    assert "ST_TENSOR_INVALID" in fails(check_files(p))


def test_nan_scan(tmp_path):
    import struct

    p = write_safetensors(tmp_path / "m.safetensors", {"w": ("F32", [2])})
    data = bytearray(p.read_bytes())
    data[-4:] = struct.pack("<f", float("nan"))
    p.write_bytes(bytes(data))
    assert "NAN_OR_INF_WEIGHTS" in fails(check_files(p, scan_values=True))


def test_shard_index(tmp_path):
    write_safetensors(tmp_path / "model-00001.safetensors", {"a": ("F32", [2])})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001.safetensors", "b": "model-00002.safetensors"}}))
    assert "SHARD_MISSING" in fails(check_files(tmp_path))


def test_partial_download(tmp_path):
    write_safetensors(tmp_path / "model.safetensors", {"a": ("F32", [2])})
    (tmp_path / "model.safetensors.incomplete").write_bytes(b"x")
    assert "PARTIAL_DOWNLOAD" in fails(check_files(tmp_path))


def test_gguf_valid_truncated_and_bad_magic(tmp_path):
    md = {"general.architecture": (8, "qwen2"), "general.alignment": (4, 32)}
    assert fails(check_files(write_gguf(tmp_path / "ok.gguf", metadata=md))) == set()
    assert "TRUNCATED_FILE" in fails(check_files(write_gguf(tmp_path / "t.gguf", metadata=md, truncate=200)))
    assert "GGUF_HEADER_INVALID" in fails(check_files(write_gguf(tmp_path / "b.gguf", magic=b"GGML")))


# -- layer 2 ---------------------------------------------------------------------

def test_healthy_hf_dir(tmp_path):
    d = make_hf_dir(tmp_path / "m", eos_ids=(2, 3))
    _, fs = check_config(d, memory=False)
    assert fails(fs) == set()
    assert "EOS_OK" in {f.code for f in fs}


def test_eos_mismatch(tmp_path):
    d = make_hf_dir(tmp_path / "m", eos_ids=(2,))  # template ends turns with <|im_end|> (id 3)
    _, fs = check_config(d, memory=False)
    f = next(x for x in fs if x.code == "EOS_MISMATCH")
    assert f.evidence["turn_end"] == "<|im_end|>" and f.evidence["turn_end_id"] == 3


def test_template_missing(tmp_path):
    _, fs = check_config(make_hf_dir(tmp_path / "m", template=None), memory=False)
    assert "TEMPLATE_MISSING" in fails(fs)


def test_template_render_error_and_no_gen_prompt(tmp_path):
    _, fs = check_config(make_hf_dir(tmp_path / "a", template="{% for m in messages %}{{ m.content "), memory=False)
    assert "TEMPLATE_RENDER_ERROR" in fails(fs)
    no_gen = "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    _, fs = check_config(make_hf_dir(tmp_path / "b", template=no_gen, eos_ids=(3,)), memory=False)
    assert "NO_GENERATION_PROMPT" in fails(fs)


def test_double_bos(tmp_path):
    d = make_hf_dir(tmp_path / "m", eos_ids=(3,), bos_in_template=True, add_bos=True)
    _, fs = check_config(d, memory=False)
    assert "DOUBLE_BOS" in fails(fs)


def test_tokenizer_exceeds_embeddings(tmp_path):
    d = make_hf_dir(tmp_path / "m", eos_ids=(3,), vocab_n=40, embed_rows=16)
    _, fs = check_config(d, memory=False)
    assert "TOKENIZER_EXCEEDS_EMBEDDINGS" in fails(fs)


def test_template_conflict(tmp_path):
    d = make_hf_dir(tmp_path / "m", eos_ids=(3,))
    (d / "chat_template.jinja").write_text(CHATML.replace("assistant", "model"))
    _, fs = check_config(d, memory=False)
    assert "TEMPLATE_CONFLICT" in fails(fs)


def test_gguf_spec_and_low_bit(tmp_path):
    md = {
        "general.architecture": (8, "qwen2"), "general.file_type": (4, 10),  # Q2_K
        "qwen2.context_length": (4, 4096), "qwen2.block_count": (4, 2),
        "qwen2.attention.head_count": (4, 2), "qwen2.attention.head_count_kv": (4, 1),
        "qwen2.embedding_length": (4, 8),
        "tokenizer.ggml.tokens": ("arr8", ["<s>", "<|im_start|>", "<|endoftext|>", "<|im_end|>"] + ["a"] * 12),
        "tokenizer.ggml.eos_token_id": (4, 2),
        "tokenizer.chat_template": (8, CHATML),
    }
    p = write_gguf(tmp_path / "m.gguf", metadata=md, tensors=[("token_embd.weight", [8, 16], 0)])
    spec = load_gguf(p)
    assert spec.special[3] == "<|im_end|>" and spec.embed_rows == 16
    assert 3 in spec.stop_ids  # llama.cpp treats <|im_end|> as end-of-generation
    _, fs = check_config(p, memory=False)
    assert "LOW_BIT_SMALL_MODEL" in fails(fs)
    assert "EOS_MISMATCH" not in fails(fs)
