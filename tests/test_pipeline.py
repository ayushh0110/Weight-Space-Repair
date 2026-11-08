"""End-to-end: fake servers whose failures have a known cause; the verdict must name it."""
import json

from conftest import completion, make_hf_dir

from wsr.cli import main
from wsr.known_issues import match
from wsr.pipeline import TriageInput, triage
from wsr.probes import Endpoint, run_probes
from wsr.request_checks import check_request

LOOP = "Step one is to open the file. " + "Then we continue. " * 40
GOOD = "Open the file, read each line, and count the words in it."


def ep(server, **kw):
    return Endpoint(server.url, "m", budget_s=30, timeout=10, **kw)


def test_penalty_caused_loop_is_a_sampling_verdict(fake_server):
    # Loops only while repeat_penalty is set; the differential must find that.
    s = fake_server(lambda b: completion(LOOP if b.get("repeat_penalty") else GOOD,
                                         "length" if b.get("repeat_penalty") else "stop"))
    inp = TriageInput(endpoint=ep(s), messages=[{"role": "user", "content": "How do I count words?"}],
                      params={"repeat_penalty": 1.5, "temperature": 0.7, "max_tokens": 200},
                      probes=[], run_differential=True)
    rep = triage(inp)
    assert rep.verdict["verdict"] == "sampling"
    assert rep.verdict["primary"]["code"] == "DIFF_FLIP:penalties_neutral"


def test_static_eos_mismatch_explains_invented_turns(tmp_path):
    d = make_hf_dir(tmp_path / "m", eos_ids=(2,))
    raw = completion("Sure!<|im_end|>\nuser: and another question?\nassistant: ok", "length")
    rep = triage(TriageInput(model_dir=d, response=raw, params={"max_tokens": 64}))
    assert rep.verdict["verdict"] == "config"
    assert rep.verdict["primary"]["code"] == "EOS_MISMATCH"


RUNON = ("Paris.<|im_end|>\n<|im_start|>user\nWhat about Germany?<|im_end|>\n<|im_start|>assistant\nBerlin.")


def test_config_only_dir_does_not_blame_files(tmp_path):
    d = make_hf_dir(tmp_path / "m", eos_ids=(2,))
    (d / "model.safetensors").unlink()
    rep = triage(TriageInput(model_dir=d, response=completion(RUNON, "length"), params={"max_tokens": 64}))
    assert rep.verdict["primary"]["code"] == "EOS_MISMATCH"


def test_leak_with_correct_files_is_runtime_not_weights(tmp_path):
    d = make_hf_dir(tmp_path / "m", eos_ids=(2, 3))
    rep = triage(TriageInput(model_dir=d, response=completion(RUNON, "length"), params={"max_tokens": 64}))
    assert rep.verdict["verdict"] == "runtime"


def test_clean_full_precision_model_gives_weight_level(tmp_path):
    d = make_hf_dir(tmp_path / "m", eos_ids=(2, 3))
    rep = triage(TriageInput(model_dir=d, response=completion(LOOP, "length"), params={"max_tokens": 200}))
    assert rep.verdict["verdict"] == "weight_level"


def test_unexplained_without_model_files():
    rep = triage(TriageInput(response=completion(LOOP, "length")))
    assert rep.verdict["verdict"] == "unexplained"


def test_healthy_output_no_failure():
    rep = triage(TriageInput(response=completion(GOOD)))
    assert rep.verdict["verdict"] == "no_failure_observed"


def test_stop_probe_and_smoke(fake_server):
    def h(b):
        if b["messages"][-1]["content"].startswith("Reply with exactly"):
            return completion("OK OK OK OK OK OK OK OK OK OK", "length")
        if b.get("max_tokens") == 8:
            return completion("1, 2, 3, 4", "stop")  # server hides truncation
        return completion("Hello!")

    fs = {f.code: f for f in run_probes(ep(fake_server(h)), ["smoke", "stop", "finish"])}
    assert fs["SMOKE_OK"].severity.value == "pass"
    assert fs["STOP_PROBE_RUNON"].failed
    assert fs["FINISH_REASON_UNRELIABLE"].failed


def test_context_probe_detects_silent_truncation(fake_server):
    def h(b):
        content = b["messages"][-1]["content"]
        n = len(content) // 4
        if n > 1500:  # server keeps only the last 1500 tokens
            return completion("I only see code " + content.split("Remember code ")[-1][:8], prompt_tokens=1500)
        return completion(" ".join(content.split("Remember code ")[i][:8] for i in (1, 2)) if "Remember" in content
                          else "x", prompt_tokens=n)

    fs = run_probes(ep(fake_server(h)), ["context"])
    trunc = [f for f in fs if f.code == "CONTEXT_TRUNCATION"]
    assert trunc and "start" in trunc[0].message


def test_known_issue_matching():
    fs = match("gemma-4-e4b", {"TAIL_LOOP"}, backend="ollama", conditions={"constrained", "gguf"})
    ids = {f.code for f in fs}
    assert "KNOWN_ISSUE:gemma4-json-format-loop" in ids
    assert "KNOWN_ISSUE:gemma4-repetition-collapse" in ids
    assert "KNOWN_ISSUE:llama3-eot-vs-eos" in {f.code for f in match("llama-3.2-1b", {"INVENTED_TURN"})}
    assert not match("phi-4-mini", {"TAIL_LOOP"}, backend="vllm")


def test_request_checks():
    codes = {f.code for f in check_request({"options": {"repeat_penalty": 1.4}, "stop": ["\n"]}, backend="ollama")}
    assert {"REPEAT_PENALTY_HIGH", "STOP_TOO_BROAD", "OLLAMA_DEFAULT_CONTEXT"} <= codes


def test_cli_output_and_json(tmp_path, capsys):
    p = tmp_path / "out.txt"
    p.write_text(LOOP, encoding="utf-8")
    assert main(["output", str(p), "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert any(f["code"] == "TAIL_LOOP" for f in data["findings"])
    p.write_text(GOOD, encoding="utf-8")
    assert main(["output", str(p)]) == 0


def test_cli_triage_saved_response(tmp_path, capsys):
    d = make_hf_dir(tmp_path / "m", eos_ids=(2,))
    r = tmp_path / "resp.json"
    r.write_text(json.dumps(completion("Hi<|im_end|>\nuser: more?", "length")), encoding="utf-8")
    code = main(["triage", "--model-dir", str(d), "--response", str(r), "--handoff", str(tmp_path / "h")])
    out = capsys.readouterr().out
    assert code == 1 and "VERDICT: CONFIG" in out and "EOS_MISMATCH" in out
    assert (tmp_path / "h" / "report.json").exists()
