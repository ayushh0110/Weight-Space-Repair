"""Images: where requests put them, what servers say, whether the model saw them.

Error texts are the ones llama-server b10244 returned with a vision projector loaded.
"""
import base64

from conftest import completion

from wsr.media import (check_media, content_text, decode_data_uri, image_not_seen_score, image_part, images_as_parts,
                       solid_png)
from wsr.pipeline import TriageInput, _prompt_tokens_estimate, triage
from wsr.probes import Endpoint, run_probes
from wsr.response import check_response

PNG = solid_png((220, 20, 20))
B64 = base64.b64encode(PNG).decode()
URI = "data:image/png;base64," + B64
Q = {"type": "text", "text": "What colour is this image?"}


def ep(server):
    e = Endpoint(server.url, "m", budget_s=30, timeout=10)
    e.reasoning = False
    return e


def codes(fs):
    return {f.code for f in fs}


def test_base64_is_not_counted_as_prompt_text():
    big = "data:image/jpeg;base64," + "A" * 40_000
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": big}}, Q]}]
    assert _prompt_tokens_estimate(msgs) < 20
    assert content_text(msgs[0]["content"]) == Q["text"]


def test_static_checks():
    ok = [{"role": "user", "content": [image_part(PNG), Q]}]
    assert check_media(ok, api="openai") == []
    assert "IMAGE_FIELD_IGNORED" in codes(check_media([{"role": "user", "content": "x", "images": [B64]}],
                                                      api="openai"))
    assert check_media([{"role": "user", "content": "x", "images": [B64]}], api="ollama") == []
    assert "IMAGE_FIELD_IGNORED" in codes(check_media([{"role": "user", "content": "x"}], {"images": [B64]}))
    assert "IMAGE_URL_NOT_OBJECT" in codes(check_media([{"role": "user", "content": [
        {"type": "image_url", "image_url": URI}, Q]}]))
    assert "IMAGE_REMOTE_URL" in codes(check_media([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}}, Q]}]))
    assert "IMAGE_NO_VISION" in codes(check_media(ok, vision=False))


def test_bad_image_bytes():
    assert decode_data_uri(URI) == (PNG, None)
    assert "invalid base64" in decode_data_uri("data:image/png;base64,@@@")[1]
    assert "cut short" in decode_data_uri("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA")[1]
    assert "not a known image" in decode_data_uri("data:image/png;base64," + base64.b64encode(b"hello").decode())[1]


def test_server_image_errors_are_named():
    for msg, cause in (("Invalid base64 value", "request"), ("Failed to load image or audio file", "request"),
                       ("Failed to download image", "request"), ("error: cannot make GET request", "request"),
                       ("image input is not supported - hint: you may need to provide the mmproj", "config")):
        _, fs = check_response({"error": {"code": 500, "message": msg}})
        f = next(f for f in fs if f.code == "IMAGE_REJECTED")
        assert f.cause.value == cause, msg


def test_not_seen_phrases():
    assert image_not_seen_score("Please provide the image you are referring to.") == 1.0
    assert image_not_seen_score("I can't see any image in your message.") == 1.0
    assert image_not_seen_score("It looks like no image was attached.") == 1.0
    assert image_not_seen_score("The image shows a red barn under a blue sky.") == 0.0


def test_images_moved_into_parts():
    msgs, params = images_as_parts([{"role": "user", "content": "Describe it", "images": [B64]}], {"temperature": 0})
    part = msgs[0]["content"][0]
    assert part["image_url"]["url"].startswith("data:image/png;base64,")
    assert msgs[0]["content"][1] == {"type": "text", "text": "Describe it"} and "images" not in msgs[0]
    assert images_as_parts([{"role": "user", "content": "x"}], {}) is None


def _vision_server(sees=True, colour=None):
    def h(b):
        c = b["messages"][-1]["content"]
        parts = c if isinstance(c, list) else []
        if not any(p.get("type") == "image_url" for p in parts) or not sees:
            return completion("Please provide the image you are referring to.")
        url = next(p for p in parts if p.get("type") == "image_url")["image_url"]["url"]
        raw = base64.b64decode(url.split(",", 1)[1])
        guess = colour or ("Red" if raw == solid_png((220, 20, 20)) else "Blue")
        return completion(guess)
    return h


def test_dropped_images_field_is_blamed_on_the_request(fake_server):
    rep = triage(TriageInput(endpoint=ep(fake_server(_vision_server())),
                             messages=[{"role": "user", "content": "What colour is it?", "images": [B64]}],
                             params={"temperature": 0, "max_tokens": 50}, probes=[], run_differential=True))
    c = codes(rep.findings)
    assert {"IMAGE_FIELD_IGNORED", "IMAGE_NOT_SEEN", "DIFF_FLIP:images_as_parts"} <= c
    assert rep.verdict["verdict"] == "request"


def test_working_image_request_is_healthy(fake_server):
    rep = triage(TriageInput(endpoint=ep(fake_server(_vision_server(colour="The image is a solid red square."))),
                             messages=[{"role": "user", "content": [image_part(PNG), Q]}],
                             params={"temperature": 0, "max_tokens": 50}, probes=[], run_differential=True))
    assert rep.verdict["verdict"] == "no_failure_observed", rep.verdict


def test_vision_probe(fake_server):
    assert "VISION_OK" in codes(run_probes(ep(fake_server(_vision_server())), ["vision"]))
    assert "VISION_IGNORED" in codes(run_probes(ep(fake_server(_vision_server(sees=False))), ["vision"]))
    assert "VISION_WRONG" in codes(run_probes(ep(fake_server(_vision_server(colour="Green"))), ["vision"]))
    err = fake_server(lambda b: {"error": {"code": 500, "message": "image input is not supported"}})
    assert "VISION_UNSUPPORTED" in codes(run_probes(ep(err), ["vision"]))


def test_thinking_budget_is_not_blamed_when_the_request_turns_thinking_off(fake_server):
    def h(b):
        if (b.get("chat_template_kwargs") or {}).get("enable_thinking") is False:
            return completion("", "stop")
        if (b.get("max_tokens") or 0) <= 64:
            return completion("", "length", reasoning_content="Thinking...")
        return completion("OK", "stop", reasoning_content="Thinking...")
    e = Endpoint(fake_server(h).url, "m", budget_s=30, timeout=10)
    rep = triage(TriageInput(endpoint=e, messages=[{"role": "user", "content": "hi"}],
                             params={"max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False}},
                             probes=["reasoning"]))
    starve = [f for f in rep.findings if f.code == "REASONING_STARVES_ANSWER"]
    assert starve and not starve[0].failed
    assert (rep.verdict.get("primary") or {}).get("code") != "REASONING_STARVES_ANSWER"
