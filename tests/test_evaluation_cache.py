import json

import pytest

from misinformation_simulation.topic_drift.provenance import EvaluationCache, input_hash


@pytest.mark.parametrize("namespace", ["extraction", "judge", "vad_llm", "vad_model"])
def test_namespace_cache_survives_restart(tmp_path, namespace):
    inputs = {"text": "same text", "model": "test-model"}
    output = {"score": 0.25}
    filename = f"{namespace}_{input_hash(inputs)}.json"

    EvaluationCache(tmp_path).put(namespace, inputs, output)

    assert (tmp_path / namespace / filename).is_file()
    assert not (tmp_path / filename).exists()
    assert EvaluationCache(tmp_path).get(namespace, inputs) == output
    assert EvaluationCache(tmp_path).get(namespace, {**inputs, "model": "other"}) is None
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("namespace", ["extraction", "judge", "vad_llm", "vad_model"])
def test_legacy_flat_cache_is_reused_without_rewriting_it(tmp_path, namespace):
    inputs = {"text": "historical text"}
    output = {"score": 0.75}
    path = tmp_path / f"{namespace}_{input_hash(inputs)}.json"
    payload = json.dumps({"inputs": inputs, "output": output}).encode()
    path.write_bytes(payload)

    assert EvaluationCache(tmp_path).get(namespace, inputs) == output
    assert path.read_bytes() == payload
    assert not (tmp_path / namespace).exists()


def test_namespace_entry_takes_precedence_over_legacy_copy(tmp_path):
    inputs = {"text": "same input"}
    legacy = tmp_path / f"judge_{input_hash(inputs)}.json"
    legacy.write_text(json.dumps({"inputs": inputs, "output": {"score": 0.75}}))
    EvaluationCache(tmp_path).put("judge", inputs, {"score": 0.25})

    assert EvaluationCache(tmp_path).get("judge", inputs) == {"score": 0.25}
    assert json.loads(legacy.read_text())["output"] == {"score": 0.75}


def test_namespace_cache_still_rejects_input_mismatch(tmp_path):
    inputs = {"text": "expected"}
    directory = tmp_path / "judge"
    directory.mkdir()
    path = directory / f"judge_{input_hash(inputs)}.json"
    path.write_text(json.dumps({"inputs": {"text": "other"}, "output": {}}))

    with pytest.raises(ValueError, match="input mismatch"):
        EvaluationCache(tmp_path).get("judge", inputs)
