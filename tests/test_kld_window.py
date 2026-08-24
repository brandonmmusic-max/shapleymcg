import json

import pytest

from quant_pipeline.evaluation.kld_window import seal_kld_window_from_texts, verify_kld_window


def test_glm_style_kld_window_is_sealed_for_target_tokenizer(tmp_path):
    texts = ["", "  ", "alpha beta", "gamma delta epsilon"]

    def encode(text, maximum):
        return list(range(maximum))

    artifact = seal_kld_window_from_texts(
        texts,
        tmp_path / "kld",
        encode,
        {"class": "FakeQwenTokenizer", "expected_model_revision": "a" * 40, "files": {}},
        {"repo": "Salesforce/wikitext", "config": "wikitext-2-raw-v1", "split": "test", "revision": "b" * 40},
        context_length=8,
    )
    assert artifact["construction"]["join_separator"] == "\\n\\n"
    assert artifact["construction"]["prefix_characters_requested"] == 40
    assert artifact["token_ids"] == list(range(8))
    verify_kld_window(artifact, tmp_path / "kld")
    on_disk = json.loads((tmp_path / "kld" / "kld-window.json").read_text())
    verify_kld_window(on_disk, tmp_path / "kld")


def test_kld_window_rejects_tampering(tmp_path):
    artifact = seal_kld_window_from_texts(
        ["enough source text"],
        tmp_path / "kld",
        lambda text, maximum: list(range(maximum)),
        {"class": "Fake"},
        {"revision": "b" * 40},
        context_length=4,
    )
    artifact["token_ids"][0] = 99
    with pytest.raises(ValueError, match="body hash"):
        verify_kld_window(artifact, tmp_path / "kld")
