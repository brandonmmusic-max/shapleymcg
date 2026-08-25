import json

from quant_pipeline.calibration.windows import document_split, seal_corpus, verify_sealed_corpus


def test_document_split_and_seal_prevent_role_leakage(tmp_path):
    path = tmp_path / "documents.jsonl"
    rows = []
    for domain in ("law", "code", "science", "prose"):
        for index in range(8):
            rows.append({"id": f"{domain}-{index}", "domain": domain, "text": "abcdefgh"})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "sealed.json"
    sealed = seal_corpus(
        path,
        output,
        lambda text: list(text.encode()),
        window_tokens=4,
        role_limits={
            role: 4
            for role in ("fit", "conditional_fit", "selection", "confirmation", "final")
        },
        seed=9,
        tokenizer_identity={"id": "test", "revision": "deadbeef"},
    )
    role_ids = [{window["document_id"] for window in sealed["windows"][role]} for role in sealed["windows"]]
    assert not any(
        role_ids[i] & role_ids[j]
        for i in range(len(role_ids))
        for j in range(i + 1, len(role_ids))
    )
    assert output.exists()
    verify_sealed_corpus(sealed)
    sealed["windows"]["final"][0]["token_ids"][0] += 1
    try:
        verify_sealed_corpus(sealed)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("tampered seal was accepted")


def test_document_split_is_deterministic():
    documents = [{"id": str(index), "domain": "x", "text": "x"} for index in range(20)]
    assert document_split(documents, 4) == document_split(documents, 4)
