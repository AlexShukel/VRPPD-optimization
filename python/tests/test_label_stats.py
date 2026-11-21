import json
import os

from vrppd_nn import fileio, label_stats


def test_outputs_are_written_as_tmp_then_renamed(tmp_path, write_labels, monkeypatch):
    renames = []
    real_replace = os.replace

    def record(src, dst):
        renames.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(fileio.os, "replace", record)
    labels = write_labels("train.jsonl", 6, per_instance=2)
    out, md = tmp_path / "stats.json", tmp_path / "stats.md"
    label_stats.main(["--labels", labels, "--out", str(out), "--md", str(md)])
    assert sorted(renames) == sorted((str(p) + ".tmp", str(p)) for p in (out, md))
    assert json.loads(out.read_text())["n_states"] == 12
    assert md.read_text().startswith(f"Label statistics for `{labels}`: 12 states")
    assert sorted(os.listdir(tmp_path)) == ["stats.json", "stats.md", "train.jsonl"]
