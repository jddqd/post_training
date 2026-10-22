import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # eval/bfcl is not a package

from run_bfcl import count_inference_errors


def test_failed_generations_are_counted(tmp_path: Path) -> None:
    # A timeout is stored as a result string: BFCL would score it as a correct "no call".
    results = tmp_path / "Qwen_Qwen3-0.6B-FC" / "non_live"
    results.mkdir(parents=True)
    (results / "BFCL_v4_irrelevance_result.json").write_text(
        json.dumps({"id": "irrelevance_0", "result": "Error during inference: Request timed out."})
        + "\n"
        + json.dumps({"id": "irrelevance_1", "result": "I cannot help with that."})
        + "\n"
    )

    assert count_inference_errors(tmp_path) == 1
