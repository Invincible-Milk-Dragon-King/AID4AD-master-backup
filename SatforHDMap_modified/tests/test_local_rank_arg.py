import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _local_rank_option_strings(script_name: str) -> list[str]:
    source = (ROOT / script_name).read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue

        option_strings = [
            arg.value for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        ]
        if "--local_rank" in option_strings or "--local-rank" in option_strings:
            return option_strings

    raise AssertionError(f"No local rank argument found in {script_name}")


@pytest.mark.parametrize("script_name", ["train.py", "train_amp.py", "test.py"])
def test_scripts_accept_hyphenated_local_rank_argument(script_name):
    option_strings = _local_rank_option_strings(script_name)

    assert "--local_rank" in option_strings
    assert "--local-rank" in option_strings
