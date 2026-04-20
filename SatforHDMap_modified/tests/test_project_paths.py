import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from project_paths import ensure_repo_root_on_sys_path  # noqa: E402


def test_ensure_repo_root_on_sys_path_inserts_parent_repo_once():
    repo_root = PROJECT_ROOT.resolve()
    repo_root_str = str(repo_root)
    original = list(sys.path)

    while repo_root_str in sys.path:
        sys.path.remove(repo_root_str)

    try:
        returned = ensure_repo_root_on_sys_path()
        assert returned == repo_root
        assert sys.path[0] == repo_root_str

        ensure_repo_root_on_sys_path()
        assert sys.path.count(repo_root_str) == 1
    finally:
        sys.path[:] = original
