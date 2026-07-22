"""Shared sys.path setup so the three MMShield feature modules (which live
in different corners of the repo with different import conventions) can all
be imported from this script regardless of cwd.

  - saa/src/  has no __init__.py and uses bare sibling imports
    (`from entropy import entropy_manual`) -- must be added to sys.path
    directly, not imported as a package.
  - typographic/ and adversarial/ ARE proper packages (have __init__.py)
    that import each other as `typographic.xxx` / `adversarial.xxx` -- the
    repo ROOT must be on sys.path for those imports to resolve.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # .../MM_Shield
SAA_SRC = REPO_ROOT / "saa" / "src"
AATFN_DIR = REPO_ROOT / "aatfn"


def setup_sys_path():
    for p in (str(REPO_ROOT), str(SAA_SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)
