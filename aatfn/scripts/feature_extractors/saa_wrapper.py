"""Thin wrapper around saa/src/extractor.py's stego_analyzer().

21-dim frozen SAA vector. See saa/FEATURE_SPEC.md for the feature order --
do not touch that file or extractor.py from here.
"""
from .paths import setup_sys_path

setup_sys_path()
import extractor  # noqa: E402  (saa/src/extractor.py, added to sys.path above)

SAA_FEATURE_NAMES = extractor.FEATURE_NAMES
SAA_DIM = 21


def extract_saa_features(image_path: str):
    """-> np.ndarray shape (21,)"""
    return extractor.stego_analyzer(str(image_path))
