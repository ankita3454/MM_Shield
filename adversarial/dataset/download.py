"""Phase 1: ensure the shared training datasets are downloaded and normalized.

FUNSD/CORD/SROIE/DocLayNet are attack-type-agnostic - the same Hugging Face
sources, the same unified-record normalization, regardless of whether a
typographic injection or an adversarial patch gets applied downstream. This
module reuses typographic.dataset.download directly rather than
reimplementing it; its only job is making sure everything this module needs
is present on disk (a no-op if the Typographic module already downloaded it,
which it will have in this project).
"""

from typographic.dataset.download import download_all, download_dataset, download_external_sample

__all__ = ["download_all", "download_dataset", "download_external_sample", "ensure_datasets_ready"]


def ensure_datasets_ready(force: bool = False) -> None:
    download_all(force=force)


if __name__ == "__main__":
    ensure_datasets_ready()
