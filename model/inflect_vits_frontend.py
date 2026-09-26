from __future__ import annotations

from deployment_frontend import FrontendOutput, process_input


VitsFrontendOutput = FrontendOutput


def run_vits_frontend(
    text: str | None = None,
    *,
    phonemes: str | None = None,
) -> VitsFrontendOutput:
    return process_input(text, phonemes=phonemes)


def run_vits_frontend_batch(
    texts: list[str],
    *,
    jobs: int = 1,
) -> list[VitsFrontendOutput]:
    del jobs
    return [process_input(text) for text in texts]
