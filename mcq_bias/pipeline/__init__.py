"""Data pipeline for the sycophancy bias eval.

Builds eval samples from live sources rather than shipping data files:

- ``sources``    — canonical MCQRecords from public HF datasets (mmlu, ...)
- ``records``    — question canonicalization matching the original
                   cot-transparency codebase: canonical question text, sha1
                   question id, deterministically chosen biased option
- ``injectors``  — bias cues as deterministic transforms, reproducing the
                   data released with the original codebase
- ``build``      — matched biased/control datasets, paired by sample id,
                   aligned by construction
"""
