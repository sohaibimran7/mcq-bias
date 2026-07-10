"""Inspect-native data pipeline for the sycophancy bias eval.

Instead of loading frozen JSONL dumps, this package builds eval samples live:

- ``sources``    — canonical MCQRecords from public HF datasets (mmlu, ...)
- ``records``    — legacy-exact canonicalization: parsed text, sha1 question
                   hash, deterministic-random biased option
- ``injectors``  — bias cues as deterministic transforms (golden-tested
                   byte-identical to the released dumps)
- ``build``      — matched biased/control datasets, paired by sample id,
                   aligned by construction
"""
