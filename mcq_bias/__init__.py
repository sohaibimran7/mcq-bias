"""MCQ Bias — an Inspect AI eval of answer-switching under biased MCQ prompts.

Self-contained: tasks, solver, scorers, answer parsers, the data-generation
pipeline, and the wrong-argument stores all live in this package. See README.md.
"""

from mcq_bias.tasks import mcq_bias, mcq_bias_unbiased

__all__ = ["mcq_bias", "mcq_bias_unbiased"]
