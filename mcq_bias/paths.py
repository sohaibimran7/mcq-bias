"""Where mcq_bias reads and writes data (frozen datasets, wrong-argument stores).

Nothing ships with the package and nothing is ever written into the installed
package directory: the data root is ``$MCQ_BIAS_DATA_DIR`` when set, else
``~/.cache/mcq_bias``. Every task, scorer, and CLI entry point also accepts an
explicit ``dataset_dir``/``data_dir`` override, which always wins.
"""

import os
from pathlib import Path


def data_root() -> Path:
    return Path(os.environ.get("MCQ_BIAS_DATA_DIR") or Path.home() / ".cache" / "mcq_bias").expanduser()


def generated_dir() -> Path:
    """Default home of frozen datasets (the materialize-once cache)."""
    return data_root() / "generated"
