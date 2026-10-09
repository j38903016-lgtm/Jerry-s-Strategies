from pathlib import Path
import os

from optimizer.dashboard import render


render(os.environ.get(
    "HOLDING_OPTIMIZER_DB",
    str(Path(__file__).resolve().parent / "data" / "optimizer.sqlite3"),
))

