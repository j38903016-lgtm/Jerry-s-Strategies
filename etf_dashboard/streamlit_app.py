"""Streamlit Community Cloud entry point."""
from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).resolve().with_name("dashboard.py")),
    run_name="__main__",
)
