"""
app.py — Hugging Face Spaces entry point
-----------------------------------------
HF Spaces expects the Streamlit app file at the path declared in the
Space's README.md front-matter (app_file: app.py).

This thin shim just executes the real dashboard so the project layout
stays intact.  All logic lives in src/app/dashboard.py.
"""
# Re-export the dashboard module so Streamlit discovers its page config.
import runpy, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
runpy.run_path(str(Path(__file__).parent / "src" / "app" / "dashboard.py"), run_name="__main__")
