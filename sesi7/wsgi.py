"""WSGI entry for the free PythonAnywhere web app.

The Web tab loads this module and uses the Flask server inside the Dash app.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_FILE = ROOT / "sesi7" / "app_dash_callback.py"

spec = importlib.util.spec_from_file_location("app_dash_callback", APP_FILE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

application = module.app.server
