"""WSGI entry for the Sesi 6 Plotly dashboard on PythonAnywhere."""
import importlib.util
from pathlib import Path

APP_FILE = Path(__file__).resolve().parent / "app_dash_plotly.py"
spec = importlib.util.spec_from_file_location("app_dash_plotly", APP_FILE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

application = module.app.server
