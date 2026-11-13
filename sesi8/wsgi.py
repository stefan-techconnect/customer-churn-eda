"""WSGI entry for the Sesi 8 dashboard on PythonAnywhere.

The Web tab's WSGI file should only contain this, with the new account name:

    import sys
    sys.path.insert(0, "/home/YOUR_USERNAME/customer-churn-eda")
    from sesi8.wsgi import application
"""
import importlib.util
from pathlib import Path

APP_FILE = Path(__file__).resolve().parent / "app_final_dashboard.py"
spec = importlib.util.spec_from_file_location("app_final_dashboard", APP_FILE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

application = module.app.server
