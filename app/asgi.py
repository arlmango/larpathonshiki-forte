import os

from .main import create_app

if os.environ.get("SYNTHETIC_DETECTOR") and os.environ.get("ALLOW_SYNTHETIC_DETECTOR") != "1":
    raise RuntimeError("synthetic detector is test/demo only; refusing to start")

app = create_app()
