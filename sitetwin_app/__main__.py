"""Run SiteTwin alone: ``SITETWIN_DEV_ACCOUNT=me python -m sitetwin_app``."""

import os

from .app import create_app

if __name__ == "__main__":
    os.environ.setdefault("SITETWIN_DEV_ACCOUNT", "dev")
    create_app().run(port=int(os.environ.get("PORT", 5055)), debug=True)
