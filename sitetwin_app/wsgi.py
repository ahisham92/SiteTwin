"""WSGI entry point, for a site that mounts SiteTwin or for running it alone.

A site that mounts SiteTwin (as Project Control does in front of ``/sitetwin``)
imports ``application`` and ``is_public`` from here, signs the person in itself,
and puts who they are into the request under ``SITE_KEY``. Set
``SITETWIN_DATA_DIR`` to a folder outside the code.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from .app import SITE_KEY, create_app, is_public

__all__ = ["SITE_KEY", "application", "get_app", "is_public"]

_app = None


def get_app():
    global _app
    if _app is None:
        _app = create_app()
    return _app


def application(environ: dict[str, Any], start_response: Callable) -> Iterable[bytes]:
    return get_app()(environ, start_response)
