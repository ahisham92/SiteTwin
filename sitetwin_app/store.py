"""SiteTwin's projects, on disk: one SQLite database and one folder per project.

Everything lives under ``SITETWIN_DATA_DIR`` (``./data`` when it is not set), so
the code can be replaced without touching anybody's models or videos.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    owner       TEXT NOT NULL,
    name        TEXT NOT NULL,
    created     REAL NOT NULL,
    latitude    REAL,
    longitude   REAL,
    elevation   REAL,
    location_from TEXT,            -- 'ifc' or 'typed'
    model_name  TEXT,
    model_summary TEXT             -- JSON from ifc.summarise
);
"""


class StoreError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def data_folder() -> Path:
    return Path(os.environ.get("SITETWIN_DATA_DIR") or Path.cwd() / "data")


class Store:
    def __init__(self, folder: Path | None = None):
        self.folder = Path(folder or data_folder())
        self.folder.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript(SCHEMA)

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.folder / "sitetwin.db")
        db.row_factory = sqlite3.Row
        return db

    def project_folder(self, pid: int) -> Path:
        folder = self.folder / "projects" / str(pid)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        out["model_summary"] = json.loads(out["model_summary"]) if out["model_summary"] else None
        return out

    def projects(self, owner: str) -> list[dict[str, Any]]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM projects WHERE owner=? ORDER BY created DESC", (owner,))
            return [self._row(r) for r in rows]

    def project(self, owner: str, pid: int) -> dict[str, Any]:
        with self._db() as db:
            row = db.execute("SELECT * FROM projects WHERE id=? AND owner=?", (pid, owner)).fetchone()
        if row is None:
            raise StoreError(404, "There is no such project.")
        return self._row(row)

    def create(self, owner: str, name: str) -> dict[str, Any]:
        name = (name or "").strip()
        if not name:
            raise StoreError(400, "Give the project a name.")
        with self._db() as db:
            cur = db.execute("INSERT INTO projects (owner, name, created) VALUES (?,?,?)",
                             (owner, name[:200], time.time()))
            pid = cur.lastrowid
        return self.project(owner, pid)

    def set_model(self, owner: str, pid: int, filename: str, summary: dict[str, Any]) -> dict[str, Any]:
        project = self.project(owner, pid)
        fields: dict[str, Any] = {"model_name": filename, "model_summary": json.dumps(summary)}
        # The model's own location wins over nothing, never over one typed in.
        if project["location_from"] != "typed" and summary.get("latitude") is not None:
            fields.update(latitude=summary["latitude"], longitude=summary["longitude"],
                          elevation=summary.get("elevation"), location_from="ifc")
        self._update(pid, fields)
        return self.project(owner, pid)

    def set_location(self, owner: str, pid: int, latitude: float, longitude: float,
                     elevation: float | None = None) -> dict[str, Any]:
        self.project(owner, pid)
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise StoreError(400, "That is not a place on Earth.")
        self._update(pid, {"latitude": latitude, "longitude": longitude,
                           "elevation": elevation, "location_from": "typed"})
        return self.project(owner, pid)

    def delete(self, owner: str, pid: int) -> None:
        self.project(owner, pid)
        with self._db() as db:
            db.execute("DELETE FROM projects WHERE id=?", (pid,))

    def _update(self, pid: int, fields: dict[str, Any]) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._db() as db:
            db.execute(f"UPDATE projects SET {cols} WHERE id=?", (*fields.values(), pid))
