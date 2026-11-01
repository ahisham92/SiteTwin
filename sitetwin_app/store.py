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
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    owner       TEXT NOT NULL,
    quality     TEXT NOT NULL,
    seconds     REAL NOT NULL,
    status      TEXT NOT NULL,       -- queued, running, done, failed
    percent     REAL NOT NULL DEFAULT 0,
    message     TEXT,
    worker      TEXT,
    created     REAL NOT NULL,
    started     REAL,
    finished    REAL,
    video       TEXT
);
"""

QUALITIES = ("preview", "draft", "final", "4k")
#: A job a worker took and has not reported on for this long is offered again.
STALE_SECONDS = 30 * 60


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
            db.execute("DELETE FROM jobs WHERE project_id=?", (pid,))
            db.execute("DELETE FROM projects WHERE id=?", (pid,))

    # ---- render jobs ----------------------------------------------------------------

    def queue_render(self, owner: str, pid: int, quality: str = "draft", seconds: float = 90) -> dict[str, Any]:
        project = self.project(owner, pid)
        if not project["model_name"]:
            raise StoreError(400, "Upload the IFC model first.")
        if project["latitude"] is None:
            raise StoreError(400, "Set the location first: the surroundings are fetched from it.")
        if quality not in QUALITIES:
            raise StoreError(400, "Unknown quality.")
        seconds = float(seconds)
        if not 10 <= seconds <= 600:
            raise StoreError(400, "A film runs between 10 seconds and 10 minutes.")
        with self._db() as db:
            busy = db.execute("SELECT id FROM jobs WHERE project_id=? AND status IN ('queued','running')",
                              (pid,)).fetchone()
            if busy:
                raise StoreError(409, "A video of this project is already being made.")
            cur = db.execute("INSERT INTO jobs (project_id, owner, quality, seconds, status, created) "
                             "VALUES (?,?,?,?,'queued',?)", (pid, owner, quality, seconds, time.time()))
            jid = cur.lastrowid
        return self.job(jid)

    def job(self, jid: int) -> dict[str, Any]:
        with self._db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        if row is None:
            raise StoreError(404, "There is no such job.")
        return dict(row)

    def jobs(self, owner: str, pid: int) -> list[dict[str, Any]]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM jobs WHERE project_id=? AND owner=? ORDER BY id DESC",
                              (pid, owner)).fetchall()
        return [dict(r) for r in rows]

    def claim(self, worker: str) -> dict[str, Any] | None:
        """The oldest waiting job (or one whose worker went quiet), now running for this worker."""
        now = time.time()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM jobs WHERE status='queued' OR (status='running' AND started < ?) "
                "ORDER BY id LIMIT 1", (now - STALE_SECONDS,)).fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running', worker=?, started=?, percent=0, message=? WHERE id=?",
                       (worker[:100], now, "Picked up", row["id"]))
        return self.job(row["id"])

    def progress(self, jid: int, percent: float, message: str) -> None:
        with self._db() as db:
            db.execute("UPDATE jobs SET percent=?, message=?, started=? WHERE id=? AND status='running'",
                       (max(0.0, min(100.0, float(percent))), (message or "")[:300], time.time(), jid))

    def finish(self, jid: int, video: str | None, message: str = "") -> dict[str, Any]:
        status = "done" if video else "failed"
        with self._db() as db:
            db.execute("UPDATE jobs SET status=?, video=?, message=?, finished=?, percent=? WHERE id=?",
                       (status, video, (message or "")[:1000], time.time(), 100.0 if video else 0.0, jid))
        return self.job(jid)

    def _update(self, pid: int, fields: dict[str, Any]) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._db() as db:
            db.execute(f"UPDATE projects SET {cols} WHERE id=?", (*fields.values(), pid))
