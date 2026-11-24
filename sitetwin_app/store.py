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
CREATE TABLE IF NOT EXISTS seen (
    name        TEXT PRIMARY KEY,  -- 'worker': the last time a render machine asked for work
    at          REAL NOT NULL,
    detail      TEXT
);
"""

QUALITIES = ("preview", "draft", "final", "4k")
#: Columns added after the first release, added to an older database when it is opened.
LATER_COLUMNS = {
    "jobs": [("stage", "TEXT"), ("frames_done", "INTEGER"), ("frames_total", "INTEGER"),
             ("gpus", "INTEGER"), ("updated", "REAL"), ("cost", "REAL"), ("preview", "TEXT"),
             ("kind", "TEXT")],             # 'film' (NULL in older rows) or 'survey'
    # the project's drone survey: what was sent, how far it is (queued, running, ready, failed),
    # what the worker said, survey.json once made, and where the person placed it
    "projects": [("survey_name", "TEXT"), ("survey_state", "TEXT"), ("survey_message", "TEXT"),
                 ("survey_info", "TEXT"), ("survey_place", "TEXT")],
}
#: What a survey's placement may hold, with its limits.
PLACE_FIELDS = {"east": (-5000.0, 5000.0), "north": (-5000.0, 5000.0), "up": (-500.0, 500.0),
                "turn": (-360.0, 360.0), "width": (0.0, 20000.0), "scale": (0.01, 100.0)}
#: What a render goes through, in order, as the worker reports it.
JOB_STAGES = ("queued", "starting", "surroundings", "scene", "frames", "joining", "uploading", "done")
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
            for table, cols in LATER_COLUMNS.items():
                have = {r["name"] for r in db.execute(f"PRAGMA table_info({table})")}
                for name, kind in cols:
                    if name not in have:
                        db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")

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
        for key in ("survey_info", "survey_place"):
            out[key] = json.loads(out[key]) if out.get(key) else None
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

    def rename(self, owner: str, pid: int, name: str) -> dict[str, Any]:
        self.project(owner, pid)
        name = (name or "").strip()
        if not name:
            raise StoreError(400, "Give the project a name.")
        self._update(pid, {"name": name[:200]})
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
            busy = db.execute("SELECT id FROM jobs WHERE project_id=? AND status IN ('queued','running') "
                              "AND COALESCE(kind,'film')='film'", (pid,)).fetchone()
            if busy:
                raise StoreError(409, "A video of this project is already being made.")
            if project.get("survey_state") in ("queued", "running"):
                raise StoreError(409, "The drone survey is still being made: the film waits for it.")
            cur = db.execute("INSERT INTO jobs (project_id, owner, quality, seconds, status, stage, created, kind) "
                             "VALUES (?,?,?,?,'queued','queued',?,'film')", (pid, owner, quality, seconds, time.time()))
            jid = cur.lastrowid
        return self.job(jid)

    def job(self, jid: int) -> dict[str, Any]:
        with self._db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        if row is None:
            raise StoreError(404, "There is no such job.")
        return dict(row)

    def jobs(self, owner: str, pid: int, kind: str = "film") -> list[dict[str, Any]]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM jobs WHERE project_id=? AND owner=? AND COALESCE(kind,'film')=? "
                              "ORDER BY id DESC", (pid, owner, kind)).fetchall()
        return [dict(r) for r in rows]

    # ---- the drone survey -------------------------------------------------------------

    def queue_survey(self, owner: str, pid: int, name: str) -> dict[str, Any]:
        """A survey was sent (a video, photos or a finished model): the worker makes it ready."""
        self.project(owner, pid)
        with self._db() as db:
            db.execute("UPDATE jobs SET status='failed', message='Replaced by a newer survey.', finished=? "
                       "WHERE project_id=? AND kind='survey' AND status IN ('queued','running')", (time.time(), pid))
            cur = db.execute("INSERT INTO jobs (project_id, owner, quality, seconds, status, stage, created, kind) "
                             "VALUES (?,?,'survey',0,'queued','queued',?,'survey')", (pid, owner, time.time()))
            jid = cur.lastrowid
        self._update(pid, {"survey_name": name[:200], "survey_state": "queued", "survey_info": None,
                           "survey_message": "Waiting for the cloud machine"})
        return self.job(jid)

    def survey_done(self, jid: int, info: dict[str, Any] | None, message: str = "") -> dict[str, Any]:
        """The worker finished a survey job: ``info`` is its survey.json, or None if it failed."""
        job = self.job(jid)
        with self._db() as db:
            db.execute("UPDATE jobs SET status=?, message=?, finished=?, updated=?, percent=?, stage=? WHERE id=?",
                       ("done" if info else "failed", (message or "")[:1000], time.time(), time.time(),
                        100.0 if info else 0.0, "done" if info else None, jid))
        self._update(job["project_id"], {"survey_state": "ready" if info else "failed",
                                         "survey_info": json.dumps(info) if info else None,
                                         "survey_message": (message or "")[:1000]})
        return self.job(jid)

    def survey_progress(self, pid: int, message: str) -> None:
        self._update(pid, {"survey_state": "running", "survey_message": (message or "")[:300]})

    def set_survey_place(self, owner: str, pid: int, place: dict[str, Any]) -> dict[str, Any]:
        self.project(owner, pid)
        clean: dict[str, float] = {}
        for key, (lo, hi) in PLACE_FIELDS.items():
            v = place.get(key)
            if v in (None, ""):
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                raise StoreError(400, f"The survey's {key} must be a number.")
            if not lo <= f <= hi:
                raise StoreError(400, f"The survey's {key} must be between {lo:g} and {hi:g}.")
            clean[key] = f
        self._update(pid, {"survey_place": json.dumps(clean)})
        return self.project(owner, pid)

    def drop_survey(self, owner: str, pid: int) -> None:
        self.project(owner, pid)
        with self._db() as db:
            db.execute("UPDATE jobs SET status='failed', message='Survey removed.', finished=? "
                       "WHERE project_id=? AND kind='survey' AND status IN ('queued','running')", (time.time(), pid))
        self._update(pid, {"survey_name": None, "survey_state": None, "survey_info": None,
                           "survey_message": None, "survey_place": None})

    def claim(self, worker: str) -> dict[str, Any] | None:
        """The oldest waiting job (or one whose worker went quiet), now running for this worker."""
        now = time.time()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM jobs WHERE status='queued' OR (status='running' AND COALESCE(updated, started) < ?) "
                "ORDER BY id LIMIT 1", (now - STALE_SECONDS,)).fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running', worker=?, started=?, updated=?, percent=0, message=?, "
                       "stage='starting', frames_done=0 WHERE id=?",
                       (worker[:100], now, now, "Picked up by the render machine", row["id"]))
        return self.job(row["id"])

    def progress(self, jid: int, percent: float, message: str, **more: Any) -> None:
        """What the worker says it is doing. ``more`` may carry stage, frames_done, frames_total, gpus, cost."""
        fields: dict[str, Any] = {"percent": max(0.0, min(100.0, float(percent))),
                                  "message": (message or "")[:300], "updated": time.time()}
        if more.get("stage") in JOB_STAGES:
            fields["stage"] = more["stage"]
        for key in ("frames_done", "frames_total", "gpus"):
            if more.get(key) is not None:
                fields[key] = int(more[key])
        if more.get("cost") is not None:
            fields["cost"] = float(more["cost"])
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._db() as db:
            db.execute(f"UPDATE jobs SET {cols} WHERE id=? AND status='running'", (*fields.values(), jid))

    def set_preview(self, jid: int, name: str) -> None:
        with self._db() as db:
            db.execute("UPDATE jobs SET preview=?, updated=? WHERE id=?", (name, time.time(), jid))

    def finish(self, jid: int, video: str | None, message: str = "") -> dict[str, Any]:
        status = "done" if video else "failed"
        with self._db() as db:
            db.execute("UPDATE jobs SET status=?, video=?, message=?, finished=?, updated=?, percent=?, stage=? "
                       "WHERE id=?", (status, video, (message or "")[:1000], time.time(), time.time(),
                                      100.0 if video else 0.0, "done" if video else None, jid))
        return self.job(jid)

    def cancel(self, owner: str, pid: int, jid: int) -> dict[str, Any]:
        job = self.job(jid)
        if job["project_id"] != pid or job["owner"] != owner:
            raise StoreError(404, "There is no such job.")
        if job["status"] not in ("queued", "running"):
            raise StoreError(409, "That video is not being made.")
        return self.finish(jid, None, "Stopped by you.")

    def saw(self, name: str, detail: str = "") -> None:
        with self._db() as db:
            db.execute("INSERT INTO seen (name, at, detail) VALUES (?,?,?) "
                       "ON CONFLICT(name) DO UPDATE SET at=excluded.at, detail=excluded.detail",
                       (name, time.time(), detail[:200]))

    def last_seen(self, name: str) -> dict[str, Any] | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM seen WHERE name=?", (name,)).fetchone()
        return dict(row) if row else None

    def _update(self, pid: int, fields: dict[str, Any]) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._db() as db:
            db.execute(f"UPDATE projects SET {cols} WHERE id=?", (*fields.values(), pid))
