# SiteTwin

**The build, played back before it happens.** SiteTwin takes a project's IFC model
exported from Revit, sets it in the *real* surroundings of its site, and renders the
construction sequence as a film a few minutes long: the structure going up in the
order it is built, with the cranes, piling rigs, trucks, pumps and crews that build it.

Not a cartoon. The aim throughout is that a client, a planner or a site manager looks
at it and recognises a real site.

SiteTwin is the ninth application on Project Control (`ahisham92/Project_Management`),
mounted at `/sitetwin` behind the site's sign-in, the same way Iris and Triton are.

## How a project goes

| Stage | What happens | Status |
|---|---|---|
| **Model** | Upload the `.ifc` from Revit. It is read for its elements, storeys and containment. | Working |
| **Location** | Taken from the model's `IfcSite` (latitude, longitude, elevation), or typed in. | Working |
| **Environment** | The real surroundings are pulled in around that point: terrain, existing buildings, roads, water, and photo-real 3D tiles. | Planned |
| **Sequence** | A first draft is read straight from the model (storey by storey, foundations → frame → floors → envelope → services). The planner edits it, or brings in a P6 / MS Project programme, and each step gets its plant and crews. | Draft working |
| **Video** | The sequence rendered as a film with real lighting, shadows, dust and moving equipment and people. | Planned |

## Architecture

The web app here is deliberately light (Flask + SQLite, pure Python, nothing to compile)
so it runs inside Project Control on PythonAnywhere. The heavy lifting, which needs a GPU,
runs as a separate **render worker** that picks up jobs and posts the finished video back.

```
 Revit ──IFC──▶ SiteTwin web app ──job──▶ Render worker (GPU)
                 (projects, model,          ├─ IFC → meshes (IfcOpenShell)
                  location, sequence)       ├─ Environment at the site's coordinates
                       ▲                    ├─ Sequence → animated timeline
                       └──── video ─────────└─ Render (Unreal Engine 5 / Blender Cycles)
```

**Making it real** (the plan for the worker):

- **Geometry:** IfcOpenShell converts IFC to glTF with Revit materials kept; `IfcMapConversion`
  or the site's lat/long places it exactly.
- **Environment:** Google Photorealistic 3D Tiles (or Cesium ion) for the real surroundings,
  with Copernicus / SRTM terrain and OpenStreetMap buildings and roads as the fallback where
  tiles are thin. The site itself is cut back to a construction site: hoarding, a cleared and
  graded plot, laydown areas, site cabins.
- **Equipment and people:** a library of real, rigged plant (tower and crawler cranes, piling
  rigs, excavators, concrete pumps and trucks, MEWPs) and crews in PPE, placed per sequence
  step and animated (crane lifts, pours, deliveries) rather than elements just appearing.
- **Rendering:** Unreal Engine 5 (Lumen, Nanite, Movie Render Queue) for photo-real output,
  with Blender Cycles as the scriptable headless alternative. Real sun position from the
  site's coordinates and the programme dates.

## Running it

```bash
pip install -r requirements-dev.txt
SITETWIN_DEV_ACCOUNT=me python -m sitetwin_app      # http://localhost:5055
python -m pytest
```

Data (database, models, videos) goes to `SITETWIN_DATA_DIR`, or `./data` when that is not set.

## Mounted in Project Control

Project Control imports `sitetwin_app.wsgi` and uses `application`, `get_app()`,
`is_public(path)` and `SITE_KEY`: it signs the person in, checks their account has been
given SiteTwin, and puts `{"id", "name", "login"}` into the request under `SITE_KEY`.
SiteTwin has no sign-in of its own. To update the site, copy the `sitetwin_app` folder
into the site's repository beside `app` and note the commit in its `SOURCE.txt`.
