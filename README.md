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
| **Environment** | The real surroundings are fetched around that point, automatically: terrain, aerial imagery, existing buildings, roads, railways and water. | Working (render worker) |
| **Sequence** | A first draft is read straight from the model (storey by storey, foundations → frame → floors → envelope → services). The planner edits it, or brings in a P6 / MS Project programme, and each step gets its plant and crews. | Draft working |
| **Video** | The sequence rendered as a film with the real sun for the site and date, plant and crews at work, captions and credits. Press **Make the video** on the project page. | Working (render worker) |

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

**Making it real** (the original plan; see "The render worker" for what is built):

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

## The render worker

Nothing is done site by site. Press **Make the video** on a project and a render
machine does the rest:

1. **Model.** IfcOpenShell turns every element of the IFC into a mesh in metres.
2. **Surroundings**, from the location alone (`sitetwin_worker/environment.py`), each optional:
   - terrain: AWS Terrain Tiles (SRTM, Copernicus, national surveys), with the site graded flat,
     an excavation for the foundations, and the sea where the tiles say there is sea;
   - aerial imagery draped over the terrain: Esri World Imagery by default, or any XYZ service
     (`SITETWIN_IMAGERY_URL`, e.g. Mapbox, MapTiler or a national orthophoto);
   - OpenStreetMap through Overpass: the neighbouring buildings at their mapped heights or
     storeys (believable heights where they are not mapped), roads, rail, water, parks.
     Buildings on the site itself are cleared.
3. **Sequence** (`sequence.py`). Storey by storey, foundations → frame → floors → stairs →
   roof → envelope → finishes → services, each element placed in turn: concrete grows from its
   base, beams, precast and glazing are lowered in, finishes appear.
4. **Plant and crews** (`equipment.py`). A tower crane sized to the building (its jib swings to
   each lifted element, the hook pays out to it), a piling rig and an excavator in the dig,
   mixer and flatbed trucks coming and going through the gate, a boom pump at the pours, a
   scissor lift round the facade, and the crew for each step on the floor being built. Hoarding,
   a gate on the land side, site cabins and a laydown area. Each is modelled in code; drop a
   real glTF model (e.g. `tower_crane.glb`) into the folder `SITETWIN_ASSETS` names and it is used
   instead.
5. **Render** (`scene.py`). Blender Cycles, on the GPU when there is one. Physical sky and the sun at
   the site's real latitude/longitude and date, AgX colour, procedural concrete, steel, glass,
   soil and asphalt, a slow orbiting camera. Each frame is captioned with the step being built
   and the data credits; ffmpeg joins them into an MP4, sent back to the project.

| Quality | Size | A frame on a mid-range GPU | A frame on a 4-core CPU |
|---|---|---|---|
| preview | 640×360 | under 1 s | about 5 s |
| draft | 1280×720 | 1–2 s | about 25 s |
| final | 1920×1080 | 3–6 s | minutes |

A 90-second film at 30 fps is 2,700 frames, so a final film wants a GPU.

### Running a render machine

```bash
pip install -r requirements-worker.txt        # Python 3.13; ffmpeg on PATH
export SITETWIN_WORKER_TOKEN=...              # the same token set on the site
python -m sitetwin_worker serve --site https://your.site/sitetwin
```

It only calls out to the site, so it works behind an office firewall, on a PC or on a cloud GPU.
To render one model directly, without the site:

```bash
python -m sitetwin_worker render model.ifc --out film/ --quality draft --seconds 90
python -m sitetwin_worker render model.ifc --frames 1,400,900 --quality preview   # a few stills to check
python tools/make_sample_ifc.py sample.ifc --lat 33.66 --lon 35.42                # a test building
```

### Making it more real, next

- Google Photorealistic 3D Tiles for cities that have them (needs a Google Maps Platform key).
- Real plant and crew models (manufacturer or photogrammetry glTFs) in `SITETWIN_ASSETS`.
- Formwork, rebar cages and props shown before each pour; scaffolding on the facade.
- Dust, vehicle tracks and weather; the programme's real dates for the sun and the seasons.
- Bringing in the planner's own programme (P6 / MS Project XML, or IFC 4D tasks) instead of the draft.

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
