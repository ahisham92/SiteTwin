import io
import json
import zipfile

from sitetwin_worker import assets


def _zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_models_are_downloaded_once_and_credited(tmp_path):
    calls = []

    def get(url, headers):
        calls.append(url)
        if "api.sketchfab.com" in url:
            assert headers["Authorization"] == "Token secret"
            return json.dumps({"gltf": {"url": "https://files/x.zip"}}).encode()
        assert headers == {}
        return _zip({"scene.gltf": "{}", "scene.bin": "x", "textures/a.png": "p"})

    files, credits = assets.prepare(tmp_path, token="secret", get=get, log=lambda m: None)
    assert set(files) == set(assets.CATALOG)
    assert all(f.endswith("scene.gltf") for f in files.values())
    assert len(credits) == len(assets.CATALOG) and "doty_aecom" in " ".join(credits)
    n = len(calls)
    again, _ = assets.prepare(tmp_path, token="secret", get=get, log=lambda m: None)
    assert again == files and len(calls) == n                 # cached, nothing fetched again


def test_without_a_token_or_on_failure_the_built_in_shapes_stand_in(tmp_path):
    assert assets.prepare(tmp_path, token="") == ({}, [])
    said = []

    def broken(url, headers):
        raise OSError("401 for Token secret")

    files, credits = assets.prepare(tmp_path, token="secret", get=broken, log=said.append)
    assert files == {} and credits == []
    assert said and not any("secret" in s for s in said)       # the token is never logged


def test_a_zip_cannot_write_outside_its_folder(tmp_path):
    def get(url, headers):
        if "api.sketchfab.com" in url:
            return json.dumps({"gltf": {"url": "https://files/x.zip"}}).encode()
        return _zip({"../../evil.gltf": "{}"})

    files, _ = assets.prepare(tmp_path / "c", token="t", get=get, log=lambda m: None)
    assert files == {}
    assert not (tmp_path / "evil.gltf").exists()
