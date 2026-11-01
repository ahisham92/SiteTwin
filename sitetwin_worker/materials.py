"""Physically based materials for the scene, built from Blender shader nodes.

No texture files are needed: concrete, steel, glass, asphalt, soil, water and the
rest are procedural, with enough variation (noise in colour and roughness, a little
bump) that surfaces do not read as flat CG colours.
"""

from __future__ import annotations

from typing import Any

_cache: dict[str, Any] = {}


def _principled(name: str):
    import bpy
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nodes = m.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    return m, bsdf, m.node_tree


def _noise_mix(tree, bsdf, c1, c2, scale=6.0, detail=8.0, rough=(0.6, 0.9), bump=0.15):
    n = tree.nodes
    tex_coord = n.new("ShaderNodeTexCoord")
    noise = n.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = detail
    tree.links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])
    ramp = n.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*c1, 1)
    ramp.color_ramp.elements[1].color = (*c2, 1)
    tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    rr = n.new("ShaderNodeMapRange")
    rr.inputs["To Min"].default_value, rr.inputs["To Max"].default_value = rough
    tree.links.new(noise.outputs["Fac"], rr.inputs["Value"])
    tree.links.new(rr.outputs["Result"], bsdf.inputs["Roughness"])
    if bump:
        b = n.new("ShaderNodeBump")
        b.inputs["Strength"].default_value = bump
        fine = n.new("ShaderNodeTexNoise")
        fine.inputs["Scale"].default_value = scale * 12
        tree.links.new(tex_coord.outputs["Object"], fine.inputs["Vector"])
        tree.links.new(fine.outputs["Fac"], b.inputs["Height"])
        tree.links.new(b.outputs["Normal"], bsdf.inputs["Normal"])


def get(name: str):
    """A material by its role, made once."""
    if name in _cache:
        return _cache[name]
    m, bsdf, tree = _principled("st_" + name)
    if name == "concrete":
        _noise_mix(tree, bsdf, (0.40, 0.39, 0.37), (0.52, 0.51, 0.48), scale=3, rough=(0.75, 0.95))
    elif name == "concrete_fresh":
        _noise_mix(tree, bsdf, (0.33, 0.33, 0.32), (0.42, 0.42, 0.41), scale=4, rough=(0.5, 0.8))
    elif name == "steel":
        bsdf.inputs["Base Color"].default_value = (0.32, 0.34, 0.36, 1)
        bsdf.inputs["Metallic"].default_value = 0.9
        bsdf.inputs["Roughness"].default_value = 0.4
    elif name == "steel_red":           # primed steel
        _noise_mix(tree, bsdf, (0.35, 0.08, 0.05), (0.45, 0.12, 0.07), scale=8, rough=(0.5, 0.7), bump=0.05)
    elif name == "glass":
        bsdf.inputs["Base Color"].default_value = (0.55, 0.65, 0.68, 1)
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 0.03
        bsdf.inputs["Transmission Weight"].default_value = 0.85
        bsdf.inputs["IOR"].default_value = 1.5
    elif name == "mullion":
        bsdf.inputs["Base Color"].default_value = (0.18, 0.19, 0.2, 1)
        bsdf.inputs["Metallic"].default_value = 0.8
        bsdf.inputs["Roughness"].default_value = 0.35
    elif name == "wood":
        _noise_mix(tree, bsdf, (0.30, 0.18, 0.09), (0.42, 0.26, 0.13), scale=20, rough=(0.5, 0.7), bump=0.05)
    elif name == "soil":
        _noise_mix(tree, bsdf, (0.27, 0.20, 0.14), (0.40, 0.31, 0.22), scale=0.4, detail=12, rough=(0.85, 1.0), bump=0.4)
    elif name == "gravel":
        _noise_mix(tree, bsdf, (0.26, 0.24, 0.21), (0.38, 0.35, 0.30), scale=1.5, detail=14, rough=(0.8, 1.0), bump=0.5)
    elif name == "ground":
        _noise_mix(tree, bsdf, (0.33, 0.28, 0.18), (0.20, 0.25, 0.11), scale=0.03, detail=12, rough=(0.85, 1.0), bump=0.2)
    elif name == "grass":
        _noise_mix(tree, bsdf, (0.13, 0.22, 0.07), (0.22, 0.30, 0.10), scale=0.2, detail=10, rough=(0.8, 1.0), bump=0.3)
    elif name == "sand":
        _noise_mix(tree, bsdf, (0.72, 0.63, 0.47), (0.80, 0.72, 0.56), scale=0.3, rough=(0.85, 1.0), bump=0.2)
    elif name == "asphalt":
        _noise_mix(tree, bsdf, (0.07, 0.07, 0.075), (0.13, 0.13, 0.135), scale=2, detail=12, rough=(0.75, 0.95), bump=0.2)
    elif name == "rail":
        _noise_mix(tree, bsdf, (0.20, 0.17, 0.14), (0.28, 0.25, 0.22), scale=3, rough=(0.8, 1.0))
    elif name == "water":
        bsdf.inputs["Base Color"].default_value = (0.02, 0.07, 0.09, 1)
        bsdf.inputs["Roughness"].default_value = 0.06
        bsdf.inputs["Specular IOR Level"].default_value = 0.6
        n = tree.nodes
        tc = n.new("ShaderNodeTexCoord")
        wave = n.new("ShaderNodeTexNoise")
        wave.inputs["Scale"].default_value = 0.35
        wave.inputs["Detail"].default_value = 6
        tree.links.new(tc.outputs["Object"], wave.inputs["Vector"])
        b = n.new("ShaderNodeBump")
        b.inputs["Strength"].default_value = 0.08
        tree.links.new(wave.outputs["Fac"], b.inputs["Height"])
        tree.links.new(b.outputs["Normal"], bsdf.inputs["Normal"])
    elif name == "facade":              # the neighbours: rendered plaster and stone
        _noise_mix(tree, bsdf, (0.62, 0.58, 0.52), (0.78, 0.75, 0.70), scale=0.08, rough=(0.7, 0.9), bump=0.1)
    elif name == "facade_dark":
        _noise_mix(tree, bsdf, (0.35, 0.36, 0.38), (0.48, 0.49, 0.50), scale=0.08, rough=(0.5, 0.8), bump=0.1)
    elif name == "roof":
        _noise_mix(tree, bsdf, (0.40, 0.39, 0.37), (0.55, 0.53, 0.50), scale=0.3, rough=(0.8, 1.0), bump=0.2)
    elif name == "crane_yellow":
        _noise_mix(tree, bsdf, (0.85, 0.55, 0.02), (0.92, 0.65, 0.06), scale=10, rough=(0.35, 0.55), bump=0.03)
    elif name == "plant_yellow":
        _noise_mix(tree, bsdf, (0.80, 0.50, 0.03), (0.88, 0.60, 0.08), scale=10, rough=(0.3, 0.5), bump=0.03)
    elif name == "truck_white":
        bsdf.inputs["Base Color"].default_value = (0.85, 0.85, 0.83, 1)
        bsdf.inputs["Roughness"].default_value = 0.25
    elif name == "drum_orange":
        bsdf.inputs["Base Color"].default_value = (0.85, 0.32, 0.05, 1)
        bsdf.inputs["Roughness"].default_value = 0.35
    elif name == "rubber":
        bsdf.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1)
        bsdf.inputs["Roughness"].default_value = 0.8
    elif name == "dark_metal":
        bsdf.inputs["Base Color"].default_value = (0.08, 0.08, 0.09, 1)
        bsdf.inputs["Metallic"].default_value = 0.7
        bsdf.inputs["Roughness"].default_value = 0.5
    elif name == "cab_glass":
        bsdf.inputs["Base Color"].default_value = (0.05, 0.07, 0.08, 1)
        bsdf.inputs["Roughness"].default_value = 0.05
    elif name == "hivis":
        bsdf.inputs["Base Color"].default_value = (1.0, 0.35, 0.0, 1)
        bsdf.inputs["Roughness"].default_value = 0.6
    elif name == "hivis_yellow":
        bsdf.inputs["Base Color"].default_value = (0.85, 0.95, 0.05, 1)
        bsdf.inputs["Roughness"].default_value = 0.6
    elif name == "workwear":
        bsdf.inputs["Base Color"].default_value = (0.06, 0.08, 0.13, 1)
        bsdf.inputs["Roughness"].default_value = 0.9
    elif name == "skin":
        bsdf.inputs["Base Color"].default_value = (0.55, 0.38, 0.27, 1)
        bsdf.inputs["Roughness"].default_value = 0.6
        bsdf.inputs["Subsurface Weight"].default_value = 0.1
    elif name == "helmet":
        bsdf.inputs["Base Color"].default_value = (0.92, 0.92, 0.9, 1)
        bsdf.inputs["Roughness"].default_value = 0.3
    elif name == "hoarding":
        _noise_mix(tree, bsdf, (0.10, 0.25, 0.45), (0.12, 0.28, 0.50), scale=4, rough=(0.4, 0.6), bump=0.02)
    elif name == "cabin":
        _noise_mix(tree, bsdf, (0.85, 0.85, 0.82), (0.92, 0.92, 0.9), scale=4, rough=(0.4, 0.6), bump=0.03)
    elif name == "formwork":
        _noise_mix(tree, bsdf, (0.55, 0.36, 0.12), (0.65, 0.45, 0.18), scale=6, rough=(0.5, 0.7), bump=0.03)
    elif name == "rebar":
        bsdf.inputs["Base Color"].default_value = (0.25, 0.12, 0.07, 1)
        bsdf.inputs["Metallic"].default_value = 0.6
        bsdf.inputs["Roughness"].default_value = 0.7
    else:
        bsdf.inputs["Base Color"].default_value = (0.6, 0.6, 0.6, 1)
    _cache[name] = m
    return m


def imagery(path: str, name: str = "st_imagery"):
    """The aerial photo draped over the terrain, slightly rough so it does not shine."""
    import bpy
    if name in _cache:
        return _cache[name]
    m, bsdf, tree = _principled(name)
    tex = tree.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(path)
    tex.extension = "EXTEND"
    tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.9
    bsdf.inputs["Specular IOR Level"].default_value = 0.2
    _cache[name] = m
    return m


# The role each IFC class plays.
FOR_KIND = {
    "IFCPILE": "concrete_fresh", "IFCFOOTING": "concrete_fresh", "IFCCOLUMN": "concrete",
    "IFCWALL": "concrete", "IFCWALLSTANDARDCASE": "concrete", "IFCSLAB": "concrete",
    "IFCBEAM": "concrete", "IFCMEMBER": "steel", "IFCPLATE": "steel", "IFCSTAIR": "concrete",
    "IFCSTAIRFLIGHT": "concrete", "IFCRAMP": "concrete", "IFCRAMPFLIGHT": "concrete",
    "IFCROOF": "roof", "IFCCURTAINWALL": "glass", "IFCWINDOW": "glass", "IFCDOOR": "wood",
    "IFCRAILING": "mullion", "IFCCOVERING": "facade",
}


def for_element(kind: str):
    return get(FOR_KIND.get(kind, "steel" if "FLOW" in kind or "SEGMENT" in kind else "concrete"))


def reset() -> None:
    _cache.clear()
