"""Physically based materials for the scene, built from Blender shader nodes.

No texture files are needed: concrete, steel, glass, asphalt, soil, water and the
rest are procedural, with enough variation (noise in colour and roughness, a little
bump) that surfaces do not read as flat CG colours.
"""

from __future__ import annotations

from typing import Any

_cache: dict[str, Any] = {}
#: Photo textures to use where they were fetched (see assets.fetch_textures): role -> maps.
TEXTURES: dict[str, dict[str, str]] = {}
#: How many metres one repeat of each texture covers.
TEXTURE_SIZE = {"gravel": 3.0, "soil": 4.0, "concrete": 2.5, "concrete_fresh": 2.5, "asphalt": 4.0,
                "formwork": 2.4}


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


#: Ground seen from a drone is never one even colour: wide damp and dry patches, worn tracks,
#: spilt sand. (low, high) brightness the photo texture is scaled between, over blotches this many
#: metres across.
MACRO = {"gravel": (0.62, 1.18, 18.0), "soil": (0.7, 1.15, 12.0), "asphalt": (0.75, 1.12, 22.0), "asphalt_road": (0.75, 1.12, 22.0),
         "concrete": (0.85, 1.08, 15.0)}


def _macro(tree, tc, colour, low: float, high: float, across: float):
    """``colour`` scaled by large soft blotches (two noises, ``across`` metres and a third of it)."""
    n, ln = tree.nodes, tree.links
    big = n.new("ShaderNodeTexNoise")
    big.inputs["Scale"].default_value = 1.0 / across
    big.inputs["Detail"].default_value = 3.0
    small = n.new("ShaderNodeTexNoise")
    small.inputs["Scale"].default_value = 3.0 / across
    small.inputs["Detail"].default_value = 6.0
    for t in (big, small):
        ln.new(tc.outputs["Object"], t.inputs["Vector"])
    both = n.new("ShaderNodeMix")
    both.data_type = "FLOAT"
    both.inputs["Factor"].default_value = 0.35
    ln.new(big.outputs["Fac"], both.inputs["A"])
    ln.new(small.outputs["Fac"], both.inputs["B"])
    ramp = n.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value = 0.3
    ramp.inputs["From Max"].default_value = 0.7
    ramp.inputs["To Min"].default_value = low
    ramp.inputs["To Max"].default_value = high
    ln.new(both.outputs["Result"], ramp.inputs["Value"])
    mul = n.new("ShaderNodeMix")
    mul.data_type = "RGBA"
    mul.blend_type = "MULTIPLY"
    mul.inputs["Factor"].default_value = 1.0
    ln.new(colour, mul.inputs["A"])
    comb = n.new("ShaderNodeCombineColor")
    for k in ("Red", "Green", "Blue"):
        ln.new(ramp.outputs["Result"], comb.inputs[k])
    ln.new(comb.outputs["Color"], mul.inputs["B"])
    return mul.outputs["Result"]


def _textured(name: str, maps: dict[str, str], size: float, tint: tuple | None = None):
    """A material from photo maps, projected on all six sides in world metres (no UVs needed)."""
    import bpy
    m, bsdf, tree = _principled("st_" + name)
    n = tree.nodes
    tc = n.new("ShaderNodeTexCoord")
    mp = n.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / size, 1 / size, 1 / size)
    tree.links.new(tc.outputs["Object"], mp.inputs["Vector"])

    def image(key: str, colour: bool):
        t = n.new("ShaderNodeTexImage")
        t.image = bpy.data.images.load(maps[key], check_existing=True)
        if not colour:
            t.image.colorspace_settings.name = "Non-Color"
        t.projection = "BOX"
        t.projection_blend = 0.25
        tree.links.new(mp.outputs["Vector"], t.inputs["Vector"])
        return t

    diff = image("diff", True)
    colour = diff.outputs["Color"]
    if tint:
        mix = n.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        tree.links.new(colour, mix.inputs["A"])
        mix.inputs["B"].default_value = (*tint, 1)
        colour = mix.outputs["Result"]
    if name in MACRO:
        colour = _macro(tree, tc, colour, *MACRO[name])
    tree.links.new(colour, bsdf.inputs["Base Color"])
    tree.links.new(image("rough", False).outputs["Color"], bsdf.inputs["Roughness"])
    nm = n.new("ShaderNodeNormalMap")
    nm.inputs["Strength"].default_value = 0.8
    tree.links.new(image("nor", False).outputs["Color"], nm.inputs["Color"])
    tree.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def _facade_real(tree, bsdf) -> None:
    """The rebuilt neighbours: each face's wall and roof colours (sampled from Google's photos of
    that building) on rendered plaster, with a floor every 3.2 m and a window bay every 3 m.
    Reads the face attributes ``wall``, ``roof`` and ``base`` that scene.rebuild_near writes."""
    n, ln = tree.nodes, tree.links

    def attr(name):
        a = n.new("ShaderNodeAttribute")
        a.attribute_name = name
        return a

    def math(op, a, b=None, value=None):
        m = n.new("ShaderNodeMath")
        m.operation = op
        ln.new(a, m.inputs[0]) if not isinstance(a, float) else m.inputs[0].__setattr__("default_value", a)
        if b is not None:
            if isinstance(b, float):
                m.inputs[1].default_value = b
            else:
                ln.new(b, m.inputs[1])
        return m.outputs[0]

    def mix(fac, a, b):
        m = n.new("ShaderNodeMix")
        m.data_type = "RGBA"
        ln.new(fac, m.inputs["Factor"])
        for sock, v in (("A", a), ("B", b)):
            if isinstance(v, tuple):
                m.inputs[sock].default_value = (*v, 1.0)
            else:
                ln.new(v, m.inputs[sock])
        return m.outputs["Result"]

    geo = n.new("ShaderNodeNewGeometry")
    pos = n.new("ShaderNodeSeparateXYZ")
    ln.new(geo.outputs["Position"], pos.inputs[0])
    nor = n.new("ShaderNodeSeparateXYZ")
    ln.new(geo.outputs["Normal"], nor.inputs[0])
    wall, roof, base, top = attr("wall"), attr("roof"), attr("base"), attr("top")
    rel = math("SUBTRACT", pos.outputs["Z"], base.outputs["Fac"])
    storey = math("FRACT", math("DIVIDE", rel, 3.2))
    nx = math("ABSOLUTE", nor.outputs["X"])
    ny = math("ABSOLUTE", nor.outputs["Y"])
    side = math("GREATER_THAN", nx, ny)                     # facing ±X: windows run along Y
    u = math("ADD", math("MULTIPLY", pos.outputs["Y"], side),
             math("MULTIPLY", pos.outputs["X"], math("SUBTRACT", 1.0, side)))
    bay = math("FRACT", math("DIVIDE", u, 3.0))
    win = math("MULTIPLY", math("GREATER_THAN", bay, 0.2), math("LESS_THAN", bay, 0.8))
    win = math("MULTIPLY", win, math("GREATER_THAN", storey, 0.3))
    win = math("MULTIPLY", win, math("LESS_THAN", storey, 0.85))
    win = math("MULTIPLY", win, math("GREATER_THAN", rel, 0.9))
    win = math("MULTIPLY", win, math("LESS_THAN", pos.outputs["Z"], math("SUBTRACT", top.outputs["Fac"], 0.4)))
    is_roof = math("GREATER_THAN", math("ABSOLUTE", nor.outputs["Z"]), 0.6)
    win = math("MULTIPLY", win, math("SUBTRACT", 1.0, is_roof))
    slab = math("MULTIPLY", math("LESS_THAN", storey, 0.07), math("SUBTRACT", 1.0, is_roof))
    noise = n.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.6
    noise.inputs["Detail"].default_value = 8
    ln.new(geo.outputs["Position"], noise.inputs["Vector"])
    grain = n.new("ShaderNodeMapRange")
    grain.inputs["To Min"].default_value = 0.85
    grain.inputs["To Max"].default_value = 1.1
    ln.new(noise.outputs["Fac"], grain.inputs["Value"])
    surface = mix(is_roof, wall.outputs["Color"], roof.outputs["Color"])
    shade = n.new("ShaderNodeMix")
    shade.data_type = "RGBA"
    shade.blend_type = "MULTIPLY"
    shade.inputs["Factor"].default_value = 1.0
    ln.new(surface, shade.inputs["A"])
    ln.new(grain.outputs["Result"], shade.inputs["B"])
    banded = mix(math("MULTIPLY", slab, 0.35), shade.outputs["Result"], (0.1, 0.1, 0.1))
    colour = mix(win, banded, (0.035, 0.045, 0.055))
    ln.new(colour, bsdf.inputs["Base Color"])
    rough = n.new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value = 0.85
    rough.inputs["To Max"].default_value = 0.06
    ln.new(win, rough.inputs["Value"])
    ln.new(rough.outputs["Result"], bsdf.inputs["Roughness"])
    spec = n.new("ShaderNodeMapRange")
    spec.inputs["To Min"].default_value = 0.3
    spec.inputs["To Max"].default_value = 1.0
    ln.new(win, spec.inputs["Value"])
    if "Specular IOR Level" in bsdf.inputs:
        ln.new(spec.outputs["Result"], bsdf.inputs["Specular IOR Level"])
    bump = n.new("ShaderNodeBump")                          # windows set back in the wall
    bump.inputs["Strength"].default_value = 0.6
    bump.inputs["Distance"].default_value = 0.15
    ln.new(math("SUBTRACT", 1.0, math("MAXIMUM", win, math("MULTIPLY", slab, -1.0))), bump.inputs["Height"])
    ln.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def get(name: str):
    """A material by its role, made once."""
    if name in _cache:
        return _cache[name]
    role = {"concrete_fresh": "concrete", "asphalt_road": "asphalt"}.get(name, name)
    tint = {"concrete_fresh": (0.85, 0.85, 0.85), "asphalt_road": (0.45, 0.45, 0.47)}.get(name)
    if role in TEXTURES:
        try:
            m = _textured(name, TEXTURES[role], TEXTURE_SIZE.get(name, 3.0), tint)
            _cache[name] = m
            return m
        except Exception:                                 # noqa: BLE001 - procedural below
            pass
    m, bsdf, tree = _principled("st_" + name)
    if name == "concrete":
        # plain cast concrete: an even light grey with faint blotches, no pattern
        _noise_mix(tree, bsdf, (0.38, 0.37, 0.35), (0.45, 0.44, 0.41), scale=0.6, detail=4, rough=(0.8, 0.92), bump=0.04)
    elif name == "concrete_fresh":
        _noise_mix(tree, bsdf, (0.36, 0.36, 0.35), (0.40, 0.40, 0.39), scale=0.8, detail=4, rough=(0.45, 0.7), bump=0.03)
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
    elif name in ("asphalt", "asphalt_road"):
        _noise_mix(tree, bsdf, (0.07, 0.07, 0.075), (0.13, 0.13, 0.135), scale=2, detail=12, rough=(0.75, 0.95), bump=0.2)
    elif name == "paving":
        # pavement slabs, 0.6 x 0.4 m, light and worn, with dark joints
        n, ln = tree.nodes, tree.links
        tc = n.new("ShaderNodeTexCoord")
        br = n.new("ShaderNodeTexBrick")
        br.inputs["Scale"].default_value = 1.0
        br.inputs["Brick Width"].default_value = 0.6
        br.inputs["Row Height"].default_value = 0.4
        br.inputs["Mortar Size"].default_value = 0.008
        br.inputs["Color1"].default_value = (0.50, 0.48, 0.44, 1)
        br.inputs["Color2"].default_value = (0.43, 0.41, 0.37, 1)
        br.inputs["Mortar"].default_value = (0.22, 0.21, 0.19, 1)
        ln.new(tc.outputs["Object"], br.inputs["Vector"])
        ln.new(_macro(tree, tc, br.outputs["Color"], 0.8, 1.1, 10.0), bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.85
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
    elif name == "facade_real":
        _facade_real(tree, bsdf)
    elif name == "road_paint":
        bsdf.inputs["Base Color"].default_value = (0.82, 0.82, 0.78, 1)
        bsdf.inputs["Roughness"].default_value = 0.7
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
        # painted steel site hoarding, dark green and weathered (not a toy blue)
        _noise_mix(tree, bsdf, (0.07, 0.13, 0.10), (0.13, 0.19, 0.15), scale=2.5, rough=(0.55, 0.8), bump=0.08)
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
