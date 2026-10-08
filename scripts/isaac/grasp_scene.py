"""Derived contact geometry for a first stationary Isaac grasp experiment.

The official hand USDs have visual meshes but empty collision containers.
These flat pad boxes approximate their visual pad subset bounds. They are a
project simulation design, not vendor collision geometry or hardware ratings.
Call after Isaac/pxr initialization and before physics initialization/reset.
This module does not create an application or run any physics steps.
"""
from __future__ import annotations

import math


PAD_CENTER_M = (-0.0015, 0.0, -0.0305)
PAD_FULL_SIZE_M = (0.003, 0.032, 0.060)
# The pad midpoint is wrist-local z=-0.2316. An intentionally lower grasp
# frame leaves a 9 mm pad/table clearance for a 60 mm tall upright container.
GRASP_FRAME_IN_WRIST_M = (0.0, 0.0, -0.2406)


def _nonnegative(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _vector(values, name, *, positive=False):
    values = tuple(float(x) for x in values)
    if len(values) != 3 or not all(math.isfinite(x) for x in values):
        raise ValueError(f"{name} must contain three finite values")
    if positive and not all(x > 0 for x in values):
        raise ValueError(f"{name} entries must be positive")
    return values


def _material(stage, path, friction):
    from pxr import UsdPhysics, UsdShade

    if stage.GetPrimAtPath(path).IsValid():
        raise ValueError(f"Refusing to overwrite existing material: {path}")
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr().Set(friction)
    api.CreateDynamicFrictionAttr().Set(friction)
    api.CreateRestitutionAttr().Set(0.0)
    return material


def _box(stage, path, center, full_size, material, *, invisible=False,
         contact_offset=0.001, color=(0.2, 0.25, 0.3)):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    if stage.GetPrimAtPath(path).IsValid():
        raise ValueError(f"Refusing to overwrite existing geometry: {path}")
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr([Gf.Vec3f(-0.5), Gf.Vec3f(0.5)])
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    cube.AddScaleOp().Set(Gf.Vec3f(*full_size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if invisible:
        cube.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(True)
    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision.CreateContactOffsetAttr().Set(contact_offset)
    collision.CreateRestOffsetAttr().Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )
    return prim


def add_finger_contact_geometry(stage, robot_prim_path="/World/RBY1", friction=0.8):
    """Attach four collision-only pad boxes to existing finger rigid bodies.

    Existing body mass/inertia, finger joints and native mimic definitions are
    untouched. Leader q=-a and follower q=+a nominally give a 2a visual opening.
    The proxy boxes are hidden in RGB; the original visual pads remain visible.
    """
    from pxr import UsdPhysics

    friction = _nonnegative(friction, "friction")
    root = str(robot_prim_path).rstrip("/")
    if not root.startswith("/") or not root:
        raise ValueError("robot_prim_path must be an absolute USD prim path")
    bodies = []
    for side, letter in (("left", "l"), ("right", "r")):
        for index in (1, 2):
            path = f"{root}/{side}_gripper/ee_finger_{letter}{index}"
            body = stage.GetPrimAtPath(path)
            if not body.IsValid() or not body.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError(f"Expected existing finger rigid body: {path}")
            proxy = path + "/project_contact_pad"
            if stage.GetPrimAtPath(proxy).IsValid():
                raise ValueError(f"Contact proxy already exists: {proxy}")
            bodies.append((side, index, path, body, proxy))
    material_path = root + "/project_grasp_pad_material"
    material = _material(stage, material_path, friction)
    patches = []
    for side, index, path, body, proxy in bodies:
        visuals = stage.GetPrimAtPath(path + "/visuals")
        was_instanceable = bool(visuals.IsInstanceable())
        # The original instanced finger meshes disappear in the PhysX contact
        # view/render combination. Expand only these small visual subtrees in
        # the composed stage; original USD and physical joints stay unchanged.
        visuals.SetInstanceable(False)
        unchanged = {name: body.GetAttribute(name).Get() for name in (
            "physics:mass", "physics:centerOfMass", "physics:diagonalInertia", "physics:principalAxes"
        )}
        pad = _box(stage, proxy, PAD_CENTER_M, PAD_FULL_SIZE_M, material, invisible=True)
        assert not pad.HasAPI(UsdPhysics.RigidBodyAPI)
        assert not pad.HasAPI(UsdPhysics.MassAPI)
        assert all(body.GetAttribute(name).Get() == value for name, value in unchanged.items())
        patches.append({"side": side, "finger_index": index, "body_path": path,
                        "collision_prim_path": proxy, "center_body_local_m": list(PAD_CENTER_M),
                        "full_size_m": list(PAD_FULL_SIZE_M), "collision_enabled": True,
                        "render_visible": False, "contact_offset_m": 0.001, "rest_offset_m": 0.0,
                        "extra_rigid_body": False, "extra_mass_authored": False,
                        "visual_instanceable_before": was_instanceable, "visual_instanceable_after": False,
                        "existing_mass_inertia_unchanged": True})
    return {"status": "derived_contact_geometry; physical_grasp_not_yet_verified",
            "patches": patches, "material_path": material_path,
            "static_friction": friction, "dynamic_friction": friction, "restitution": 0.0,
            "grasp_frame_in_wrist_m": list(GRASP_FRAME_IN_WRIST_M),
            "pad_midpoint_in_wrist_m": [0.0, 0.0, -0.2316],
            "geometry_source": "RB gripper USD visual subset_91_191_191_006 body-local bounds: x[-3,0]mm, y[-16,16]mm, z[-60.5,-0.5]mm, rounded to a box.",
            "assumptions": [
                "A flat 3x32x60mm box approximates the visual pad wedge/chamfers; this is not vendor contact geometry.",
                "Friction is a simulation assumption, not a measured hardware coefficient.",
                "Native joint signs/limits/mimic definitions are unchanged; their physical response must be measured.",
                "The grasp frame is 9mm below pad midpoint to leave pad/table clearance for the nominal 60mm container.",
                "Pad-only geometry does not add metal finger/housing collisions or validate whole-hand/self-collision behavior.",
                "Original USD files are unchanged; collision children are authored only in the derived stage."
            ]}


def add_stationary_grasp_scene(stage, prefix="/World/StationaryGrasp", *,
                               table_top=0.80, object_size=(0.04, 0.05, 0.06),
                               object_mass=0.10, source_xy=(0.40, -0.30),
                               target_xy=(0.38, -0.43), friction=0.6, transport_layout=False):
    """Add a static table and one free, massive, collidable upright container.

    This helper only authors the initial scene. It provides no attachment,
    kinematic object motion, robot pose override or controller.
    """
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    size = _vector(object_size, "object_size", positive=True)
    mass = _nonnegative(object_mass, "object_mass")
    friction = _nonnegative(friction, "friction")
    top = float(table_top)
    source_xy, target_xy = tuple(map(float, source_xy)), tuple(map(float, target_xy))
    if mass == 0 or not math.isfinite(top) or top <= 0.05:
        raise ValueError("Positive object mass and table_top>0.05m required")
    if any(len(x) != 2 or not all(math.isfinite(v) for v in x) for x in (source_xy, target_xy)):
        raise ValueError("source_xy and target_xy must contain two finite values")
    prefix = str(prefix).rstrip("/")
    if not prefix.startswith("/") or stage.GetPrimAtPath(prefix).IsValid():
        raise ValueError("Use a new absolute scene prefix")
    UsdGeom.Xform.Define(stage, prefix)
    material = _material(stage, prefix + "/surface_material", friction)
    table_path = prefix + "/table_top"
    cy, sy = (-.40, .30) if transport_layout else (-.36, .50)
    _box(stage, table_path, (0.60, cy, top - 0.025), (0.54, sy, 0.05), material,
         color=(0.42, 0.49, 0.53))
    legs = []
    for index, (x, y) in enumerate(((.82,cy-sy/2+.05),(.82,cy+sy/2-.05),(.46,cy-sy/2+.05),(.46,cy+sy/2-.05))):
        path = prefix + f"/table_leg_{index}"
        _box(stage, path, (x, y, (top - 0.05) / 2), (0.04, 0.04, top - 0.05), material)
        legs.append(path)
    object_path = prefix + "/object"
    center = (*source_xy, top + size[2] / 2 + 0.001)
    obj = _box(stage, object_path, center, size, material, color=(0.85, 0.16, 0.10))
    rigid = UsdPhysics.RigidBodyAPI.Apply(obj)
    rigid.CreateRigidBodyEnabledAttr().Set(True)
    rigid.CreateKinematicEnabledAttr().Set(False)
    PhysxSchema.PhysxRigidBodyAPI.Apply(obj).CreateDisableGravityAttr().Set(False)
    inertia = tuple(mass / 12 * (size[(i + 1) % 3] ** 2 + size[(i + 2) % 3] ** 2) for i in range(3))
    mass_api = UsdPhysics.MassAPI.Apply(obj)
    mass_api.CreateMassAttr().Set(mass)
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0, 0, 0))
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*inertia))
    mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
    return {"status": "initial_stationary_scene; no_physics_run",
            "table_top_prim": table_path, "table_leg_prims": legs, "table_top_world_z_m": top,
            "table_center_xy_m": [0.60, cy], "table_full_size_m": [0.54, sy, 0.05],
            "object_prim": object_path, "object_initial_world_center_m": list(center),
            "object_full_size_m": list(size), "object_mass_kg": mass,
            "object_diagonal_inertia_kg_m2": list(inertia),
            "place_target_world_center_m": [*target_xy, top + size[2] / 2],
            "object_dynamic": True, "object_gravity_enabled": True, "object_attached": False,
            "surface_friction_assumption": friction, "contact_offset_m": 0.001,
            "limitations": ["Known initial pose; no perception/QP/LLM/navigation/tray is added.",
                            "Physics results and contact/clearance gates must be measured separately."]}
