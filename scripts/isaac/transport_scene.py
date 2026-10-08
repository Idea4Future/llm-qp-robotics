"""Derived physical tray and B workcell. Dimensions/materials are design assumptions."""
from __future__ import annotations

TRAY_CENTER = (.40, -.12, .78)
TRAY_TOP_IN_BASE = .788


def add_transport_scene(stage, base_path="/World/RBY1/base", destination_x=3.0, destination_y=0., initial_base_xy=(0.,0.)):
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics
    from grasp_scene import _box, _material
    root = "/World/CarryTray"
    tray = UsdGeom.Xform.Define(stage, root)
    tray.AddTranslateOp().Set(Gf.Vec3d(TRAY_CENTER[0]+initial_base_xy[0],TRAY_CENTER[1]+initial_base_xy[1],TRAY_CENTER[2]))
    UsdPhysics.RigidBodyAPI.Apply(tray.GetPrim())
    PhysxSchema.PhysxRigidBodyAPI.Apply(tray.GetPrim()).CreateDisableGravityAttr(False)
    mass = UsdPhysics.MassAPI.Apply(tray.GetPrim())
    mass.CreateMassAttr(1.0)
    # Approximation for tray, walls and support posts as one rigid accessory.
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, -.12))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.025, .025, .007))
    material = _material(stage, "/World/TrayMaterial", .7)
    _box(stage, root + "/floor", (0, 0, 0), (.18, .18, .016), material, color=(.08, .28, .38))
    for sign in (-1, 1):
        _box(stage, root + f"/wall_x_{sign:+d}".replace("+", "p").replace("-", "m"),
             (sign * .095, 0, .018), (.01, .20, .02), material, color=(.10, .35, .45))
        _box(stage, root + f"/wall_y_{sign:+d}".replace("+", "p").replace("-", "m"),
             (0, sign * .095, .018), (.18, .01, .02), material, color=(.10, .35, .45))
        _box(stage, root + f"/post_{sign:+d}".replace("+", "p").replace("-", "m"),
             (0, sign * .065, -.24), (.024, .024, .464), material, color=(.12, .16, .19))
    mount = UsdPhysics.FixedJoint.Define(stage, "/World/RBY1/joints/project_tray_mount")
    mount.CreateBody0Rel().SetTargets([Sdf.Path(base_path)])
    mount.CreateBody1Rel().SetTargets([Sdf.Path(root)])
    mount.CreateLocalPos0Attr(Gf.Vec3f(*TRAY_CENTER))
    mount.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    mount.CreateLocalRot0Attr(Gf.Quatf(1.))
    mount.CreateLocalRot1Attr(Gf.Quatf(1.))
    table_material = _material(stage, "/World/TableBMaterial", .6)
    table = "/World/TableB"
    UsdGeom.Xform.Define(stage, table)
    _box(stage, table + "/top", (destination_x + .60, destination_y-.40, .775), (.54, .30, .05), table_material,
         color=(.3, .5, .48))
    for i, (x, y) in enumerate(((.82, -.50), (.82, -.30), (.46, -.50), (.46, -.30))):
        _box(stage, table + f"/leg_{i}", (destination_x+x, destination_y+y, .375), (.04, .04, .75), table_material)
    return {"tray_prim": root, "tray_floor_prim": root + "/floor", "tray_mass_kg": 1.,
            "tray_center_in_base_m": list(TRAY_CENTER), "tray_top_in_base_m": TRAY_TOP_IN_BASE,
            "tray_inner_half_width_m": .09, "tray_friction_assumption": .7,
            "destination_table_prim": table + "/top", "destination_base": [destination_x, destination_y, 0.],
            "destination_object_center": [destination_x+.4, destination_y-.3, .83],
            "geometry_status": "physical geometry authored; task outcome requires runtime validation"}
