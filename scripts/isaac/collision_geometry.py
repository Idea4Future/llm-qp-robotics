"""Measured collider envelope using current native articulation link transforms."""
from __future__ import annotations
import itertools
import numpy as np


def articulation_envelope(stage, articulation_view, axle_xy):
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.spatial.transform import Rotation
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy,UsdGeom.Tokens.guide],False,True)
    native=np.asarray(articulation_view._physics_view.get_link_transforms())[0]
    records=[]
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        enabled=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if enabled is False:continue
        body=prim
        while body.IsValid() and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
        if not body.IsValid():continue
        if body.GetName() not in articulation_view.body_names:continue
        try:index=articulation_view.get_body_index(body.GetName())
        except (ValueError,KeyError):continue
        if index is None:continue
        bound=cache.ComputeRelativeBound(prim,body).ComputeAlignedRange()
        lo,hi=np.asarray(bound.GetMin()),np.asarray(bound.GetMax())
        if bound.IsEmpty() or not np.isfinite(np.r_[lo,hi]).all():raise ValueError(f"Invalid collider extent: {prim.GetPath()}")
        corners=np.array(list(itertools.product(*zip(lo,hi))))
        transform=native[index]
        world=corners@Rotation.from_quat(transform[3:]).as_matrix().T+transform[:3]
        radius=float(np.max(np.linalg.norm(world[:,:2]-np.asarray(axle_xy),axis=1)))
        records.append({"path":str(prim.GetPath()),"body":body.GetName(),"radius_m":radius,"world_bounds_m":[world.min(axis=0).tolist(),world.max(axis=0).tolist()]})
    if not records:raise ValueError("No articulation colliders measured")
    bodies={r["body"] for r in records}
    for required in ("base", "CarryTray", "project_wrist_camera_mount", "ee_finger_r1", "ee_finger_r2"):
        if required not in bodies:raise ValueError("Required carry collider missing: "+required)
    return {"radius_m":max(r['radius_m'] for r in records),"colliders":records,"method":"native current link pose with USD body-local conservative collider bounding boxes; circle about axle"}
