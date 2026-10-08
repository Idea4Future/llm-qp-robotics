"""Remove one proven duplicate floor in a derived USD stage, before reset.

No Kit/GPU imports or engine launch. The original warehouse asset is untouched.
Keep the project's default physical floor/material and all visible meshes.
Reject any unexpected geometry/transform instead of guessing which floor wins.
"""
from __future__ import annotations

import math

WAREHOUSE_PLANE = '/World/Warehouse/GroundPlane/CollisionPlane'
DEFAULT_ROOT = '/World/defaultGroundPlane'
TOLERANCE_M = 1e-7
TOLERANCE_NORMAL = 1e-7


def deduplicate_warehouse_ground(stage):
    """Disable only the warehouse CollisionAPI when two enabled Z=0 planes match.

Call after warehouse composition and BEFORE world.reset/PhysX initialization.
Returns before/after metadata. This is a scene correction, not a claim about
the cause or resolution of measured velocity jitter. Calls are intentionally
not silently idempotent: an already-disabled plane fails the expected contract.
"""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    if not stage:
        raise ValueError('A composed USD stage is required')
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError('Floor deduplication requires a Z-up stage')
    if abs(float(UsdGeom.GetStageMetersPerUnit(stage))-1.) > 1e-12:
        raise ValueError('Floor comparison requires stage units in meters')
    warehouse = stage.GetPrimAtPath(WAREHOUSE_PLANE)
    root = stage.GetPrimAtPath(DEFAULT_ROOT)
    if not warehouse.IsValid() or not root.IsValid():
        raise ValueError('Expected warehouse/default floor prims are missing')

    def enabled_collision(prim):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            return False
        value = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if not isinstance(value, bool):
            raise ValueError('Collision enabled state is unavailable: '+str(prim.GetPath()))
        return value

    default_enabled = [prim for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                       if enabled_collision(prim)]
    if len(default_enabled) != 1 or not default_enabled[0].IsA(UsdGeom.Plane):
        raise ValueError('Default floor requires exactly one enabled Plane collider and no other enabled collision shape')
    default = default_enabled[0]
    if not warehouse.IsA(UsdGeom.Plane) or not enabled_collision(warehouse):
        raise ValueError('Warehouse floor must be an enabled Plane collider')
    warehouse_root = stage.GetPrimAtPath('/World/Warehouse/GroundPlane')
    warehouse_enabled = [prim for prim in Usd.PrimRange(warehouse_root, Usd.TraverseInstanceProxies())
                         if enabled_collision(prim)]
    if len(warehouse_enabled) != 1 or warehouse_enabled[0].GetPath() != warehouse.GetPath():
        raise ValueError('Unexpected additional enabled warehouse floor collision geometry')
    # All checks precede the single scene mutation.
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())

    def describe(prim):
        if prim.IsInstanceProxy():
            raise ValueError('Floor collision is instance-proxied; reject rather than modify original asset')
        axis = UsdGeom.Plane(prim).GetAxisAttr().Get()
        if axis != UsdGeom.Tokens.z:
            raise ValueError('Expected authored Plane axis Z: '+str(prim.GetPath()))
        matrix = transforms.GetLocalToWorldTransform(prim)
        if not all(math.isfinite(float(matrix[i][j])) for i in range(4) for j in range(4)):
            raise ValueError('Nonfinite floor transform')
        determinant = float(matrix.GetDeterminant())
        if not math.isfinite(determinant) or abs(determinant) < 1e-15:
            raise ValueError('Singular floor transform')
        origin = matrix.Transform(Gf.Vec3d(0.))
        normal = matrix.GetInverse().GetTranspose().TransformDir(Gf.Vec3d(0., 0., 1.))
        length = float(normal.GetLength())
        if not math.isfinite(length) or length <= 0.:
            raise ValueError('Invalid floor normal')
        normal /= length
        origin_values, normal_values = list(origin), list(normal)
        if abs(origin_values[2]) > TOLERANCE_M or math.dist(normal_values, [0., 0., 1.]) > TOLERANCE_NORMAL:
            raise ValueError('Expected an upward world Z=0 plane: '+str(prim.GetPath()))
        return {'path': str(prim.GetPath()), 'type': prim.GetTypeName(), 'authored_axis': str(axis),
                'world_origin_m': origin_values, 'world_normal': normal_values,
                'collision_enabled_before': True,
                'render_visibility_before': str(UsdGeom.Imageable(prim).ComputeVisibility())}

    warehouse_info, default_info = describe(warehouse), describe(default)
    origin_error = math.dist(warehouse_info['world_origin_m'], default_info['world_origin_m'])
    normal_error = math.dist(warehouse_info['world_normal'], default_info['world_normal'])
    if origin_error > TOLERANCE_M or normal_error > TOLERANCE_NORMAL:
        raise ValueError('Warehouse/default plane world origins or normals differ')
    # Author into the stage-owned session layer even if the caller accidentally
    # left an external asset layer as its edit target. No referenced layer/file
    # is changed; flattened stage exports still include this derived override.
    session = stage.GetSessionLayer()
    if not session or not session.anonymous:
        raise ValueError('Require an anonymous stage session layer for the derived floor override')
    attribute = UsdPhysics.CollisionAPI(warehouse).GetCollisionEnabledAttr()
    with Usd.EditContext(stage, session):
        if not attribute.Set(False) or attribute.Get() is not False:
            raise RuntimeError('Derived warehouse floor collision override could not be authored')
    warehouse_info['collision_enabled_after'] = attribute.Get()
    default_info['collision_enabled_after'] = UsdPhysics.CollisionAPI(default).GetCollisionEnabledAttr().Get()
    warehouse_info['render_visibility_after'] = str(UsdGeom.Imageable(warehouse).ComputeVisibility())
    default_info['render_visibility_after'] = str(UsdGeom.Imageable(default).ComputeVisibility())
    return {'applied': True, 'changed_attribute': WAREHOUSE_PLANE+'.physics:collisionEnabled',
            'warehouse_floor': warehouse_info, 'retained_default_floor': default_info,
            'world_origin_difference_m': origin_error, 'world_normal_difference': normal_error,
            'comparison_tolerance_m': TOLERANCE_M, 'comparison_normal_tolerance': TOLERANCE_NORMAL,
            'override_layer': session.identifier,
            'visible_meshes_and_material_bindings_modified': False,
            'robot_or_object_state_modified': False, 'original_assets_modified': False,
            'must_run_before_physics_initialization': True,
            'scope': 'derived USD floor deduplication only; physical effects require a separate measured trial'}
