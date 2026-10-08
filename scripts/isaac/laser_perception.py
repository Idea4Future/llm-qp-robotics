"""Ideal project planar LiDAR using the current PhysX collision scene.

Importing this file never starts Kit, reads object GT poses or steps physics.
The caller owns physics steps, localization, static-map admission, footprint,
motion commands and the stopping policy. A scan is an instantaneous geometric
snapshot with no sensor noise, rolling exposure or moving-object prediction.

After World.reset with enable_scene_query_support=True::

    lidar = PlanarLidar()
    scan = lidar.observe(base_prim.get_world_pose(), world.current_time)
    gate = directional_coverage(scan, world.current_time, travel_direction_rad=0.)
    if not scan['valid'] or not gate['accepted']: stop_command()
    # Use scan['points_world_xy_m'] with the admitted static map/local QP.

The pinned A USD has no Camera/LiDAR prim/type/schema/name matches in385
prims including instance proxies (SHA256 below). This says nothing about
the sensors fitted to the physical RB-Y1. This helper is a NEW project sensor.

References: installed omni.physx107.3.26 _physx.pyi:2197,3600; its
PhysxSceneQueryInterface.py test_physics_raycast_all; NVIDIA107.3 Python API:
https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physx/docs/api/python.html
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np


PINNED_ROBOT_USD_SHA256 = "44596dc52c0f897b11ff27aa0f162d9168239c82e5891e16b67c13a80af00962"
ROBOT_SENSOR_AUDIT = {
    "asset": "third_party/rby1_isaac/assets/model_v_1_2_a.usd",
    "sha256": PINNED_ROBOT_USD_SHA256,
    "prim_count_including_instance_proxies": 385,
    "camera_lidar_name_type_schema_matches": [],
    "scope": "pinned USD static audit; no conclusion about physical RB-Y1 hardware",
}
LIDAR_MOUNT_PROPOSAL = {
    "status": "source-default-pose static geometry proposal; not a measured Isaac mounting result",
    "front_origin_base_m": [.32,0.,.4], "rear_origin_base_m": [-.32,0.,.4],
    "front_yaw_base_rad": 0., "rear_yaw_base_rad": math.pi,
    "base_visual_bbox_min_m": [-.334625,-.2595,.00016],
    "base_visual_bbox_max_m": [.304138,.2595,.358069],
    "proposed_housing_radius_m": .035, "proposed_housing_height_m": .04,
    "housing_bottom_above_chassis_visual_top_m": .021931,
    "torso0_collision_capsule_top_m": .385501,
    "torso1_collision_capsule_bbox_min_m": [-.105,-.115,.2955],
    "torso1_collision_capsule_bbox_max_m": [.125,.115,.7455],
    "rear_housing_box_to_torso0_capsule_conservative_clearance_m": math.hypot(.055,.0995)-.105,
    "rear_housing_max_axle_rear_extent_m": .32+.035+.228,
    "geometry_assumptions": [
        "Default torso joints and source authored geometry; native deformation/tilt/control errors not evaluated.",
        "Housing is render-only. This is not a mechanical standoff design or a physical mass/collision result.",
        "Source guide-purpose/invisible colliders included in the AABB audit.",
    ],
}


def _vector(value, n, name):
    out=np.asarray(value,float)
    if out.shape!=(n,) or not np.isfinite(out).all():
        raise ValueError(f"{name} must contain {n} finite values")
    return out


def _rotation(quaternion):
    q=_vector(quaternion,4,"quaternion_wxyz")
    if np.linalg.norm(q)<1e-12:raise ValueError("Zero quaternion")
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def _pose(basepose,axle_offset):
    """Accept native (position,wxyz), or navigation_runtime.gt_base_state dict."""
    upright_assumption=False
    if isinstance(basepose,dict):
        p=_vector(basepose.get("base_position_m",basepose.get("position_m")),3,"base position")
        q=basepose.get("quaternion_wxyz",basepose.get("base_quaternion_wxyz"))
        if q is None:
            yaw=float(basepose["base_yaw_rad"])
            if not math.isfinite(yaw):raise ValueError("Invalid yaw")
            q=[math.cos(yaw/2),0,0,math.sin(yaw/2)];upright_assumption=True
        R=_rotation(q)
    else:
        p,q=basepose;p=_vector(p,3,"base position");R=_rotation(q)
    yaw=math.atan2(R[1,0],R[0,0]);c,s=math.cos(yaw),math.sin(yaw)
    Raxle=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    return p,R,p+R@axle_offset,Raxle,yaw,upright_assumption


def _under(path,prefix):
    return path==prefix or path.startswith(prefix.rstrip('/')+'/')


def _field(hit,name,alternate=None):
    if isinstance(hit,dict):return hit.get(name,hit.get(alternate) if alternate else None)
    result=getattr(hit,name,None)
    return result if result is not None or alternate is None else getattr(hit,alternate,None)


def _float3(value):
    # Native RaycastHit vectors are carb.Float3. Dictionary mocks may use tuples.
    if hasattr(value,"x"):return _vector([value.x,value.y,value.z],3,"hit position")
    return _vector(value,3,"hit position")


def audit_robot_sensor_prims(stage,robot_prim_path="/World/RBY1"):
    """Read composed robot sensor-looking prims; never alter USD or instantiate sensors."""
    from pxr import Usd
    root=stage.GetPrimAtPath(robot_prim_path)
    if not root.IsValid():raise ValueError("Robot prim is missing")
    matches=[];count=0
    for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
        count+=1
        tokens=[str(prim.GetPath()),prim.GetTypeName(),*prim.GetAppliedSchemas(),
                *[a.GetName() for a in prim.GetAttributes()]]
        if any(any(key in token.lower() for key in ("lidar","laser","camera","rtxsensor")) for token in tokens):
            matches.append({"path":str(prim.GetPath()),"type":prim.GetTypeName(),"schemas":prim.GetAppliedSchemas()})
    return {"prim_count_including_instance_proxies":count,"matches":matches,
            "scope":"composed model audit; project-added sensors may appear; not physical hardware inventory"}


def add_lidar_visual(stage,base_prim_path="/World/RBY1/base",*,origin_base_m=(.32,0,.4),name="project_planar_lidar_visual"):
    """Optional render-only mount marker; adds no mass, collision or real sensor API.

    This visual is deliberately not a physical housing claim. The raycast
    observer works without it. Physical housing mass/collisions are unmodeled.
    """
    from pxr import Gf,UsdGeom,UsdPhysics
    base=stage.GetPrimAtPath(base_prim_path)
    if not base.IsValid() or not base.HasAPI(UsdPhysics.RigidBodyAPI):raise ValueError("Base rigid body missing")
    if not isinstance(name,str) or not name.isidentifier():raise ValueError("Visual name must be one USD identifier")
    path=base_prim_path.rstrip('/')+'/'+name
    if stage.GetPrimAtPath(path).IsValid():raise ValueError("LiDAR visual already exists")
    cylinder=UsdGeom.Cylinder.Define(stage,path)
    cylinder.CreateAxisAttr(UsdGeom.Tokens.z);cylinder.CreateRadiusAttr(.035);cylinder.CreateHeightAttr(.04)
    cylinder.AddTranslateOp().Set(Gf.Vec3d(*_vector(origin_base_m,3,"origin_base_m")))
    cylinder.CreateDisplayColorAttr([Gf.Vec3f(.12,.22,.25)])
    return {"path":path,"origin_base_m":list(origin_base_m),"radius_m":.035,"height_m":.04,
            "render_only":True,"extra_mass_kg":0.,"collision_enabled":False,
            "status":"optional project sensor visual; physical housing mass/collision unmodeled"}


class PlanarLidar:
    """A timestamped, self-occlusion-aware scan of actual PhysX colliders.

    Default360beams,360deg,3m,10Hz. Base-local(.32,0,.4) is a proposed front
    mounting point, NOT a vendor sensor location. Rays blocked first by robot
    colliders are unknown/self_occluded, not clear or magically transmitted.
    Obstacles in front of a self collider remain valid measurements.

    query_interface/query_ready injection is for lightweight unit tests.
    Runtime default obtains native PhysX lazily and checks running/attached
    scene and scene-query support. Calling observe never advances simulation.
    """
    def __init__(self,*,rays=360,fov_deg=360.,maximum_range_m=3.,rate_hz=10.,
                 maximum_age_s=.15,origin_base_m=(.32,0,.4),axle_offset_base_m=(.228,0,0),
                 sensor_yaw_base_rad=0.,self_prim_prefixes=("/World/RBY1",),query_interface=None,query_ready=None):
        if isinstance(rays,bool) or not isinstance(rays,int) or rays<2:raise ValueError("rays must be integer>=2")
        for value in (maximum_range_m,rate_hz,maximum_age_s):
            if not math.isfinite(float(value)) or float(value)<=0:raise ValueError("Range/rate/age must be finite positive")
        if not 0<float(fov_deg)<=360:raise ValueError("fov_deg must be in(0,360]")
        self.rays,self.fov_deg=rays,float(fov_deg)
        self.maximum_range_m,self.period_s,self.maximum_age_s=float(maximum_range_m),1/float(rate_hz),float(maximum_age_s)
        self.origin_base_m=_vector(origin_base_m,3,"origin_base_m")
        self.sensor_yaw_base_rad=float(sensor_yaw_base_rad)
        if not math.isfinite(self.sensor_yaw_base_rad):raise ValueError("Invalid sensor yaw")
        self.axle_offset_base_m=_vector(axle_offset_base_m,3,"axle_offset_base_m")
        self.self_prefixes=tuple(str(p).rstrip('/') for p in self_prim_prefixes)
        if not self.self_prefixes or any(not p.startswith('/') or p=='/' for p in self.self_prefixes):
            raise ValueError("Self masks must be explicit absolute prim prefixes")
        self.angles=np.linspace(-math.radians(self.fov_deg)/2,math.radians(self.fov_deg)/2,rays,endpoint=self.fov_deg<360)+self.sensor_yaw_base_rad
        self._query,self._query_ready=query_interface,query_ready
        self._injected=query_interface is not None
        self._native_float3=None
        self._last=None;self._sequence=0

    def metadata(self):
        return {"sensor":"project ideal planar LiDAR; PhysX raycast_all",
            "pinned_robot_sensor_audit":copy.deepcopy(ROBOT_SENSOR_AUDIT),
            "mount_selection_static_geometry":copy.deepcopy(LIDAR_MOUNT_PROPOSAL),
            "origin_base_m":self.origin_base_m.tolist(),"scan_height_base_m":float(self.origin_base_m[2]),
            "sensor_yaw_base_rad":self.sensor_yaw_base_rad,
            "rays":self.rays,"fov_deg":self.fov_deg,"maximum_range_m":self.maximum_range_m,
            "rate_hz":1/self.period_s,"maximum_age_s":self.maximum_age_s,
            "self_prim_prefixes":list(self.self_prefixes),"self_occlusion_policy":"nearest robot hit blocks farther external hits",
            "axle_offset_base_m":self.axle_offset_base_m.tolist(),
            "localization":"caller-provided known base pose; no AMCL/EKF/odometry estimation",
            "footprint":"caller must supply/check actual loaded robot+arms+tray+camera envelope; self masking is not footprint inflation",
            "limitations":["Ideal collision-geometry rays, not RTX reflectance/intensity/dropout simulation or a physical stock sensor.",
                "Only geometry intersecting the0.4m plane is sensed: tabletops/overhangs and obstacles below this height may be missed. Static3D collision map must complement this scan.",
                "Self-occluded sectors are unknown. The intended travel direction needs its own coverage gate.",
                "10Hz cached hits are old geometric snapshots; no dynamic velocity estimate or braking-distance guarantee.",
                "No-hit means no collider in3m along that ray, not free space beyond range or between finite-angle rays.",
                "Optional sensor visual has no physical mass/collision; it is not hardware installation validation."]}

    def _ready(self):
        if self._injected:
            if self._query_ready is None:return False,"injected_query_readiness_not_declared"
            try:return bool(self._query_ready() if callable(self._query_ready) else self._query_ready),"query_not_ready"
            except Exception:return False,"query_readiness_error"
        try:
            from omni.physx import get_physx_interface,get_physx_simulation_interface,get_physx_scene_query_interface
            from isaacsim.core.api import SimulationContext
            import carb
            from pxr import UsdGeom
            context=SimulationContext.instance()
            if context is None or not get_physx_interface().is_running():return False,"physics_not_running"
            if not math.isclose(UsdGeom.GetStageMetersPerUnit(context.stage),1.,abs_tol=1e-10,rel_tol=0):
                return False,"scene_units_not_meters"
            # Kit timeline owns attachment in normal World execution; the low-level
            # simulation interface can report no manually attached stage here.
            # Validate the actual tensor view used by this running World instead.
            view=context.physics_sim_view
            if view is None or not view.is_valid:return False,"physics_view_not_valid"
            if not context.get_physics_context().get_enable_scene_query_support():return False,"scene_query_support_disabled"
            self._query=get_physx_scene_query_interface()
            self._native_float3=carb.Float3
            return bool(self._query is not None),"physx_query_interface_unavailable"
        except Exception as error:return False,"physx_query_interface_unavailable:"+type(error).__name__

    def _invalid(self,reason,now=None):
        return {"valid":False,"reason":reason,"timestamp_s":None,"current_time_s":now,
                "stop_required":True,"points_world_m":[],"points_world_xy_m":[],"points_axle_m":[],
                "points_axle_xy_m":[],"hit_paths":[],"sensor":"project ideal PhysX planar LiDAR"}

    def observe(self,basepose,timestamp_s,*,force=False):
        """Return fresh scan or≤0.1s cache, transformed into the current axle frame.

        World points remain at their acquisition pose/time. Current axle points
        apply current localization to that old snapshot; they do not predict
        moving obstacles. All ray callbacks execute before the caller's next
        physics step. Missing/backend/malformed data is fail-closed.
        """
        try:
            now=float(timestamp_s)
            if not math.isfinite(now) or now<0:raise ValueError("Invalid simulation timestamp")
            pose=_pose(basepose,self.axle_offset_base_m)
        except Exception:return self._invalid("invalid_timestamp_or_base_pose")
        ready,reason=self._ready()
        if not ready:return self._invalid(reason,now)
        if self._last is not None and now<float(self._last["timestamp_s"])-1e-8:
            self._last=None;return self._invalid("simulation_time_moved_backwards",now)
        if not force and self._last is not None and now-self._last["timestamp_s"]<self.period_s-1e-9:
            return self._current(self._last,pose,now,cached=True)
        p,R,axle,Raxle,yaw,upright=pose
        origin=p+R@self.origin_base_m
        states=[];ranges=[];hits=[];directions=[];self_count=0;callback_count=0
        started=time.perf_counter()
        for ray_index,angle in enumerate(self.angles):
            direction=R@np.array([math.cos(angle),math.sin(angle),0.]);directions.append(direction.tolist())
            external=[];own=[];errors=[]
            def callback(hit):
                nonlocal callback_count
                callback_count+=1
                try:
                    distance=float(_field(hit,"distance"));position=_float3(_field(hit,"position"))
                    collision=str(_field(hit,"collision") or "")
                    body=str(_field(hit,"rigid_body","rigidBody") or "")
                    if not math.isfinite(distance) or distance<0 or distance>self.maximum_range_m+1e-5:
                        raise ValueError("Invalid hit range")
                    if not collision.startswith('/') and not body.startswith('/'):
                        raise ValueError("Hit has no absolute collision/body path")
                    if np.linalg.norm(position-origin-distance*direction)>max(1e-3,distance*1e-4):
                        raise ValueError("Hit position/range do not match ray in meter units")
                    row={"ray_index":ray_index,"distance_m":distance,"point_world_m":position.tolist(),
                         "collision_path":collision,"rigid_body_path":body}
                    masked=any(_under(path,prefix) for prefix in self.self_prefixes for path in (collision,body) if path)
                    (own if masked else external).append(row)
                except Exception as error:errors.append(type(error).__name__+":"+str(error))
                return True  # all hits are unsorted; continue and choose nearest ourselves
            try:
                query_origin=tuple(origin) if self._native_float3 is None else self._native_float3(*origin)
                query_direction=tuple(direction) if self._native_float3 is None else self._native_float3(*direction)
                result=self._query.raycast_all(query_origin,query_direction,self.maximum_range_m,callback,True)
                if not isinstance(result,(bool,int,np.bool_,np.integer)):
                    raise ValueError("Unexpected raycast_all return type")
            except Exception as error:
                self._last=None;return {**self._invalid("raycast_failed:"+type(error).__name__,now),"failed_ray_index":ray_index}
            if errors:
                self._last=None;return {**self._invalid("malformed_hit",now),"failed_ray_index":ray_index,"errors":errors}
            nearest_own=min(own,key=lambda h:h["distance_m"]) if own else None
            nearest=min(external,key=lambda h:h["distance_m"]) if external else None
            self_count+=len(own)
            if nearest_own is not None and (nearest is None or nearest_own["distance_m"]<=nearest["distance_m"]+1e-5):
                states.append("self_occluded");ranges.append(None)
            elif nearest is not None:
                states.append("hit");ranges.append(nearest["distance_m"]);hits.append(nearest)
            else:states.append("no_hit_within_range");ranges.append(self.maximum_range_m)
        self._sequence+=1
        scan={"valid":any(s!="self_occluded" for s in states),"reason":"accepted",
              "timestamp_s":now,"sequence":self._sequence,"stop_required":False,
              "sensor":"project ideal PhysX planar LiDAR","ray_count":self.rays,
              "ray_angles_base_rad_at_scan":self.angles.tolist(),"ray_directions_world_at_scan":directions,
              "ray_ranges_m":ranges,"ray_states":states,"hit_paths":hits,
              "points_world_m":[h["point_world_m"] for h in hits],
              "points_world_xy_m":[h["point_world_m"][:2] for h in hits],
              "sensor_origin_world_m":origin.tolist(),"scan_plane_normal_world":R[:,2].tolist(),
              "axle_origin_world_at_scan_m":axle.tolist(),"base_yaw_at_scan_rad":yaw,
              "observed_fraction":float(np.mean([s!="self_occluded" for s in states])),
              "self_occluded_ray_count":states.count("self_occluded"),"self_hit_callback_count":self_count,
              "total_hit_callback_count":callback_count,"query_wall_time_s":time.perf_counter()-started,
              "maximum_range_m":self.maximum_range_m,"maximum_age_s":self.maximum_age_s,
              "base_input_upright_assumption":upright,"full_fov_observed":all(s!="self_occluded" for s in states),
              "point_semantics":"visible first external collider surfaces; no-hit endpoints are never obstacle points",
              "collision_safety_guarantee":False}
        points=np.asarray(scan["points_world_m"],float).reshape(-1,3)
        scan["points_axle_at_scan_m"]=((points-axle)@Raxle).tolist()
        if not scan["valid"]:scan.update(reason="all_rays_self_occluded",stop_required=True)
        self._last=scan
        return self._current(scan,pose,now,cached=False)

    def _current(self,scan,pose,now,*,cached):
        age=now-float(scan["timestamp_s"])
        if age< -1e-8 or age>self.maximum_age_s:return self._invalid("stale_or_future_scan",now)
        result=copy.deepcopy(scan);_,_,axle,Raxle,_,_=pose
        points=np.asarray(scan["points_world_m"],float).reshape(-1,3)
        current=(points-axle)@Raxle
        directions=np.asarray(scan["ray_directions_world_at_scan"],float)@Raxle
        result.update(current_time_s=now,scan_age_s=age,cached=cached,
                      points_axle_m=current.tolist(),points_axle_xy_m=current[:,:2].tolist(),
                      axle_origin_world_current_m=axle.tolist(),
                      ray_angles_axle_current_rad=np.arctan2(directions[:,1],directions[:,0]).tolist(),
                      moving_obstacle_prediction=False)
        return result

    def latest(self,current_time_s,basepose):
        """Read cache only; a missing/stale/backend-invalid record requests stop."""
        if self._last is None:return self._invalid("missing_scan",current_time_s)
        ready,reason=self._ready()
        if not ready:return self._invalid(reason,current_time_s)
        try:
            now=float(current_time_s)
            if not math.isfinite(now):raise ValueError("Invalid time")
            return self._current(self._last,_pose(basepose,self.axle_offset_base_m),now,cached=True)
        except Exception:return self._invalid("invalid_timestamp_or_base_pose")


def directional_coverage(scan,current_time_s,*,travel_direction_rad=0.,half_cone_rad=math.radians(15),minimum_fraction=1.,maximum_plane_tilt_rad=math.radians(5)):
    """Fail-closed freshness/beam-coverage admission for a proposed direction.

    It is NOT a clearance, stopping-distance or collision-safety test. Empty
    obstacle points can be a valid open scan; unknown/self-masked beams cannot.
    For reversing pass travel_direction_rad=pi, and do not bypass a failed gate.
    """
    rejected={"accepted":False,"stop_required":True}
    if not isinstance(scan,dict) or not scan.get("valid",False):return {**rejected,"reason":"invalid_or_missing_scan"}
    try:
        now,stamp=float(current_time_s),float(scan["timestamp_s"])
        maxage=float(scan["maximum_age_s"]);age=now-stamp
        if not math.isfinite(age) or age< -1e-8 or age>maxage:return {**rejected,"reason":"stale_or_future_scan"}
        tilt_limit=float(maximum_plane_tilt_rad)
        normal=_vector(scan["scan_plane_normal_world"],3,"scan plane normal")
        if not 0<=tilt_limit<math.pi/2:raise ValueError("Invalid plane tilt limit")
        if normal[2]<math.cos(tilt_limit)-1e-8:return {**rejected,"reason":"scan_plane_not_horizontal_for_planar_control"}
        direction=float(travel_direction_rad);half=float(half_cone_rad);minimum=float(minimum_fraction)
        if not math.isfinite(direction) or not 0<half<=math.pi or not 0<minimum<=1:raise ValueError("Invalid cone")
        angles=np.asarray(scan["ray_angles_axle_current_rad"],float);states=scan["ray_states"]
        if len(states)!=len(angles) or not np.isfinite(angles).all():raise ValueError("Malformed beams")
        delta=np.arctan2(np.sin(angles-direction),np.cos(angles-direction))
        selected=np.flatnonzero(np.abs(delta)<=half+1e-8)
        if not len(selected):return {**rejected,"reason":"travel_direction_outside_scan_fov"}
        known=sum(states[i] in ("hit","no_hit_within_range") for i in selected)
        fraction=known/len(selected);accepted=fraction+1e-12>=minimum
        return {"accepted":accepted,"stop_required":not accepted,
                "reason":"coverage_accepted" if accepted else "travel_direction_self_occluded_or_unknown",
                "beam_count_in_cone":len(selected),"observed_fraction_in_cone":fraction,"scan_age_s":age,
                "scope":"data/freshness/coverage only; separate footprint+barrier+contact gates required"}
    except Exception:return {**rejected,"reason":"malformed_scan"}
