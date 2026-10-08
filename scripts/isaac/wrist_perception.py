"""Isaac wrist RGBD/ArUco helpers; importing this file never starts Kit.

Before reset: ``meta = create_wrist_sensor(stage, wrist_path, object_path,
assetdir)``. After reset: ``camera = initialize_wrist_camera(meta)`` and
``observer = WristArucoObserver(meta['marker'], meta['quality_gate'])``.
After a rendered frame, call ``observe_wrist_camera(camera, observer,
world.current_time)``. Check ``valid`` before using ``T_world_object``.

Only images, calibration and known camera kinematics enter the detector.
Object truth belongs exclusively in the optional offline comparison helper.
The generic 100g physical mount is a project design, not a commercial sensor.
This port requires a new Isaac physics/perception trial; MuJoCo results do
not establish its visibility, interference, contact or detection performance.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


USD_TO_CV = np.diag([1.0, -1.0, -1.0])
LENS_POSITION_M = np.array([0.0, -0.1, -0.185])
CAMERA_RX_DEG = 31.0
BRACKET_ENDPOINTS_M = np.array([
    [[0.0, -.035, -.15], [0.0, -.075, -.15]],
    [[0.0, -.075, -.15], [0.0, -.115, -.173]],
])


def _rotation_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def _transform(rotation, position):
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = position
    return out


def _quat_gf(rotation):
    from pxr import Gf
    # Gf uses row-vector matrices; our rotations act on column vectors.
    return Gf.Matrix3d(*np.asarray(rotation).T.ravel().tolist()).ExtractRotation().GetQuat()


def _numpy_rotation(quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Expected a finite nonzero wxyz quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def world_camera_from_wrist(position, quaternion_wxyz):
    """Known wrist kinematics -> OpenCV camera frame (+x right,+y down,+z forward)."""
    R = _numpy_rotation(quaternion_wxyz)
    return _transform(R, np.asarray(position, float)) @ _transform(
        _rotation_x(math.radians(CAMERA_RX_DEG)) @ USD_TO_CV, LENS_POSITION_M)


def generate_marker_texture(path, marker_id=17):
    """Generate a real OpenCV ID17 marker: black30mm inside white40mm panel.

    This is a deterministic fiducial texture, not generated scene imagery.
    The 800px panel has a 600px black-square footprint and 100px white margins.
    Existing content is verified and never silently overwritten.
    """
    import cv2
    path = Path(path)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    black = cv2.aruco.generateImageMarker(dictionary, marker_id, 600)
    panel = np.full((800, 800), 255, np.uint8)
    panel[100:700, 100:700] = black
    panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2RGB)
    if path.exists():
        existing = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if existing is None or not np.array_equal(existing, panel):
            raise ValueError(f"Existing marker texture differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), panel):
            raise OSError(f"Could not write marker texture: {path}")
    return {"file": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "dictionary": "DICT_4X4_50", "id": marker_id, "panel_pixels": 800,
            "black_square_pixels": 600, "white_margin_pixels": 100}


def _capsule_inertia(mass, radius, length, direction):
    """Uniform capsule inertia at its center (cylinder plus two hemispheres)."""
    vc, vs = math.pi*radius**2*length, 4*math.pi*radius**3/3
    mc, ms = mass*vc/(vc+vs), mass*vs/(vc+vs)
    axial = .5*mc*radius**2 + .4*ms*radius**2
    transverse = mc*(3*radius**2+length**2)/12 + ms*(
        .4*radius**2 + length**2/4 + 3*length*radius/8)
    axis = np.asarray(direction, float)
    axis /= np.linalg.norm(axis)
    return transverse*np.eye(3) + (axial-transverse)*np.outer(axis, axis)


def _mount_inertia():
    rotation = _rotation_x(math.radians(CAMERA_RX_DEG))
    center = LENS_POSITION_M + rotation @ np.array([0., 0., .0155])
    sizes, mass = np.array([.04, .03, .025]), .08
    box_i = mass/12*np.diag([
        sizes[1]**2+sizes[2]**2, sizes[0]**2+sizes[2]**2, sizes[0]**2+sizes[1]**2])
    components = [(mass, center, rotation @ box_i @ rotation.T)]
    for start, end in BRACKET_ENDPOINTS_M:
        delta = end-start
        components.append((.01, (start+end)/2, _capsule_inertia(.01, .003, np.linalg.norm(delta), delta)))
    com = sum(m*p for m, p, _ in components)/.1
    inertia = sum(I + m*(np.dot(p-com,p-com)*np.eye(3)-np.outer(p-com,p-com))
                  for m, p, I in components)
    eigenvalues, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    return center, com, inertia, eigenvalues, axes


def create_wrist_sensor(stage, wrist_path, object_path, assetdir, marker_id=17):
    """Author moving child camera, massive welded accessory and top marker.

    Call before physics reset. Returns JSON-serializable metadata only; it
    neither creates a Camera render product nor starts/steps a simulation.
    The camera prim follows the wrist. The mount is a separate rigid link,
    connected by a fixed joint, with100g explicit mass/inertia and collisions.
    Printed texture has no collision or mass; the object's existing mass is
    untouched. The object may be a scaled unit Cube; its scale is compensated.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Vt

    wrist_path, object_path = str(wrist_path), str(object_path)
    wrist, obj = stage.GetPrimAtPath(wrist_path), stage.GetPrimAtPath(object_path)
    if not wrist.IsValid() or not wrist.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError("Wrist must be an existing rigid body")
    if not obj.IsValid() or not obj.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError("Object must be an existing rigid body")
    if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1., rel_tol=0., abs_tol=1e-10):
        raise ValueError("This meter-based helper requires metersPerUnit=1")
    camera_path = wrist_path + "/project_wrist_rgbd"
    robot_root = str(wrist.GetParent().GetPath())
    body_path = robot_root + "/project_wrist_camera_mount"
    joint_path = robot_root + "/joints/project_wrist_camera_mount_joint"
    marker_path = object_path + f"/project_aruco_{marker_id}"
    for path in (camera_path, body_path, joint_path, marker_path):
        if stage.GetPrimAtPath(path).IsValid():
            raise ValueError(f"Refusing duplicate sensor prim: {path}")
    texture_path = Path(assetdir)/f"aruco_{marker_id}.png"
    generate_marker_texture(texture_path, marker_id)
    # Configuration is recorded here so the physical registration cannot drift
    # if a separate legacy config is subsequently edited.
    marker = {"dictionary": "DICT_4X4_50", "id": marker_id,
              "black_square_length_m": .03, "white_panel_length_m": .04,
              "object_dimensions_m": [.04,.05,.06],
              "object_to_marker_translation_m": [0.,0.,.03014],
              "object_to_marker_rotation": np.eye(3).tolist()}
    gate = {"minimum_edge_px": 20., "maximum_reprojection_error_px": 1.5,
            "minimum_depth_m": .005, "maximum_depth_m": 2.,
            "image_border_margin_px": 3., "maximum_depth_disagreement_m": .02}
    rotation = _rotation_x(math.radians(CAMERA_RX_DEG))
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.AddTranslateOp().Set(Gf.Vec3d(*LENS_POSITION_M))
    camera.AddOrientOp().Set(Gf.Quatf(_quat_gf(rotation)))
    # USD focal/aperture values share units (tenths of a stage unit); their
    # ratio determines projection. Focal=20 and vertical aperture=23.094 gives
    # precisely60deg vertical FoV and square pixels at960x720.
    focal, height, width = 20., 720, 960
    vertical = 2*focal*math.tan(math.radians(60.)/2)
    camera.CreateFocalLengthAttr(focal)
    camera.CreateVerticalApertureAttr(vertical)
    camera.CreateHorizontalApertureAttr(vertical*width/height)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.005, 5.))
    camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    camera.CreateFocusDistanceAttr(0.)
    camera.CreateFStopAttr(0.)
    camera.CreateShutterOpenAttr(0.)
    camera.CreateShutterCloseAttr(0.)
    # Keep the mount out of nested-rigid-body hierarchies; author its initial
    # frame exactly at the wrist frame and weld through local identity anchors.
    cache = UsdGeom.XformCache()
    body = UsdGeom.Xform.Define(stage, body_path)
    body.AddTransformOp().Set(cache.GetLocalToWorldTransform(wrist) *
                             cache.GetLocalToWorldTransform(wrist.GetParent()).GetInverse())
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateRigidBodyEnabledAttr(True)
    center, com, tensor, principal, axes = _mount_inertia()
    mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass_api.CreateMassAttr(.1)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*com))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*principal))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(_quat_gf(axes)))
    material = UsdShade.Material.Define(stage, body_path+"/contact_material")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(.5); api.CreateDynamicFrictionAttr(.5); api.CreateRestitutionAttr(0.)

    def collision(geom):
        prim = geom.GetPrim()
        geom.CreateDisplayColorAttr([Gf.Vec3f(.10,.12,.14)])
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        p = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        p.CreateContactOffsetAttr(.001); p.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")

    shell = UsdGeom.Cube.Define(stage, body_path+"/housing")
    shell.CreateSizeAttr(1.)
    shell.AddTranslateOp().Set(Gf.Vec3d(*center))
    shell.AddOrientOp().Set(Gf.Quatf(_quat_gf(rotation)))
    shell.AddScaleOp().Set(Gf.Vec3f(.04,.03,.025)); collision(shell)
    for i, (start,end) in enumerate(BRACKET_ENDPOINTS_M):
        capsule = UsdGeom.Capsule.Define(stage, body_path+f"/bracket_{i}")
        capsule.CreateAxisAttr(UsdGeom.Tokens.z)
        capsule.CreateRadiusAttr(.003); capsule.CreateHeightAttr(float(np.linalg.norm(end-start)))
        capsule.AddTranslateOp().Set(Gf.Vec3d(*(start+end)/2))
        capsule.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0,0,1),Gf.Vec3d(*(end-start))).GetQuat()))
        collision(capsule)
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.GetBody0Rel().SetTargets([Sdf.Path(wrist_path)])
    joint.GetBody1Rel().SetTargets([Sdf.Path(body_path)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0)); joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1)); joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateBreakForceAttr(float("inf")); joint.CreateBreakTorqueAttr(float("inf"))
    add_object_marker(stage, object_path, assetdir, marker_id)
    scale = np.array(Gf.Transform(UsdGeom.Xformable(obj).GetLocalTransformation()).GetScale(), float)
    expected_K=np.array([[height/(2*math.tan(math.pi/6)),0,width/2],
                         [0,height/(2*math.tan(math.pi/6)),height/2],[0,0,1.]])
    return {"status":"authored_only; new Isaac physics/perception validation required",
            "camera_path":camera_path,"resolution":[width,height],"fovy_deg":60.,
            "camera_local_position_m":LENS_POSITION_M.tolist(),"camera_rx_deg":CAMERA_RX_DEG,
            "T_wrist_camera_cv":_transform(rotation@USD_TO_CV,LENS_POSITION_M).tolist(),
            "expected_K":expected_K.tolist(),"K_source":"read actual Isaac Camera.get_intrinsics_matrix after initialize",
            "near_m":.005,"far_m":5.,"camera_axes":"USD +X right,+Y up,-Z forward; CV flips Y and Z",
            "physical_mount":{"body_path":body_path,"joint_path":joint_path,"mass_kg":.1,
                "housing_mass_kg":.08,"housing_full_size_m":[.04,.03,.025],"housing_center_wrist_m":center.tolist(),
                "housing_behind_lens_camera_z_m":[.003,.028],"bracket_mass_kg":.02,"bracket_radius_m":.003,
                "bracket_endpoints_wrist_m":BRACKET_ENDPOINTS_M.tolist(),"center_of_mass_wrist_m":com.tolist(),
                "inertia_tensor_wrist_kg_m2":tensor.tolist(),"principal_inertia_kg_m2":principal.tolist(),
                "collision_enabled":True,"fixed_joint_added":1,"degrees_of_freedom_added":0,
                "joint_collision_enabled":False,"friction_assumption":.5},
            "marker":marker,"marker_path":marker_path,"marker_parent_scale_compensation":scale.tolist(),
            "marker_texture_sha256":hashlib.sha256(texture_path.read_bytes()).hexdigest(),
            "quality_gate":gate,
            "assumptions":["Ideal pinhole RGBD: zero lens distortion/no noise/blur/exposure calibration errors.",
                "Camera/world kinematics and base localization are known; object pose is estimated from image corners.",
                "Printed marker registration/size are known. Texture has zero added mass and no collision.",
                "100g mount is generic physical bounding geometry, not a calibrated hardware camera.",
                "Wrist-to-mount joint collision is disabled; existing robot selfcollision policy remains. Other-body interference needs a new trial.",
                "Runtime camera/marker visibility, texture UV direction, planar PnP ambiguity and physical mount effects require Isaac validation."]}


def initialize_wrist_camera(metadata, frequency=None):
    """Bind an already-authored child camera after World.reset; no world steps."""
    from isaacsim.sensors.camera import Camera
    camera=Camera(metadata["camera_path"],resolution=tuple(metadata["resolution"]),frequency=frequency)
    camera.initialize()
    camera.set_clipping_range(metadata["near_m"],metadata["far_m"])
    camera.add_distance_to_image_plane_to_frame()
    K=np.asarray(camera.get_intrinsics_matrix(),float)
    if not np.allclose(K,np.asarray(metadata["expected_K"]),rtol=2e-5,atol=2e-5):
        raise RuntimeError(f"Actual camera K differs from authored projection: {K}")
    return camera


class WristArucoObserver:
    """Calibrated image-only detector; invalid results never contain a pose."""
    def __init__(self,marker=None,quality_gate=None):
        import cv2
        self.cv2=cv2
        self.marker=marker or {"dictionary":"DICT_4X4_50","id":17,"black_square_length_m":.03,
            "object_to_marker_translation_m":[0,0,.03014],"object_to_marker_rotation":np.eye(3).tolist()}
        self.gate=quality_gate or {"minimum_edge_px":20.,"maximum_reprojection_error_px":1.5,
            "minimum_depth_m":.005,"maximum_depth_m":2.,"image_border_margin_px":3.,"maximum_depth_disagreement_m":.02}
        dictionary=cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco,self.marker["dictionary"]))
        parameters=cv2.aruco.DetectorParameters();parameters.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector=cv2.aruco.ArucoDetector(dictionary,parameters)
        h=self.marker["black_square_length_m"]/2
        self.points=np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]],float)
        self.T_object_marker=_transform(np.asarray(self.marker["object_to_marker_rotation"],float),
                                       np.asarray(self.marker["object_to_marker_translation_m"],float))

    def observe(self,rgb,K,T_world_camera_cv,timestamp_s,depth_m=None,*,now_s=None,max_age_s=.01):
        cv2=self.cv2
        try:
            timestamp_s=float(timestamp_s)
        except (TypeError,ValueError):
            return {"valid":False,"timestamp_s":None,"marker_id":int(self.marker["id"]),"reason":"invalid_timestamp"}
        result={"valid":False,"timestamp_s":timestamp_s,"marker_id":int(self.marker["id"])}
        def reject(reason):return {**result,"reason":reason}
        if not math.isfinite(float(timestamp_s)):
            return reject("invalid_timestamp")
        if now_s is not None:
            age=float(now_s)-float(timestamp_s);result["frame_age_s"]=age
            if not math.isfinite(age) or age < -1e-6 or age > max_age_s:
                return reject("stale_or_future_frame")
        if (not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3
                or rgb.shape[2]!=3 or min(rgb.shape[:2])<4):
            return reject("invalid_rgb_image")
        try:
            K,T=np.asarray(K,float),np.asarray(T_world_camera_cv,float)
        except (TypeError,ValueError):
            return reject("invalid_camera_calibration")
        if K.shape!=(3,3) or T.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(T).all():
            return reject("invalid_camera_calibration")
        if K[0,0]<=0 or K[1,1]<=0 or not np.allclose(K[2],[0,0,1]) or not np.allclose(T[3],[0,0,0,1]):
            return reject("invalid_camera_calibration")
        if not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5) or np.linalg.det(T[:3,:3])<.99999:
            return reject("invalid_camera_rotation")
        if depth_m is not None and np.shape(depth_m)!=rgb.shape[:2]:
            return reject("unaligned_depth_image")
        corners,ids,rejected=self.detector.detectMarkers(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))
        result.update(detected_ids=[] if ids is None else ids.ravel().astype(int).tolist(),rejected_candidate_count=len(rejected))
        matches=[] if ids is None else np.flatnonzero(ids.ravel()==self.marker["id"])
        if len(matches)!=1:return reject("marker_not_found" if len(matches)==0 else "duplicate_marker_id")
        xy=np.asarray(corners[int(matches[0])][0],float)
        edge=np.linalg.norm(np.roll(xy,-1,axis=0)-xy,axis=1)
        result.update(corners_px=xy.tolist(),minimum_edge_px=float(edge.min()))
        if edge.min()<self.gate["minimum_edge_px"]:return reject("marker_too_small")
        margin=self.gate["image_border_margin_px"];height,width=rgb.shape[:2]
        if xy[:,0].min()<margin or xy[:,0].max()>width-1-margin or xy[:,1].min()<margin or xy[:,1].max()>height-1-margin:
            return reject("marker_at_image_boundary")
        try:
            solved,rvecs,tvecs,_=cv2.solvePnPGeneric(self.points,xy,K,np.zeros(5),flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:return reject("pnp_error")
        candidates=[]
        if solved:
            for rvec,tvec in zip(rvecs,tvecs):
                R,_=cv2.Rodrigues(rvec);p=np.asarray(tvec).reshape(3)
                if not np.isfinite(R).all() or not np.isfinite(p).all() or np.min((R@self.points.T+p[:,None])[2])<=0:continue
                projected,_=cv2.projectPoints(self.points,rvec,tvec,K,np.zeros(5))
                error=float(np.sqrt(np.mean(np.sum((projected[:,0]-xy)**2,axis=1))))
                candidates.append((error,R,p))
        if not candidates:return reject("pnp_no_positive_depth_solution")
        candidates.sort(key=lambda c:c[0]);rms,R,p=candidates[0]
        result.update(reprojection_error_px=rms,pnp_depth_m=float(p[2]),pnp_candidate_count=len(candidates),
                      pnp_candidate_reprojection_errors_px=[c[0] for c in candidates])
        if rms>self.gate["maximum_reprojection_error_px"]:return reject("pnp_reprojection_error")
        if not self.gate["minimum_depth_m"]<=p[2]<=self.gate["maximum_depth_m"]:return reject("pnp_depth_out_of_range")
        if depth_m is not None:
            # At an oblique plane, marker-center pixel ray depth is slightly
            # different from translation.z. Compare to ray/estimated-plane
            # intersection, rather than to the depth at an unrelated pixel.
            center=np.rint(xy.mean(axis=0)).astype(int)
            patch=np.asarray(depth_m)[max(0,center[1]-2):center[1]+3,max(0,center[0]-2):center[0]+3]
            finite=patch[np.isfinite(patch)&(patch>0)]
            if not len(finite):return reject("invalid_depth_at_marker")
            ray=np.linalg.solve(K,np.array([*center,1.],float));normal=R[:,2]
            denominator=float(normal@ray)
            if abs(denominator)<1e-8:return reject("marker_plane_parallel_to_depth_ray")
            expected=float(normal@p/denominator)
            measured=float(np.median(finite))
            result.update(measured_depth_m=measured,predicted_center_ray_depth_m=expected,depth_disagreement_m=abs(measured-expected))
            if expected<=0 or result["depth_disagreement_m"]>self.gate["maximum_depth_disagreement_m"]:return reject("rgb_depth_disagreement")
        T_world_marker=T@_transform(R,p)
        return {**result,"valid":True,"reason":"accepted","T_world_marker":T_world_marker.tolist(),
                "T_world_object":(T_world_marker@np.linalg.inv(self.T_object_marker)).tolist(),
                "pose_method":"RGB ArUco corners + calibrated IPPE_SQUARE; depth only quality gate",
                "camera_world_pose_assumed_known":True}


def observe_wrist_camera(camera,observer,current_time_s,*,T_world_camera_cv=None,max_frame_age_s=.01):
    """Consume one timestamped RGBD frame; does not render, step or read object GT.

    Caller renders at the current physical state. For moving robots a supplied
    T_world_camera_cv MUST be the kinematic snapshot at the image timestamp.
    Otherwise native Camera's known world pose is used, requiring render/state
    synchronization. Stale/future frames are rejected before pose estimation.
    """
    frame=camera.get_current_frame(clone=True)
    stamp=float(frame.get("rendering_time",float("nan")))
    result={"valid":False,"timestamp_s":stamp,"marker_id":int(observer.marker["id"])}
    rgba=frame.get("rgb")
    if rgba is None or np.asarray(rgba).size==0:return {**result,"reason":"camera_frame_not_ready"}
    rgba=np.asarray(rgba)
    if rgba.ndim!=3 or rgba.shape[2] not in (3,4):return {**result,"reason":"invalid_rgb_image"}
    if T_world_camera_cv is None:
        p,q=camera.get_world_pose(camera_axes="usd")
        T_world_camera_cv=_transform(_numpy_rotation(q)@USD_TO_CV,np.asarray(p,float))
        source="native_camera_pose_known; caller must synchronize pose with exposure"
    else:source="caller camera-kinematic snapshot at exposure"
    depth=frame.get("distance_to_image_plane")
    if depth is None:return {**result,"reason":"depth_frame_not_ready"}
    if np.asarray(depth).ndim==3 and np.asarray(depth).shape[2]==1:depth=np.asarray(depth)[...,0]
    observation=observer.observe(rgba[...,:3],np.asarray(camera.get_intrinsics_matrix(),float),
        np.asarray(T_world_camera_cv,float),stamp,np.asarray(depth),now_s=current_time_s,max_age_s=max_frame_age_s)
    observation.update(camera_pose_source=source,frame_timestamp_source="Isaac rendering_time",depth_kind="distance_to_image_plane meters")
    return observation


def compare_observation_to_truth(observation,T_world_object_truth):
    """Offline evaluator ONLY; no truth read occurs in detector/capture helpers."""
    if not observation.get("valid",False):return {"evaluated":False,"reason":observation.get("reason","invalid")}
    estimated=np.asarray(observation["T_world_object"],float);truth=np.asarray(T_world_object_truth,float)
    if truth.shape!=(4,4) or not np.isfinite(truth).all():raise ValueError("Expected finite4x4 truth transform")
    rotation=estimated[:3,:3]@truth[:3,:3].T
    angle=math.acos(float(np.clip((np.trace(rotation)-1)/2,-1,1)))
    return {"evaluated":True,"translation_error_m":float(np.linalg.norm(estimated[:3,3]-truth[:3,3])),
            "rotation_error_rad":angle,"ground_truth_role":"offline evaluation only"}


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Generate the deterministic marker asset; no Kit launch")
    parser.add_argument("--generate-marker",type=Path,required=True)
    options=parser.parse_args()
    print(json.dumps(generate_marker_texture(options.generate_marker),indent=2))


def add_object_marker(stage, object_path, assetdir, marker_id=17):
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
    obj=stage.GetPrimAtPath(object_path)
    marker_path=object_path+f"/project_aruco_{marker_id}"
    texture_path=Path(assetdir)/f"aruco_{marker_id}.png"
    generate_marker_texture(texture_path,marker_id)
    marker={"object_dimensions_m":[.04,.05,.06],"object_to_marker_translation_m":[0.,0.,.03014]}
    # A dynamic unit Cube scales its children. Compensate translation AND
    # dimensions: S*T(origin/S)*S^-1 gives the intended origin and meter sizes.
    local = UsdGeom.Xformable(obj).GetLocalTransformation()
    scale = np.array(Gf.Transform(local).GetScale(), float)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Object scale must be finite, positive and shear-free")
    decomposition=Gf.Transform(local)
    local_rotation=np.array(Gf.Matrix3d(decomposition.GetRotation())).T
    if not np.allclose(np.array(local)[:3,:3].T, local_rotation@np.diag(scale), atol=1e-9):
        raise ValueError("Object shear cannot be compensated by diagonal scale")
    if obj.IsA(UsdGeom.Cube):
        actual_dimensions=scale*float(UsdGeom.Cube(obj).GetSizeAttr().Get())
        if not np.allclose(actual_dimensions,marker["object_dimensions_m"],atol=1e-7,rtol=0):
            raise ValueError("Fixed marker registration requires a 40x50x60mm object")
    registration = UsdGeom.Xform.Define(stage, marker_path)
    registration.AddTranslateOp().Set(Gf.Vec3d(*(np.array(marker["object_to_marker_translation_m"])/scale)))
    registration.AddScaleOp().Set(Gf.Vec3f(*(1/scale)))
    face = UsdGeom.Mesh.Define(stage, marker_path+"/printed_panel")
    h=.02
    # Vertices TL,TR,BR,BL in object/marker XY; CCW indices give +Z normal.
    face.CreatePointsAttr([Gf.Vec3f(-h,h,0),Gf.Vec3f(h,h,0),Gf.Vec3f(h,-h,0),Gf.Vec3f(-h,-h,0)])
    face.CreateFaceVertexCountsAttr([4]); face.CreateFaceVertexIndicesAttr([0,3,2,1])
    face.CreateExtentAttr([Gf.Vec3f(-h,-h,0),Gf.Vec3f(h,h,0)])
    face.CreateDoubleSidedAttr(False); face.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    st = UsdGeom.PrimvarsAPI(face).CreatePrimvar("st",Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0,1),Gf.Vec2f(1,1),Gf.Vec2f(1,0),Gf.Vec2f(0,0)]))
    marker_mat=UsdShade.Material.Define(stage,marker_path+"/printed_material")
    surface=UsdShade.Shader.Define(stage,marker_path+"/printed_material/surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(1.)
    surface.CreateInput("metallic",Sdf.ValueTypeNames.Float).Set(0.)
    surface.CreateInput("opacity",Sdf.ValueTypeNames.Float).Set(1.)
    reader=UsdShade.Shader.Define(stage,marker_path+"/printed_material/st_reader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname",Sdf.ValueTypeNames.Token).Set("st")
    texture=UsdShade.Shader.Define(stage,marker_path+"/printed_material/texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file",Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture_path.resolve())))
    texture.CreateInput("sourceColorSpace",Sdf.ValueTypeNames.Token).Set("raw")
    texture.CreateInput("wrapS",Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("wrapT",Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("st",Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),"result")
    surface.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(),"rgb")
    marker_mat.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(),"surface")
    UsdShade.MaterialBindingAPI.Apply(face.GetPrim()).Bind(marker_mat)
    return marker_path
