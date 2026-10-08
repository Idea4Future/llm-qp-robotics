#!/usr/bin/env python3
"""Physical tray loading and transport integration gate."""
from __future__ import annotations
import argparse
import json
import sys
import time
import traceback
import signal
import fcntl
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--mode", choices=("load", "drive", "full"), default="load")
parser.add_argument("--events", type=Path)
parser.add_argument("--layout", choices=("legacy","rack-v1"), default="legacy")
parser.add_argument("--rack", choices=("A","B","C"), default="A")
parser.add_argument("--moving-person", action="store_true")
parser.add_argument("--people-count",type=int,choices=(1,3),default=3)
parser.add_argument("--task-spec", type=Path)
parser.add_argument("--vision", action="store_true")
parser.add_argument("--controller", choices=("dls", "qp"), default="dls")
parser.add_argument("--navigation-preflight", action="store_true")
parser.add_argument("--dynamic-obstacle", action="store_true", help="Optional sensed cart stop/wait/resume trial during full transport")
parser.add_argument("--probe-duration",type=float,default=2.)
parser.add_argument("--probe-only", action="store_true", help="Only settle and inspect native FK/Jacobian")
parser.add_argument("--scene", choices=("minimal", "warehouse"), default="minimal")
parser.add_argument("--video", action="store_true", help="Record actual rendered physics frames at 25 simulation fps")
parser.add_argument("--inspect-gripper", action="store_true", help="Close-up camera and finger visual/physics diagnostics")
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()
if (args.dynamic_obstacle or args.moving_person) and args.mode != "full":
    parser.error("--dynamic-obstacle requires --mode full")
if args.dynamic_obstacle and args.moving_person:
    parser.error("Choose the cart or the person, not both")
args.enable_cameras = True
if args.mode == "full":
    args.vision = True
    args.controller = "qp"
if args.layout == "rack-v1":
    from rack_scene import HOME, RACKS, OUT_DOCK
    if args.task_spec:
        early_spec=json.loads(args.task_spec.read_text())
        source_id=early_spec['source_id']
        if source_id not in ('station_rack_A','station_rack_B','station_rack_C'):
            parser.error('rack-v1 requires a registered rack source')
        args.rack=source_id[-1]
    initial_xy=HOME[:2] if args.mode=='full' and not args.probe_only else RACKS[args.rack]
else:
    initial_xy=(0.,0.)
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
engine_lock = (ROOT / ".cache/isaac_transport.lock").open("a+")
fcntl.flock(engine_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
def emit(stage, detail):
    if args.events:
        args.events.parent.mkdir(parents=True, exist_ok=True)
        with args.events.open("a", encoding="utf8") as f:
            f.write(json.dumps({"type":"stage", "stage":stage, "detail":detail}, ensure_ascii=False)+"\n")
def emit_preview(path):
    """Notify the chat about a real saved frame using a job-relative path."""
    if args.events:
        from command_runtime import append_event
        path=Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        relative=path.relative_to(args.events.resolve().parent)
        append_event(args.events,"preview",path=str(relative))
def cancelled(signum, frame):
    raise KeyboardInterrupt("User requested stop")
signal.signal(signal.SIGTERM, cancelled)
emit("simulation_start", "Isaac Sim 장면과 물리 로봇을 초기화합니다.")
app = AppLauncher(dict(vars(args)), multi_gpu=False, width=640 if args.headless else 1280, height=480 if args.headless else 720, fast_shutdown=True).app
signal.signal(signal.SIGTERM, cancelled)
signal.signal(signal.SIGINT, cancelled)
samples = []
people_samples = []
people_initial_samples = []
video_writer = None
summary = {"passed": False, "scope": "stationary contact grasp with known initial pose; no navigation/QP/LLM", "probe_only": args.probe_only}
started = time.monotonic()

try:
    import numpy as np
    from scipy.spatial.transform import Rotation
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, PhysxSchema
    from isaacsim.core.api import World
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.prims import SingleRigidPrim, RigidPrim
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.sensors.camera")
    from isaacsim.sensors.camera import Camera
    from vendor_task import load_standalone_task
    from grasp_scene import add_finger_contact_geometry, add_stationary_grasp_scene, GRASP_FRAME_IN_WRIST_M
    from validate_grasp import validate
    from transport_scene import add_transport_scene, TRAY_TOP_IN_BASE
    sys.path.insert(0, str(ROOT / "third_party/rby1_isaac/src"))
    from gripper_servers.rb_gripper import RbGripperServer
    VendorTask = load_standalone_task(ROOT)

    class GraspTask(VendorTask):
        def _add_gripper_to_robot(self, stage):
            super()._add_gripper_to_robot(stage)
            for side in ("left", "right"):
                stage.GetPrimAtPath(self._gripper_side_root_path(side)).RemoveAPI(UsdPhysics.ArticulationRootAPI)
            self.pad_metadata = add_finger_contact_geometry(stage)

        def set_up_scene(self, scene):
            super().set_up_scene(scene)
            self.deinstanced_visual_paths=[]
            for prim in list(Usd.PrimRange(scene.stage.GetPrimAtPath("/World/RBY1"))):
                if prim.GetName()=="visuals" and prim.IsInstanceable():
                    prim.SetInstanceable(False)
                    self.deinstanced_visual_paths.append(str(prim.GetPath()))
            if args.layout == 'rack-v1':
                from rack_scene import add_rack_scene
                self.scene_metadata=add_rack_scene(scene.stage,args.rack)
                self.transport_metadata=add_transport_scene(scene.stage,destination_x=OUT_DOCK[0],destination_y=OUT_DOCK[1])
            else:
                self.scene_metadata = add_stationary_grasp_scene(scene.stage, transport_layout=True)
                self.transport_metadata = add_transport_scene(scene.stage)
            if args.dynamic_obstacle:
                from dynamic_obstacle import add_dynamic_cart
                self.dynamic_metadata = add_dynamic_cart(scene.stage)
            from laser_perception import add_lidar_visual
            self.laser_metadata = [add_lidar_visual(scene.stage, origin_base_m=(sign*.32,0,.4), name=name) for sign,name in ((1,"front"),(-1,"rear"))]
            if args.vision:
                from wrist_perception import create_wrist_sensor
                self.sensor_metadata = create_wrist_sensor(scene.stage, "/World/RBY1/link_right_arm_6", self.scene_metadata["object_prim"], ROOT / "assets/isaac", marker_id=self.scene_metadata.get("marker_id",17))
                if args.layout=='rack-v1':
                    from wrist_perception import add_object_marker
                    for label,record in self.scene_metadata['all_racks'].items():
                        if label!=args.rack:add_object_marker(scene.stage,record['object_prim'],ROOT/'assets/isaac',record['marker_id'])
            if args.scene == "warehouse":
                from isaacsim.storage.native import get_assets_root_path
                from isaacsim.core.utils.stage import add_reference_to_stage
                asset_root = get_assets_root_path()
                if not asset_root:
                    raise RuntimeError("Warehouse asset server unavailable")
                url = asset_root + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
                add_reference_to_stage(url, "/World/Warehouse")
                if not scene.stage.GetPrimAtPath("/World/Warehouse").GetChildren():
                    raise RuntimeError("Warehouse did not compose")
                UsdGeom.Imageable(scene.stage.GetPrimAtPath("/World/defaultGroundPlane")).MakeInvisible()
                self.scene_metadata["warehouse_url"] = url
                from warehouse_ground import deduplicate_warehouse_ground
                self.scene_metadata['ground_collision']=deduplicate_warehouse_ground(scene.stage)
            if args.moving_person:
                if args.scene!='warehouse' or args.layout!='rack-v1':
                    raise RuntimeError('Walking workers require the wide rack warehouse layout')
                from warehouse_people import author_warehouse_people
                from navigation_runtime import extract_static_collision_rectangles, workcell_rectangles
                from rack_scene import mission_config, robot_waiting_zones
                people_config=mission_config(args.rack)
                people_snapshot=extract_static_collision_rectangles(scene.stage,slice_mesh=True)
                people_snapshot['rectangles'].extend(workcell_rectangles(people_config))
                (out/'people_collision_snapshot.json').write_text(json.dumps(people_snapshot,indent=2))
                self.people_metadata=author_warehouse_people(scene.stage,people_snapshot,
                    people_config['navigation']['map_bounds_xy_m'],worker_count=args.people_count,
                    robot_waiting_zones=robot_waiting_zones(robot_radius_m=.8,clearance_margin_m=.15))
                self.people_metadata.update(motion_model='robot-priority-yield-v1',
                    motion_trigger='nominal_patrol_with_robot_priority_yield',
                    motion_independent_of_robot_stop=False,
                    human_robot_radius_design_cap_m=.8,
                    source='project scripted pedestrian yielding; native robot state and admitted path are external scenario inputs only')
                self.people_metadata['limitations']=[
                    'Scripted kinematic people react to simulator robot state and reserved route; no human sensing or learned behavior.',
                    'Robot controller still uses its own native state and LiDAR; pedestrian targets are not robot-control inputs.',
                    'One body capsule and visual-only gait; no full-limb contact or general safety guarantee.']
                for worker_metadata in self.people_metadata['workers']:
                    worker_metadata.update(motion_model='robot-priority-yield-v1',
                        motion_trigger='nominal_patrol_with_robot_priority_yield',
                        source='scripts/isaac/yielding_people.py',
                        human_locomotion_controller=False,
                        behavior_inputs='native robot state and reserved route; external scripted person behavior only')
            # Contact report flags must be authored before PhysX creates actors.
            # Post-reset views cannot recover initialization collision impulses.
            for prim in scene.stage.Traverse():
                if prim.HasAPI(UsdPhysics.RigidBodyAPI) and (str(prim.GetPath()).startswith('/World/RBY1/') or str(prim.GetPath())=='/World/CarryTray'):
                    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            ground_material = PhysicsMaterial("/World/ProjectGroundMaterial", static_friction=0.8, dynamic_friction=0.8, restitution=0.0)
            scene.get_object("default_ground_plane").apply_physics_material(ground_material)
            # Vendor base support cylinders are fixed skids, not articulated casters.
            # Model their small rolling resistance with a disclosed low-friction
            # contact approximation; drive-wheel material is unchanged.
            skid = scene.stage.GetPrimAtPath("/World/Physics_Materials/model_a_extra_geometry_material")
            if not skid.IsValid():raise RuntimeError("Expected model A support material missing")
            UsdPhysics.MaterialAPI(skid).CreateStaticFrictionAttr(.02)
            UsdPhysics.MaterialAPI(skid).CreateDynamicFrictionAttr(.02)
            PhysxSchema.PhysxMaterialAPI.Apply(skid).CreateFrictionCombineModeAttr("min")
            self.robot.set_solver_position_iteration_count(16)
            self.robot.set_solver_velocity_iteration_count(4)

        def _set_default_robot_state(self):
            super()._set_default_robot_state()
            if initial_xy != (0.,0.):
                initial_position, initial_quaternion=self.robot.get_world_pose()
                self.robot.set_world_pose(position=np.asarray(initial_position)+np.array([*initial_xy,0.]),orientation=initial_quaternion)
            q = np.zeros(self.num_joints)
            init = {"left": [-.4, .15, 0, -1, 0, 1.4, 0],
                    "right": [-.4591, -.3124, -.1156, -1.4535, -.2808, 1.9189, -.0087]}
            for side, values in init.items():
                for i, value in enumerate(values):
                    q[self.dof_names.index(f"{side}_arm_{i}")] = value
                letter = side[0]
                q[self.dof_names.index(f"gripper_finger_{letter}1")] = -.05
                q[self.dof_names.index(f"gripper_finger_{letter}2")] = .05
            # Initial state only; all later motion uses physics and bounded effort.
            self.robot.set_joint_positions(q)
            self.reference = q[self.joint_indices].copy()
            self.finger_target = -.05
            self.base_command = np.zeros(2)
            indices = np.array(self.gripper_command_indices)
            self.robot._articulation_view.set_gains(kps=np.array([[2000., 2000.]]), kds=np.array([[20., 20.]]), joint_indices=indices)
            self.robot._articulation_view.set_max_efforts(np.array([[12., 12.]]), joint_indices=indices)

        def _build_reference_target(self, simulation_time):
            mode = np.zeros(len(self.model_config.cpp_joint_names), dtype=bool)
            mode[:self.model_config.mobility_dof] = True
            target = self.reference.copy()
            from navigation_runtime import cpp_wheel_reference
            mode, target = cpp_wheel_reference(self.model_config.cpp_joint_names, target, *self.base_command)
            self.robot._articulation_view.set_joint_position_targets(
                np.array([[-.05, self.finger_target]]), joint_indices=self.gripper_command_indices)
            # Sim 5.1 gravity wrapper indexes a generalized floating-base vector.
            gravity = self.robot._articulation_view.get_generalized_gravity_forces(
                joint_indices=np.arange(self.num_joints, dtype=np.int64) + 6)[0]
            self.pd_controller.update_feedforward_term(gravity[self.joint_indices])
            return mode, target

    class TransportWorld(World):
        """Mirror measured external workers to render-only transforms."""
        def render(self, *render_args, **render_kwargs):
            people=getattr(self,'people_visual_group',None)
            synced=people.sync_visual_from_physics(self.current_time) if people is not None else None
            result=super().render(*render_args,**render_kwargs)
            if synced is not None:
                visual_record=self.people_visual_audit.update(synced)
                if not visual_record['passed']:
                    raise RuntimeError('Worker render/physics mismatch: '+str(visual_record.get('error')))
            return result

    world = TransportWorld(physics_dt=.002, rendering_dt=.04, backend="numpy", device="cpu",
                  stage_units_in_meters=1.0, sim_params={"use_gpu_pipeline": False, "enable_scene_query_support": True})
    task = GraspTask(robot_model="a", gripper_enabled=True, gripper_server=RbGripperServer())
    world.add_task(task)
    world.reset()
    dynamic_cart = None
    if args.dynamic_obstacle:
        from dynamic_obstacle import ScriptedCart
        dynamic_cart = ScriptedCart(world.stage, task.dynamic_metadata)
        dynamic_cart.initialize()
    people_group=None
    if args.moving_person:
        from yielding_people import WarehousePeople
        people_group=WarehousePeople(world.stage,task.people_metadata)
        people_group.initialize()
        from people_render_audit import WorkerVisualAudit
        world.people_visual_audit=WorkerVisualAudit(world.stage)
        world.people_visual_group=people_group
    robot, view = task.robot, task.robot._articulation_view
    obj_path, table_path = task.scene_metadata["object_prim"], task.scene_metadata["table_top_prim"]
    obj = SingleRigidPrim(obj_path, name="test_object")
    obj.initialize()
    base = SingleRigidPrim("/World/RBY1/base", name="measured_base")
    base.initialize()
    if people_group is not None:
        from navigation_runtime import gt_base_state
        people_group.configure_robot(lambda: gt_base_state(base),emit=emit)
    fingers = RigidPrim("/World/RBY1/right_gripper/ee_finger_r[12]",
                       name="finger_contacts", reset_xform_properties=False,
                       contact_filter_prim_paths_expr=[obj_path], max_contact_count=64)
    fingers.initialize()
    support = RigidPrim(obj_path, name="object_support_contact", reset_xform_properties=False,
                        contact_filter_prim_paths_expr=[table_path], max_contact_count=32)
    support.initialize()
    tray = SingleRigidPrim(task.transport_metadata["tray_prim"], name="tray_measurement", reset_xform_properties=False)
    tray.initialize()
    tray_contact = RigidPrim(obj_path, name="tray_contact", reset_xform_properties=False, contact_filter_prim_paths_expr=[task.transport_metadata["tray_prim"]], max_contact_count=64)
    tray_contact.initialize()
    if args.vision:
        from wrist_perception import initialize_wrist_camera, WristArucoObserver, observe_wrist_camera, world_camera_from_wrist
        wrist_camera = initialize_wrist_camera(task.sensor_metadata, frequency=None)
        observer = WristArucoObserver(task.sensor_metadata["marker"], task.sensor_metadata["quality_gate"])
    wrist = robot.link_right_arm_6
    arm_indices = np.array([robot.dof_names.index(f"right_arm_{i}") for i in range(7)])
    cpp_indices = np.array([task.model_config.cpp_joint_names.index(f"right_arm_{i}") for i in range(7)])
    body_index = view.get_body_index("link_right_arm_6")
    limits = view.get_dof_limits()[0, arm_indices]
    offset = np.array(GRASP_FRAME_IN_WRIST_M)
    source = np.array(task.scene_metadata['object_initial_world_center_m']) - [0,0,.001]
    dock_xy=np.array(RACKS[args.rack] if args.layout=='rack-v1' else (0.,0.))
    destination = np.r_[dock_xy+[.40,-.12], TRAY_TOP_IN_BASE + .032]
    hover, lifted, dst_hover = source + [0, 0, .14], source + [0, 0, .12], destination + [0, 0, .12]

    def tcp_pose():
        p, quat = wrist.get_world_pose()
        r = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
        return np.asarray(p) + r @ offset, r

    qp_records = []
    previous_arm_velocity = np.zeros(7)
    use_qp = args.controller == "qp" or args.task_spec is not None
    if use_qp:
        sys.path.insert(0, str(ROOT / "src"))
        from opti_robot.convex_control import ArmVelocityQP, BaseVelocityQP, task_spec_qp_config
        task_spec = json.loads(args.task_spec.read_text()) if args.task_spec else None
        qp_config = task_spec_qp_config(task_spec) if task_spec else {"arm_limits": {"joint_speed_max": .3, "joint_accel_max": .5}, "arm_weights": None, "base_limits": {"speed_max": (.25 if args.layout=="rack-v1" else .1), "accel_max": .1, "yaw_rate_max": .25, "margin": .15}, "base_weights": None}
        for key, cap in (("joint_speed_max", .3), ("joint_accel_max", .5)):
            qp_config["arm_limits"][key] = min(cap, qp_config["arm_limits"].get(key,cap))
        for key, cap in (("speed_max", .25 if args.layout=="rack-v1" else .15), ("accel_max", .1), ("yaw_rate_max", .3)):
            qp_config["base_limits"][key] = min(cap, qp_config["base_limits"].get(key,cap))
        task.admitted_spec = task_spec
        if task_spec and args.events:
            from command_runtime import emit_spec_event
            emit_spec_event(args.events, task_spec, arm_qp_active=True, base_qp_active=False,layout=args.layout)
        arm_qp = ArmVelocityQP(limits[:,0], limits[:,1], **qp_config["arm_limits"], weights=qp_config["arm_weights"], joint_position_gain=1.)

    def command_tcp(target):
        global previous_arm_velocity
        tcp, rotation = tcp_pose()
        error = np.r_[target - tcp, Rotation.from_matrix(rotation.T).as_rotvec()]
        jacobians = view.get_jacobians()
        if jacobians.shape[-1] != robot.num_dof + 6:
            raise RuntimeError(f"Unexpected floating Jacobian shape: {jacobians.shape}")
        jac = jacobians[0, body_index][:, arm_indices + 6].copy()
        r = rotation @ offset
        skew = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        jac[:3] -= skew @ jac[3:]
        current = robot.get_joint_positions()[arm_indices]
        if use_qp:
            velocity = np.r_[2.*error[:3], 2.*error[3:]]
            velocity[:3] *= min(1., .08/max(np.linalg.norm(velocity[:3]),1e-9))
            result = arm_qp.solve(current, jac, velocity, previous_arm_velocity, .02, reference_q=task.reference[cpp_indices])
            qp_records.append({"time": float(world.current_time), "controller": "arm", **result})
            if not result["accepted"]:
                raise RuntimeError("Arm QP rejected: "+str(result["reason"]))
            previous_arm_velocity = np.asarray(result["solution"])
            task.reference[cpp_indices] += .02*previous_arm_velocity
            if np.max(np.abs(task.reference[cpp_indices]-current))>.08:
                raise RuntimeError("Arm servo reference tracking error exceeded 0.08 rad")
            return
        jac[3:] *= .3
        error[3:] *= .3
        dq = jac.T @ np.linalg.solve(jac @ jac.T + .015**2 * np.eye(6), error)
        dq *= min(1.0, .012 / max(np.abs(dq).max(), 1e-9))
        current = robot.get_joint_positions()[arm_indices]
        desired_reference = task.reference[cpp_indices] + .15*dq
        bounded_reference = current + np.clip(desired_reference - current, -.04, .04)
        task.reference[cpp_indices] = np.clip(bounded_reference, limits[:, 0] + .005, limits[:, 1] - .005)

    light = UsdLux.DomeLight.Define(world.stage, "/World/Light")
    light.CreateIntensityAttr(1200.)
    camera = Camera("/World/GraspCamera", resolution=(640, 480))
    camera.initialize()
    eye, focus = Gf.Vec3d(2.3, -2.6, 1.8), Gf.Vec3d(.25, -.2, .85)
    if args.layout=='rack-v1':
        eye,focus=Gf.Vec3d(*(np.r_[dock_xy,0.]+[1.8,2.4,1.8])),Gf.Vec3d(*(np.r_[dock_xy,0.]+[.5,-.25,.9]))
    if args.inspect_gripper:
        eye, focus = Gf.Vec3d(.85, -1.0, 1.02), Gf.Vec3d(.4, -.3, .91)
    quat = Gf.Matrix4d().SetLookAt(eye, focus, Gf.Vec3d(0, 0, 1)).GetInverse().ExtractRotationQuat()
    camera.set_world_pose(np.array(eye), np.array([quat.GetReal(), *quat.GetImaginary()]), camera_axes="usd")
    camera.prim.GetAttribute("focalLength").Set(28.)
    camera.set_clipping_range(0.005, 100.0)
    if not args.headless:
        from isaacsim.core.utils.viewports import set_active_viewport_camera
        # Native viewport uses its own free perspective camera. Recording
        # follows GraspCamera independently and never resets the user's view.
        from isaacsim.core.utils.viewports import set_camera_view
        set_active_viewport_camera('/OmniverseKit_Persp')
        set_camera_view(eye=np.array([0.,6.8,6.]),target=np.array([0.,-.5,.8]),camera_prim_path='/OmniverseKit_Persp')
        from omni.kit.viewport.utility import get_active_viewport_window
        viewport_window=get_active_viewport_window()
        if viewport_window is None:raise RuntimeError('Requested Isaac viewport is unavailable')
        viewport_window.viewport_api.fill_frame=False
        viewport_window.viewport_widget.resolution=(1280,720)
        perspective=UsdGeom.Camera(world.stage.GetPrimAtPath('/OmniverseKit_Persp'))
        perspective.GetFocalLengthAttr().Set(12.)
        summary['native_viewer']={'enabled':True,'camera_path':'/OmniverseKit_Persp',
            'render_resolution':list(viewport_window.viewport_widget.resolution),
            'focal_length_mm':float(perspective.GetFocalLengthAttr().Get()),
            'recording_camera_is_separate':True}
    world.set_simulation_dt(.002, .04)
    if args.scene == "warehouse":
        from navigation_runtime import extract_static_collision_rectangles
        warehouse_snapshot = extract_static_collision_rectangles(world.stage,slice_mesh=args.layout=="rack-v1")
        if args.layout=='rack-v1':
            rack_snapshot=extract_static_collision_rectangles(world.stage,prefix='/World/ProjectRacks')
            # Free small objects belong to sensing/contact monitoring, not a
            # permanent global-map obstacle at their initial location.
            rack_snapshot['rectangles']=[r for r in rack_snapshot['rectangles'] if not r['path'].endswith('/object')]
            warehouse_snapshot['rectangles'].extend(rack_snapshot['rectangles'])
            warehouse_snapshot['project_racks']=rack_snapshot
        (out / "warehouse_collision_snapshot.json").write_text(json.dumps(warehouse_snapshot, indent=2))
    manipulation_environment_contact=None
    if args.layout=='rack-v1':
        rack_paths=[r['path'] for r in warehouse_snapshot['project_racks']['rectangles']]
        manipulation_contact_paths=list(dict.fromkeys([r['path'] for r in warehouse_snapshot['rectangles']]
            +[task.transport_metadata['destination_table_prim']]+['/World/TableB/leg_'+str(i) for i in range(4)]
            +(task.people_metadata['rigid_body_paths'] if args.moving_person else [])))
        rack_contact_indices=[manipulation_contact_paths.index(p) for p in rack_paths]
        body_paths=[str(prim.GetPath()) for prim in world.stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() in view.body_names]
        manipulation_environment_contact=RigidPrim(body_paths,name='rack_manipulation_contacts',reset_xform_properties=False,
            contact_filter_prim_paths_expr=[manipulation_contact_paths for _ in body_paths],max_contact_count=8192)
        manipulation_environment_contact.initialize()
        summary['manipulation_contact_monitor']={'filter_paths':manipulation_contact_paths,'interval_s':.002,
            'maximum_admitted_force_n':.05,'includes_people':args.moving_person}
    initial_tcp, initial_rot = tcp_pose()
    from collision_geometry import articulation_envelope
    (out/'initial_geometry.json').write_text(json.dumps({'base_pose':np.asarray(base.get_world_pose()[0]).tolist(),
       'root_pose':np.asarray(robot.get_world_pose()[0]).tolist(),'tray_pose':np.asarray(tray.get_world_pose()[0]).tolist(),
       'colliders':articulation_envelope(world.stage,view,np.asarray(base.get_world_pose()[0])[:2]+[.228,0.])},indent=2))
    summary.update(pad_geometry=task.pad_metadata, scene=task.scene_metadata, transport=task.transport_metadata,
                   deinstanced_visual_paths=task.deinstanced_visual_paths, initial_tcp=initial_tcp.tolist(), initial_rotation=initial_rot.tolist(), initial_arm_q=robot.get_joint_positions()[arm_indices].tolist(), arm_joint_limits=limits.tolist(), mode=args.mode,
                   joint_names=list(robot.dof_names), arm_indices=arm_indices.tolist(),
                   jacobian_shape=list(view.get_jacobians().shape), wrist_body_index=int(body_index),
                   gripper_leader_force_cap_n=12., friction_is_assumption=True,
                   self_collision_enabled=False, object_attachment=False, gravity_enabled=True, support_skid_friction_assumption=.02, support_skid_friction_combine="min",
                   object_pose_overrides_during_execution=False)
    if args.mode=='full' and not args.probe_only:
        summary['rest_definition_revision']='windowed-rest-v1'
    summary['articulation_solver_iterations']={'position':16,'velocity':4}
    summary['joint_velocity_feedback']='unchanged vendor finite-difference moving average'
    if args.dynamic_obstacle:
        summary["dynamic_obstacle"] = task.dynamic_metadata
    if args.moving_person:
        summary["walking_workers"] = task.people_metadata
    if args.video:
        import imageio.v2 as imageio
        video_writer = imageio.get_writer(str(out / "replay.mp4"), fps=25, codec="libx264", quality=7)
        summary["video_timing"] = "25 fps in simulation time; does not represent measured wall-time speed"
    (out / "setup.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.layout=='rack-v1':
        record_pose=camera.get_world_pose(camera_axes='usd')
        overview_eye=Gf.Vec3d(0.,6.8,6.);overview_focus=Gf.Vec3d(0.,-.5,.8)
        capture_focal=camera.prim.GetAttribute("focalLength").Get()
        camera.prim.GetAttribute("focalLength").Set(12.)
        overview_q=Gf.Matrix4d().SetLookAt(overview_eye,overview_focus,Gf.Vec3d(0,0,1)).GetInverse().ExtractRotationQuat()
        camera.set_world_pose(np.asarray(overview_eye),np.array([overview_q.GetReal(),*overview_q.GetImaginary()]),camera_axes='usd')
        for _ in range(8):world.render()
        Image.fromarray(np.asarray(camera.get_rgba())[...,:3].astype(np.uint8)).save(out/'layout_overview.png')
        emit_preview(out/'layout_overview.png')
        if not args.probe_only:
            camera.set_world_pose(*record_pose,camera_axes='usd')
            camera.prim.GetAttribute('focalLength').Set(capture_focal)

    phases = [("settle", 2., hover, hover, -.05),
              ("approach", 3., hover, source, -.05),
              ("approach_settle", 2., source, source, -.05),
              ("close", 2., source, source, -.015),
              ("lift", 3., source, lifted, -.015),
              ("hold", 3., lifted, lifted, -.015),
              ("transfer", 7., lifted, dst_hover, -.015),
              ("transfer_settle", 2., dst_hover, dst_hover, -.015),
              ("lower", 3., dst_hover, destination, -.015),
              ("lower_settle", 2., destination, destination, -.015),
              ("open", 1., destination, destination, -.05),
              ("retreat", 2., destination, destination + [0, 0, .14], -.05),
              ("place_settle", 1., destination + [0, 0, .14], destination + [0, 0, .14], -.05)]
    if use_qp:
        scale = max(1., .3/float(np.min(arm_qp.speed)))
        phases = [(name, seconds*scale, begin, end, finger) for name,seconds,begin,end,finger in phases]
    if args.probe_only:
        if not 2.<=args.probe_duration<=60.:raise ValueError("Probe duration must be2..60simseconds")
        phases = [("settle",args.probe_duration,hover,hover,-.05)]
    hold_rel, hold_rot = None, None
    base_initial = np.asarray(base.get_world_pose()[0])
    from physical_rest import MeasuredRestWindow
    from navigation_runtime import gt_base_state
    final_rest_monitor=MeasuredRestWindow()
    def execute_phases(phases):
        global hold_rel, hold_rot, source
        summary.setdefault("phase_durations_s",{}).update({name:seconds for name,seconds,*_ in phases})
        for phase, seconds, begin, end, finger in phases:
            print("Phase:", phase, flush=True)
            labels={"settle":"지정 선반에서 자세와 손목 카메라를 확인합니다.","approach":"영상으로 인식한 물체로 접근합니다.","close":"양쪽 손가락 접촉으로 물체를 집습니다.","lift":"물체를 들어 올립니다.","hold":"접촉과 들림을 유지하는지 확인합니다.","transfer":"물체를 트레이 위로 옮깁니다.","lower":"트레이에 적재합니다.","open":"그리퍼를 열어 물체를 놓습니다.","retreat":"운반 자세로 손을 물립니다.",
                    "unload_hover":"트레이 위로 손목 카메라를 이동합니다.","unload_approach":"관측한 트레이 물체로 접근합니다.","unload_close":"트레이 물체를 다시 집습니다.","unload_lift":"트레이에서 물체를 들어 올립니다.","unload_hold":"양쪽 손가락 접촉과 들림을 확인합니다.","unload_transfer":"물체를 출고대 위로 옮깁니다.","unload_lower":"출고대의 목표 위치로 내립니다.","unload_open":"그리퍼를 열어 출고대에 놓습니다.","unload_retreat":"놓은 물체에서 손을 물립니다."}
            emit(phase, labels.get(phase,"목표 자세와 물체의 정착 상태를 확인합니다."))
            reach_phase=phase=="settle" or phase.endswith(("hover_settle","approach_settle","transfer_settle","lower_settle"))
            nominal_steps=round(seconds/.002)
            maximum_steps=nominal_steps+(3000 if reach_phase else 0)
            for step in range(maximum_steps):
                t = min(1., (step + 1) * .002 / seconds)
                target = begin + (end - begin) * t * t * (3 - 2 * t)
                task.finger_target = finger
                if step % 10 == 0:
                    command_tcp(target)
                if people_group is not None and step%250==0:
                    from collision_geometry import articulation_envelope
                    from navigation_runtime import gt_base_state
                    measured_envelope=articulation_envelope(world.stage,view,gt_base_state(base)['axle_xy_m'])
                    summary.setdefault('people_manipulation_envelope_samples',[]).append({
                        'time':float(world.current_time),'phase':phase,
                        'radius_m':measured_envelope['radius_m'],'design_cap_m':.8})
                    if measured_envelope['radius_m']>.8:
                        raise RuntimeError('Manipulation envelope exceeded pedestrian waiting-zone design cap')
                if people_group is not None:people_group.step(world.current_time)
                if not app.is_running():raise KeyboardInterrupt('Isaac Sim window closed')
                world.step(render=False)
                if people_group is not None and step%10==0:
                    people_samples.append({"phase":phase,**people_group.measure(world.current_time)})
                if step % 20 == 19:
                    world.render()
                    if video_writer is not None:
                        rgba = camera.get_rgba()
                        if rgba is not None and rgba.size:
                            video_writer.append_data(np.asarray(rgba)[..., :3].astype(np.uint8))
                tcp, rot = tcp_pose()
                p, quat = obj.get_world_pose()
                objrot = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
                min_z = float(p[2] - np.abs(objrot[2]) @ (np.array([.04, .05, .06]) / 2))
                force = np.linalg.norm(fingers.get_contact_force_matrix(dt=.002)[:, 0, :], axis=1)
                table_force = float(np.linalg.norm(support.get_contact_force_matrix(dt=.002)))
                bp, bq = base.get_world_pose()
                up_z = float(1 - 2 * (bq[1]**2 + bq[2]**2))
                rel = rot.T @ (p - tcp)
                rel_rot = rot.T @ objrot
                if phase.endswith("hold") and hold_rel is None:
                    hold_rel, hold_rot = rel.copy(), rel_rot.copy()
                row = {"time": float(world.current_time), "phase": phase, "object": np.asarray(p).tolist(), "object_quaternion_wxyz":np.asarray(quat).tolist(),
                       "tcp": tcp.tolist(), "arm_q": robot.get_joint_positions()[arm_indices].tolist(), "arm_reference": task.reference[cpp_indices].tolist(), "tcp_error_m": float(np.linalg.norm(target - tcp)),
                       "tcp_rotation_error_rad": float(np.linalg.norm(Rotation.from_matrix(rot).as_rotvec())),
                       "clearance_m": min_z - .8, "finger_normal_force_n": force.tolist(), "support_force_n": table_force,
                       "relative_translation_drift_m": 0. if hold_rel is None else float(np.linalg.norm(rel - hold_rel)),
                       "relative_rotation_drift_rad": 0. if hold_rot is None else float(np.linalg.norm(Rotation.from_matrix(rel_rot @ hold_rot.T).as_rotvec())),
                       "base_xy_drift_m": float(np.linalg.norm(np.asarray(bp)[:2] - base_initial[:2])), "base_up_z": up_z,
                       "base_linear_velocity_m_s":np.asarray(base.get_linear_velocity()).tolist(),
                       "base_angular_velocity_rad_s":np.asarray(base.get_angular_velocity()).tolist(),
                       "object_velocity": np.asarray(obj.get_linear_velocity()).tolist(),
                       "object_angular_velocity": np.asarray(obj.get_angular_velocity()).tolist()}
                tray_position, tray_quat = tray.get_world_pose()
                tray_rot = Rotation.from_quat(np.asarray(tray_quat)[[1,2,3,0]]).as_matrix()
                row["tray_position_m"]=np.asarray(tray_position).tolist()
                row["tray_quaternion_wxyz"]=np.asarray(tray_quat).tolist()
                row["object_in_tray"] = (tray_rot.T @ (p - tray_position)).tolist()
                row["tray_support_force_n"] = float(np.linalg.norm(tray_contact.get_contact_force_matrix(dt=.002)))
                if "support_B" in globals():
                    row["B_support_force_n"]=float(np.linalg.norm(support_B.get_contact_force_matrix(dt=.002)))
                if manipulation_environment_contact is not None:
                    contacts=np.linalg.norm(manipulation_environment_contact.get_contact_force_matrix(dt=.002),axis=-1)
                    row['robot_rack_contact_max_n']=float(contacts[:,rack_contact_indices].max())
                    row['robot_environment_contact_max_n']=float(contacts.max())
                    if contacts.max()>.05:
                        ij=np.unravel_index(np.argmax(contacts),contacts.shape)
                        (out/'manipulation_collision.json').write_text(json.dumps({'time':float(world.current_time),'body':body_paths[ij[0]],'environment':manipulation_contact_paths[ij[1]],'force_n':float(contacts.max())},indent=2))
                        world.render();Image.fromarray(np.asarray(camera.get_rgba())[...,:3].astype(np.uint8)).save(out/'manipulation_collision.png')
                        raise RuntimeError('Robot/environment contact during manipulation')
                samples.append(row)
                if phase=='unload_place_settle':
                    row['base_state']=gt_base_state(base)
                    final_rest_monitor.update(float(world.current_time),row['base_state'])
                if phase in ("lift", "hold", "transfer", "transfer_settle", "lower", "lower_settle", "unload_lift", "unload_hold", "unload_transfer", "unload_transfer_settle", "unload_lower", "unload_lower_settle") and min(force)<.02:
                    raise RuntimeError("Bilateral grasp contact lost during "+phase)
                if not np.isfinite(np.r_[p, tcp, force, robot.get_joint_positions()]).all():
                    raise RuntimeError("Non-finite physical state")
                if row["base_xy_drift_m"] > .03 or up_z < .98:
                    raise RuntimeError("Base exceeded stationary drift/tilt bounds")
                if reach_phase and step+1>=nominal_steps and step%20==19:
                    if row["tcp_error_m"]<=.0035:
                        break
                    if step+1==nominal_steps:
                        emit(phase,"목표 위치 도착을 추가 확인합니다. 도착 기준은 4 mm이며 최대 6초 더 기다립니다.")
            summary["phase_durations_s"][phase]=(step+1)*.002
            rgba = camera.get_rgba()
            if rgba is not None and rgba.size:
                Image.fromarray(np.asarray(rgba)[..., :3].astype(np.uint8)).save(out / f"{phase}.png")
                emit_preview(out / f"{phase}.png")
            if args.inspect_gripper:
                from pxr import Usd
                cache = UsdGeom.XformCache()
                inspection = []
                for i in (1, 2):
                    body_path = f"/World/RBY1/right_gripper/ee_finger_r{i}"
                    body = world.stage.GetPrimAtPath(body_path)
                    entries = []
                    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube):
                            imageable = UsdGeom.Imageable(prim)
                            entries.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                                            "visibility": str(imageable.ComputeVisibility()),
                                            "purpose": str(imageable.ComputePurpose()),
                                            "world_transform": np.asarray(cache.GetLocalToWorldTransform(prim)).tolist()})
                    positions, orientations = fingers.get_world_poses()
                    body_index_in_view = fingers.prim_paths.index(body_path)
                    position, orientation = positions[body_index_in_view], orientations[body_index_in_view]
                    inspection.append({"body": body_path, "position": np.asarray(position).tolist(),
                                       "quaternion_wxyz": np.asarray(orientation).tolist(), "geometries": entries})
                (out / f"{phase}_finger_visuals.json").write_text(json.dumps(inspection, indent=2) + "\n")
            if phase == "settle" and args.vision:
                for _ in range(8):
                    world.render()
                wp,wq=wrist.get_world_pose()
                observation=observe_wrist_camera(wrist_camera,observer,world.current_time,T_world_camera_cv=world_camera_from_wrist(wp,wq))
                (out/"source_observation.json").write_text(json.dumps(observation,indent=2))
                Image.fromarray(np.asarray(wrist_camera.get_rgba())[...,:3].astype(np.uint8)).save(out/"source_wrist.png")
                emit_preview(out/"source_wrist.png")
                if not observation["valid"]:
                    raise RuntimeError("Wrist source observation rejected: "+observation["reason"])
                observed=np.asarray(observation["T_world_object"])[:3,3]
                if np.linalg.norm(observed-source)>.015:
                    raise RuntimeError("Observed source outside registered workcell tolerance")
                # Preserve mutable waypoint references; no physical object state is changed.
                source[:]=observed
            if (phase in ("settle", "approach_settle", "transfer_settle", "lower_settle") or phase.endswith(("hover_settle", "approach_settle", "transfer_settle", "lower_settle"))) and samples[-1]["tcp_error_m"] > .004:
                raise RuntimeError(f"{phase} TCP did not reach target: {samples[-1]['tcp_error_m']}")
            if phase.endswith("close") and min(samples[-1]["finger_normal_force_n"]) < .02:
                raise RuntimeError("No bilateral finger/object contact before lift")
    if people_group is not None:
        summary['people_clock_start']=people_group.start(world.current_time,robot_axle_xy_m=np.asarray(base.get_world_pose()[0])[:2]+[.228,0.],robot_exclusion_radius_m=.7)
    if args.layout=='rack-v1' and args.mode=='full' and not args.probe_only:
        from transport_navigation import navigate_loaded
        from navigation_runtime import gt_base_state
        from rack_scene import approach_config
        # Settle the initial physical pose before measuring a navigation envelope.
        for initial_step in range(500):
            if people_group is not None:people_group.step(world.current_time)
            world.step(render=False)
            if people_group is not None and initial_step%10==0:
                people_initial_samples.append({'phase':'initial_settle',**people_group.measure(world.current_time)})
        state=gt_base_state(base)
        approach_out=out/'approach';approach_out.mkdir()
        approach=navigate_loaded(world,task,base,obj,tray,tray_contact,camera,video_writer,approach_out,warehouse_snapshot,qp_config,emit,
            config_override=approach_config(args.rack,[*state['base_position_m'][:2],state['base_yaw_rad']]),
            loaded=False,skip_undock=True,preview=emit_preview if args.events else None,people=people_group)
        summary['approach_navigation']=approach
        if not approach['passed']:raise RuntimeError('Initial navigation to rack rejected')
        base_initial=np.asarray(base.get_world_pose()[0])
    execute_phases(phases)
    if args.probe_only:
        summary["passed"] = True
        summary["scope"] = "native FK/Jacobian and stationary pose probe only; not a grasp"
    else:
        final = [s for s in samples if s["phase"] == "place_settle"]
        hold = [s for s in samples if s["phase"] == "hold"]
        checks = {"held_above_table": min(s["clearance_m"] for s in hold) > .05,
                  "bilateral_hold": all(min(s["finger_normal_force_n"]) > .02 for s in hold),
                  "tray_supported": min(s["tray_support_force_n"] for s in final) > .02,
                  "gripper_released": max(max(s["finger_normal_force_n"]) for s in final) < .02,
                  "inside_tray": all(abs(s["object_in_tray"][0]) < .065 and abs(s["object_in_tray"][1]) < .060 for s in final),
                  "tray_height": all(abs(s["object_in_tray"][2] - .038) < .01 for s in final)}
        summary.update(passed=all(checks.values()), checks=checks, final_object_in_tray=final[-1]["object_in_tray"],
                       scope="stationary physical pick and tray load only; driving/LLM not yet evaluated by this mode")
    if args.navigation_preflight:
        from transport_navigation import navigate_loaded
        summary["navigation_preflight"] = navigate_loaded(world,task,base,obj,tray,tray_contact,camera,video_writer,out,warehouse_snapshot,qp_config,emit,preflight_only=True)
        summary["scope"] = "Native geometry/LiDAR/Nav2 binding preflight only; no loaded transport"
    if args.mode == "full" and not args.probe_only:
        if not summary["passed"]:
            raise RuntimeError("Tray loading gate rejected; navigation is not allowed")
        if args.scene != "warehouse":
            raise RuntimeError("Full transport requires the warehouse collision map")
        from transport_navigation import navigate_loaded
        summary["load_checks"] = summary["checks"]
        if task_spec and args.events:
            emit_spec_event(args.events, task_spec, arm_qp_active=True, base_qp_active=True,layout=args.layout)
        from rack_scene import mission_config
        navigation = navigate_loaded(world,task,base,obj,tray,tray_contact,camera,video_writer,out,warehouse_snapshot,qp_config,emit,cart=dynamic_cart,preview=emit_preview if args.events else None,
            config_override=mission_config(args.rack) if args.layout=='rack-v1' else None,people=people_group)
        summary["navigation"] = navigation
        if people_group is not None:
            summaries=[summary['approach_navigation'].get('dynamic_obstacle',{}),navigation.get('dynamic_obstacle',{})]
            summary['people_encounter_exercised']=any(x.get('encounter_exercised',False) for x in summaries)
            summary['people_encounter_count']=sum(x.get('encounter_count',0) for x in summaries)
            summary['people_corridor_entry_exercised']=any(x.get('corridor_entry_exercised',False) for x in summaries)
            for key in ('corridor_entry_count','completed_corridor_entry_episode_count','preemptive_release_count'):
                summary['people_'+key]=sum(x.get(key,0) for x in summaries)
        if navigation.get("passed") is not True:
            raise RuntimeError("Navigation or optional obstacle validation did not pass")
        base_initial = np.asarray(base.get_world_pose()[0])
        hold_rel, hold_rot = None, None
        previous_arm_velocity = np.zeros(7)
        current, _ = tcp_pose()
        tp,tq=tray.get_world_pose()
        tr=Rotation.from_quat(np.asarray(tq)[[1,2,3,0]]).as_matrix()
        tray_expected=np.asarray(tp)+tr@np.array([0.,0.,.038])
        tray_hover=tray_expected+[0,0,.14]
        execute_phases([("unload_hover",6.,current,tray_hover,-.05),("unload_hover_settle",3.,tray_hover,tray_hover,-.05)])
        for _ in range(8):world.render()
        wp,wq=wrist.get_world_pose()
        observation=observe_wrist_camera(wrist_camera,observer,world.current_time,T_world_camera_cv=world_camera_from_wrist(wp,wq))
        (out/"tray_observation.json").write_text(json.dumps(observation,indent=2))
        Image.fromarray(np.asarray(wrist_camera.get_rgba())[...,:3].astype(np.uint8)).save(out/"tray_wrist.png")
        if not observation["valid"]:raise RuntimeError("Tray observation rejected: "+observation["reason"])
        unload_source=np.asarray(observation["T_world_object"])[:3,3]
        if np.linalg.norm(unload_source-tray_expected)>.02:raise RuntimeError("Observed object outside registered tray slot")
        target=np.asarray(task.transport_metadata["destination_object_center"])
        raised=unload_source+[0,0,.14]
        target_hover=target+[0,0,.14]
        support_B=RigidPrim(obj_path,name="B_support",reset_xform_properties=False,contact_filter_prim_paths_expr=[task.transport_metadata["destination_table_prim"]],max_contact_count=32)
        support_B.initialize()
        execute_phases([("unload_approach",6.,tray_hover,unload_source,-.05),
                        ("unload_approach_settle",3.,unload_source,unload_source,-.05),
                        ("unload_close",3.,unload_source,unload_source,-.015),
                        ("unload_lift",6.,unload_source,raised,-.015),
                        ("unload_hold",3.,raised,raised,-.015),
                        ("unload_transfer",10.,raised,target_hover,-.015),
                        ("unload_transfer_settle",3.,target_hover,target_hover,-.015),
                        ("unload_lower",6.,target_hover,target,-.015),
                        ("unload_lower_settle",3.,target,target,-.015),
                        ("unload_open",2.,target,target,-.05),
                        ("unload_retreat",4.,target,target+[0,0,.14],-.05),
                        ("unload_place_settle",2.,target+[0,0,.14],target+[0,0,.14],-.05)])
        final=[row for row in samples if row['phase']=='unload_place_settle']
        checks={"on_B_target":all(np.linalg.norm(np.asarray(row['object'])[:2]-target[:2])<.015 for row in final),
                "B_support":all(row['B_support_force_n']>.02 for row in final),
                "released":all(max(row['finger_normal_force_n'])<.02 for row in final),
                "settled":all(np.linalg.norm(row['object_velocity'])<.01 for row in final),
                "upright_height":all(abs(row['object'][2]-.83)<.005 for row in final)}
        unload_hold=[row for row in samples if row["phase"]=="unload_hold"]
        checks["unload_bilateral_hold"]=all(min(row["finger_normal_force_n"])>.02 for row in unload_hold)
        checks["unload_lift_clearance"]=all(row["object_in_tray"][2]>.10 for row in unload_hold)
        summary['final_rest_evidence']=final_rest_monitor.evidence()
        checks['final_base_windowed_rest']=summary['final_rest_evidence']['ready']
        summary.update(passed=all(checks.values()),unload_checks=checks,mode="full",scope="continuous image-guided physical pick/tray/wheel transport/unload with local arm and base QPs",final_object=final[-1]['object'])
        from validate_transport import validate as validate_transport_trace
        independent = validate_transport_trace(samples, task.scene_metadata, task.transport_metadata, mode="full",
            navigation_trace=json.loads((out/"navigation_trace.json").read_text()), phase_durations_s=summary["phase_durations_s"],
            approach_trace=json.loads((out/"approach/navigation_trace.json").read_text()) if args.layout=="rack-v1" else None,
            final_rest_evidence=summary['final_rest_evidence'],rest_definition_revision=summary['rest_definition_revision'],
            **({'manipulation_contact_monitor':summary['manipulation_contact_monitor']} if 'manipulation_contact_monitor' in summary else {}))
        (out/"trace_validation.json").write_text(json.dumps(independent,indent=2))
        summary["independent_validation_passed"]=independent["passed"]
        summary["passed"]=summary["passed"] and independent["passed"]
        if people_group is not None:
            from validate_people import validate_people
            people_audit=validate_people(summary,people_samples,
                json.loads((out/'people_collision_snapshot.json').read_text()),
                initial_records=people_initial_samples,inventory_role='authoring',
                approach_trace=json.loads((out/'approach/people_trace.json').read_text()),
                navigation_trace=json.loads((out/'people_trace.json').read_text()))
            (out/'people_saved_audit.json').write_text(json.dumps(people_audit,indent=2))
            summary['independent_people_validation_passed']=people_audit['passed']
            summary['passed']=summary['passed'] and people_audit['passed']
        if args.layout=='rack-v1':
            # Final overview is a render of the completed physical state. It
            # never changes the user's independent native perspective camera.
            camera.prim.GetAttribute('focalLength').Set(12.)
            camera.set_world_pose(np.asarray(overview_eye),np.array([overview_q.GetReal(),*overview_q.GetImaginary()]),camera_axes='usd')
            for _ in range(8):world.render()
            Image.fromarray(np.asarray(camera.get_rgba())[...,:3].astype(np.uint8)).save(out/'final_overview.png')
            emit_preview(out/'final_overview.png')
        emit("verify", "출고대 지지·그리퍼 해제·물체 정착을 측정합니다.")
    if args.mode == "drive" and summary["passed"]:
        from navigation_runtime import gt_base_state
        probe_start = gt_base_state(base)
        drive_records = []
        for label, seconds, command in [("reverse", 4., [-.08, 0.]), ("brake", 2., [0., 0.]), ("turn", 4., [0., .12]), ("rest", 2., [0., 0.])]:
            print("Drive probe:", label, flush=True)
            for step in range(round(seconds/.002)):
                task.base_command += np.clip(np.array(command)-task.base_command, [-.1*.002,-.25*.002],[.1*.002,.25*.002])
                world.step(render=False)
                if step%20 == 19:
                    world.render()
                    if video_writer is not None:
                        video_writer.append_data(np.asarray(camera.get_rgba())[...,:3].astype(np.uint8))
                state=gt_base_state(base)
                pp,pq=tray.get_world_pose()
                rr=Rotation.from_quat(np.asarray(pq)[[1,2,3,0]]).as_matrix()
                relative=rr.T@(obj.get_world_pose()[0]-pp)
                state.update(time=float(world.current_time),phase=label,object_in_tray=relative.tolist(),command=task.base_command.tolist(),tray_support_force_n=float(np.linalg.norm(tray_contact.get_contact_force_matrix(dt=.002))))
                drive_records.append(state)
                state["wheel_velocity_native_rad_s"] = robot.get_joint_velocities()[[robot.dof_names.index("right_wheel"),robot.dof_names.index("left_wheel")]].tolist()
                state["wheel_pd_torque_before_clip_nm"] = task.pd_controller.target_torque[:2].tolist()
                state["wheel_applied_effort_nm"] = np.clip(task.pd_controller.target_torque[:2],-task.max_efforts[:2],task.max_efforts[:2]).tolist()
                state["wheel_native_effort_cap_nm"] = task.max_efforts[:2].tolist()
                if not args.probe_only and (abs(relative[0])>.065 or abs(relative[1])>.060 or abs(relative[2]-.038)>.02):
                    raise RuntimeError("Object left tray during drive probe")
        reverse_end=[r for r in drive_records if r["phase"]=="brake"][-1]
        turn_end=drive_records[-1]
        checks={"reverse_sign":reverse_end["base_position_m"][0]-probe_start["base_position_m"][0]<-.15,
                "yaw_sign":turn_end["base_yaw_rad"]-reverse_end["base_yaw_rad"]>.2,
                "rest_speed":turn_end["measured_planar_speed_m_s"]<.01,
                "tray_support":args.probe_only or min(r["tray_support_force_n"] for r in drive_records)>.02}
        if args.probe_only: del checks["tray_support"]
        summary.update(passed=all(checks.values()),drive_probe_checks=checks,drive_end=turn_end,scope="empty wheel probe only" if args.probe_only else "physical tray loading, reverse/turn/braking probe; not a full transport task")
        (out/"drive_samples.json").write_text(json.dumps(drive_records))

except (Exception, KeyboardInterrupt) as error:
    traceback.print_exc()
    summary.update(passed=False, error=str(error), error_type=type(error).__name__)
finally:
    try:
        if args.mode == "full" and not args.probe_only:
            # A load-only milestone or interrupted navigation can never be
            # mistaken for completion when Kit's fast close exits with zero.
            summary["passed"] = bool(summary.get("passed") and summary.get("independent_validation_passed") is True
                                      and summary.get("navigation", {}).get("passed") is True)
        summary.update(wall_seconds=time.monotonic() - started, physical_samples=len(samples))
        if 'world' in globals() and hasattr(world,'people_visual_audit'):
            visual_audit=world.people_visual_audit.summary()
            summary['people_visual_audit']=visual_audit
            summary['passed']=bool(summary['passed'] and visual_audit['passed'])
            (out/'people_render_samples.json').write_text(json.dumps(world.people_visual_audit.records))
        # Persist the raw measurements first, including after a gate failure.
        if people_samples:(out/"people_manipulation_samples.json").write_text(json.dumps(people_samples))
        if people_initial_samples:(out/"people_initial_samples.json").write_text(json.dumps(people_initial_samples))
        if 'people_group' in globals() and people_group is not None and hasattr(people_group,'navigation_reservations'):
            (out/'people_yield_events.json').write_text(json.dumps({
                'motion_model':'robot-priority-yield-v1','events':people_group.events,
                'navigation_reservations':people_group.navigation_reservations,
                'scope':'External scripted pedestrian behavior; not robot sensor observations.'}))
            summary['people_yield_behavior']={'motion_model':'robot-priority-yield-v1',
                'mode_transition_count':len(people_group.events),
                'reservation_count':len(people_group.navigation_reservations),
                'robot_ground_truth_used_by_people_only':True}
        (out / "state_samples.json").write_text(json.dumps(samples) + "\n")
        if "qp_records" in globals():
            (out / "qp_records.json").write_text(json.dumps(qp_records))
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    finally:
        try:
            if video_writer is not None:
                video_writer.close()
        finally:
            app.close(skip_cleanup=True)
