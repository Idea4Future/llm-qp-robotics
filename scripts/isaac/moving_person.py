"""Project-created animated worker and an independently timed crossing script.

This is a procedural visual person with PPE and a native PhysX kinematic
capsule. It is NOT a force-driven humanoid, walking controller, human behavior
model, or NVIDIA character asset. The limb animation is visual; the capsule is
an approximate body collider and does not reproduce individual moving limbs.
Importing this file never launches Kit or advances physics.

    metadata = add_moving_person(stage)             # BEFORE World.reset
    person = ScriptedPerson(stage, metadata)
    world.reset()
    person.initialize()                            # AFTER reset
    scenario = plan_person_crossing(path_xy, radius_m=robot_radius,
                      margin_m=margin, static_rectangles=obstacles,
                      map_bounds_xy_m=bounds, start_delay_s=10.)
    # At the navigation phase start, after undocking:
    event = person.setup(scenario, world.current_time, robot_axle_xy_m=axle_xy)
    # Before EVERY world.step, independently of robot stop/release state:
    target = person.step(world.current_time)
    world.step(render=False)
    actual = person.measure(world.current_time)     # evaluation ONLY

setup is one explicit, logged external-person scenario placement. All later
body motion uses native set_kinematic_targets through the existing adapter.
No robot/tray/carried-object pose is edited. No person GT position/velocity is
an input to robot control; real scene-query LiDAR points must drive its gate.
The accepted but ignored release argument is only an adapter compatibility
field: a stopped robot cannot postpone this person's schedule.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from dynamic_obstacle import ScriptedCart, _positive, _vector


OFFICIAL_ASSET_REVIEW = {
    "isaac_version": "5.1.0",
    "official_character_doc": "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/assets/usd_assets_props.html",
    "official_actor_control_doc": "https://docs.isaacsim.omniverse.nvidia.com/5.1.0/action_and_event_data_generation/ext_replicator-agent/actor_control.html",
    "installed_people_extension": "omni.anim.people-0.7.9+107.3.3",
    "installed_extension_character_usd_count": 0,
    "official_assets_exist": True,
    "official_assets_downloaded_or_used": False,
    "decision": "Use a reproducible small project-created worker first; no downloaded mesh/texture/animation-graph dependencies.",
}


def _guard_visual_isolation(stage, visual_path):
    """Reject a visual subtree that could move physics or inherit a stale root.

    The body's native PhysX pose may advance without USD/Fabric writeback of
    its root. A separately transformed, nonphysical Visual subtree is the
    display of that measured pose; it must never contain a collider/body.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    prim = stage.GetPrimAtPath(visual_path)
    if not prim.IsValid() or not visual_path.endswith('/Visual'):
        raise ValueError('Worker isolated Visual prim missing')
    visual = UsdGeom.Xformable(prim)
    if not visual.GetResetXformStack():
        raise ValueError('Worker Visual must reset its parent transform stack')
    if [op.GetOpName() for op in visual.GetOrderedXformOps()] != [
            'xformOp:translate', 'xformOp:orient', 'xformOp:scale']:
        raise ValueError('Worker Visual requires translate/orient/scale in order')
    for child in Usd.PrimRange(prim):
        if any(child.HasAPI(api) for api in (
                UsdPhysics.CollisionAPI, UsdPhysics.RigidBodyAPI,
                UsdPhysics.ArticulationRootAPI)):
            raise ValueError('Worker Visual subtree must contain no physics API')
    return visual


def add_moving_person(stage, prim_path="/World/ProjectDynamicWorker", *,
                      initial_position_m=(1.5, 2.2, 0.), radius_m=.22,
                      height_m=1.7, mass_kg=70.):
    """Author one kinematic body with a visible procedural worker, before reset.

    The70kg authored mass does not govern its imposed trajectory: kinematic
    contacts have effectively infinite mass. The0.4m scan intersects the body
    capsule, not a learned/person-recognition sensor. Placement must be checked
    in the caller's actual warehouse. No original USD asset is changed.
    """
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    position=_vector(initial_position_m,3,'initial_position_m')
    radius=_positive(radius_m,'radius_m');height=_positive(height_m,'height_m')
    mass_kg=_positive(mass_kg,'mass_kg')
    if height<=2*radius:raise ValueError('Worker capsule must have a positive cylindrical section')
    if not prim_path.startswith('/World/ProjectDynamic') or any(not p.isidentifier() for p in prim_path.strip('/').split('/')):
        raise ValueError('Only a new /World/ProjectDynamic* external person is allowed')
    if stage.GetPrimAtPath(prim_path).IsValid():raise ValueError('Worker already exists')
    if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage),1.,abs_tol=1e-8) or UsdGeom.GetStageUpAxis(stage)!=UsdGeom.Tokens.z:
        raise ValueError('Worker requires meter units and Z-up')

    root=UsdGeom.Xform.Define(stage,prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*position));root.AddOrientOp().Set(Gf.Quatf(1.))
    rigid=UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rigid.CreateRigidBodyEnabledAttr(True);rigid.CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim()).CreateDisableGravityAttr(True)
    mass=UsdPhysics.MassAPI.Apply(root.GetPrim());mass.CreateMassAttr(mass_kg)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0,0,height/2))
    # Explicit cylindrical approximation; no human anatomical inertia claim.
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(mass_kg*(3*radius**2+height**2)/12,
                                           mass_kg*(3*radius**2+height**2)/12,mass_kg*radius**2/2))
    material=UsdShade.Material.Define(stage,prim_path+'/ContactMaterial')
    material_api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(.4);material_api.CreateDynamicFrictionAttr(.4)
    material_api.CreateRestitutionAttr(0.)
    capsule=UsdGeom.Capsule.Define(stage,prim_path+'/body_collision')
    capsule.CreateAxisAttr(UsdGeom.Tokens.z);capsule.CreateRadiusAttr(radius)
    capsule.CreateHeightAttr(height-2*radius);capsule.AddTranslateOp().Set(Gf.Vec3d(0,0,height/2))
    capsule.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(capsule.GetPrim()).CreateCollisionEnabledAttr(True)
    collision=PhysxSchema.PhysxCollisionAPI.Apply(capsule.GetPrim())
    collision.CreateContactOffsetAttr(.002);collision.CreateRestOffsetAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(capsule.GetPrim()).Bind(material,materialPurpose='physics')

    scale=height/1.7
    visual=UsdGeom.Xform.Define(stage,prim_path+'/Visual')
    # Declare the independent render transform before reset/Fabric population.
    # Only this nonphysical subtree receives measured-pose USD writes later.
    visual_translate=visual.AddTranslateOp()
    visual_orient=visual.AddOrientOp()
    visual_scale=visual.AddScaleOp()
    visual.SetXformOpOrder([visual_translate,visual_orient,visual_scale],resetXformStack=True)
    visual_translate.Set(Gf.Vec3d(*position))
    visual_orient.Set(Gf.Quatf(1.))
    visual_scale.Set(Gf.Vec3d(scale,scale,scale))
    clothes=(.12,.21,.32);skin=(.66,.43,.29);vest=(.78,.90,.06)
    black=(.035,.04,.045);helmet=(.98,.66,.045);reflective=(.83,.88,.86)
    def shape(path,kind,center,size,color):
        obj=kind.Define(stage,path)
        if kind==UsdGeom.Cube:obj.CreateSizeAttr(1.)
        else:obj.CreateRadiusAttr(1.)
        obj.AddTranslateOp().Set(Gf.Vec3d(*center));obj.AddScaleOp().Set(Gf.Vec3d(*size))
        obj.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        return obj
    v=prim_path+'/Visual'
    shape(v+'/Pelvis',UsdGeom.Sphere,(0,0,.85),(.125,.145,.115),clothes)
    shape(v+'/Jacket',UsdGeom.Cube,(0,0,1.16),(.255,.315,.46),clothes)
    shape(v+'/SafetyVest',UsdGeom.Cube,(0,0,1.20),(.267,.322,.35),vest)
    for x,label in ((.137,'Front'),(-.137,'Back')):
        for z,index in ((1.14,0),(1.26,1)):
            shape(v+f'/Reflective{label}{index}',UsdGeom.Cube,(x,0,z),(.007,.328,.027),reflective)
        for sign,suffix in ((-1,'R'),(1,'L')):
            shape(v+f'/ReflectiveVertical{label}{suffix}',UsdGeom.Cube,(x,sign*.085,1.31),(.007,.028,.17),reflective)
    shape(v+'/Neck',UsdGeom.Sphere,(0,0,1.415),(.045,.045,.068),skin)
    shape(v+'/Head',UsdGeom.Sphere,(.012,0,1.515),(.105,.099,.106),skin)
    shape(v+'/Nose',UsdGeom.Sphere,(.113,0,1.505),(.016,.016,.025),skin)
    for sign,label in ((-1,'R'),(1,'L')):
        shape(v+'/Eye'+label,UsdGeom.Sphere,(.112,sign*.036,1.54),(.009,.010,.009),black)
    shape(v+'/HardHat',UsdGeom.Sphere,(.005,0,1.65),(.125,.118,.05),helmet)
    shape(v+'/HardHatBrim',UsdGeom.Cube,(.031,0,1.616),(.30,.26,.012),helmet)

    def joint(path,position):
        j=UsdGeom.Xform.Define(stage,path)
        j.AddTranslateOp().Set(Gf.Vec3d(*position));j.AddRotateYOp().Set(0.)
    def limb(path,center,radius,length,color):
        obj=UsdGeom.Capsule.Define(stage,path);obj.CreateAxisAttr(UsdGeom.Tokens.z)
        obj.CreateRadiusAttr(radius);obj.CreateHeightAttr(length)
        obj.AddTranslateOp().Set(Gf.Vec3d(*center));obj.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    animation_paths={}
    for sign,label in ((-1,'R'),(1,'L')):
        hip=v+'/Hip'+label;joint(hip,(0,sign*.085,.83))
        limb(hip+'/Thigh',(0,0,-.17),.062,.28,clothes)
        knee=hip+'/Knee';joint(knee,(0,0,-.36))
        limb(knee+'/Shin',(0,0,-.16),.049,.27,clothes)
        ankle=knee+'/Ankle';joint(ankle,(0,0,-.36))
        shape(ankle+'/Boot',UsdGeom.Cube,(.038,0,-.045),(.19,.12,.12),black)
        shoulder=v+'/Shoulder'+label;joint(shoulder,(0,sign*.184,1.355))
        limb(shoulder+'/Sleeve',(0,0,-.125),.05,.20,clothes)
        elbow=shoulder+'/Elbow';joint(elbow,(0,0,-.27))
        limb(elbow+'/Forearm',(0,0,-.12),.037,.20,clothes)
        shape(elbow+'/Hand',UsdGeom.Sphere,(.008,0,-.27),(.04,.039,.045),skin)
        animation_paths[label]={'hip':hip,'knee':knee,'ankle':ankle,'shoulder':shoulder,'elbow':elbow}
    _guard_visual_isolation(stage, prim_path+'/Visual')
    return {'prim_path':prim_path,'collision_path':str(capsule.GetPath()),
            'initial_position_m':position.tolist(),'size_m':[2*radius,2*radius,height],
            'radius_m':radius,'height_m':height,'authored_mass_kg':mass_kg,
            'kinematic':True,'motion_model':'predefined simulation-clock trajectory; no robot-stop dependency',
            'motion_trigger':'navigation_phase_clock','release_signal_used':False,
            'visual_asset':'project-created procedural articulated-looking PPE worker',
            'source':'scripts/isaac/moving_person.py','official_asset_review':copy.deepcopy(OFFICIAL_ASSET_REVIEW),
            'external_mesh_texture_assets':[],'animation_paths':animation_paths,
            'visual_prim_path':prim_path+'/Visual',
            'visual_pose_source':'native measured PhysX',
            'visual_reset_xform_stack':True,
            'visual_pose_sync':'Measured native body pose to isolated Visual immediately before rendering; no physical pose write.',
            'animation_is_visual_only':True,'human_locomotion_controller':False,
            'collision_model':'one invisible kinematic vertical capsule; moving limb shapes/contact are approximated',
            'static_sweep_visual_radius_m':.35*scale,
            'physical_model_status':'authored_not_live_validated',
            'mass_meaning':'70kg authored approximation; effective infinite contact mass under kinematic motion',
            'limitations':['No human intent, balance, contact gait, stochastic behavior or reaction to the robot.',
                           'Arms/boots can extend beyond the body capsule; the visual is not a validated full-body collision model.',
                           'The0.4m LiDAR plane intersects the capsule; it is not semantic human recognition.',
                           'No actual PhysX execution/visibility/avoidance result is implied by source construction.']}


def plan_person_crossing(path_xy, *, radius_m, margin_m=.15, person_radius_m=.22,
                         height_m=1.7, static_rectangles=None,map_bounds_xy_m=None,
                         progress_fraction=.6,start_delay_s=10.,maximum_speed_m_s=.15,
                         minimum_end_distance_m=.8):
    """Pick an independently timed transverse walk through a path-middle point.

    The entire straight worker sweep is checked with a conservative0.35m visual
    radius against supplied2D rectangles. The person may cross the robot route;
    robot/person collision avoidance is the later sensing/control experiment.
    Timing is a fixed scenario input, never delayed by an observed robot stop.
    """
    from dynamic_obstacle import _rectangles,_clear_center
    points=np.asarray(path_xy,float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.isfinite(points).all():raise ValueError('Invalid finite path')
    points=points[np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-8]]
    if len(points)<2:raise ValueError('Zero-length path')
    radius=_positive(radius_m,'radius_m');margin=_positive(margin_m,'margin_m')
    person_radius=_positive(person_radius_m,'person_radius_m');height=_positive(height_m,'height_m')
    speed=_positive(maximum_speed_m_s,'maximum_speed_m_s')
    if not math.isfinite(float(start_delay_s)) or start_delay_s<0:raise ValueError('Invalid start delay')
    if not 0<float(progress_fraction)<1:raise ValueError('Invalid path fraction')
    minimum_end=_positive(minimum_end_distance_m,'minimum_end_distance_m')
    visual_radius=max(person_radius,.35*height/1.7)
    offset=radius+margin+visual_radius+.30
    rectangles=_rectangles(static_rectangles)
    bounds=None if map_bounds_xy_m is None else _rectangles([map_bounds_xy_m])[0]
    vectors=np.diff(points,axis=0);lengths=np.linalg.norm(vectors,axis=1);starts=np.r_[0.,np.cumsum(lengths)]
    for fraction in dict.fromkeys([float(progress_fraction),.7,.5,.4,.3]):
        progress=fraction*starts[-1]
        if min(progress,starts[-1]-progress)<minimum_end:continue
        i=min(int(np.searchsorted(starts,progress,side='right')-1),len(lengths)-1)
        tangent=vectors[i]/lengths[i];center=points[i]+(progress-starts[i])*tangent
        normal=np.array([-tangent[1],tangent[0]])
        # A bend's local normal can run into a workcell. Add fixed diagonal
        # directions with positive Y starts; every direction retains the same
        # >=30deg crossing angle, swept-body and complete-path endpoint gates.
        # Increase side distance so its transverse component stays >=offset.
        diagonals=[np.array([math.cos(math.radians(a)),math.sin(math.radians(a))])
                   for a in (45,135,30,60,120,150)]
        for direction in (normal,np.array([0.,1.]),np.array([1.,0.]),*diagonals):
            crossing_sine=abs(float(tangent[0]*direction[1]-tangent[1]*direction[0]))
            if crossing_sine<.5:continue
            side_distance=offset/crossing_sine
            start=center+side_distance*direction;end=center-side_distance*direction
            swept=start+np.linspace(0.,1.,math.ceil(2*side_distance/.025)+1)[:,None]*(end-start)
            inflated=visual_radius+.025
            if bounds is not None and (np.any(swept-inflated<bounds[:2]) or np.any(swept+inflated>bounds[2:])):continue
            if any(not _clear_center(point,rect,inflated) for point in swept for rect in rectangles):continue
            endpoint_clearances=[]
            for endpoint in (start,end):
                fractions=np.clip(np.sum((endpoint-points[:-1])*vectors,axis=1)/lengths**2,0.,1.)
                endpoint_clearances.append(float(np.min(np.linalg.norm(endpoint-(points[:-1]+fractions[:,None]*vectors),axis=1))))
            if min(endpoint_clearances)<radius+margin+person_radius+.15:continue
            # Cubic smoothstep has zero endpoint speed, peak1.5*distance/duration.
            duration=1.5*np.linalg.norm(end-start)/speed
            return {'start_position_m':[*start,0.],'end_position_m':[*end,0.],
                    'blocking_position_m':[*start,0.],'clear_position_m':[*end,0.],
                    'crossing_point_xy_m':center.tolist(),'yaw_rad':math.atan2(*(end-start)[::-1]),
                    'size_m':[2*person_radius,2*person_radius,height],
                    'person_radius_m':person_radius,'robot_radius_m':radius,'margin_m':margin,
                    'start_delay_s':float(start_delay_s),'walk_duration_s':float(duration),
                    'clear_duration_s':float(start_delay_s+duration),'maximum_script_speed_m_s':speed,
                    'path_fraction':fraction,'path_progress_m':float(progress),'side_offset_m':float(side_distance),
                    'crossing_direction_xy':direction.tolist(),'endpoint_distances_to_path_m':endpoint_clearances,
                    'visual_sweep_radius_m':visual_radius,'static_rectangles_checked':len(rectangles),
                    'clock_origin':'setup at navigation phase start after undock',
                    'motion_independent_of_robot_stop':True,
                    'scope':'fixed external-person scenario;2D geometry check only, no physical avoidance result'}
    raise ValueError('No clear transverse worker crossing: choose a wider open region/another path')


class ScriptedPerson(ScriptedCart):
    """Use one external kinematic body; animate its procedural visual limbs."""
    def __init__(self,stage,metadata,*,maximum_update_gap_s=.05):
        super().__init__(stage,metadata,maximum_update_gap_s=maximum_update_gap_s)
        self._stage=stage;self._clock_origin=None;self._start_delay=0.
        self._walk_duration=None;self._walk_distance=None
        self._visual_path = self.path+'/Visual'
        self._visual_xform = _guard_visual_isolation(stage,self._visual_path)
        visual_prim = stage.GetPrimAtPath(self._visual_path)
        self._visual_translate = visual_prim.GetAttribute('xformOp:translate')
        self._visual_orient = visual_prim.GetAttribute('xformOp:orient')
        self._visual_scale = visual_prim.GetAttribute('xformOp:scale')
        self._last_visual_sync_time = None
        self._animation_attributes={label:{name:stage.GetPrimAtPath(path).GetAttribute('xformOp:rotateY')
                         for name,path in parts.items()} for label,parts in metadata['animation_paths'].items()}
        if any(not attr.IsValid() for parts in self._animation_attributes.values() for attr in parts.values()):raise ValueError('Worker visual animation joints missing')

    def sync_visual_from_physics(self,simulation_time):
        """Copy actual native XYZ+XYZW to isolated Visual, never to the body.

        Call after physics and immediately before render. The return contains
        native measurements and written visual targets, not verified rendered
        positions. This is display synchronization, not a robot-control input
        or a replacement of native set_kinematic_targets. Repeated renders at
        one simulation timestamp are valid; backwards time and invalid state
        raise before any USD pose write.
        """
        from pxr import Gf
        if self._view is None:
            raise RuntimeError('Initialize native worker view before visual sync')
        if isinstance(simulation_time,(bool,np.bool_)):
            raise ValueError('Worker visual sync clock must be finite/nonnegative')
        now=float(simulation_time)
        if (not math.isfinite(now) or now<0. or
                (self._last_visual_sync_time is not None and
                 now<self._last_visual_sync_time-1e-9)):
            raise ValueError('Worker visual sync clock invalid/reversed')
        _guard_visual_isolation(self._stage,self._visual_path)
        actual=np.asarray(self._view.get_transforms(),dtype=float)
        if actual.shape!=(1,7) or not np.isfinite(actual).all():
            raise RuntimeError('Native worker transform invalid for visual sync')
        quaternion=actual[0,3:7]
        if abs(float(np.linalg.norm(quaternion))-1.)>1e-3:
            raise RuntimeError('Native worker quaternion is not unit length')
        visual_scale=np.asarray(self._visual_scale.Get(),dtype=float)
        if visual_scale.shape!=(3,) or not np.isfinite(visual_scale).all() or np.any(visual_scale<=0.):
            raise RuntimeError('Worker visual scale must be finite and positive')
        position=actual[0,:3]
        # No root/body_collision setter, native target or physics step occurs.
        if self._visual_translate.Set(Gf.Vec3d(*map(float,position))) is False:
            raise RuntimeError('Worker Visual translation write failed')
        if self._visual_orient.Set(Gf.Quatf(float(quaternion[3]),
                                         Gf.Vec3f(*map(float,quaternion[:3])))) is False:
            raise RuntimeError('Worker Visual orientation write failed')
        self._last_visual_sync_time=now
        return {'timestamp_s':now,'prim_path':self.path,
                'visual_prim_path':self._visual_path,
                'native_position_m':position.tolist(),
                'native_quaternion_xyzw':quaternion.tolist(),
                'height_m':float(self.metadata['height_m']),
                'visual_scale':visual_scale.tolist(),
                'visual_local_offset_m':[0.,0.,0.],
                'visual_position_target_m':position.tolist(),
                'visual_quaternion_target_wxyz':[float(quaternion[3]),*quaternion[:3].tolist()],
                'visual_pose_source':'native measured PhysX',
                'visual_reset_xform_stack':True,'visual_only':True,
                'physical_pose_written':False}

    def setup(self,scenario,simulation_time,*,robot_axle_xy_m=None):
        scenario=copy.deepcopy(scenario)
        start=_vector(scenario['start_position_m'],3,'start_position_m')
        end=_vector(scenario['end_position_m'],3,'end_position_m')
        duration=_positive(scenario['walk_duration_s'],'walk_duration_s')
        delay=float(scenario.get('start_delay_s',0.))
        if not math.isfinite(delay) or delay<0:raise ValueError('Invalid worker start delay')
        if not np.allclose(scenario['size_m'],self.metadata['size_m']):raise ValueError('Worker scenario/collider size mismatch')
        if abs(start[2])>1e-8 or abs(end[2])>1e-8:raise ValueError('Worker feet must remain on the flat scenario floor')
        facade={**scenario,'blocking_position_m':start.tolist(),'clear_position_m':end.tolist(),
                'clear_duration_s':duration,'cart_circumradius_m':self.metadata['radius_m']}
        event=super().setup(facade,simulation_time,robot_axle_xy_m=robot_axle_xy_m)
        self._clock_origin=float(simulation_time);self._start_delay=delay;self._walk_duration=duration
        self._walk_distance=float(np.linalg.norm(end-start))
        self.metadata['clear_duration_s']=delay+duration
        self.metadata['scheduled_start_s']=self._clock_origin+delay
        event.update(type='external_person_scenario_setup',scenario=copy.deepcopy(scenario),
                     motion_independent_of_robot_stop=True,clock_origin_s=self._clock_origin)
        return event

    def step(self,simulation_time,*,release=False):
        if self._clock_origin is None:raise RuntimeError('Worker scenario has not been set up')
        now=float(simulation_time)
        scheduled=now>=self._clock_origin+self._start_delay-1e-9
        target=super().step(now,release=scheduled)
        u=float(target['motion_fraction']);blend=3*u*u-2*u*u*u
        phase=2*math.pi*self._walk_distance*blend/.65
        moving=0.<u<1.
        for label,parts in self._animation_attributes.items():
            sign=1. if label=='L' else -1.
            swing=math.sin(phase)*sign if moving else 0.
            hip=18.*swing;knee=22.*max(0.,-swing)
            parts['hip'].Set(hip);parts['knee'].Set(knee);parts['ankle'].Set(-hip-knee)
            parts['shoulder'].Set(-13.*swing);parts['elbow'].Set(-7.)
        target.update(state='waiting_for_clock' if not scheduled else 'walking' if u<1. else 'finished',
                     source='external-worker independent simulation-clock script',
                     ignored_robot_release_signal=bool(release),release_signal_used=False,
                     scheduled_start_s=self._clock_origin+self._start_delay,
                     walking_visual_only=True,clear_duration_s=self._start_delay+self._walk_duration)
        return target

    def measure(self,simulation_time):
        measured=super().measure(simulation_time)
        measured.update(source='native PhysX external-worker GT; evaluation only',
                        motion_independent_of_robot_stop=True,visual_limb_animation_is_not_contact_gait=True)
        return measured


class PersonStopGate:
    """LiDAR-only stop/wait/resume gate for one independently timed worker.

    Collision-path labels associate visible returns with the scripted test
    worker. They are not semantic person detection. The worker's GT position,
    target, velocity and release event are never read by control. A previous
    positive observation inside the planned swept corridor is required before
    a later positive clear observation can permit resume. Vanishing/occlusion
    is never evidence of clearance. The margins are project design inputs, not
    a certified braking distance or guarantee on unobserved body extent.

    Compatible constructor with the cart gate, plus optional setup_time_s.
    The caller records scheduled walking start in release_time for information
    only; after_step always returns False so it cannot trigger person motion.
    """
    def __init__(self, person_path, radius, margin, clear_duration_s, path_xy, *,
                 setup_time_s=None, stop_trigger_buffer_m=.90,
                 maximum_clear_scan_gap_s=.15):
        if not isinstance(person_path,str) or not person_path.startswith('/'):
            raise ValueError('Person gate requires an absolute collision actor path')
        self.path=person_path
        self.circle=_positive(radius,'radius')+_positive(margin,'margin')
        self.clear_duration=_positive(clear_duration_s,'clear_duration_s')
        self.trigger_buffer=_positive(stop_trigger_buffer_m,'stop_trigger_buffer_m')
        self.maximum_clear_scan_gap=_positive(maximum_clear_scan_gap_s,'maximum_clear_scan_gap_s')
        if setup_time_s is not None and not math.isfinite(float(setup_time_s)):
            raise ValueError('Person gate requires a finite setup time')
        self.setup_time=None if setup_time_s is None else float(setup_time_s)
        self.setup_time_source='first_control_time' if setup_time_s is None else 'explicit_scenario_setup_time'
        points=np.asarray(path_xy,dtype=float).copy()
        if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.isfinite(points).all():
            raise ValueError('Person gate needs a finite planned Nav2 polyline')
        points=points[np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-8]]
        if len(points)<2:raise ValueError('Person gate path has zero length')
        points.setflags(write=False)
        self.path_xy=points;self.path_vectors=np.diff(points,axis=0)
        self.path_lengths_squared=np.sum(self.path_vectors**2,axis=1)
        self.mode='tracking'
        self.first_detection=self.stop_request=self.rest_since=None
        self.rest_confirmed=self.release_time=self.clear_since=self.resume_time=None
        self.resumed_motion_time=self.corridor_entry_time=None
        self.corridor_entry_latched=False
        self.path_surface_clearance=self.resume_path_surface_clearance=None
        self.hit_records=self.qp_rejections=0
        self.minimum_detected_distance=float('inf')
        self.maximum_contact_n=0.;self.load_gates_held=True
        self.path_clearance_trace=[]
        self._last_clearance_stamp=self._last_positive_scan_s=None
        self._last_control_time=self._last_after_step_time=None

    def control(self, now, phase, scans, axle_xy, desired):
        now=float(now)
        if not math.isfinite(now) or (self._last_control_time is not None and now<self._last_control_time-1e-9):
            raise RuntimeError('Person gate control time is invalid or reversed')
        self._last_control_time=now
        if self.setup_time is None:self.setup_time=now
        axle=np.asarray(axle_xy,dtype=float);desired=np.asarray(desired,dtype=float)
        if axle.shape!=(2,) or desired.shape!=(2,) or not np.isfinite(axle).all() or not np.isfinite(desired).all():
            raise RuntimeError('Person gate requires finite axle/command vectors')
        distance=None
        if phase=='navigate':
            if not isinstance(scans,(list,tuple)) or not scans:
                raise RuntimeError('Person gate requires nonempty fresh native LiDAR scans')
            values=[];person_points=[];person_stamps=[]
            for scan in scans:
                try:
                    stamp=float(scan['timestamp_s']);age_limit=float(scan.get('maximum_age_s',.15))
                except (KeyError,TypeError,ValueError,AttributeError) as exc:
                    raise RuntimeError('Person gate scan timestamp is missing/invalid') from exc
                if not scan.get('valid',False) or not math.isfinite(stamp) or not math.isfinite(age_limit) or age_limit<=0. or not -1e-8<=now-stamp<=age_limit:
                    raise RuntimeError('Person gate requires fresh valid native LiDAR scans')
                hits=scan.get('hit_paths')
                if not isinstance(hits,list):raise RuntimeError('Person gate scan hit list is missing/invalid')
                for hit in hits:
                    hit_path=hit.get('collision_path') if isinstance(hit,dict) else None
                    if not isinstance(hit_path,str):raise RuntimeError('Person gate hit collision path is invalid')
                    if hit_path==self.path or hit_path.startswith(self.path+'/'):
                        point=np.asarray(hit.get('point_world_m'),dtype=float)
                        if point.shape!=(3,) or not np.isfinite(point).all():raise RuntimeError('Person gate native return is invalid')
                        person_points.append(point[:2]);person_stamps.append(stamp)
                        values.append(float(np.linalg.norm(point[:2]-axle)))
            self.path_surface_clearance=None
            stamp=None
            if values:
                distance=min(values);stamp=min(person_stamps)
                offsets=np.asarray(person_points)[:,None,:]-self.path_xy[:-1][None,:,:]
                fractions=np.clip(np.sum(offsets*self.path_vectors[None,:,:],axis=2)/self.path_lengths_squared[None,:],0.,1.)
                self.path_surface_clearance=float(np.min(np.linalg.norm(offsets-fractions[:,:,None]*self.path_vectors[None,:,:],axis=2)))
                if self.path_surface_clearance<=self.circle+.10:
                    self.corridor_entry_latched=True
                    if self.corridor_entry_time is None:self.corridor_entry_time=stamp
                self.hit_records+=1
                self.minimum_detected_distance=min(self.minimum_detected_distance,distance)
                if self.first_detection is None:self.first_detection=now
                if self._last_clearance_stamp is None or stamp>self._last_clearance_stamp+1e-9:
                    self.path_clearance_trace.append({'scan_timestamp_s':stamp,'decision_time_s':now,
                        'person_surface_return_count':len(person_points),
                        'minimum_return_distance_to_nav_polyline_m':self.path_surface_clearance,
                        'required_resume_distance_m':self.circle+.15,
                        'minimum_radial_distance_from_robot_m':distance,
                        'corridor_entry_latched':self.corridor_entry_latched,'mode':self.mode})
                    self._last_clearance_stamp=stamp
            if self.mode=='tracking' and distance is not None and distance<=self.circle+self.trigger_buffer:
                self.mode='braking';self.stop_request=now
            if self.mode=='braking' and now-self.stop_request>5.:
                raise RuntimeError('Person gate: physical stop was not confirmed within5simseconds')
            if self.mode=='waiting':
                # Time here is NEW positive sensor-frame time, not elapsed
                # control calls over a cached frame. A missing/large scan gap
                # breaks the consecutive clearance evidence.
                clear=(self.corridor_entry_latched and stamp is not None and
                       self.path_surface_clearance>=self.circle+.15 and
                       self.rest_confirmed is not None and stamp>=self.rest_confirmed-1e-9)
                if clear:
                    if self._last_positive_scan_s is not None and stamp-self._last_positive_scan_s>self.maximum_clear_scan_gap+1e-9:
                        self.clear_since=None
                    if self.clear_since is None:self.clear_since=stamp
                    if stamp-self.clear_since>=.2-1e-9:
                        self.mode='resumed';self.resume_time=now
                        self.resume_path_surface_clearance=self.path_surface_clearance
                else:self.clear_since=None
            if self.mode=='resumed' and self.path_surface_clearance is not None and self.path_surface_clearance<self.circle+.10:
                raise RuntimeError('Person returns re-entered the swept corridor after resume')
            self._last_positive_scan_s=stamp
            if self.mode in ('braking','waiting') and now>self.setup_time+self.clear_duration+30.:
                raise RuntimeError('Person gate: no observed crossing/clearance before scenario deadline')
        return (np.zeros(2) if self.mode in ('braking','waiting') else desired.copy()),distance

    def after_step(self, now, state, contact_n, load_held):
        now=float(now)
        speed=float(state['measured_planar_speed_m_s']);yaw=float(state['measured_yaw_rate_rad_s'])
        contact=float(contact_n)
        if not all(math.isfinite(x) for x in (now,speed,yaw,contact)) or speed<0. or contact<0. or (self._last_after_step_time is not None and now<self._last_after_step_time-1e-9):
            raise RuntimeError('Person gate physical measurement is invalid or reversed')
        if self._last_after_step_time is not None and now-self._last_after_step_time>.05+1e-9:
            self.rest_since=None
        self._last_after_step_time=now
        self.maximum_contact_n=max(self.maximum_contact_n,contact)
        self.load_gates_held=self.load_gates_held and bool(load_held)
        rest=speed<.006 and abs(yaw)<.015
        if self.mode=='braking':
            if rest:
                if self.rest_since is None:self.rest_since=now
                if now-self.rest_since>=.5-1e-9:
                    self.rest_confirmed=now;self.mode='waiting'
            else:self.rest_since=None
        if self.mode=='resumed' and speed>.02 and self.resumed_motion_time is None:
            self.resumed_motion_time=now
        return False

    def summary(self):
        checks={'person_detected_by_native_lidar':self.first_detection is not None,
                'stop_requested_from_sensor_distance':self.stop_request is not None,
                'physical_rest_continuous_0p5s':self.rest_confirmed is not None,
                'person_entered_planned_corridor_by_lidar':self.corridor_entry_latched,
                'clearance_confirmed_by_positive_lidar':self.resume_time is not None,
                'physical_motion_resumed':self.resumed_motion_time is not None,
                'all_base_qps_accepted':self.qp_rejections==0,
                'no_robot_person_or_environment_contact':self.maximum_contact_n<=.05,
                'load_support_region_and_tilt_held':bool(self.load_gates_held)}
        return {'passed':all(checks.values()),'checks':checks,'mode':self.mode,
                'first_detection_s':self.first_detection,'stop_request_s':self.stop_request,
                'rest_started_s':self.rest_since,'rest_confirmed_s':self.rest_confirmed,
                'person_clock_walk_begin_s':self.release_time,
                'person_motion_independent_of_robot_stop':True,
                'release_time_used_for_control':False,'after_step_can_release_person':False,
                'corridor_entry_scan_timestamp_s':self.corridor_entry_time,
                'resume_command_s':self.resume_time,'resumed_motion_s':self.resumed_motion_time,
                'person_hit_record_count':self.hit_records,
                'minimum_sensed_person_distance_m':None if not math.isfinite(self.minimum_detected_distance) else self.minimum_detected_distance,
                'maximum_environment_contact_n':self.maximum_contact_n,'qp_rejection_count':self.qp_rejections,
                'person_gt_control_input':False,'simulated_collision_path_labels_used':True,
                'scenario_setup_time_s':self.setup_time,'scenario_setup_time_source':self.setup_time_source,
                'waiting_deadline_s':None if self.setup_time is None else self.setup_time+self.clear_duration+30.,
                'stop_trigger_buffer_m':self.trigger_buffer,'resume_clearance_buffer_m':.15,
                'incoming_corridor_latch_buffer_m':.10,
                'resume_clearance_metric':'minimum positive person LiDAR return distance to the complete planned Nav2 polyline',
                'latest_person_return_path_distance_m':self.path_surface_clearance,
                'resume_person_return_path_distance_m':self.resume_path_surface_clearance,
                'resume_required_path_distance_m':self.circle+.15,
                'path_clearance_trace':list(self.path_clearance_trace),
                'scope':'independently timed person scenario; native LiDAR stop/wait/resume only, no dynamic-route avoidance or safety guarantee',
                'clearance_observability_limit':'Visible finite body surface returns only; unseen limb/body extent and other heights are not certified. The0.15m buffer is a design input.',
                'stop_thresholds':{'planar_speed_m_s':.006,'yaw_rate_rad_s':.015,'continuous_rest_s':.5},
                'clearance_thresholds':{'new_positive_scan_duration_s':.2,'maximum_scan_gap_s':self.maximum_clear_scan_gap}}
