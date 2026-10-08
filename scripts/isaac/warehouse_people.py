"""Procedural warehouse workers; independent clock loops and LiDAR-only gate.

No import starts Kit. Route authoring uses the caller's actual height-aware
static collision inventory. GT worker poses/targets are scenario and evaluation
data ONLY, never inputs to PeopleStopGate or the robot QP. The visible PPE
workers use one kinematic capsule each; this is not humanoid gait/control.

    meta = author_warehouse_people(stage, snapshot, bounds, worker_count=3)
    world.reset()
    group = WarehousePeople(stage, meta)
    group.initialize()
    group.start(world.current_time)  # once, after initial world settling
    # BEFORE every manipulation/navigation/idle physics step:
    group.step(world.current_time)
    world.step(render=False)
    actual = group.measure(world.current_time)  # logging/evaluation only

One worker selects P3 (the main aisle). Group routes never depend on robot
stop/release. Each navigation leg constructs its own PeopleStopGate with the
fresh Nav2 path. No encounter is a normal leg result by default; a separate
require_encounter=True trial or a cross-leg aggregate must exercise avoidance.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from dynamic_obstacle import _positive, _rectangles, _vector
from moving_person import ScriptedPerson, add_moving_person


def _point_segment_distance(points, a, b):
    points=np.asarray(points,float);vector=np.asarray(b)-a
    alpha=np.clip(np.sum((points-a)*vector,axis=-1)/(vector@vector),0.,1.)
    return np.linalg.norm(points-(a+alpha[...,None]*vector),axis=-1)


def _segment_distance(a, b, c, d):
    """Exact planar segment distance, including an interior intersection."""
    u=b-a;v=d-c
    cross=lambda p,q:float(p[0]*q[1]-p[1]*q[0])
    denominator=cross(u,v)
    if abs(denominator)>1e-12:
        alpha=cross(c-a,v)/denominator;beta=cross(c-a,u)/denominator
        if 0.<=alpha<=1. and 0.<=beta<=1.:return 0.
    return float(min(*_point_segment_distance(np.array([a,b]),c,d),
                     *_point_segment_distance(np.array([c,d]),a,b)))


def _worker_routes(snapshot, bounds_xy_m, worker_count, static_margin_m,
                   human_separation_m, robot_waiting_zones=None):
    if isinstance(snapshot,dict):
        if not snapshot.get('coverage_admitted',False):raise ValueError('People require an admitted actual static collision inventory')
        height=np.asarray(snapshot.get('height_range_m',()),float)
        if height.shape!=(2,) or height[0]>.06 or height[1]<1.7:
            raise ValueError('People inventory must cover body-height obstacles, not only the LiDAR plane')
        entries=snapshot.get('rectangles')
    else:raise ValueError('People authoring requires collision snapshot metadata and rectangles')
    rectangles=np.array(_rectangles(entries),float)
    if rectangles.size==0:raise ValueError('Empty warehouse collision inventory')
    bounds=np.asarray(_rectangles([bounds_xy_m])[0])
    count=int(worker_count)
    if count!=worker_count or count not in (1,2,3):raise ValueError('worker_count must be1,2or3')
    margin=_positive(static_margin_m,'static_margin_m')
    separation=_positive(human_separation_m,'human_separation_m')
    if margin<.60-1e-9 or separation<1.5-1e-9:raise ValueError('Do not relax the worker0.60m static/1.5m pairwise design gates')
    # Caller supplies scene-config waiting positions and conservative robot
    # envelopes. Do not assume the transport envelope also covers arm motion.
    # None preserves older callers, but metadata never claims that gate ran.
    waiting_zones=[]
    if robot_waiting_zones is not None:
        if not isinstance(robot_waiting_zones,(list,tuple)) or not robot_waiting_zones:
            raise ValueError('Provide nonempty robot waiting zones, or None explicitly')
        ids=set()
        for zone in robot_waiting_zones:
            if not isinstance(zone,dict):raise ValueError('Robot waiting zone must be a dictionary')
            label=zone.get('id')
            if not isinstance(label,str) or not label.strip() or label in ids:
                raise ValueError('Robot waiting zone IDs must be unique nonempty strings')
            center=_vector(zone.get('axle_xy_m'),2,'robot waiting axle center')
            radius_value=zone.get('robot_radius_m')
            clearance_value=zone.get('clearance_margin_m',.15)
            if isinstance(radius_value,(bool,np.bool_)) or isinstance(clearance_value,(bool,np.bool_)):
                raise ValueError('Robot waiting radius/margin must be numeric, not boolean')
            radius=_positive(radius_value,'robot waiting radius')
            clearance=_positive(clearance_value,'robot waiting margin')
            if clearance<.15-1e-9:raise ValueError('Do not relax the robot waiting0.15m margin')
            ids.add(label)
            waiting_zones.append({'id':label,'axle_xy_m':center.tolist(),
                                  'robot_radius_m':radius,'clearance_margin_m':clearance})
    # CPU USD audit of all visible limbs over101gait phases found a body-local
    # complete-AABB corner radius0.364516m. Round UP for the visual sweep gate;
    # the native body collider remains the original0.22m capsule.
    visual_radius=.38;inflated=visual_radius+margin+.025
    specs=[{'id':'P1','axis':1,'lane':-1.9,'ends':(-4.8,.2),'speed':.25,'delay':0.},
           {'id':'P2','axis':1,'lane':3.1,'ends':(-4.8,.2),'speed':.28,'delay':7.},
           {'id':'P3','axis':0,'lane':2.2,'ends':(-5.,5.),'speed':.30,'delay':14.}]
    if count==1:specs=[specs[2]]
    elif count==2:specs=[specs[0],specs[2]]
    # Prefer the requested gap. Wider alternatives are explicit in metadata;
    # they do not bypass any map/static/pairwise clearance gate.
    offsets=[0.]+[sign*amount for amount in (.2,.4,.6,.8,1.,1.2,1.5,2.,2.5,3.,3.5,4.,4.5)
                  for sign in (-1.,1.)]
    candidates=[]
    for spec in specs:
        choices=[]
        for offset in offsets:
            lane=spec['lane']+offset
            a=np.array([lane,spec['ends'][0]]) if spec['axis']==1 else np.array([spec['ends'][0],lane])
            b=np.array([lane,spec['ends'][1]]) if spec['axis']==1 else np.array([spec['ends'][1],lane])
            if np.any(np.minimum(a,b)-inflated<bounds[:2]) or np.any(np.maximum(a,b)+inflated>bounds[2:]):continue
            distance=np.linalg.norm(b-a)
            samples=a+np.linspace(0.,1.,math.ceil(distance/.025)+1)[:,None]*(b-a)
            delta=np.maximum(rectangles[None,:,:2]-samples[:,None,:],0.)+np.maximum(samples[:,None,:]-rectangles[None,:,2:],0.)
            center_clearance=float(np.sqrt(np.sum(delta**2,axis=2)).min())
            if center_clearance<inflated:continue
            waiting_checks=[]
            for zone in waiting_zones:
                center_distance=float(_point_segment_distance(np.asarray([zone['axle_xy_m']]),a,b)[0])
                required=zone['robot_radius_m']+zone['clearance_margin_m']+visual_radius+.025
                waiting_checks.append({'zone_id':zone['id'],
                    'minimum_all_phase_center_distance_m':center_distance,
                    'required_center_distance_m':required,
                    'remaining_margin_m':center_distance-required})
            if any(check['remaining_margin_m']<0. for check in waiting_checks):continue
            duration=1.5*distance/spec['speed'];turn=2.
            choices.append({'worker_id':spec['id'],'start_position_m':[*a,0.],
                'end_position_m':[*b,0.],'requested_lane_m':spec['lane'],'chosen_lane_m':lane,
                'lane_offset_m':offset,'axis':spec['axis'],'maximum_script_speed_m_s':spec['speed'],
                'start_delay_s':spec['delay'],'leg_duration_s':float(duration),
                'endpoint_turn_duration_s':turn,'loop_period_s':float(2*(duration+turn)),
                'visual_sweep_radius_m':visual_radius,'static_surface_margin_m':margin,
                'sample_buffer_m':.025,'swept_sample_count':len(samples),
                'minimum_center_static_distance_m':center_clearance,
                'minimum_static_visual_surface_clearance_lower_bound_m':center_clearance-visual_radius-.025,
                'robot_waiting_zone_clearances':waiting_checks,
                'minimum_map_boundary_margin_m':float(np.minimum(np.minimum(a,b)-inflated-bounds[:2],bounds[2:]-(np.maximum(a,b)+inflated)).min()),
                'yaw_start_rad':math.atan2(*(b-a)[::-1]),'size_m':[.44,.44,1.7]})
        if not choices:raise ValueError('No static/map/waiting-clear worker lane for '+spec['id']+'; do not weaken margins')
        candidates.append(choices)
    def select(i, selected):
        if i==len(candidates):return selected
        for route in candidates[i]:
            a=np.asarray(route['start_position_m'])[:2];b=np.asarray(route['end_position_m'])[:2]
            if any(_segment_distance(a,b,np.asarray(old['start_position_m'])[:2],np.asarray(old['end_position_m'])[:2])<separation for old in selected):continue
            result=select(i+1,selected+[route])
            if result is not None:return result
        return None
    routes=select(0,[])
    if routes is None:raise ValueError('No route set meets1.5m center separation for ALL loop phases')
    pairwise=[]
    for i,a in enumerate(routes):
        for b in routes[i+1:]:
            distance=_segment_distance(np.asarray(a['start_position_m'])[:2],np.asarray(a['end_position_m'])[:2],
                                       np.asarray(b['start_position_m'])[:2],np.asarray(b['end_position_m'])[:2])
            pairwise.append({'worker_ids':[a['worker_id'],b['worker_id']],'minimum_all_phase_center_distance_m':distance})
    return routes,pairwise,len(rectangles),bounds,waiting_zones


def plan_warehouse_people(collision_snapshot, map_bounds_xy_m, *, worker_count=3,
                          static_margin_m=.60, human_separation_m=1.5,
                          robot_waiting_zones=None):
    """Pure CPU sweep gates; caller supplies config-derived robot waiting zones.

    Each zone contains id, axle_xy_m, robot_radius_m and optional margin>=.15m.
    Its radius must bound every applicable waiting/manipulation arm pose, not
    just the stowed transport pose. No supplied zones means no waiting claim.
    """
    routes,pairs,count,bounds,zones=_worker_routes(collision_snapshot,map_bounds_xy_m,worker_count,
                                                static_margin_m,human_separation_m,robot_waiting_zones)
    return {'routes':routes,'pairwise_route_clearances':pairs,'static_rectangle_count':count,
            'map_bounds_xy_m':bounds.tolist(),'static_surface_margin_m':static_margin_m,
            'human_center_separation_m':human_separation_m,
            'robot_waiting_zones':zones,
            'robot_waiting_zone_gate_applied':robot_waiting_zones is not None,
            'route_geometry_accepted':True,'physical_execution_validated':False,
            'scope':'CPU body-height2D sweeps and supplied fixed robot waiting circles only; no moving robot clearance/physics guarantee'}


def author_warehouse_people(stage, collision_snapshot, map_bounds_xy_m, *, worker_count=3,
                            static_margin_m=.60, human_separation_m=1.5,
                            robot_waiting_zones=None):
    """Author every body at its loop start BEFORE reset, without runtime jumps."""
    from pxr import Gf
    plan=plan_warehouse_people(collision_snapshot,map_bounds_xy_m,worker_count=worker_count,
                              static_margin_m=static_margin_m,human_separation_m=human_separation_m,
                              robot_waiting_zones=robot_waiting_zones)
    paths=['/World/ProjectDynamicWorker_'+route['worker_id'] for route in plan['routes']]
    if any(stage.GetPrimAtPath(path).IsValid() for path in paths):raise ValueError('People paths already exist')
    workers=[]
    for route,path in zip(plan['routes'],paths):
        metadata=add_moving_person(stage,path,initial_position_m=route['start_position_m'])
        yaw=route['yaw_start_rad']
        initial_orientation=Gf.Quatf(math.cos(yaw/2),Gf.Vec3f(0.,0.,math.sin(yaw/2)))
        stage.GetPrimAtPath(path).GetAttribute('xformOp:orient').Set(initial_orientation)
        # Initial authoring only; the independent Visual must also face the
        # route until its native measured-pose render synchronization begins.
        stage.GetPrimAtPath(path+'/Visual').GetAttribute('xformOp:orient').Set(initial_orientation)
        metadata.update(route=copy.deepcopy(route),motion_trigger='warehouse_people_clock_loop')
        metadata['static_sweep_visual_radius_m']=route['visual_sweep_radius_m']
        workers.append(metadata)
    return {**plan,'workers':workers,'prim_paths':paths,'rigid_body_paths':paths,
            'collision_paths':[x['collision_path'] for x in workers],
            'motion_trigger':'warehouse_people_clock_loop','release_signal_used':False,
            'no_runtime_setup_pose_jump':True,'source':'scripts/isaac/warehouse_people.py',
            'limitations':['One capsule per worker; moving arms/feet collision is approximated.',
                           'Native physical visibility, contact separation and braking require live verification.',
                           'No human intent/reaction/learned gait/semantic detector is modeled.']}


def loop_target(route, elapsed_s):
    """Deterministic cubic travel and smooth180deg endpoint turns, CPU-only."""
    now=float(elapsed_s)
    if not math.isfinite(now) or now<0.:raise ValueError('Invalid worker loop clock')
    a=np.asarray(route['start_position_m'],float);b=np.asarray(route['end_position_m'],float)
    leg=route['leg_duration_s'];turn=route['endpoint_turn_duration_s']
    delay=route['start_delay_s'];yaw=route['yaw_start_rad']
    if now<delay:return a.copy(),yaw,False,0.,'stagger_wait'
    clock=now-delay;period=2*(leg+turn);cycle=math.floor(clock/period);phase=clock-cycle*period
    smooth=lambda u:3*u*u-2*u*u*u
    length=float(np.linalg.norm(b-a))
    if phase<leg:
        f=smooth(phase/leg);return a+f*(b-a),yaw,True,(2*cycle+f)*length,'outbound'
    if phase<leg+turn:
        f=smooth((phase-leg)/turn);return b.copy(),yaw+math.pi*f,False,(2*cycle+1)*length,'turn_at_end'
    if phase<2*leg+turn:
        f=smooth((phase-leg-turn)/leg);return b+f*(a-b),yaw+math.pi,True,(2*cycle+1+f)*length,'return'
    f=smooth((phase-2*leg-turn)/turn)
    return a.copy(),yaw+math.pi*(1+f),False,(2*cycle+2)*length,'turn_at_start'


class _LoopWorker(ScriptedPerson):
    def start_loop(self,simulation_time):
        if self._view is None or self._scenario is not None:raise RuntimeError('Initialize each worker and start once')
        now=float(simulation_time)
        if not math.isfinite(now) or now<0:raise ValueError('Invalid worker clock start')
        route=copy.deepcopy(self.metadata['route']);position=np.asarray(route['start_position_m'])
        actual=np.asarray(self._view.get_transforms(),float)
        if actual.shape!=(1,7) or not np.isfinite(actual).all() or np.linalg.norm(actual[0,:3]-position)>.02:
            raise RuntimeError('Authored worker drifted before clock start; reject rather than reset its pose')
        self._scenario=route;self._clock_origin=now;self._last_time=now;self._last_position=position.copy()
        return {'prim_path':self.path,'timestamp_s':now,'clock_started':True,'external_pose_jump':False,
                'robot_or_load_pose_changed':False,'start_delay_s':route['start_delay_s']}

    def step(self,simulation_time,*,release=False):
        if self._scenario is None:raise RuntimeError('Worker clock was not started')
        now=float(simulation_time);dt=now-self._last_time
        if not math.isfinite(now) or dt<-1e-9 or dt>self.maximum_gap+1e-9:raise RuntimeError('Worker loop target clock missing/reversed')
        target,yaw,moving,distance,state=loop_target(self._scenario,max(0.,now-self._clock_origin))
        if np.linalg.norm(target-self._last_position)>self._scenario['maximum_script_speed_m_s']*dt+1e-7:
            raise RuntimeError('Worker loop target exceeded design peak speed')
        self._view.set_kinematic_targets(self._transform(target,yaw),np.array([0],dtype=np.uint32))
        phase=2*math.pi*distance/.65
        for label,parts in self._animation_attributes.items():
            swing=math.sin(phase)*(1. if label=='L' else -1.) if moving else 0.
            hip=18.*swing;knee=22.*max(0.,-swing)
            for name,value in {'hip':hip,'knee':knee,'ankle':-hip-knee,'shoulder':-13.*swing,'elbow':-7.}.items():parts[name].Set(value)
        self._last_time=now;self._last_position=target
        return {'timestamp_s':now,'prim_path':self.path,'state':state,'position_target_m':target.tolist(),
                'yaw_target_rad':yaw,'native_target_only':True,'release_signal_used':False,
                'ignored_robot_release_signal':bool(release),'external_pose_jump':False,
                'external_obstacle_gt_for_evaluation_only':True}


class WarehousePeople:
    def __init__(self,stage,metadata,*,maximum_update_gap_s=.05):
        self.metadata=copy.deepcopy(metadata)
        self.workers=[_LoopWorker(stage,m,maximum_update_gap_s=maximum_update_gap_s) for m in metadata['workers']]
        self._start_time=None

    def initialize(self,simulation_view=None,*,body_views=None):
        if body_views is not None and len(body_views)!=len(self.workers):raise ValueError('One mock/native body view per worker is required')
        for i,worker in enumerate(self.workers):worker.initialize(simulation_view,body_view=None if body_views is None else body_views[i])

    def start(self,simulation_time,*,robot_axle_xy_m=None,robot_exclusion_radius_m=None):
        if self._start_time is not None:raise RuntimeError('WarehousePeople clock starts only once')
        if (robot_axle_xy_m is None)!=(robot_exclusion_radius_m is None):raise ValueError('Provide both initial robot exclusion inputs, or neither')
        if robot_axle_xy_m is not None:
            axle=_vector(robot_axle_xy_m,2,'initial_robot_axle');radius=_positive(robot_exclusion_radius_m,'initial_robot_exclusion_radius')
            if any(np.linalg.norm(np.asarray(w.metadata['route']['start_position_m'])[:2]-axle)<=radius+w.metadata['route']['visual_sweep_radius_m'] for w in self.workers):
                raise RuntimeError('Worker initial route pose overlaps the robot scenario exclusion circle')
        self._start_time=float(simulation_time)
        events=[w.start_loop(simulation_time) for w in self.workers]
        return {'timestamp_s':self._start_time,'motion_independent_of_robot_stop':True,'workers':events,'external_pose_jump':False}

    def step(self,simulation_time,*,release=False):
        if self._start_time is None:raise RuntimeError('People clock was not started')
        return {'timestamp_s':float(simulation_time),'workers':[w.step(simulation_time,release=release) for w in self.workers],
                'release_signal_used':False,'motion_independent_of_robot_stop':True}

    def sync_visual_from_physics(self,simulation_time):
        """Render-only native-pose synchronization; requires initialized views.

        Does not require started motion clocks, which allows an initial render
        after reset and view initialization while the workers are stationary.
        The caller owns the subsequent world.render() and every physics step.
        """
        now=float(simulation_time)
        return {'timestamp_s':now,
                'workers':[w.sync_visual_from_physics(simulation_time) for w in self.workers],
                'visual_only':True,'physical_pose_written':False}

    def measure(self,simulation_time):
        return {'timestamp_s':float(simulation_time),'workers':[w.measure(simulation_time) for w in self.workers],
                'source':'native external-worker GT; evaluation ONLY; never robot control input'}


class PeopleStopGate:
    """Aggregate all sensor-observed active people risks; do not use their GT.

    Only a previously observed active risk is held on disappearance. Workers
    outside sensor range do not by themselves freeze the robot: the caller's
    native LiDAR travel-coverage and QP barrier checks are still required.
    A corridor-entry risk must visibly leave the FORWARD remaining swept
    corridor. A never-entered preemptive risk instead needs new positive
    returns outside its original activation envelope, with observed outward
    progress after measured rest. The two release reasons are recorded apart.
    Returning loop workers can open a new stop episode. Positive-return labels
    are simulation collision-path association, not learned human recognition.
    """
    def __init__(self,person_paths,radius,margin,path_xy,*,setup_time_s,
                 require_encounter=False,stop_trigger_buffer_m=.90,
                 maximum_episode_wait_s=150.,maximum_clear_scan_gap_s=.15,
                 preemptive_release_hysteresis_m=.05,preemptive_outward_progress_m=.02):
        paths=list(person_paths)
        if not paths or len(set(paths))!=len(paths) or any(not isinstance(p,str) or not p.startswith('/') for p in paths):
            raise ValueError('People gate needs distinct absolute rigid actor paths')
        if any(a.startswith(b+'/') for a in paths for b in paths if a!=b):raise ValueError('People paths must not be nested aliases')
        self.paths=tuple(paths);self.circle=_positive(radius,'radius')+_positive(margin,'margin')
        self.trigger_buffer=_positive(stop_trigger_buffer_m,'stop_trigger_buffer')
        self.maximum_wait=_positive(maximum_episode_wait_s,'maximum_episode_wait')
        self.maximum_scan_gap=_positive(maximum_clear_scan_gap_s,'maximum_clear_scan_gap')
        # Design thresholds for this sensor policy, not calibrated stopping or
        # sensor-error bounds. Activation requires BOTH radial/path distances
        # below R+trigger; max(radial,path) therefore measures that envelope.
        self.preemptive_hysteresis=_positive(preemptive_release_hysteresis_m,'preemptive_release_hysteresis')
        self.preemptive_outward_progress=_positive(preemptive_outward_progress_m,'preemptive_outward_progress')
        self.setup_time=float(setup_time_s);self.require_encounter=bool(require_encounter)
        if not math.isfinite(self.setup_time):raise ValueError('Finite setup time required')
        points=np.asarray(path_xy,float).copy()
        if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.isfinite(points).all():raise ValueError('Finite Nav2 polyline required')
        points=points[np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-8]]
        if len(points)<2:raise ValueError('Nonzero Nav2 path required')
        points.setflags(write=False);self.path_xy=points;self._vectors=np.diff(points,axis=0)
        self._lengths=np.linalg.norm(self._vectors,axis=1);self._starts=np.r_[0.,np.cumsum(self._lengths)]
        self.path_progress=0.;self.mode='tracking'
        self.first_detection=self.stop_request=self.rest_since=self.rest_confirmed=None
        self.release_time=self.resume_time=self.resumed_motion_time=None
        self.maximum_contact_n=0.;self.load_gates_held=True;self.qp_rejections=0
        from physical_rest import MeasuredRestWindow
        self._rest_window=MeasuredRestWindow()
        self.active={};self.retired={};self.episodes=[];self._episode=None
        self.path_clearance_trace=[];self.person_hit_records={p:0 for p in paths}
        self.minimum_detected_distance=float('inf');self._last_control=self._last_after_step=None

    def _remaining_path(self,axle):
        alpha=np.clip(np.sum((axle-self.path_xy[:-1])*self._vectors,axis=1)/self._lengths**2,0.,1.)
        projected=self.path_xy[:-1]+alpha[:,None]*self._vectors
        progress=self._starts[:-1]+alpha*self._lengths
        allowed=(progress>=self.path_progress-.1)&(progress<=self.path_progress+1.)
        distance=np.linalg.norm(projected-axle,axis=1);distance[~allowed]=np.inf
        if not np.isfinite(distance).any():raise RuntimeError('People gate lacks a local forward path projection')
        i=int(np.argmin(distance));self.path_progress=max(self.path_progress,float(progress[i]))
        index=min(int(np.searchsorted(self._starts,self.path_progress,side='right')-1),len(self._lengths)-1)
        start=self.path_xy[index]+self._vectors[index]*(self.path_progress-self._starts[index])/self._lengths[index]
        remaining=np.vstack([start,self.path_xy[index+1:]])
        # At the goal the forward swept corridor includes the endpoint disk.
        return remaining

    def _scan_people(self,now,scans,axle,remaining):
        if not isinstance(scans,(list,tuple)) or not scans:raise RuntimeError('People gate requires nonempty fresh native LiDAR scans')
        returns={p:[] for p in self.paths};stamps={p:[] for p in self.paths}
        for scan in scans:
            try:stamp=float(scan['timestamp_s']);age=float(scan.get('maximum_age_s',.15))
            except (KeyError,ValueError,TypeError,AttributeError) as exc:raise RuntimeError('People scan timestamp missing/invalid') from exc
            if not scan.get('valid',False) or not math.isfinite(stamp) or not math.isfinite(age) or age<=0 or not -1e-8<=now-stamp<=min(age,.15):
                raise RuntimeError('People gate requires fresh valid native LiDAR scans')
            hits=scan.get('hit_paths')
            if not isinstance(hits,list):raise RuntimeError('People scan hit list missing/invalid')
            for hit in hits:
                path=hit.get('collision_path') if isinstance(hit,dict) else None
                if not isinstance(path,str):raise RuntimeError('People collision-path label missing/invalid')
                for person in self.paths:
                    if path==person or path.startswith(person+'/'):
                        point=np.asarray(hit.get('point_world_m'),float)
                        if point.shape!=(3,) or not np.isfinite(point).all():raise RuntimeError('Native worker return invalid')
                        returns[person].append(point[:2]);stamps[person].append(stamp);break
        result={}
        for path,points in returns.items():
            if not points:continue
            points=np.asarray(points);radial=float(np.linalg.norm(points-axle,axis=1).min())
            vectors=np.diff(remaining,axis=0);length_squared=np.sum(vectors**2,axis=1)
            valid=length_squared>1e-14
            if not np.any(valid):
                clearance=float(np.linalg.norm(points-remaining[0],axis=1).min())
            else:
                offsets=points[:,None,:]-remaining[:-1][None,valid,:]
                fractions=np.clip(np.sum(offsets*vectors[None,valid,:],axis=2)/length_squared[None,valid],0.,1.)
                clearance=float(np.linalg.norm(offsets-fractions[:,:,None]*vectors[None,valid,:],axis=2).min())
            result[path]={'stamp':min(stamps[path]),'radial_m':radial,'path_distance_m':clearance,'return_count':len(points)}
            self.person_hit_records[path]+=1;self.minimum_detected_distance=min(self.minimum_detected_distance,radial)
            if self.first_detection is None:self.first_detection=now
        return result

    def _activate(self,path,now,seen):
        if self.mode in ('tracking','resumed'):
            self.mode='braking';self.stop_request=now;self.rest_since=self.rest_confirmed=None
            self._rest_window.reset()
            self._episode={'stop_request_s':now,'risk_paths':[],'rest_confirmed_s':None,
                           'resume_command_s':None,'resumed_motion_s':None,'clearances_m':{}}
        self.active[path]={'entered':seen['path_distance_m']<=self.circle+.10,
                           'entry_scan_s':seen['stamp'] if seen['path_distance_m']<=self.circle+.10 else None,
                           'clear_since':None,'last_positive_stamp':None,'last_trace_stamp':None,
                           'activation_time_s':now,'last_observation':seen,
                           'clear_reason':None,'clear_start_trigger_distance_m':None,
                           'last_clear_trigger_distance_m':None,'observed_outward_progress_m':0.}
        self._episode['risk_paths'].append(path);self.retired.pop(path,None)

    def control(self,now,phase,scans,axle_xy,desired):
        now=float(now);axle=_vector(axle_xy,2,'axle');desired=_vector(desired,2,'desired')
        if not math.isfinite(now) or (self._last_control is not None and now<self._last_control-1e-9):raise RuntimeError('People control clock invalid/reversed')
        self._last_control=now;nearest=None
        if phase=='navigate':
            remaining=self._remaining_path(axle);observed=self._scan_people(now,scans,axle,remaining)
            if observed:nearest=min(v['radial_m'] for v in observed.values())
            for path,seen in observed.items():
                old=self.retired.get(path)
                # Outgoing cleared people should not immediately reopen the
                # same stop solely because .90trigger>.15resume. A NEW inward
                # surface-distance decrease or corridor entry rearms it.
                if old is not None:
                    old['maximum_clearance_m']=max(old['maximum_clearance_m'],seen['path_distance_m'])
                    inward=seen['stamp']>old['resume_scan_s']+1e-9 and seen['path_distance_m']<old['maximum_clearance_m']-.02
                    if old.get('release_reason')=='preemptive_observed_retreat':
                        metric=max(seen['radial_m'],seen['path_distance_m'])
                        old['maximum_trigger_distance_m']=max(old['maximum_trigger_distance_m'],metric)
                        inward=inward or (seen['stamp']>old['resume_scan_s']+1e-9
                            and metric<old['maximum_trigger_distance_m']-.02)
                    rearm=inward or seen['path_distance_m']<=self.circle+.10
                else:rearm=True
                risky=seen['radial_m']<=self.circle+self.trigger_buffer and seen['path_distance_m']<=self.circle+self.trigger_buffer
                if path not in self.active and rearm and risky:self._activate(path,now,seen)
            for path,risk in self.active.items():
                seen=observed.get(path)
                if seen is None:
                    risk.update(clear_since=None,last_positive_stamp=None,clear_reason=None,
                                clear_start_trigger_distance_m=None,last_clear_trigger_distance_m=None,
                                observed_outward_progress_m=0.)
                    continue
                risk['last_observation']=seen;stamp=seen['stamp']
                if seen['path_distance_m']<=self.circle+.10:
                    risk['entered']=True
                    if risk['entry_scan_s'] is None:risk['entry_scan_s']=stamp
                after_rest=(self.mode=='waiting' and self.rest_confirmed is not None
                            and stamp>=self.rest_confirmed-1e-9)
                metric=max(seen['radial_m'],seen['path_distance_m'])
                reason=None
                if after_rest and seen['path_distance_m']>=self.circle+.15:
                    if risk['entered']:reason='observed_corridor_exit'
                    elif metric>=self.circle+self.trigger_buffer+self.preemptive_hysteresis:
                        reason='preemptive_observed_retreat'
                clear=reason is not None
                if clear:
                    new_positive=(risk['last_positive_stamp'] is None
                                  or stamp>risk['last_positive_stamp']+1e-9)
                    gap=(risk['last_positive_stamp'] is not None
                         and stamp-risk['last_positive_stamp']>self.maximum_scan_gap+1e-9)
                    inward=(reason=='preemptive_observed_retreat'
                            and new_positive
                            and risk['last_clear_trigger_distance_m'] is not None
                            and metric<risk['last_clear_trigger_distance_m']-.02)
                    if gap or inward or risk['clear_reason']!=reason:
                        risk['clear_since']=None
                    if risk['clear_since'] is None:
                        risk['clear_since']=stamp;risk['clear_start_trigger_distance_m']=metric
                    risk['clear_reason']=reason
                    if new_positive:
                        risk['observed_outward_progress_m']=metric-risk['clear_start_trigger_distance_m']
                        risk['last_clear_trigger_distance_m']=metric
                else:
                    risk.update(clear_since=None,clear_reason=None,clear_start_trigger_distance_m=None,
                                last_clear_trigger_distance_m=None,observed_outward_progress_m=0.)
                risk['last_positive_stamp']=stamp
                if risk['last_trace_stamp'] is None or stamp>risk['last_trace_stamp']+1e-9:
                    self.path_clearance_trace.append({'decision_time_s':now,'scan_timestamp_s':stamp,'person_path':path,
                        'mode':self.mode,'forward_path_progress_m':self.path_progress,**seen,
                        'corridor_entry_latched':risk['entered'],'clear_since_scan_s':risk['clear_since'],
                        'release_candidate_reason':risk['clear_reason'],
                        'observed_outward_progress_m':risk['observed_outward_progress_m']})
                    risk['last_trace_stamp']=stamp
            if self.mode=='braking' and now-self.stop_request>5.:raise RuntimeError('People physical rest not confirmed within5simseconds')
            if self.mode in ('braking','waiting') and now-self.stop_request>self.maximum_wait:raise RuntimeError('People observed-risk clearance timed out')
            if self.mode=='waiting' and self.active and all(r['clear_since'] is not None and
                    r['last_positive_stamp']-r['clear_since']>=.2-1e-9
                    and (r['clear_reason']=='observed_corridor_exit'
                         or r['observed_outward_progress_m']>=self.preemptive_outward_progress-1e-9)
                    for r in self.active.values()):
                self.mode='resumed';self.resume_time=now
                self._episode['release_reasons']={path:risk['clear_reason'] for path,risk in self.active.items()}
                self._episode['corridor_entry_count']=sum(bool(r['entered']) for r in self.active.values())
                self._episode['risk_release_details']={}
                for path,risk in self.active.items():
                    seen=risk['last_observation'];self._episode['clearances_m'][path]=seen['path_distance_m']
                    self._episode['risk_release_details'][path]={
                        'release_reason':risk['clear_reason'],'corridor_entry_exercised':bool(risk['entered']),
                        'entry_scan_s':risk['entry_scan_s'],'clear_since_scan_s':risk['clear_since'],
                        'clear_confirmed_scan_s':seen['stamp'],'path_distance_m':seen['path_distance_m'],
                        'radial_distance_m':seen['radial_m'],
                        'observed_outward_progress_m':risk['observed_outward_progress_m']}
                    self.retired[path]={'resume_scan_s':seen['stamp'],'maximum_clearance_m':seen['path_distance_m'],
                        'maximum_trigger_distance_m':max(seen['radial_m'],seen['path_distance_m']),
                        'release_reason':risk['clear_reason']}
                self._episode['rest_confirmed_s']=self.rest_confirmed;self._episode['resume_command_s']=now
                self.episodes.append(self._episode);self.active={}
        return (np.zeros(2) if self.mode in ('braking','waiting') else desired.copy()),nearest

    def after_step(self,now,state,contact_n,load_held):
        now=float(now);speed=float(state['measured_planar_speed_m_s']);yaw=float(state['measured_yaw_rate_rad_s']);contact=float(contact_n)
        if not all(math.isfinite(x) for x in (now,speed,yaw,contact)) or speed<0 or contact<0 or (self._last_after_step is not None and now<self._last_after_step-1e-9):
            raise RuntimeError('People physical-state measurement invalid/reversed')
        if self._last_after_step is not None and now-self._last_after_step>.05+1e-9:self.rest_since=None
        self._last_after_step=now;self.maximum_contact_n=max(self.maximum_contact_n,contact)
        self.load_gates_held=self.load_gates_held and bool(load_held)
        if self.mode=='braking':
            if self._rest_window.update(now,state):
                self.rest_confirmed=now;self.rest_since=now-.5;self.mode='waiting'
                self._episode['rest_evidence']=self._rest_window.evidence()
        if self.mode=='resumed' and speed>.02 and self._episode is not None and self._episode['resumed_motion_s'] is None:
            self._episode['resumed_motion_s']=now
            if self.resumed_motion_time is None:self.resumed_motion_time=now
        return False

    def summary(self):
        count=len(self.episodes)+(1 if self.active else 0)
        corridor_count=sum(e.get('corridor_entry_count',0) for e in self.episodes)+sum(bool(r['entered']) for r in self.active.values())
        preemptive_releases=sum(reason=='preemptive_observed_retreat'
            for e in self.episodes for reason in e.get('release_reasons',{}).values())
        resolved=not self.active and all(e['rest_confirmed_s'] is not None and e['resume_command_s'] is not None and e['resumed_motion_s'] is not None for e in self.episodes)
        checks={'all_observed_risks_physically_stopped_cleared_and_resumed':resolved,
                'required_encounter_observed':not self.require_encounter or count>=1,
                'all_base_qps_accepted':self.qp_rejections==0,
                'no_robot_person_or_environment_contact':self.maximum_contact_n<=.05,
                'load_support_region_and_tilt_held':bool(self.load_gates_held)}
        return {'passed':all(checks.values()),'checks':checks,'mode':self.mode,
                'rest_definition_revision':'windowed-rest-v1',
                'encounter_exercised':count>=1,'encounter_count':count,'completed_encounter_count':len(self.episodes),
                'corridor_entry_exercised':corridor_count>0,'corridor_entry_count':corridor_count,
                'corridor_entry_count_unit':'risk activations with sensed corridor entry; repeated activations count separately',
                'completed_corridor_entry_episode_count':sum(e.get('corridor_entry_count',0)>0
                    and e['rest_confirmed_s'] is not None and e['resume_command_s'] is not None
                    and e['resumed_motion_s'] is not None for e in self.episodes),
                'preemptive_release_count':preemptive_releases,
                'require_encounter':self.require_encounter,'active_risk_paths':list(self.active),
                'person_paths':list(self.paths),'person_hit_record_count':dict(self.person_hit_records),
                'first_detection_s':self.first_detection,'stop_request_s':self.stop_request,
                'rest_confirmed_s':self.rest_confirmed,'resume_command_s':self.resume_time,
                'resumed_motion_s':self.resumed_motion_time,'person_clock_information_s':self.release_time,
                'episodes':copy.deepcopy(self.episodes),'path_clearance_trace':list(self.path_clearance_trace),
                'minimum_sensed_person_distance_m':None if not math.isfinite(self.minimum_detected_distance) else self.minimum_detected_distance,
                'maximum_environment_contact_n':self.maximum_contact_n,'qp_rejection_count':self.qp_rejections,
                'stop_trigger_buffer_m':self.trigger_buffer,'resume_clearance_buffer_m':.15,
                'incoming_corridor_latch_buffer_m':.10,'maximum_episode_wait_s':self.maximum_wait,
                'preemptive_release_hysteresis_m':self.preemptive_hysteresis,
                'preemptive_outward_progress_required_m':self.preemptive_outward_progress,
                'preemptive_release_metric':'max(radial return distance, forward remaining-path return distance)',
                'preemptive_release_required_metric_m':self.circle+self.trigger_buffer+self.preemptive_hysteresis,
                'preemptive_release_requires_path_clearance_m':self.circle+.15,
                'forward_path_progress_m':self.path_progress,'person_gt_control_input':False,
                'release_time_used_for_control':False,'after_step_can_release_person':False,
                'simulated_collision_path_labels_used':True,
                'scope':'sensor-only aggregation of independently timed workers; stop/wait/resume, no dynamic replanning or safety guarantee',
                'observability_limit':'Only positively observed active risks are tracked. Missing active returns hold the robot; never-seen/out-of-range workers are not certified absent. Visible surfaces at one scan height do not certify unseen limbs.',
                'motion_reentry_policy':'Previously cleared worker rearms on fresh sensed inward path-distance decrease>0.02m or corridor re-entry; preemptively released workers also rearm on trigger-metric decrease>0.02m. No GT velocity/route input.',
                'design_threshold_limit':'Trigger, hysteresis and outward-progress thresholds are policy design values, not calibrated braking/uncertainty bounds or a safety guarantee.',
                'stop_thresholds':{'planar_speed_m_s':.006,'yaw_rate_rad_s':.015,'continuous_rest_s':.5},
                'clearance_thresholds':{'new_positive_scan_duration_s':.2,'maximum_scan_gap_s':self.maximum_scan_gap}}
