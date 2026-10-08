"""Continuous torque-driven loaded navigation; the caller owns the live Isaac world."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from navigation_runtime import (load_config, gt_base_state, request_nav2_route, PolylineReference,
                                straight_dock_reference, point_rectangle_distance,
                                require_native_base_view)
from collision_geometry import articulation_envelope


def require_navigation_physics(world, base):
    """Fail when GUI pause/STOP/reset makes native mission data unavailable."""
    if not world.is_playing():
        raise RuntimeError('Isaac simulation was paused/stopped during the mission; native stop measurement unavailable')
    require_native_base_view(base)


def observe_exceptional_stop(world, task, base, obj, tray, tray_contact,
                             environment_contact, *, trigger_exception,
                             loaded, accel_max, emit, people=None, cart=None,
                             cart_setup=None):
    """Set only a zero reference, then observe recovery iff physics is live.

Never restarts/reset/plays a paused world. Secondary recovery errors become
metadata so the caller can re-raise the ORIGINAL mission exception.
"""
    from physical_rest import MeasuredRestWindow
    initial_command = np.asarray(task.base_command).copy()
    task.base_command = np.zeros(2)
    rest = MeasuredRestWindow()
    result = {'trigger_failure':str(trigger_exception) or type(trigger_exception).__name__,
              'trigger_error_type':type(trigger_exception).__name__,
              'zero_reference_set':True, 'actual_rest_0p5s_observed':False,
              'rest_definition_revision':'windowed-rest-v1', 'rest_evidence':rest.evidence(),
              'measurement_status':'unavailable', 'simulation_duration_s':0.,
              'initial_state':None, 'initial_command_v_omega':initial_command.tolist(),
              'emergency_zero_may_violate_command_slew':bool(np.any(np.abs(initial_command)>np.array([accel_max,.25])*.02+1e-9)),
              'maximum_contact_n':None, 'load_gates_held':None,
              'stop_error':None, 'records':[], 'physics_steps_attempted':0,
              'scope':'exceptional zero reference only until native rest is measured; original experiment remains rejected'}
    start = None
    try:
        # No state read or physical step is allowed if STOP/pause invalidates it.
        require_navigation_physics(world, base)
        result['initial_state'] = gt_base_state(base)
        start = float(world.current_time)
        if not math.isfinite(start):
            raise RuntimeError('Isaac simulation time is not finite during exceptional stop')
        result['measurement_status'] = 'attempted'
        load_held = True
        contact_peak = 0.
        emit('dynamic_emergency_stop','시험 오류를 보존하고 바퀴 0 명령의 실제 감속·정지를 별도로 확인합니다.')
        previous_time = start
        for stop_step in range(1000):
            require_navigation_physics(world, base)
            if cart_setup is not None:cart.step(world.current_time,release=False)
            if people is not None:people.step(world.current_time)
            result['physics_steps_attempted'] += 1
            world.step(render=False)
            require_navigation_physics(world, base)
            now = float(world.current_time)
            if not math.isfinite(now) or now <= previous_time:
                raise RuntimeError('Isaac simulation clock did not advance during exceptional stop; measurement unavailable')
            result['simulation_duration_s'] = now-start
            previous_time = now
            measured = gt_base_state(base)
            p,q = tray.get_world_pose()
            rot = Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
            relative = rot.T@(obj.get_world_pose()[0]-p)
            support_force = float(np.linalg.norm(tray_contact.get_contact_force_matrix(dt=.002)))
            peak = float(np.max(np.linalg.norm(environment_contact.get_contact_force_matrix(dt=.002),axis=-1)))
            contact_peak = max(contact_peak,peak)
            load_held = load_held and (not loaded or (abs(relative[0])<=.065 and abs(relative[1])<=.060 and abs(relative[2]-.038)<=.012 and support_force>=.02)) and rot[2,2]>=.98
            result['maximum_contact_n'] = contact_peak
            result['load_gates_held'] = bool(load_held)
            ready = rest.update(now,measured)
            if stop_step%10==0:
                result['records'].append({'time':now,'state':measured,
                    'object_in_tray':relative.tolist(),'tray_support_force_n':support_force,
                    'environment_contact_n':peak})
            if ready:
                result['actual_rest_0p5s_observed'] = True
                result['measurement_status'] = 'rest_observed'
                break
        if not result['actual_rest_0p5s_observed']:
            result['measurement_status'] = 'incomplete'
            result['stop_error'] = 'Exceptional stop did not satisfy the measured rest window within2simseconds'
    except (Exception, KeyboardInterrupt) as recovery_error:
        result['stop_error'] = str(recovery_error) or type(recovery_error).__name__
        result['measurement_status'] = 'incomplete' if result['records'] else 'unavailable'
    result['rest_evidence'] = rest.evidence()
    return result


class _CartStopGate:
    """Experiment stop/wait/resume policy driven by LiDAR returns and base twist.

    Simulated collision-path labels identify the test cart; its ground-truth
    transform is never used here. The .30m trigger buffer is a design value,
    not a physical braking-distance guarantee. QP rejection always rejects
    this experiment, even if the separate emergency stop is measured to work.
    """
    def __init__(self, cart_path, radius, margin, clear_duration_s, path_xy):
        self.path=cart_path
        self.circle=float(radius+margin)
        self.clear_duration=float(clear_duration_s)
        points=np.asarray(path_xy,dtype=float).copy()
        if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.isfinite(points).all():
            raise ValueError('Dynamic cart gate needs a finite planned Nav2 polyline')
        points=points[np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-8]]
        if len(points)<2:raise ValueError('Dynamic cart gate path has zero length')
        self.path_xy=points
        self.path_vectors=np.diff(points,axis=0)
        self.path_lengths_squared=np.sum(self.path_vectors**2,axis=1)
        self.path_xy.setflags(write=False)
        self.path_surface_clearance=self.resume_path_surface_clearance=None
        self.path_clearance_trace=[];self._last_clearance_stamp=None
        self.mode='tracking'
        self.first_detection=self.stop_request=self.rest_since=None
        self.rest_confirmed=self.release_time=self.clear_since=self.resume_time=None
        self.resumed_motion_time=None
        self.hit_records=self.qp_rejections=0
        self.minimum_detected_distance=float('inf')
        self.maximum_contact_n=0.
        self.load_gates_held=True

    def control(self, now, phase, scans, axle_xy, desired):
        distance=None
        if phase=='navigate':
            values=[];cart_points=[];cart_stamps=[]
            for scan in scans:
                stamp=float(scan['timestamp_s'])
                if not scan.get('valid',False) or not math.isfinite(stamp) or not -1e-8<=now-stamp<=float(scan.get('maximum_age_s',.15)):
                    raise RuntimeError('Dynamic cart gate requires fresh valid native LiDAR scans')
                for hit in scan['hit_paths']:
                    path=hit['collision_path']
                    if path==self.path or path.startswith(self.path+'/'):
                        point=np.asarray(hit['point_world_m'],dtype=float)
                        if point.shape!=(3,) or not np.isfinite(point).all():raise RuntimeError('Invalid native cart return')
                        cart_points.append(point[:2]);cart_stamps.append(stamp)
                        values.append(float(np.linalg.norm(point[:2]-axle_xy)))
            self.path_surface_clearance=None
            if values:
                distance=min(values)
                # Distance to every finite planned segment includes end caps.
                # Robot/cart radial separation alone cannot show that the cart
                # has vacated the future swept corridor (dynamic20 failure).
                offsets=np.asarray(cart_points)[:,None,:]-self.path_xy[:-1][None,:,:]
                fractions=np.clip(np.sum(offsets*self.path_vectors[None,:,:],axis=2)/self.path_lengths_squared[None,:],0.,1.)
                self.path_surface_clearance=float(np.min(np.linalg.norm(offsets-fractions[:,:,None]*self.path_vectors[None,:,:],axis=2)))
                stamp=min(cart_stamps)
                if self._last_clearance_stamp is None or stamp>self._last_clearance_stamp+1e-9:
                    self.path_clearance_trace.append({'scan_timestamp_s':stamp,'decision_time_s':float(now),
                        'cart_surface_return_count':len(cart_points),'minimum_return_distance_to_nav_polyline_m':self.path_surface_clearance,
                        'required_distance_m':self.circle+.15,'minimum_radial_distance_from_robot_m':distance,'mode':self.mode})
                    self._last_clearance_stamp=stamp
                self.hit_records+=1
                self.minimum_detected_distance=min(self.minimum_detected_distance,distance)
                if self.first_detection is None:self.first_detection=float(now)
            if self.mode=='tracking' and distance is not None and distance<=self.circle+.30:
                self.mode='braking';self.stop_request=float(now)
            if self.mode=='braking' and now-self.stop_request>5.:
                raise RuntimeError('Dynamic cart: physical stop was not confirmed within5simseconds')
            if self.mode=='waiting':
                # Cached frames cannot make the clearance evidence newer. A
                # positive return of the cart must persist for .2s of scan time;
                # disappearance/occlusion alone does NOT mean the cart is clear.
                stamp=min(cart_stamps) if cart_stamps else None
                if self.release_time is not None and self.path_surface_clearance is not None and self.path_surface_clearance>=self.circle+.15 and stamp>=self.release_time:
                    if self.clear_since is None:self.clear_since=stamp
                    if stamp-self.clear_since>=.2-1e-9:
                        self.mode='resumed';self.resume_time=float(now)
                        self.resume_path_surface_clearance=self.path_surface_clearance
                else:self.clear_since=None
                if self.release_time is not None and now-self.release_time>self.clear_duration+10.:
                    raise RuntimeError('Dynamic cart: no persistent LiDAR clearance after release')
            if self.mode=='resumed' and self.path_surface_clearance is not None and self.path_surface_clearance<self.circle+.10:
                raise RuntimeError('Dynamic cart returns re-entered the planned swept corridor after resume')
        return (np.zeros(2) if self.mode in ('braking','waiting') else np.asarray(desired)),distance

    def after_step(self, now, state, contact_n, load_held):
        self.maximum_contact_n=max(self.maximum_contact_n,float(contact_n))
        self.load_gates_held=self.load_gates_held and bool(load_held)
        rest=state['measured_planar_speed_m_s']<.006 and abs(state['measured_yaw_rate_rad_s'])<.015
        release=False
        if self.mode=='braking':
            if rest:
                if self.rest_since is None:self.rest_since=float(now)
                if now-self.rest_since>=.5-1e-9:
                    self.rest_confirmed=float(now);self.mode='waiting';release=True
            else:self.rest_since=None
        if self.mode=='resumed' and state['measured_planar_speed_m_s']>.02 and self.resumed_motion_time is None:
            self.resumed_motion_time=float(now)
        return release

    def summary(self):
        checks={'cart_detected_by_native_lidar':self.first_detection is not None,
                'stop_requested_from_sensor_distance':self.stop_request is not None,
                'physical_rest_continuous_0p5s':self.rest_confirmed is not None,
                'cart_release_after_measured_rest':self.release_time is not None and self.rest_confirmed is not None and self.release_time>=self.rest_confirmed,
                'clearance_confirmed_by_lidar':self.resume_time is not None,
                'physical_motion_resumed':self.resumed_motion_time is not None,
                'all_base_qps_accepted':self.qp_rejections==0,
                'no_robot_cart_or_environment_contact':self.maximum_contact_n<=.05,
                'load_support_region_and_tilt_held':bool(self.load_gates_held)}
        return {'passed':all(checks.values()),'checks':checks,'mode':self.mode,
                'first_detection_s':self.first_detection,'stop_request_s':self.stop_request,
                'rest_started_s':self.rest_since,'rest_confirmed_s':self.rest_confirmed,
                'cart_release_s':self.release_time,'resume_command_s':self.resume_time,
                'resumed_motion_s':self.resumed_motion_time,'cart_hit_record_count':self.hit_records,
                'minimum_sensed_cart_distance_m':None if not math.isfinite(self.minimum_detected_distance) else self.minimum_detected_distance,
                'maximum_environment_contact_n':self.maximum_contact_n,'qp_rejection_count':self.qp_rejections,
                'scope':'optional native LiDAR stop/wait/resume experiment; no dynamic-route avoidance or safety guarantee',
                'cart_gt_control_input':False,'simulated_collision_path_labels_used':True,
                'stop_trigger_buffer_m':.30,'resume_clearance_buffer_m':.15,
                'resume_clearance_metric':'minimum positive cart LiDAR return distance to the complete planned Nav2 polyline',
                'latest_cart_return_path_distance_m':self.path_surface_clearance,
                'resume_cart_return_path_distance_m':self.resume_path_surface_clearance,
                'resume_required_path_distance_m':self.circle+.15,
                'path_clearance_trace':list(self.path_clearance_trace),
                'clearance_observability_limit':'Visible finite surface returns only; unseen cart extent/other scan heights are not certified by this gate. The additional0.15m is a design buffer, not a full-shape or safety guarantee.',
                'stop_thresholds':{'planar_speed_m_s':.006,'yaw_rate_rad_s':.015,'continuous_rest_s':.5}}


def navigate_loaded(world,task,base,obj,tray,tray_contact,camera,writer,out,snapshot,qp_config,emit,preflight_only=False,cart=None,preview=None,config_override=None,loaded=True,skip_undock=False,people=None):
    from opti_robot.convex_control import BaseVelocityQP
    is_person=cart is not None and cart.metadata.get("motion_trigger")=="navigation_phase_clock"
    config=json.loads(json.dumps(config_override)) if config_override is not None else load_config()
    if config_override is None:config['navigation']['map_bounds_xy_m']=[-2.,-2.5,5.,3.]
    nav=config['navigation']
    state=gt_base_state(base)
    envelope=articulation_envelope(world.stage,task.robot._articulation_view,state['axle_xy_m'])
    (out/'carry_envelope.json').write_text(json.dumps(envelope,indent=2))
    object_p,object_q=obj.get_world_pose()
    corners=np.array([[x,y,z] for x in (-.02,.02) for y in (-.025,.025) for z in (-.03,.03)])
    corners=corners@Rotation.from_quat(np.asarray(object_q)[[1,2,3,0]]).as_matrix().T+object_p
    object_radius=float(np.max(np.linalg.norm(corners[:,:2]-np.asarray(state['axle_xy_m']),axis=1)))
    radius=max(envelope['radius_m'],object_radius if loaded else 0.)+.015
    envelope['object_radius_m']=object_radius
    from laser_perception import PlanarLidar, directional_coverage
    lasers=[PlanarLidar(rays=180,fov_deg=180,origin_base_m=(sign*.32,0,.4),sensor_yaw_base_rad=yaw,self_prim_prefixes=("/World/RBY1","/World/CarryTray")) for sign,yaw in ((1,0),(-1,math.pi))]
    laser_records=[]
    from isaacsim.core.prims import RigidPrim
    from pxr import UsdPhysics
    body_paths=[str(prim.GetPath()) for prim in world.stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() in task.robot._articulation_view.body_names]
    xmin,ymin,xmax,ymax=nav['map_bounds_xy_m']
    filter_paths=[r['path'] for r in snapshot['rectangles'] if r['xy_bounds_m'][0]<xmax and r['xy_bounds_m'][2]>xmin and r['xy_bounds_m'][1]<ymax and r['xy_bounds_m'][3]>ymin]
    filter_paths += [task.scene_metadata['table_top_prim'],*task.scene_metadata['table_leg_prims'], task.transport_metadata['destination_table_prim']]
    filter_paths += ["/World/TableB/leg_"+str(i) for i in range(4)]
    filter_paths=list(dict.fromkeys(filter_paths))
    if cart is not None:
        # Native filter selects rigid actors. LiDAR retains the child collider
        # path; selecting only that child could omit the cart actor's contacts.
        filter_paths.append(cart.metadata['prim_path'])
    if people is not None:filter_paths.extend(people.metadata['rigid_body_paths'])
    environment_contact=RigidPrim(body_paths,name='navigation_environment_contacts',reset_xform_properties=False,contact_filter_prim_paths_expr=[filter_paths for _ in body_paths],max_contact_count=8192)
    environment_contact.initialize()
    # Use the freshly measured circle for this actual carry posture. This is
    # a planning input, not an inherited hardware dimension or safety guarantee.
    nav['design_envelope_radius_m']=radius
    nav['planning_clearance_margin_m']=qp_config['base_limits']['margin']
    spec_path=os.environ.get('OPTI_ISAAC_WORLD_REGISTRY')
    task_spec=getattr(task,'admitted_spec',None)
    snapshot=json.loads(json.dumps(snapshot))
    if task_spec and task_spec['forbidden_zone_ids']:
        if not spec_path:raise RuntimeError('Forbidden-zone registry missing')
        registry=json.loads(Path(spec_path).read_text())
        zones={z['id']:z for z in registry['entities'] if z['kind']=='zone'}
        if isinstance(zones,list):zones={z['id']:z for z in zones}
        for name in task_spec['forbidden_zone_ids']:
            zone=zones[name]
            bounds=zone.get('bounds_design_xy_m')
            if bounds is None:raise RuntimeError('Forbidden zone lacks geometry: '+name)
            snapshot['rectangles'].append({'path':'forbidden:'+name,'xy_bounds_m':bounds})
    emit('navigation_plan','창고 충돌 지도와 운반 자세로 Nav2 경로를 계산합니다.')
    planned=request_nav2_route(out/'navigation',snapshot,measured_envelope_radius_m=radius,config=config)
    if not planned['accepted']:raise RuntimeError('Nav2 path rejected: '+str(planned.get('failure')))
    path=json.loads(Path(planned['path_file']).read_text())['xy_m']
    if people is not None and hasattr(people,'set_navigation'):
        destination=np.asarray(config['workcells']['B']['base_dock_pose_m_rad'],float)
        goal_axle=destination[:2]+.228*np.array([math.cos(destination[2]),math.sin(destination[2])])
        people.set_navigation(path,goal_axle,robot_circle_m=radius+qp_config['base_limits']['margin'],
                              timestamp_s=world.current_time)
    controller_limits=nav['limits'].copy()
    b=qp_config['base_limits']
    controller_limits.update(vmax_m_s=min(.25 if config_override is not None else .15,b['speed_max']),omega_max_rad_s=min(.3,b['yaw_rate_max']),position_tolerance_m=.009,yaw_tolerance_rad=.012)
    qp=BaseVelocityQP(speed_max=b['speed_max'],yaw_rate_max=b['yaw_rate_max'],accel_max=b['accel_max'],
                      yaw_accel_max=.25,wheel_radius=.1,wheel_separation=.53,wheel_rate_max=3.14,weights=qp_config['base_weights'])
    if preflight_only:
        scans=[laser.observe(base.get_world_pose(),world.current_time) for laser in lasers]
        (out/'preflight_scans.json').write_text(json.dumps(scans))
        if not all(scan.get('valid') for scan in scans):raise RuntimeError('Native LiDAR preflight failed: '+str([x.get('reason') for x in scans]))
        for scan,direction in zip(scans,(0.,math.pi)):
            gate=directional_coverage(scan,world.current_time,travel_direction_rad=direction)
            if not gate['accepted']:raise RuntimeError('LiDAR preflight coverage failed: '+str(gate))
        return {'passed':True,'scope':'native geometry/LiDAR/Nav2 bindings only; no loaded driving','radius_m':radius}
    reference=PolylineReference(path,limits=controller_limits)
    records=[]; previous=np.zeros(2); failed=None
    from physical_rest import MeasuredRestWindow
    rest_windows={}
    # Optional recorded snapshots, scheduled by simulation time. Reuse frames
    # after the existing render; do not add a physics step or render call.
    next_preview_time=float(world.current_time)+10.
    preview_sequence=0
    cart_records=[];cart_setup=None;dynamic_gate=None;release_cart=False;emergency_stop=None
    if people is not None:
        from warehouse_people import PeopleStopGate
        dynamic_gate=PeopleStopGate(people.metadata['prim_paths'],radius,b['margin'],path,setup_time_s=world.current_time)
        emit('dynamic_people_active',f'작업자 {len(people.metadata["prim_paths"])}명이 통로를 걷고 로봇에 양보합니다. 로봇은 LiDAR 관측으로 위험 구간을 확인합니다.')
    obstacles=planned['obstacle_rectangles']
    def pose():
        require_navigation_physics(world,base)
        r=gt_base_state(base)
        return r,np.r_[r['axle_xy_m'],r['base_yaw_rad']],np.r_[r['base_position_m'][:2],r['base_yaw_rad']]
    try:
        if cart is not None and not is_person:
            from dynamic_obstacle import plan_cart_interruption
            # Navigation starts after undocking, but scenario setup happens at
            # the current A pose. A path midpoint can be too close to that pose;
            # reject it before any external-body placement, then try later
            # middle fractions. Never move the robot to make setup fit.
            scenario=None;setup_rejections=[]
            for fraction in (.5,.6,.7,.4,.3):
                try:
                    candidate=plan_cart_interruption(path,radius_m=radius,margin_m=b['margin'],
                           cart_size_m=cart.metadata['size_m'],static_rectangles=obstacles,
                           map_bounds_xy_m=nav['map_bounds_xy_m'],progress_fraction=fraction)
                    distance=float(np.linalg.norm(np.asarray(candidate['blocking_position_m'])[:2]-state['axle_xy_m']))
                    exclusion=radius+b['margin']+candidate['cart_circumradius_m']
                    if distance<=exclusion+.025:
                        setup_rejections.append({'requested_fraction':fraction,'reason':'initial robot circle proximity',
                                                  'distance_m':distance,'required_distance_m':exclusion+.025})
                        continue
                    scenario=candidate;break
                except ValueError as error:
                    setup_rejections.append({'requested_fraction':fraction,'reason':str(error)})
            if scenario is None:raise RuntimeError('No safe external-cart scenario setup: '+str(setup_rejections))
            scenario['initial_placement_rejected_candidates']=setup_rejections
            cart_setup=cart.setup(scenario,world.current_time,robot_axle_xy_m=state['axle_xy_m'])
            dynamic_gate=_CartStopGate(cart.metadata['prim_path'],radius,b['margin'],scenario['clear_duration_s'],np.asarray(path,dtype=float).copy())
            emit('dynamic_obstacle_setup','별도 카트 시험을 경로에 배치했습니다. LiDAR 정지·대기·재개를 확인합니다.')
        phases=([('navigate',400.),('dock',50.)] if skip_undock else [('undock',50.),('navigate',400.),('dock',50.)])
        for name,timeout in phases:
            emit(name,{'undock':'선반에서 후진해 주행 통로로 나옵니다.','navigate':'Nav2 경로를 QP 바퀴 명령으로 따라갑니다.','dock':'목표 작업 구역에 저속으로 접근합니다.'}[name])
            if is_person and name=='navigate':
                from moving_person import plan_person_crossing, PersonStopGate
                scenario=plan_person_crossing(path,radius_m=radius,margin_m=b['margin'],
                    static_rectangles=obstacles,map_bounds_xy_m=nav['map_bounds_xy_m'],start_delay_s=10.)
                cart_setup=cart.setup(scenario,world.current_time,robot_axle_xy_m=gt_base_state(base)['axle_xy_m'])
                dynamic_gate=PersonStopGate(cart.metadata['prim_path'],radius,b['margin'],scenario['clear_duration_s'],np.asarray(path,dtype=float).copy(),setup_time_s=world.current_time)
                emit('dynamic_person_setup','시간표에 따라 통로를 건너는 작업자를 배치했습니다. 실제 LiDAR 관측으로 정지·재개합니다.')
            rest_monitor=MeasuredRestWindow()
            for step in range(round(timeout/.002)):
                require_navigation_physics(world,base)
                r,axle,bp=pose()
                if step%10==0:
                    if name=='navigate':desired,diagnostic=reference.reference(axle)
                    else:
                        goal=nav['A_undocked_base_pose_m_rad'] if name=='undock' else config['workcells']['B']['base_dock_pose_m_rad']
                        desired,diagnostic=straight_dock_reference(bp,goal,backwards=name=='undock',limits={**controller_limits,'vmax_m_s':min(.10,b['speed_max'])})
                    scans=[laser.observe(base.get_world_pose(),world.current_time) for laser in lasers]
                    if not all(scan.get('valid') for scan in scans):raise RuntimeError('LiDAR scan invalid: '+str([scan.get('reason') for scan in scans]))
                    chosen=1 if desired[0]<0 else 0
                    coverage=directional_coverage(scans[chosen],world.current_time,travel_direction_rad=math.pi if chosen else 0.)
                    if not coverage['accepted']:raise RuntimeError('Travel-direction LiDAR coverage rejected: '+str(coverage['reason']))
                    if step%50==0:laser_records.append({'time':float(world.current_time),'phase':name,'scans':scans,'coverage':coverage})
                    cart_distance=None
                    if dynamic_gate is not None:
                        old_mode=dynamic_gate.mode
                        desired,cart_distance=dynamic_gate.control(world.current_time,name,scans,axle[:2],desired)
                        if old_mode!=dynamic_gate.mode:
                            emit('dynamic_'+dynamic_gate.mode,{'braking':'LiDAR 검출로 QP 감속·정지를 요청합니다.',
                                  'resumed':'LiDAR에서 이동 장애물이 비켜난 것을 확인해 경로 주행을 재개합니다.'}.get(dynamic_gate.mode,dynamic_gate.mode))
                    points=[]
                    if name=='navigate':
                        for item in obstacles:
                            distance,gradient=point_rectangle_distance(axle[:2],item['xy_bounds_m'])
                            if distance<0:raise RuntimeError('Axle entered a static collision rectangle')
                            if distance<3.0:points.append(-gradient*distance)
                    for scan in scans:
                        for hit in scan['hit_paths']:
                            path=hit['collision_path']
                            # Known workcell docking is separately guarded by
                            # real robot/table contacts. Unknown sensed geometry
                            # remains a local barrier even in close docking.
                            if name!='navigate' and (path.startswith('/World/StationaryGrasp/table') or path.startswith('/World/TableB/') or path.startswith(task.scene_metadata.get('rack_prim','/nonexistent')+'/')):continue
                            relative=np.asarray(hit['point_world_m'])[:2]-axle[:2]
                            if np.linalg.norm(relative)<3.:points.append(relative)
                    result=qp.solve(desired,previous,.02,lidar_points_axle=np.asarray(points).reshape(-1,2) if points else None,
                                    heading=axle[2],radius=radius,margin=b['margin'])
                    if not result['accepted']:
                        # Preserve rejection; never turn emergency-zero recovery
                        # into an accepted experiment. The except block measures
                        # its physical stopping separately before re-raising.
                        if dynamic_gate is not None:
                            dynamic_gate.qp_rejections+=1
                            records.append({'time':float(world.current_time),'phase':name,'state':r,'tracking':diagnostic,
                                            'qp':result,'dynamic_mode':dynamic_gate.mode,'sensed_cart_distance_m':cart_distance})
                        raise RuntimeError('Base QP rejected: '+str(result['reason']))
                    previous=np.asarray(result['solution']);task.base_command=previous.copy()
                    record={'time':float(world.current_time),'phase':name,'state':r,'tracking':diagnostic,'qp':result}
                    if dynamic_gate is not None:record.update(dynamic_mode=dynamic_gate.mode,sensed_cart_distance_m=cart_distance)
                    records.append(record)
                cart_target=None
                if cart is not None and cart_setup is not None:
                    cart_target=cart.step(world.current_time,release=release_cart)
                    release_cart=False
                    if cart_target['release_started']:
                        dynamic_gate.release_time=float(cart_target['release_timestamp_s'])
                        emit('dynamic_person_walk' if is_person else 'dynamic_cart_release','작업자가 미리 정한 시간표에 따라 걷기 시작합니다.' if is_person else '실제 정지 0.5초 유지 후 외부 카트를 천천히 비켜나게 합니다.')
                if people is not None:cart_target=people.step(world.current_time)
                world.step(render=False)
                require_navigation_physics(world,base)
                p,q=tray.get_world_pose();rot=Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
                relative=rot.T@(obj.get_world_pose()[0]-p)
                force=float(np.linalg.norm(tray_contact.get_contact_force_matrix(dt=.002)))
                contact_max=float(np.max(np.linalg.norm(environment_contact.get_contact_force_matrix(dt=.002),axis=-1)))
                if dynamic_gate is not None:
                    dynamic_gate.maximum_contact_n=max(dynamic_gate.maximum_contact_n,contact_max)
                    dynamic_gate.load_gates_held=dynamic_gate.load_gates_held and bool(
                        (not loaded or (abs(relative[0])<=.065 and abs(relative[1])<=.060 and abs(relative[2]-.038)<=.012 and force>=.02)) and rot[2,2]>=.98)
                if contact_max>.05:
                    matrix=np.linalg.norm(environment_contact.get_contact_force_matrix(dt=.002),axis=-1)
                    ij=np.unravel_index(np.argmax(matrix),matrix.shape)
                    (out/'collision_event.json').write_text(json.dumps({'time':float(world.current_time),'phase':name,'force_n':contact_max,'sensor_body':environment_contact.prim_paths[ij[0]],'filter_path':filter_paths[ij[1]],'state':gt_base_state(base)},indent=2))
                    world.render()
                    require_navigation_physics(world,base)
                    from PIL import Image
                    Image.fromarray(np.asarray(camera.get_rgba())[...,:3].astype(np.uint8)).save(out/'collision.png')
                    raise RuntimeError('Robot/environment contact detected during '+name+': '+str(contact_max))
                if step%10==0:
                    records[-1].update(object=np.asarray(obj.get_world_pose()[0]).tolist(),object_velocity=np.asarray(obj.get_linear_velocity()).tolist(),object_angular_velocity=np.asarray(obj.get_angular_velocity()).tolist(),object_quaternion_wxyz=np.asarray(obj.get_world_pose()[1]).tolist(),object_in_tray=relative.tolist(),tray_position_m=np.asarray(p).tolist(),tray_quaternion_wxyz=np.asarray(q).tolist(),tray_support_force_n=force,base_up_z=float(rot[2,2]),robot_environment_contact_max_n=contact_max)
                if step%250==0:
                    live=articulation_envelope(world.stage,task.robot._articulation_view,gt_base_state(base)['axle_xy_m'])
                    if live['radius_m']>radius:raise RuntimeError('Physical carry envelope grew beyond admitted radius')
                if loaded and (abs(relative[0])>.065 or abs(relative[1])>.060 or abs(relative[2]-.038)>.012):
                    raise RuntimeError('Object left admitted tray support region during '+name)
                if loaded and force<.02:raise RuntimeError('Tray support contact lost during '+name)
                if rot[2,2]<.98:raise RuntimeError('Loaded base/tray tilt bound exceeded')
                if dynamic_gate is not None:
                    old_mode=dynamic_gate.mode
                    release_cart=dynamic_gate.after_step(world.current_time,gt_base_state(base),contact_max,True)
                    if old_mode!=dynamic_gate.mode and dynamic_gate.mode=='waiting':
                        emit('dynamic_waiting','실제 정지를 확인했습니다. LiDAR에서 통로가 비기를 기다립니다.')
                    if step%10==0:
                        cart_records.append({'time':float(world.current_time),'phase':name,'target':cart_target,
                            'actual_external_obstacle_evaluation':people.measure(world.current_time) if people is not None else cart.measure(world.current_time),'policy_mode':dynamic_gate.mode})
                if step%500==0:
                    with (out/'navigation_live.jsonl').open('a') as live_log:
                        live_log.write(json.dumps({'time':float(world.current_time),'phase':name,'state':gt_base_state(base),
                            'tracking':diagnostic,'command':np.asarray(task.base_command).tolist()})+'\n')
                if step%5000==0 and step>0:
                    support_detail='물체 트레이 지지 확인' if loaded else '물체를 집기 위한 선반 접근'
                    emit(name, f"{name} 진행 중 · 목표 위치와 거리 {diagnostic['goal_distance_m']:.2f} m · {support_detail}")
                if step%20==19:
                    # Follow the physical robot; this changes only the viewing camera.
                    from pxr import Gf
                    actual=base.get_world_pose()[0]
                    eye=Gf.Vec3d(*(actual+(np.array([1.8,2.4,1.8]) if config_override is not None else np.array([2.3,-2.6,1.8]))))
                    focus=Gf.Vec3d(*(actual+np.array([.25,-.2,.85])))
                    qt=Gf.Matrix4d().SetLookAt(eye,focus,Gf.Vec3d(0,0,1)).GetInverse().ExtractRotationQuat()
                    camera.set_world_pose(np.asarray(eye),np.array([qt.GetReal(),*qt.GetImaginary()]),camera_axes='usd')
                    world.render()
                    require_navigation_physics(world,base)
                    if writer is not None:writer.append_data(np.asarray(camera.get_rgba())[...,:3].astype(np.uint8))
                    if preview is not None and float(world.current_time)+1e-9>=next_preview_time:
                        rgba=camera.get_rgba()
                        if rgba is not None and np.asarray(rgba).size:
                            from PIL import Image
                            preview_sequence+=1
                            preview_path=out/f"navigation_{preview_sequence:04d}_{name}_{round(float(world.current_time)*1000):09d}ms.png"
                            Image.fromarray(np.asarray(rgba)[...,:3].astype(np.uint8)).save(preview_path)
                            preview(preview_path)
                            next_preview_time+=10.
                r,axle,bp=pose()
                goal_distance=diagnostic['goal_distance_m']
                # Physical rest, not merely a zero reference or a planner status.
                eligible=goal_distance<.012 and abs(diagnostic.get('goal_yaw_error_rad',1.))<.015
                rest_ready=rest_monitor.update(float(world.current_time),r,eligible=eligible)
                if rest_ready and step%10==9:
                    rest_windows[name]=rest_monitor.evidence()
                    break
            else:raise RuntimeError(name+' timeout before measured arrival/rest')
        result={'passed':True,'loaded':loaded,'skipped_undock':skip_undock,'final_state':r,'envelope_radius_m':radius,'nav2_path':planned['path_file'],'qp_count':len(records),
                'rest_definition_revision':'windowed-rest-v1','rest_windows':rest_windows}
        if dynamic_gate is not None:
            dynamic=dynamic_gate.summary()
            result.update(passed=dynamic['passed'],dynamic_obstacle=dynamic,dynamic_checks=dynamic['checks'])
        return result
    except (Exception, KeyboardInterrupt) as exc:
        failed=str(exc) or type(exc).__name__
        if cart is not None or people is not None:
            # An exceptional zero command bypasses normal command slew if needed.
            # Its effect is measured; it is not a QP solution or an accepted run.
            emergency_stop=observe_exceptional_stop(world,task,base,obj,tray,tray_contact,
                environment_contact,trigger_exception=exc,loaded=loaded,
                accel_max=b['accel_max'],emit=emit,people=people,cart=cart,cart_setup=cart_setup)
            if dynamic_gate is not None:
                if emergency_stop['maximum_contact_n'] is not None:
                    dynamic_gate.maximum_contact_n=max(dynamic_gate.maximum_contact_n,emergency_stop['maximum_contact_n'])
                if emergency_stop['load_gates_held'] is not None:
                    dynamic_gate.load_gates_held=dynamic_gate.load_gates_held and emergency_stop['load_gates_held']
        else:
            task.base_command=np.zeros(2)
        raise
    finally:
        task.base_command=np.zeros(2)
        if people is not None and hasattr(people,'clear_navigation'):
            people.clear_navigation()
        (out/'navigation_trace.json').write_text(json.dumps({'failure':failed,'records':records,'rest_definition_revision':'windowed-rest-v1','rest_windows':rest_windows,'emergency_stop':emergency_stop}))
        (out/'lidar_scans.json').write_text(json.dumps(laser_records))
        if people is not None:
            dynamic=dynamic_gate.summary() if dynamic_gate is not None else {'passed':False,'checks':{},'failure':'people gate setup incomplete'}
            if failed:dynamic.update(passed=False,failure=failed)
            (out/'people_trace.json').write_text(json.dumps({'enabled':True,'people_metadata':people.metadata,'summary':dynamic,'emergency_stop':emergency_stop,'records':cart_records}))
        if cart is not None:
            dynamic=dynamic_gate.summary() if dynamic_gate is not None else {'passed':False,'checks':{},'failure':'scenario setup incomplete'}
            if failed:dynamic.update(passed=False,failure=failed)
            (out/'dynamic_trace.json').write_text(json.dumps({'enabled':True,'cart_metadata':cart.metadata,'scenario_setup':cart_setup,
                     'summary':dynamic,'emergency_stop':emergency_stop,'records':cart_records},indent=2))
