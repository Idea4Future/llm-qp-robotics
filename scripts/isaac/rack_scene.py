"""Project-authored steel warehouse racks, real colliders, and initial catalog.

Dimensions, carton mass and friction are simulation design assumptions. Stored
cartons are fixed obstacles; only the three small color containers are dynamic.
No function changes a robot/object pose after initial scene construction.
"""
from __future__ import annotations
import copy

HOME = (-6., 3.5, 0.)
RACKS = {'A': (-5., -2.), 'B': (0., -2.), 'C': (5., -2.)}
OBJECTS = {'A': 'redbin01', 'B': 'bluebin01', 'C': 'greenbin01'}
COLORS = {'A': (.85,.16,.10), 'B': (.08,.28,.85), 'C': (.10,.65,.25)}
OUT_DOCK = (2., 4.5, 0.)


def mission_config(selected='A'):
    from navigation_runtime import load_config
    if selected not in RACKS: raise ValueError('Unknown rack')
    c=load_config(); x,y=RACKS[selected]
    c['layout']='rack-v1'
    c['layout_revision']='wide-2026-10-08'
    c['navigation'].update(map_bounds_xy_m=[-8.,-6.,8.8,7.5],
        A_undocked_base_pose_m_rad=[x-.8,y,0.], B_predock_base_pose_m_rad=[OUT_DOCK[0]-.8,OUT_DOCK[1],0.])
    c['navigation']['limits']['vmax_m_s']=.25
    c['workcells']={}
    for label,(rx,ry) in RACKS.items():
        c['workcells']['rack_'+label]={'table_center_m':[rx+1.,ry-.675,1.3],
            'table_size_m':[1.4,.85,2.6], 'base_dock_pose_m_rad':[rx,ry,0.]}
    c['workcells']['B']={'table_center_m':[OUT_DOCK[0]+.6,OUT_DOCK[1]-.4,.775], 'table_size_m':[.54,.3,.05],
                         'base_dock_pose_m_rad':list(OUT_DOCK)}
    c['selected_rack']=selected
    return c


def approach_config(selected, actual_base_pose):
    c=mission_config(selected); x,y=RACKS[selected]
    c['navigation']['A_undocked_base_pose_m_rad']=list(actual_base_pose)
    c['navigation']['B_predock_base_pose_m_rad']=[x-.8,y,0.]
    c['workcells']['B']['base_dock_pose_m_rad']=[x,y,0.]
    return c


def robot_waiting_zones(*, robot_radius_m=.8, clearance_margin_m=.15):
    """Config-derived fixed waiting circles; design bounds, not measurements.

    The caller must separately reject native robot envelopes exceeding the
    supplied radius, including manipulation poses. No actor pose is changed.
    """
    import math
    from dynamic_obstacle import _positive, _vector
    if isinstance(robot_radius_m,bool) or isinstance(clearance_margin_m,bool):
        raise ValueError('Waiting circle radius/margin must be numeric')
    radius=_positive(robot_radius_m,'waiting robot radius')
    margin=_positive(clearance_margin_m,'waiting circle margin')
    if margin<.15-1e-9:raise ValueError('Do not relax waiting circle0.15m margin')
    config=mission_config('A')
    offset=_vector(config['robot']['axle_offset_base_m'],3,'axle offset')
    result=[]
    def add(label,base_pose):
        pose=_vector(base_pose,3,'waiting base pose');yaw=float(pose[2])
        axle=[float(pose[0]+math.cos(yaw)*offset[0]-math.sin(yaw)*offset[1]),
              float(pose[1]+math.sin(yaw)*offset[0]+math.cos(yaw)*offset[1])]
        result.append({'id':label,'axle_xy_m':axle,'robot_radius_m':radius,
                       'clearance_margin_m':margin,'base_pose_m_rad':pose.tolist()})
    add('station_home_dock',HOME)
    for label in RACKS:
        rack_config=mission_config(label)
        add('station_rack_'+label+'_dock',rack_config['workcells']['rack_'+label]['base_dock_pose_m_rad'])
        add('station_rack_'+label+'_predock',rack_config['navigation']['A_undocked_base_pose_m_rad'])
    add('station_out_dock',config['workcells']['B']['base_dock_pose_m_rad'])
    add('station_out_predock',config['navigation']['B_predock_base_pose_m_rad'])
    return result


def add_rack_scene(stage, selected='A'):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    from grasp_scene import _box, _material
    root='/World/ProjectRacks'; UsdGeom.Xform.Define(stage,root)
    material=_material(stage,root+'/material',.6)
    records={}
    for index,(label,(x,y)) in enumerate(RACKS.items()):
        prefix=root+'/Rack_'+label; UsdGeom.Xform.Define(stage,prefix)
        colliders=[]
        def box(name,p,size,color):
            path=prefix+'/'+name
            prim=_box(stage,path,(x+p[0],y+p[1],p[2]),size,material,color=color)
            colliders.append(path); return prim
        # Rear-supported open-front steel rack and four physical shelf levels. The open
        # working bay has 0.8m headroom for wrist/camera above the small object.
        for i,px in enumerate((.325,1.675)):
            for j,py in enumerate((-1.075,-.775)):
                box(f'post_{i}_{j}',(px,py,1.30),(.05,.05,2.60),(.07,.18,.36))
        for k,top in enumerate((.14,.80,1.65,2.35)):
            box(f'shelf_{k}',(1.,-.735 if k==0 else -.675,top-.025),(1.40,.73 if k==0 else .85,.05),(.34,.38,.40))
            for j,py in enumerate((-1.07,-.40 if k==0 else -.28)):
                box(f'beam_{k}_{j}',(1.,py,top-.055),(1.40,.045,.06),(.96,.35,.04))
        # Back bracing uses small horizontal rails, all with collisions.
        for k,z in enumerate((.5,1.2,2.0)):
            box(f'back_rail_{k}',(1.,-1.075,z),(1.30,.025,.025),(.10,.22,.40))
        cartons=[]
        for k,(px,py,pz,sx,sy,sz) in enumerate([
            (.68,-.78,.14+.22,.40,.48,.44),(1.30,-.78,.14+.18,.45,.48,.36),
            (1.22,-.85,.80+.18,.40,.35,.36),
            (.68,-.77,1.65+.22,.42,.48,.44),(1.26,-.77,1.65+.18,.48,.48,.36),
            (.92,-.77,2.35+.18,.55,.48,.36)]):
            box(f'stored_carton_{k}',(px,py,pz),(sx,sy,sz),(.52+.03*(k%3),.35,.17))
            # Tape is render-only, on the carton top. It adds no hidden obstacle.
            tape=UsdGeom.Cube.Define(stage,prefix+f'/tape_{k}');tape.CreateSizeAttr(1.)
            tape.AddTranslateOp().Set(Gf.Vec3d(x+px,y+py,pz+sz/2+.0002))
            tape.AddScaleOp().Set(Gf.Vec3f(.035,sy,.0002));tape.CreateDisplayColorAttr([Gf.Vec3f(.73,.58,.34)])
            cartons.append({'center_m':[x+px,y+py,pz],'size_m':[sx,sy,sz]})
        # Render-only letter plate on the front beam; no fictitious collider.
        patterns={'A':['01110','10001','10001','11111','10001','10001','10001'],
                  'B':['11110','10001','10001','11110','10001','10001','11110'],
                  'C':['01111','10000','10000','10000','10000','10000','01111']}
        def sign_pixel(name,px,pz,sx,sz,color,py=-.238):
            face=UsdGeom.Cube.Define(stage,prefix+'/'+name);face.CreateSizeAttr(1.)
            face.AddTranslateOp().Set(Gf.Vec3d(x+px,y+py,pz));face.AddScaleOp().Set(Gf.Vec3f(sx,.002,sz))
            face.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        sign_pixel('zone_plate',.90,1.53,.26,.24,(.95,.95,.91),py=-.244)
        for row,line in enumerate(patterns[label]):
            for col,char in enumerate(line):
                if char=='1':sign_pixel(f'label_{row}_{col}',.90+(col-2)*.026,1.53+(3-row)*.026,.024,.024,COLORS[label])
        object_path=prefix+'/object'
        obj=_box(stage,object_path,(x+.4,y-.3,.831),(.04,.05,.06),material,color=COLORS[label])
        UsdPhysics.RigidBodyAPI.Apply(obj).CreateKinematicEnabledAttr(False)
        PhysxSchema.PhysxRigidBodyAPI.Apply(obj).CreateDisableGravityAttr(False)
        mass=UsdPhysics.MassAPI.Apply(obj);mass.CreateMassAttr(.10);mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.1/12*(.05**2+.06**2),.1/12*(.04**2+.06**2),.1/12*(.04**2+.05**2)))
        records[label]={'rack_prim':prefix,'object_prim':object_path,'object_id':OBJECTS[label], 'marker_id':17+index,
            'table_top_prim':prefix+'/shelf_1','table_leg_prims':[], 'all_static_colliders':colliders,
            'table_top_world_z_m':.8,'object_initial_world_center_m':[x+.4,y-.3,.831],
            'object_full_size_m':[.04,.05,.06],'object_mass_kg':.1,'base_dock_pose_m_rad':[x,y,0.],
            'object_dynamic':True,'object_attached':False,'object_gravity_enabled':True,'stored_cartons':cartons}
    # Colored floor outlines are render-only paint; racks and cartons above are
    # actual obstacles. User can distinguish the three zones from any viewport.
    for label,(x,y) in RACKS.items():
        for k,(cx,cy,sx,sy) in enumerate(((x+1.,y-1.16,1.5,.025),(x+1.,y-.19,1.5,.025))):
            paint=UsdGeom.Cube.Define(stage,root+f'/zone_{label}_{k}');paint.CreateSizeAttr(1.)
            paint.AddTranslateOp().Set(Gf.Vec3d(cx,cy,.001));paint.AddScaleOp().Set(Gf.Vec3f(sx,sy,.001))
            paint.CreateDisplayColorAttr([Gf.Vec3f(*COLORS[label])])
    for tag,(px,py),color in (('Home',HOME[:2],(.12,.7,.6)),('Out',OUT_DOCK[:2],(.93,.63,.1))):
        for k,(ox,oy,sx,sy) in enumerate(((0.,-.55,1.1,.025),(0.,.55,1.1,.025),(-.55,0.,.025,1.1),(.55,0.,.025,1.1))):
            paint=UsdGeom.Cube.Define(stage,root+f'/{tag}_outline_{k}');paint.CreateSizeAttr(1.)
            paint.AddTranslateOp().Set(Gf.Vec3d(px+ox,py+oy,.001));paint.AddScaleOp().Set(Gf.Vec3f(sx,sy,.001))
            paint.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    result=copy.deepcopy(records[selected]); result.update(layout='rack-v1',layout_revision='wide-2026-10-08',selected_rack=selected,all_racks=records,
        status='authored design; outcome requires fresh runtime validation',
        limitations=['Stored large cartons are fixed scenery/obstacles, not gripped objects.',
                     'Each job resets all three small containers; no inventory persistence between jobs.',
                     'Project-authored rack geometry, not a measured commercial rack.'])
    return result
