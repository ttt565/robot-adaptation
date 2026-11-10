"""Synchronized contact-view RGB and real MuJoCo motion; no learned control policy."""
import copy
import hashlib
import io
import json
import math
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import engine
from appearance import make_texture

ROOT=Path(__file__).resolve().parent
COUNTS={'train':48,'val':12,'test':12,'ood':12}


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def episode_plan(split,index):
    base={'train':1000,'val':100000,'test':200000,'ood':300000}[split]+index
    rng=np.random.default_rng(base)
    if split=='train':
        condition='correlated' if index<24 else ['normal','appearance_only','hidden_friction','motor_fault'][(index-24)//6]
    elif split=='ood':
        condition=['reversed','severe_motor','severe_hidden'][index//4]
    else:
        condition='correlated' if index<4 else ['normal','appearance_only','hidden_friction','motor_fault'][(index-4)//2]
    high=float(rng.uniform(.55,.9));low=float(rng.uniform(.025,.06));medium=float(rng.uniform(.07,.13))
    before_kind,after_kind,before_mu,after_mu='tile_grid','tile_grid',high,high
    if condition=='correlated':
        if index%3==2:
            after_kind,after_mu='parallel_stripes',medium
        else:
            after_kind,after_mu='speckled',low
    elif condition=='appearance_only':
        after_kind='speckled'
    elif condition in ['hidden_friction','severe_hidden']:
        after_mu=.015 if condition=='severe_hidden' else low
    elif condition=='reversed':
        before_mu,after_kind,after_mu=low,'speckled',high
    return {'id':f'{split}_{index:03d}','split':split,'condition':condition,
            'texture_seed':base*2,'input_seed':base+900000,'noise_seed':base+1900000,
            'before_kind':before_kind,'after_kind':after_kind,'before_mu':before_mu,'after_mu':after_mu,
            'boundary':float(rng.uniform(1.4,2.8)),
            'motor_event':float(rng.uniform(3.,7.)),
            'motor_efficiency':.25 if condition=='severe_motor' else float(rng.uniform(.45,.75)),
            'light':float(rng.uniform(.65,1.05)),'ambient':float(rng.uniform(.15,.35))}


class LearningPlant(engine.ContactPlant):
    def __init__(self,config,episode):
        xml=ET.fromstring((ROOT/'robot.xml').read_text())
        world,asset=xml.find('worldbody'),xml.find('asset')
        world.find("geom[@name='floor']").set('pos','0 0 -0.2')
        for light in world.findall('light'):
            light.set('diffuse',' '.join([str(episode['light'])]*3))
            light.set('ambient',' '.join([str(episode['ambient'])]*3))
        assets={}
        for index,kind in enumerate([episode['before_kind'],episode['after_kind']]):
            buffer=io.BytesIO()
            Image.fromarray(make_texture(kind,episode['texture_seed']+index)).save(buffer,format='PNG')
            assets[f'terrain{index}.png']=buffer.getvalue()
            ET.SubElement(asset,'texture',name=f'terrain{index}',type='2d',file=f'terrain{index}.png')
            ET.SubElement(asset,'material',name=f'terrain{index}_mat',texture=f'terrain{index}',
                          texuniform='true',texrepeat='2 2',reflectance='0')
        boundary=episode['boundary']
        contacts=ET.SubElement(xml,'contact')
        for index,(left,right,mu) in enumerate([(-8.,boundary,episode['before_mu']),
                                               (boundary,32.,episode['after_mu'])]):
            name=f'terrain_{index}'
            ET.SubElement(world,'geom',name=name,type='box',pos=f'{(left+right)/2} 0 -0.05',
                          size=f'{(right-left)/2} 20 0.05',material=f'terrain{index}_mat',group='0')
            for tire in ['left_tire','right_tire']:
                ET.SubElement(contacts,'pair',geom1=tire,geom2=name,condim='3',
                              solref='0.008 1',solimp='0.95 0.99 0.001',
                              friction=f'{mu} {mu} 0.001 0.0001 0.0001')
        chassis=world.find("body[@name='chassis']")
        ET.SubElement(chassis,'camera',name='contact_camera',pos='0 0 0.5',
                      xyaxes='1 0 0 0 1 0',fovy='60')
        for geom in chassis.iter('geom'):
            geom.set('group','1')
        settings=copy.deepcopy(config)
        settings['change_time']=episode['motor_event']
        settings['disturbance']['left_efficiency']=episode['motor_efficiency']
        settings['visualize_hidden_disturbance']=False
        scenario='left_motor_drop' if episode['condition'] in ['motor_fault','severe_motor'] else 'none'
        super().__init__(settings,scenario,ET.tostring(xml,encoding='unicode'),assets)
        self.terrains=[self.model.geom(f'terrain_{index}').id for index in [0,1]]
        self.camera_id=self.model.camera('contact_camera').id

    def motion(self):
        body=self.true_state()
        return np.r_[body[3:5],self.data.qvel[self.wheel_dofs]]

    def aligned_contacts(self):
        camera_position=self.data.cam_xpos[self.camera_id].copy()
        rotation=self.data.cam_xmat[self.camera_id].reshape(3,3)
        focal=64/(2*math.tan(math.radians(60)/2))
        pixels,errors,visible,world_points,present=[],[],[],[],[]
        for tire in self.tires:
            points=[contact.pos.copy() for contact in self.data.contact
                    if tire in [contact.geom1,contact.geom2] and
                    any(terrain in [contact.geom1,contact.geom2] for terrain in self.terrains)]
            point=np.mean(points,axis=0) if points else self.data.geom_xpos[tire].copy()
            point[2]=0.
            relative=rotation.T@(point-camera_position)
            depth=-relative[2]
            pixel=np.array([31.5+focal*relative[0]/depth,31.5-focal*relative[1]/depth])
            ray=rotation@np.array([(pixel[0]-31.5)/focal,-(pixel[1]-31.5)/focal,-1.])
            back=camera_position+(-camera_position[2]/ray[2])*ray
            pixels.append(pixel);errors.append(float(np.linalg.norm(back[:2]-point[:2])))
            visible.append(bool(depth>0 and np.all((pixel>=2)&(pixel<=61))))
            present.append(bool(points))
            world_points.append(point)
        return np.asarray(pixels),np.asarray(errors),np.asarray(visible),np.asarray(world_points),np.asarray(present)

    def terrain_audit(self):
        after=False;mus=[]
        for contact in self.data.contact:
            pair=[contact.geom1,contact.geom2]
            if any(tire in pair for tire in self.tires) and any(terrain in pair for terrain in self.terrains):
                mus.append(contact.friction[0]);after|=self.terrains[1] in pair
        return bool(after),float(np.mean(mus)) if mus else np.nan


def planned_commands(config,seed):
    rng=np.random.default_rng(seed)
    times=np.arange(round(config['duration']/config['dt']))*config['dt']
    phase=rng.uniform(-np.pi,np.pi,3)
    ramp=(1-np.cos(np.minimum(times/1.5,1)*np.pi))/2
    total=ramp*(4.8+rng.uniform(.5,1.2)*np.sin(rng.uniform(.4,.9)*times+phase[0])+
                .7*np.sin(1.7*times+phase[1]))
    difference=rng.uniform(.2,.65)*np.sin(1.1*times+phase[2])+.15*np.sin(2.3*times)
    for start in rng.uniform([3.,7.],[5.,10.]):
        total[(times>=start)&(times<start+.4)]-=3.8
    return np.clip(np.c_[(total-difference)/2,(total+difference)/2],-6,6)


def collect_episode(arguments):
    config,episode,data_directory=arguments
    directory=Path(data_directory)/'episodes'/episode['id']
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'rgb').mkdir(exist_ok=True)
    plant=LearningPlant(config,episode)
    renderer=mujoco.Renderer(plant.model,height=64,width=64)
    options=mujoco.MjvOption();options.geomgroup[1]=0;options.geomgroup[5]=0
    commands=planned_commands(config,episode['input_seed'])
    count=len(commands)
    truth=np.zeros((count+1,4));rgb=np.zeros((count+1,64,64,3),dtype=np.uint8)
    pixels=np.zeros((count+1,2,2));alignment=np.zeros((count+1,2));visible=np.zeros((count+1,2),dtype=bool)
    contact_present=np.zeros((count+1,2),dtype=bool)
    positions=np.zeros((count+1,3));contact_mu=np.full(count+1,np.nan)
    first_contact=None
    try:
        for step in range(count+1):
            truth[step]=plant.motion()
            positions[step]=plant.data.xpos[plant.body_id]
            pixels[step],alignment[step],visible[step],_,contact_present[step]=plant.aligned_contacts()
            after,contact_mu[step]=plant.terrain_audit()
            if first_contact is None and after:
                first_contact=step*config['dt']
            renderer.update_scene(plant.data,camera='contact_camera',scene_option=options)
            rgb[step]=renderer.render()
            Image.fromarray(rgb[step]).save(directory/'rgb'/f'{step:04d}.png')
            if step<count:
                plant.step(commands[step])
    finally:
        renderer.close()
    rng=np.random.default_rng(episode['noise_seed'])
    observed=truth+rng.normal(size=truth.shape)*np.array([.01,.01,.05,.05])
    condition=episode['condition']
    event=episode['motor_event'] if condition in ['motor_fault','severe_motor'] else first_contact if episode['before_mu']!=episode['after_mu'] else None
    valid=bool(np.isfinite(truth).all() and np.isfinite(observed).all() and
               positions[:,2].min()>.045 and positions[:,2].max()<.3 and
               (np.abs(positions[:,1])<19).all() and (positions[:,0]>-7).all() and
               (positions[:,0]<31).all() and visible.mean()>.9 and
               (event is not None or condition in ['normal','appearance_only']))
    file=directory/'episode.npz'
    np.savez_compressed(file,times=np.arange(count+1)*config['dt'],truth=truth,observed=observed,
                        commands=commands,rgb=rgb,wheel_pixels=pixels,alignment_error_m=alignment,
                        contact_visible=visible,contact_present=contact_present,body_position=positions,contact_mu=contact_mu)
    metadata={**episode,'file':f'episodes/{episode["id"]}/episode.npz','event_time':event,
              'first_post_boundary_contact':first_contact,'valid':valid,'frames':count+1,
              'alignment_error_max_m':float(alignment.max()),'contact_visibility_fraction':float(visible.mean()),
              'instantaneous_active_contact_fraction':float(contact_present.mean()),
              'min_body_height':float(positions[:,2].min()),'max_body_height':float(positions[:,2].max()),
              'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
    write_json(directory/'metadata.json',metadata)
    return metadata


def collect_dataset(data_root,workers=2,limit=None):
    data_root=Path(data_root);data_root.mkdir(parents=True,exist_ok=True)
    if (data_root/'manifest.json').exists():
        raise FileExistsError('Dataset already complete. Choose another --data-dir or use train/evaluate.')
    config=json.loads((ROOT/'config.json').read_text())
    config.update({'dt':.1,'duration':12.,'physics_dt':.002,'visualize_hidden_disturbance':False})
    calibration_episode=episode_plan('train',0)
    calibration_episode.update(condition='normal',texture_seed=9000001,before_kind='tile_grid',after_kind='tile_grid',before_mu=.8,after_mu=.8,boundary=20.)
    calibration=engine.calibrate(config,plant_factory=lambda settings:LearningPlant(settings,calibration_episode))
    config['physics']=calibration['physics']
    write_json(data_root/'calibration.json',calibration)
    planned=[episode_plan(split,index) for split,count in COUNTS.items() for index in range(count if limit is None else min(count,limit))]
    write_json(data_root/'plan.json',{'config':config,'episodes':planned})
    episodes=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for number,metadata in enumerate(pool.map(collect_episode,[(config,ep,str(data_root)) for ep in planned]),1):
            episodes.append(metadata)
            write_json(data_root/'manifest.partial.json',{'config':config,'episodes':episodes})
            print(f"Collected {number}/{len(planned)} {metadata['id']} {metadata['condition']} valid={metadata['valid']} elapsed={time.perf_counter()-started:.1f}s",flush=True)
    manifest={'config':config,'episodes':episodes,'split_counts':{split:sum(ep['split']==split for ep in episodes) for split in COUNTS},
              'protocol':{'dt':.1,'physics_dt':.002,'camera':'current-contact ideal downward camera, 64x64, 10Hz',
                          'alignment':'current wheel contact points projected with camera calibration; RGB and motion at identical timestamps',
                          'visibility_assumption':'robot geometry hidden only in RGB; this is an idealized contact view, not front-camera anticipation',
                          'visual_relation':'training texture family statistically associated with mu range; exceptions and reversed mappings retained',
                          'supervision':'future observed motion; physical parameter metadata and truth are audit/test scoring only',
                          'units':'v m/s, yaw rad/s, wheel rates rad/s, nominal commands N',
                          'split':'complete episodes, texture seeds, input seeds and noise seeds are disjoint'}}
    write_json(data_root/'manifest.json',manifest)
    if not all(ep['valid'] for ep in episodes):
        raise RuntimeError('Some physical episodes failed audit; all have been retained in the manifest.')
    return manifest
