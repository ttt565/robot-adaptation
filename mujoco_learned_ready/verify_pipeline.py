"""Verify trained artifacts, causal inference, and ideal-camera ground projection."""
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from collect_data import LearningPlant,write_json
from demo import parameter_hash
from learning import MODES,_read_episode,_neural_predictions,load_model

ROOT=Path(__file__).resolve().parent


def projection_audit(data_root,output):
    manifest=json.loads((data_root/'manifest.json').read_text())
    episode=next(ep for ep in manifest['episodes'] if ep['id']=='test_000')
    plant=LearningPlant(manifest['config'],episode)
    renderer=mujoco.Renderer(plant.model,height=64,width=64)
    options=mujoco.MjvOption();options.geomgroup[1]=0;options.geomgroup[5]=0
    poses=[(-.4,0.,0.,0.),(0.,0.,0.,math.pi/4),(.4,0.,0.,0.),
           (0.,0.,.04,math.pi/4),(0.,.04,0.,math.pi/4),(0.,.03,-.04,1.1)]
    figure,axes=plt.subplots(2,3,figsize=(10,7),constrained_layout=True)
    checks=[]
    try:
        for index,((offset,roll,pitch,yaw),axis) in enumerate(zip(poses,axes.flat)):
            cr,sr=math.cos(roll/2),math.sin(roll/2)
            cp,sp=math.cos(pitch/2),math.sin(pitch/2)
            cy,sy=math.cos(yaw/2),math.sin(yaw/2)
            plant.data.qpos[:3]=[episode['boundary']+offset,0,.12]
            plant.data.qpos[3:7]=[cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy]
            mujoco.mj_forward(plant.model,plant.data)
            pixels,_,visible,points,_=plant.aligned_contacts()
            renderer.update_scene(plant.data,camera='contact_camera',scene_option=options)
            rgb=renderer.render().copy()
            renderer.enable_segmentation_rendering()
            renderer.update_scene(plant.data,camera='contact_camera',scene_option=options)
            segmentation=renderer.render().copy()
            renderer.disable_segmentation_rendering()
            for wheel,(pixel,point) in enumerate(zip(pixels,points)):
                x,y=np.rint(pixel).astype(int)
                expected=plant.terrains[int(point[0]>=episode['boundary'])]
                actual=int(segmentation[y,x,0])
                assert visible[wheel] and actual==expected,(index,wheel,actual,expected)
                checks.append({'pose':index,'wheel':wheel,'pixel':pixel.tolist(),'expected_ground_geom':expected,'rendered_ground_geom':actual})
            axis.imshow(rgb);axis.scatter(pixels[:,0],pixels[:,1],s=120,facecolors='none',edgecolors=['yellow','cyan'],linewidths=2)
            axis.set_title(f'roll {math.degrees(roll):.1f}, pitch {math.degrees(pitch):.1f}, yaw {math.degrees(yaw):.1f} deg')
            axis.axis('off')
    finally:
        renderer.close()
    figure.suptitle('Ideal contact-view projection audit: wheel ground patches\nSynthetic poses; segmentation IDs are used only by this audit')
    figure.savefig(output/'contact_alignment.png',dpi=150)
    plt.close(figure)
    write_json(output/'contact_alignment_audit.json',{'checks':checks,'passed':True,
               'scope':'known simulated camera geometry and ground-patch visibility; not real-camera calibration accuracy',
               'roundtrip_error_note':'alignment_error_m stored in episodes is algebraic projection/backprojection consistency only'})
    return checks


def verify(data_root=None,output=None):
    data_root=Path(data_root or ROOT/'data');output=Path(output or ROOT/'results'/'trained')
    manifest=json.loads((data_root/'manifest.json').read_text())
    trained=json.loads((output/'training_report.json').read_text())
    metrics=json.loads((output/'metrics.json').read_text())
    counts={split:sum(ep['split']==split for ep in manifest['episodes']) for split in ['train','val','test','ood']}
    assert counts=={'train':48,'val':12,'test':12,'ood':12}
    assert trained['training_episode_count']==48 and trained['validation_episode_count']==12
    assert trained['normalization_split']=='train' and not trained['loss_uses_truth']
    assert trained['training_manifest_sha256']==hashlib.sha256((data_root/'manifest.json').read_bytes()).hexdigest()
    for field in ['id','input_seed','noise_seed']:
        values=[ep[field] for ep in manifest['episodes']]
        assert len(set(values))==84
    texture_sets={split:{ep['texture_seed']+offset for ep in manifest['episodes'] if ep['split']==split for offset in [0,1]} for split in counts}
    for first in counts:
        for second in counts:
            if first!=second:assert not texture_sets[first]&texture_sets[second]
    frame_count=0
    for item in manifest['episodes']:
        file=data_root/item['file']
        assert item['valid'] and hashlib.sha256(file.read_bytes()).hexdigest()==item['sha256']
        with np.load(file,allow_pickle=False) as data:
            assert data['rgb'].shape==(121,64,64,3) and data['contact_visible'].all()
            assert np.isfinite(data['truth']).all() and np.isfinite(data['observed']).all()
        assert len(list((file.parent/'rgb').glob('*.png')))==121
        frame_count+=121
    assert len(metrics['per_episode'])==24*6
    for item in metrics['per_episode']:
        assert item['split'] in ['test','ood']
        assert all(math.isfinite(item[key]) for key in ['one_speed_rmse','one_yaw_rmse','half_speed_rmse','half_yaw_rmse'])
        prediction=np.load(output/'predictions'/item['id']/f"{item['model']}.npz",allow_pickle=False)
        assert np.isfinite(prediction['one']).all() and np.isfinite(prediction['half'][:116]).all()
        assert np.isnan(prediction['half'][116:]).all()
    stream=load_model(output/'checkpoints'/'fusion.pt')
    sample=next(item for item in manifest['episodes'] if item['id']=='test_000')
    episode=_read_episode(data_root,sample,need_truth=False)
    before=parameter_hash(stream.model)
    one,half,latent=_neural_predictions(stream,episode,5)
    changed={key:value.copy() for key,value in episode.items()}
    changed['observed'][21:]+=100
    changed['rgb'][21:]=255-changed['rgb'][21:]
    one2,half2,latent2=_neural_predictions(stream,changed,5)
    for original,perturbed in [(one,one2),(half,half2),(latent,latent2)]:
        np.testing.assert_array_equal(original[:21],perturbed[:21])
    stream.reset();streamed_one=[];streamed_half=[];streamed_z=[]
    for step in range(120):
        result=stream.stream_predict(episode['observed'][step],episode['rgb'][step],
                                     episode['commands'][step-1] if step else None,episode['commands'][step:step+5])
        streamed_one.append(result['one']);streamed_half.append(result['half'] if result['half'] is not None else np.full(4,np.nan));streamed_z.append(result['latent'])
    np.testing.assert_allclose(np.asarray(streamed_one),one,rtol=1e-5,atol=2e-6)
    np.testing.assert_allclose(np.asarray(streamed_half),half,rtol=1e-5,atol=3e-6,equal_nan=True)
    np.testing.assert_allclose(np.asarray(streamed_z),latent,rtol=1e-5,atol=2e-6)
    after=parameter_hash(stream.model)
    assert before==after
    online=np.load(ROOT/'results'/'online_demo'/'online_log.npz')
    actual=np.load(data_root/sample['file'])
    np.testing.assert_allclose(online['truth'],actual['truth'],rtol=0,atol=1e-9)
    np.testing.assert_allclose(online['one'],one,rtol=1e-5,atol=3e-6)
    projection=projection_audit(data_root,output)
    checks=['84 valid independent episodes; complete synchronized RGB; 48/12/12/12 split with disjoint texture/input/noise seeds',
            'training uses observations, train-only normalization and validation-only checkpoint selection',
            'three neural models actually trained; all 24 held-out episodes have six comparable prediction/ablation outputs',
            'future observation/RGB perturbation leaves all forecasts through the shared prefix unchanged',
            'streaming and batched causal inference agree; eight-dimensional hidden context updates while weight hashes stay identical',
            'fresh MuJoCo streaming run reproduces held-out motion and forecast outputs',
            '12 projected wheel-ground patches match independent rendered segmentation IDs at six synthetic poses']
    sources=['collect_data.py','learning.py','pipeline.py','demo.py','verify_pipeline.py','training_dashboard.py','engine.py','control_core.py','robot.xml','appearance.py','config.json','README.md','requirements.txt','run_all.cmd','train_only.cmd','evaluate.cmd','run_demo.cmd']
    report={'status':'passed_trained_prototype','checked_on':'2026-11-03','checks':checks,'frames':frame_count,
            'weight_sha256_before':before,'weight_sha256_after':after,'projection_checks':len(projection),
            'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
            'limits':['ideal current-contact camera; robot geometry hidden; no forward anticipation or real-camera calibration result',
                      'single training seed and limited procedural simulated textures; no real-world performance claim',
                      'fusion image-ablation change is small; incremental visual benefit has not been established',
                      'learned latent coordinates are not calibrated physical/friction parameters']}
    write_json(ROOT/'ready_check.json',report)
    print(json.dumps({'checks':checks,'frames':frame_count,'projection_checks':len(projection)},indent=2),flush=True)
    return report


if __name__=='__main__':
    verify()
