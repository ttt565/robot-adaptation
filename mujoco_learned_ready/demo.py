"""Run a frozen learned predictor beside a fresh MuJoCo episode, one frame at a time."""
import argparse
import hashlib
import json
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import mujoco
import numpy as np

from collect_data import LearningPlant,planned_commands,write_json
from learning import load_model

ROOT=Path(__file__).resolve().parent


def parameter_hash(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_demo(data_root,model_root,output,episode_id=None,headless=False,max_steps=None):
    data_root,model_root,output=Path(data_root),Path(model_root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((data_root/'manifest.json').read_text())
    episode=next((ep for ep in manifest['episodes'] if ep['id']==episode_id),None) if episode_id else next(ep for ep in manifest['episodes'] if ep['split']=='test' and ep['condition']=='correlated')
    if episode is None:
        raise ValueError('Unknown episode ID.')
    config=manifest['config'];dt=config['dt']
    stream=load_model(model_root/'checkpoints'/'fusion.pt')
    before=parameter_hash(stream.model)
    plant=LearningPlant(config,episode)
    renderer=mujoco.Renderer(plant.model,height=64,width=64)
    options=mujoco.MjvOption();options.geomgroup[1]=0;options.geomgroup[5]=0
    commands=planned_commands(config,episode['input_seed'])
    count=len(commands)
    noise=np.random.default_rng(episode['noise_seed']).normal(size=(count+1,4))*[.01,.01,.05,.05]
    truth=[plant.motion()];observed=[];one=[];half=[];latent=[];latency=[]
    viewer,figure,threads=None,None,[]
    if not headless:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from mujoco import viewer as native_viewer
        existing=set(threading.enumerate())
        viewer=native_viewer.launch_passive(plant.model,plant.data,show_left_ui=False,show_right_ui=False)
        threads=[thread for thread in threading.enumerate() if thread not in existing and thread.daemon]
        with viewer.lock():
            viewer.cam.type,viewer.cam.trackbodyid=mujoco.mjtCamera.mjCAMERA_TRACKING,plant.body_id
            viewer.cam.distance,viewer.cam.azimuth,viewer.cam.elevation=2.8,135,-28
        plt.ion()
        figure,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
        figure.suptitle('Frozen neural weights; online recurrent context; no learned speed control')
    try:
        for step in range(count if max_steps is None else min(count,max_steps)):
            start=time.perf_counter()
            if viewer is not None and (not viewer.is_running() or not plt.fignum_exists(figure.number)):
                break
            with viewer.lock() if viewer is not None else nullcontext():
                renderer.update_scene(plant.data,camera='contact_camera',scene_option=options)
                rgb=renderer.render().copy()
                pixels,_,_,_,_=plant.aligned_contacts()
                obs=truth[-1]+noise[step]
                began=time.perf_counter()
                prediction=stream.stream_predict(obs,rgb,commands[step-1] if step else None,commands[step:step+5])
                latency.append((time.perf_counter()-began)*1000)
                observed.append(obs);one.append(prediction['one']);half.append(prediction['half'] if prediction['half'] is not None else np.full(4,np.nan));latent.append(prediction['latent'])
                plant.step(commands[step])
                truth.append(plant.motion())
            if viewer is not None:
                viewer.sync()
                if step%2==0:
                    for axis in axes.flat:
                        axis.clear()
                    axes[0,0].imshow(rgb);axes[0,0].scatter(pixels[:,0],pixels[:,1],s=45,facecolors='none',edgecolors='yellow')
                    axes[0,0].set_title('Current contact-view RGB');axes[0,0].axis('off')
                    times=np.arange(len(truth))*dt
                    axes[0,1].plot(times,np.asarray(truth)[:,0],label='Actual v')
                    axes[0,1].plot(times[1:],np.asarray(one)[:,0],label='Prior one-step forecast')
                    axes[0,1].set(title='Motion forecast',xlabel='Time (s)',ylabel='Speed (m/s)');axes[0,1].legend(fontsize=8)
                    axes[1,0].plot(times[:-1],np.asarray(latent)[:,:4])
                    axes[1,0].set(title='First four learned context coordinates',xlabel='Time (s)',ylabel='z (no physical units)')
                    axes[1,1].axis('off')
                    axes[1,1].text(.02,.95,f"Episode: {episode['id']}\nt = {times[-1]:.1f} s\n\nWeights fixed throughout run.\nHidden state updates each observation.\n\nInference: {latency[-1]:.1f} ms\nNo backpropagation during motion.\nInputs were planned before the run.",va='top',fontsize=11)
                    plt.pause(.001)
                remaining=dt-(time.perf_counter()-start)
                if remaining>0:
                    time.sleep(remaining)
    finally:
        renderer.close()
        if viewer is not None:
            viewer.close()
            for thread in threads:
                thread.join(timeout=3)
            plt.close(figure);plt.ioff()
    truth,one,half,latent=np.asarray(truth),np.asarray(one),np.asarray(half),np.asarray(latent)
    if not len(one):
        raise RuntimeError('No online observation was processed.')
    after=parameter_hash(stream.model)
    assert before==after,'Neural weights changed during online inference.'
    errors=one[:,:2]-truth[1:,:2]
    report={'episode_id':episode['id'],'complete':len(one)==count,'steps':len(one),
            'weights_before':before,'weights_after':after,'weights_unchanged':before==after,
            'hidden_state_coordinate_std':np.std(latent,axis=0).tolist(),
            'one_speed_rmse':float(np.sqrt(np.mean(errors[:,0]**2))),
            'one_yaw_rmse':float(np.sqrt(np.mean(errors[:,1]**2))),
            'median_inference_ms':float(np.median(latency)),
            'score_scope':'whole available episode including cold start; not the burn-in-excluded benchmark score',
            'inputs':'fresh MuJoCo observed motion + current RGB + previous nominal input; future inputs preplanned',
            'online_update':'GRU hidden state only; network weights frozen; context frozen inside each half-second rollout'}
    np.savez_compressed(output/'online_log.npz',times=np.arange(len(truth))*dt,truth=truth,observed=np.asarray(observed),
                        one=one,half=half,latent=latent,commands=commands[:len(one)],inference_ms=np.asarray(latency))
    write_json(output/'online_demo_report.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,
                        help='Dataset directory; defaults to data, or the bundled demo_assets manifest.')
    parser.add_argument('--model-dir',type=Path,default=ROOT/'results'/'trained')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'runs'/'online_demo')
    parser.add_argument('--episode')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--max-steps',type=int)
    args=parser.parse_args()
    data_dir=args.data_dir
    if data_dir is None:
        data_dir=ROOT/'data'
        if not (data_dir/'manifest.json').is_file():
            data_dir=ROOT/'demo_assets'
    run_demo(data_dir,args.model_dir,args.output_dir,args.episode,args.headless,args.max_steps)


if __name__=='__main__':
    main()
