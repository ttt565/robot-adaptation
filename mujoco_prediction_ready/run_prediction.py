"""MuJoCo prediction benchmark: fixed inputs and one shared log per physical case."""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import engine
from predictors import MODES, OnlinePredictor
from run_vision import prepare_model, write_json
from surfaces import CASES, GroundCamera, SurfacePlant

ROOT = Path(__file__).resolve().parent
PROTOCOL = {'duration': 16., 'horizon_seconds': .5, 'warmup_seconds': 3.2,
            'recovery_window_seconds': 1., 'recovery_hold_seconds': 1.,
            'recovery_speed_rmse': .03, 'recovery_yaw_rmse': .06}


def command_plan(config, seed):
    """Complete schedule exists before the plant starts; independent of states/models."""
    times = np.arange(round(config['duration']/config['dt'])) * config['dt']
    phase = np.random.default_rng(30000 + seed).uniform(-np.pi, np.pi)
    ramp = (1 - np.cos(np.minimum(times/2., 1.)*np.pi))/2
    total = ramp*(4.2 + .6*np.sin(.8*times+phase) + .4*np.sin(1.7*times+.3*phase))
    difference = ramp*(.35*np.sin(1.1*times+phase) + .15*np.sin(2.3*times))
    return np.column_stack([(total-difference)/2, (total+difference)/2])


def source_hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['engine.py','robot.xml','surfaces.py','appearance.py','control_core.py','run_prediction.py','predictors.py']}


def collect(config, case, seed, classifier, directory):
    directory.mkdir(parents=True, exist_ok=True)
    plant, commands = SurfacePlant(config, case, texture_seed=1000+seed), command_plan(config, seed)
    camera, dt = GroundCamera(plant), config['dt']
    steps = len(commands)
    states, observations = np.zeros((steps+1,5)), np.zeros((steps+1,5))
    states[0] = plant.true_state()
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=states.shape)*config['measurement_noise']['std']
    encoder_noise = rng.normal(size=(steps,2))*.05
    observations[0] = states[0]+noise[0]
    wheels = np.zeros((steps,2))
    contact_mu, heights = np.full(steps,np.nan), np.zeros(steps)
    first_contact, camera_rows, preview = None, [], []
    try:
        for step in range(steps+1):
            if step%10 == 0:
                rgb = camera.read_rgb()
                prediction = classifier.predict(rgb)
                camera_rows.append({'step':step,'time':step*dt,'label':prediction['label'],
                                    'match_score':prediction['match_score']})
                if not preview:
                    preview.append(rgb)
                elif len(preview)==1:
                    preview.append(rgb)
                else:
                    preview[-1]=rgb
            if step==steps:
                break
            states[step+1] = plant.step(commands[step])
            observations[step+1] = states[step+1]+noise[step+1]
            wheels[step] = plant.data.qvel[plant.wheel_dofs]+encoder_noise[step]
            audit = plant.audit()  # 真值接触信息仅存入离线审计，不传给预测器。
            contact_mu[step], heights[step] = audit['contact_mu'], audit['height']
            if first_contact is None and audit['after_surface_contact']:
                first_contact = (step+1)*dt
    finally:
        camera.close()
    event_time = config['change_time'] if case=='motor_fault' else first_contact if plant.mu_change else None
    finite = bool(np.isfinite(states).all() and np.isfinite(observations).all() and np.isfinite(wheels).all())
    inside = bool(np.all((states[:,0]>-5.5)&(states[:,0]<23.5)&(np.abs(states[:,1])<3.5)))
    contact_coverage = float(np.mean(np.isfinite(contact_mu)))
    valid = finite and inside and np.min(heights)>.04 and (not plant.mu_change or event_time is not None)
    metadata = {'case':case,'seed':seed,'event_time':event_time,'first_post_patch_contact':first_contact,
                'valid':bool(valid),'finite':finite,'within_contact_area_margin':inside,
                'min_body_height':float(heights.min()),'tire_contact_fraction':contact_coverage,
                'nominal_command_units':'N; actual torque conversion and hidden efficiency applied only inside plant',
                'planned_commands_sha256':hashlib.sha256(commands.tobytes()).hexdigest(),
                'observations_sha256':hashlib.sha256(observations.tobytes()).hexdigest(),
                'source_sha256':source_hashes(),'configuration':config}
    np.savez_compressed(directory/'log.npz', times=np.arange(steps+1)*dt,states=states,observations=observations,
                        commands=commands,wheel_rates_observed=wheels,contact_mu=contact_mu,body_height=heights)
    with (directory/'camera_predictions.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(camera_rows[0]));writer.writeheader();writer.writerows(camera_rows)
    for label,rgb in zip(['first','last'],preview):
        Image.fromarray(rgb).save(directory/f'{label}_camera.png')
    write_json(directory/'metadata.json',metadata)
    return metadata


def recovery(times, errors, event, config):
    if event is None:
        return {'status':'not_applicable','confirmation_delay':None}
    dt=config['dt']
    size=round(PROTOCOL['recovery_window_seconds']/dt)
    hold=round(PROTOCOL['recovery_hold_seconds']/dt)
    consecutive=0
    for end in range(size-1,len(times)):
        start=end-size+1
        block=errors[start:end+1]
        if times[start]<event or not np.isfinite(block).all():
            consecutive=0;continue
        rms=np.sqrt(np.mean(block**2,axis=0))
        good=rms[0]<=PROTOCOL['recovery_speed_rmse'] and rms[1]<=PROTOCOL['recovery_yaw_rmse']
        consecutive=consecutive+1 if good else 0
        if consecutive>=hold:
            # 最后一个预测起点的真值在0.5秒后才可用于确认。
            available=times[end]+PROTOCOL['horizon_seconds']
            return {'status':'confirmed','confirmation_delay':float(available-event),
                    'last_origin_time':float(times[end]),'confirmation_time':float(available)}
    return {'status':'not_recovered','confirmation_delay':None}


def replay(config, case, seed, mode, directory, output):
    metadata=json.loads((directory/'metadata.json').read_text())
    if not metadata['valid']:
        raise RuntimeError(f'Invalid collection retained for inspection: {directory}')
    # 采集代码的哈希保留在metadata中；允许修改预测器后复用不可变日志。
    # 当前预测实现另外记录，不将模型代码变化误判为运动数据变化。
    if metadata['configuration']!=config:
        raise RuntimeError('Replay configuration differs from the one used for this shared log.')
    log=dict(np.load(directory/'log.npz',allow_pickle=False))
    for name in ['commands','observations']:
        key='planned_commands_sha256' if name=='commands' else 'observations_sha256'
        if hashlib.sha256(log[name].tobytes()).hexdigest()!=metadata[key]:
            raise RuntimeError('Shared log hash mismatch.')
    commands,observations,states=log['commands'],log['observations'],log['states']
    steps,dt=len(commands),config['dt']
    horizon=round(PROTOCOL['horizon_seconds']/dt)
    camera_rows={int(row['step']):row for row in csv.DictReader((directory/'camera_predictions.csv').open())}
    public_settings={key:config[key] for key in ['physics','dt','estimator','measurement_noise']}
    predictor=OnlinePredictor(mode,public_settings)
    one,half=np.full((steps,2),np.nan),np.full((steps,2),np.nan)
    eta,raw,windows=np.ones((steps,2)),np.ones((steps,2)),np.zeros(steps,dtype=int)
    visual_times=[]
    for step in range(steps):
        if predictor.observe_image(camera_rows.get(step)):
            visual_times.append(step*dt)
        eta[step],raw[step],windows[step]=predictor.estimator.used(),predictor.estimator.raw,predictor.estimator.window
        # 此处预测函数只收到起点观测和已知预定输入，没有未来观测/RGB/场景信息。
        one[step]=predictor.predict(observations[step],commands[step:step+1])
        if step+horizon<=steps:
            half[step]=predictor.predict(observations[step],commands[step:step+horizon])
        predictor.update(observations[step],commands[step],observations[step+1])
    errors_one=one-states[1:,3:5]
    errors_half=np.full_like(half,np.nan)
    valid_count=steps-horizon+1
    errors_half[:valid_count]=half[:valid_count]-states[horizon:,3:5]
    observed_innovation=one-observations[1:,3:5]
    times=log['times'][:-1]
    event=metadata['event_time']
    score_start=max(PROTOCOL['warmup_seconds'],event if event is not None else PROTOCOL['warmup_seconds'])
    post=times>=score_start
    finite_half=np.isfinite(errors_half).all(axis=1)
    def rms(error,mask):
        return np.sqrt(np.mean(error[mask]**2,axis=0)).tolist() if np.any(mask) else [None,None]
    one_rmse=rms(errors_one,post)
    half_rmse=rms(errors_half,post&finite_half)
    pre=(times>=PROTOCOL['warmup_seconds'])&(times+PROTOCOL['horizon_seconds']<(event if event is not None else np.inf))
    metric={'one_speed_rmse_post':one_rmse[0],'one_yaw_rmse_post':one_rmse[1],
            'half_speed_rmse_post':half_rmse[0],'half_yaw_rmse_post':half_rmse[1],
            'half_rmse_pre':rms(errors_half,pre&finite_half),
            'one_observation_innovation_rmse_post':rms(observed_innovation,post),
            'half_speed_absolute_error_integral_post':float(np.sum(np.abs(errors_half[post&finite_half,0]))*dt),
            'half_yaw_absolute_error_integral_post':float(np.sum(np.abs(errors_half[post&finite_half,1]))*dt),
            'one_post_origin_interval':[float(times[post][0]),float(times[post][-1])],
            'half_post_origin_interval':[float(times[post&finite_half][0]),float(times[post&finite_half][-1])],
            'event_time_evaluation_only':event,'recovery':recovery(times,errors_half,event,config),
            'visual_trigger_times':visual_times,'log_observations_sha256':metadata['observations_sha256'],
            'log_commands_sha256':metadata['planned_commands_sha256'],
            'finite_predictions':bool(np.isfinite(one).all() and np.isfinite(half[:valid_count]).all()),
            'parameter_semantics':'effective planar drive gains; not friction coefficients or a learned neural latent',
            'prediction_source_sha256':{name:source_hashes()[name] for name in
                                        ['predictors.py','control_core.py','run_prediction.py']},
            'missing_half_targets':int(steps-valid_count)}
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'predictions.npz',times=times,one=one,half=half,errors_one=errors_one,
                        errors_half=errors_half,observed_innovation=observed_innovation,eta=eta,eta_raw=raw,window=windows)
    write_json(output/'metrics.json',metric)
    return {'case':case,'mode':mode,'seed':seed,'times':times,'errors_one':errors_one,'errors_half':errors_half,
            'eta':eta,'window':windows,'event_time':event,'visual_times':visual_times,'metrics':metric,'commands':commands}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-cases',action='store_true')
    parser.add_argument('--case',choices=CASES)
    parser.add_argument('--seeds',type=int,nargs='+',default=[2,3])
    parser.add_argument('--output-dir',type=Path)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--collect-only',action='store_true')
    group.add_argument('--replay-only',action='store_true')
    args=parser.parse_args()
    if args.case and args.all_cases:
        parser.error('Choose --case or --all-cases.')
    if any(seed<0 for seed in args.seeds) or len(set(args.seeds))!=len(args.seeds):
        parser.error('Use distinct nonnegative seeds.')
    if args.replay_only and not args.output_dir:
        parser.error('--replay-only needs the existing --output-dir.')
    import matplotlib
    matplotlib.use('Agg')
    output=args.output_dir.resolve() if args.output_dir else ROOT/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if not args.replay_only and output.exists() and any(output.iterdir()):
        parser.error('Use an empty output directory for a new collection.')
    output.mkdir(parents=True,exist_ok=True)
    cases=[args.case] if args.case else CASES
    if args.replay_only:
        config=json.loads((output/'config_used.json').read_text())
        classifier=None
    else:
        classifier,_=prepare_model()
        config=json.loads((ROOT/'config.json').read_text())
        config['duration']=PROTOCOL['duration']
        calibration=engine.calibrate(config,plant_factory=lambda settings:SurfacePlant(settings,'normal',texture_seed=4000))
        config['physics']=calibration['physics']
        write_json(output/'config_used.json',config)
        write_json(output/'calibration.json',calibration)
    runs,collections=[],[]
    for seed in args.seeds:
        for case in cases:
            directory=output/'datasets'/f'seed_{seed}'/case
            if not args.replay_only:
                print(f'Collecting one shared MuJoCo log: {case}, seed {seed}',flush=True)
                metadata=collect(config,case,seed,classifier,directory)
                print({key:metadata[key] for key in ['valid','event_time','within_contact_area_margin','min_body_height']},flush=True)
            else:
                metadata=json.loads((directory/'metadata.json').read_text())
            collections.append(metadata)
            if not args.collect_only:
                for mode in MODES:
                    run=replay(config,case,seed,mode,directory,output/'predictions'/f'seed_{seed}'/case/mode)
                    runs.append(run)
                    print(mode,{key:run['metrics'][key] for key in ['half_speed_rmse_post','half_yaw_rmse_post','recovery']},flush=True)
    report={'protocol':PROTOCOL,'model_modes':MODES,
            'interpretation':'same recorded inputs/observations; no model-dependent speed or controller',
            'visual_scope':'RGB appearance-change scheduling ablation, not learned image-to-dynamics inference',
            'future_inputs':'entire open-loop command plan generated before collection; conditional motion forecasts',
            'prediction_order':'current image then forecast; only afterward reveal next observation and update',
            'score_truth':'future MuJoCo states and event metadata are used only for offline scoring',
            'recovery_definition':'trailing 1 s half-horizon RMSE below preset v/yaw thresholds for 1 s; confirmation includes 0.5 s target availability',
            'collections':collections,'runs':[{'case':r['case'],'mode':r['mode'],'seed':r['seed'],'metrics':r['metrics']} for r in runs]}
    write_json(output/'report.json',report)
    if runs:
        from prediction_dashboard import save_dashboard
        save_dashboard(runs,output/'prediction_dashboard.png')
    print(f'Results saved: {output}',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
