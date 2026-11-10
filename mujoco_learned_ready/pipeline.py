"""Collect aligned MuJoCo data, train neural dynamics, evaluate online latent state."""
import argparse
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['all','collect','train','evaluate'],nargs='?',default='all')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results'/'trained')
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--limit',type=int,help='Small collection check: at most this many episodes per split')
    args=parser.parse_args()
    data=args.data_dir.resolve();output=args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if args.epochs<1 or args.workers<1:
        parser.error('epochs/workers must be positive.')
    if args.action in ['all','collect']:
        if (data/'manifest.json').exists() and args.action=='all':
            print(f'Using existing dataset: {data}',flush=True)
        else:
            from collect_data import collect_dataset
            manifest=collect_dataset(data,args.workers,args.limit)
            print(f"Saved {len(manifest['episodes'])} complete RGB-motion episodes.",flush=True)
    if args.action in ['all','train']:
        from learning import train_all
        train_all(data,output,epochs=args.epochs,seed=args.seed,device=args.device)
    if args.action in ['all','evaluate']:
        from learning import evaluate_all
        evaluate_all(data,output)
        from training_dashboard import render_report
        render_report(data,output)
    print(f'Data: {data}\nResults: {output}',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
