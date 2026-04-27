from pathlib import Path
import argparse
import json

from nuscenes.nuscenes import NuScenes


def build_manifest(dataroot: Path, version: str, scene_name: str, num_samples: int):
    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
    scene = next(s for s in nusc.scene if s['name'] == scene_name)

    samples = []
    token = scene['first_sample_token']
    idx = 0
    while token and idx < num_samples:
        sample = nusc.get('sample', token)
        entry = {
            'index': idx,
            'sample_token': sample['token'],
            'timestamp_us': int(sample['timestamp']),
            'timestamp_s': float(sample['timestamp']) / 1e6,
            'prev': sample['prev'],
            'next': sample['next'],
            'channels': {},
        }
        for channel, sd_token in sample['data'].items():
            sample_data = nusc.get('sample_data', sd_token)
            entry['channels'][channel] = {
                'token': sd_token,
                'filename': sample_data['filename'],
                'is_key_frame': bool(sample_data['is_key_frame']),
                'timestamp_us': int(sample_data['timestamp']),
            }
        samples.append(entry)
        token = sample['next']
        idx += 1

    return {
        'version': version,
        'dataroot': str(dataroot),
        'scene_name': scene_name,
        'scene_token': scene['token'],
        'description': scene['description'],
        'num_requested': num_samples,
        'num_selected': len(samples),
        'samples': samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=Path, default=Path.home() / 'datasets' / 'nuscenes')
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--scene-name', type=str, default='scene-0061')
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    manifest = build_manifest(args.dataroot, args.version, args.scene_name, args.num_samples)
    if args.output is None:
        output = Path(__file__).resolve().parents[1] / 'manifests' / f"{args.scene_name}_first{args.num_samples}.json"
    else:
        output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2))
    print(output)
    print(json.dumps({
        'scene_name': manifest['scene_name'],
        'num_selected': manifest['num_selected'],
        'sample_tokens': [s['sample_token'] for s in manifest['samples']],
    }, indent=2))


if __name__ == '__main__':
    main()
