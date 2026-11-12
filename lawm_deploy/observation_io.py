"""Lossless observation archives and offline cloud export. No hardware imports."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from lawm_deploy.model_contract import timing_metadata


def back_project(depth, K, cam_pose, near, far):
    """Pinhole depth to base XYZ in metres, using the depth optical frame."""
    v, u = np.nonzero((depth > near) & (depth < far))
    z = depth[v, u]
    cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    return cam @ cam_pose[:3, :3].T + cam_pose[:3, 3]


def save_observation(path, obs, cfg, provenance, **extra):
    """Keep legacy rounded-mm keys, plus exact median and optional sensor Z16."""
    arrays = dict(coords=obs.coords, segment=obs.segment, q_arm=obs.q_arm, q_hand=obs.q_hand,
                  flange=obs.flange, observation_info_json=json.dumps(obs.info),
                  config_json=json.dumps(cfg), provenance_json=json.dumps(provenance),
                  timing_json=json.dumps(timing_metadata()))
    raw = getattr(obs, 'raw_depth', None)
    for i, (depth, K, T) in enumerate(obs.views):
        arrays[f'depth_{i}'] = np.round(depth * 1000).clip(0, 65535).astype(np.uint16)
        arrays[f'depth_median_m_{i}'] = np.asarray(depth, dtype=np.float32)
        arrays[f'cam_{i}'] = np.concatenate([K.ravel(), T.ravel()])
        if raw is not None:
            frames, scale = raw[i]
            arrays[f'sensor_depth_z16_{i}'] = frames
            arrays[f'sensor_depth_scale_m_{i}'] = scale
    arrays.update(extra)
    np.savez_compressed(path, **arrays)


def write_ply(path, points):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite Nx3 points')
    with Path(path).open('w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n')
        np.savetxt(f, points, fmt='%.9g')


def export_clouds(source, destination):
    """Export one saved start/goal or preflight frame; never invent missing data."""
    source, destination = Path(source), Path(destination)
    with np.load(source, allow_pickle=False) as saved:
        indices = sorted(int(k[6:]) for k in saved.files if k.startswith('depth_') and k[6:].isdigit())
        if not indices:
            raise ValueError('No saved depth views in this archive')
        destination.mkdir(parents=True, exist_ok=False)
        manifest = dict(source=str(source.resolve()), units='metres', base_frame='robot_base_z_up',
                        camera_frame='depth_optical', model_input_rotation_applied=False, cameras=[])
        for i in indices:
            key = f'depth_median_m_{i}'
            if key in saved:
                depth, depth_source = saved[key], 'full_precision_median_depth'
            else:
                depth = saved[f'depth_{i}']
                depth_source = 'legacy_rounded_millimetre_median' if depth.dtype == np.uint16 else 'saved_median_depth_metres'
                depth = depth.astype(float) * (0.001 if depth.dtype == np.uint16 else 1.0)
            if f'cam_{i}' in saved:
                flat = saved[f'cam_{i}']
                K, T = flat[:9].reshape(3, 3), flat[9:].reshape(4, 4)
            else:
                K, T = saved[f'K_{i}'], saved[f'T_base_depth_{i}']
            camera = back_project(depth, K, np.eye(4), 0, np.inf)
            base = camera @ T[:3, :3].T + T[:3, 3]
            write_ply(destination / f'camera_{i}_optical.ply', camera)
            write_ply(destination / f'camera_{i}_base.ply', base)
            manifest['cameras'].append(dict(index=i, points=len(base), source=depth_source,
                robot_masked=False, cropped=False, voxel_sampled=False,
                sensor_z16_available=f'sensor_depth_z16_{i}' in saved))
        if 'coords' in saved:
            write_ply(destination / 'model_input_base.ply', saved['coords'])
            manifest['model_points'] = len(saved['coords'])
            manifest['model_input_note'] = 'Exact saved base-frame input with FK hand. Labels are in observation.npz. Apply the training rotation exactly once in the model adapter.'
        for key in ('config_json', 'provenance_json', 'observation_info_json', 'timing_json', 'camera_metadata_json'):
            if key in saved:
                manifest[key.removesuffix('_json')] = json.loads(str(saved[key]))
    shutil.copy2(source, destination / 'observation.npz')
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Export saved start/goal/preflight NPZ without hardware or LAWM')
    parser.add_argument('source')
    parser.add_argument('destination', help='new directory; existing directories are not overwritten')
    args = parser.parse_args()
    export_clouds(args.source, args.destination)
    print(f'Exported {args.destination}; no data uploaded')


if __name__ == '__main__':
    main()
