"""Stream NPZ members once to scratch, then read large arrays through mmap."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from ..artifacts import digest, verify_records, write_json
from .schema import validate_raw


def resolve(manifest, value):
    return (Path(manifest).resolve().parent / value).resolve()


def load_raw(manifest, require_vm=True):
    manifest = Path(manifest).resolve()
    meta = json.loads(manifest.read_text())
    if meta.get('kind') != 'raw_vm' or meta.get('version') != 1:
        raise ValueError('Expected a version-1 raw_vm dataset manifest')
    data = {}
    for key, record in meta['arrays'].items():
        path = resolve(manifest, record['path'])
        stat = path.stat()
        if stat.st_size != record['bytes'] or stat.st_mtime_ns != record['mtime_ns']:
            raise ValueError(f'Raw array changed since preparation: {path}')
        data[key] = np.load(path, mmap_mode='r', allow_pickle=False)
    validate_raw(data, require_vm)
    return data


def import_raw(source, destination, require_vm=True):
    """Return a raw manifest. A supplied manifest reuses existing arrays without copying."""
    source, destination = Path(source).resolve(), Path(destination)
    if source.suffix == '.json':
        load_raw(source, require_vm)
        return source
    if source.suffix != '.npz':
        raise ValueError('Raw input must be an NPZ archive or a raw dataset JSON manifest')
    destination.mkdir(parents=True, exist_ok=False)
    required = {'theta', 'coords', 'time', 'case_names'} | ({'vm'} if require_vm else set())
    records = {}
    with zipfile.ZipFile(source) as archive:
        if not {k + '.npy' for k in required}.issubset(archive.namelist()):
            raise ValueError(f'Raw archive must contain {sorted(required)}')
        for key in sorted(required | ({'vm'} if 'vm.npy' in archive.namelist() else set())):
            path = destination / (key + '.npy')
            checksum = hashlib.sha256()
            with archive.open(key + '.npy') as src, path.open('xb') as out:
                for block in iter(lambda: src.read(8 * 1024 * 1024), b''):
                    out.write(block)
                    checksum.update(block)
            stat = path.stat()
            records[key] = dict(path=path.name, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                sha256=checksum.hexdigest())
    manifest = destination / 'dataset.json'
    write_json(manifest, dict(kind='raw_vm', version=1, source=str(source), arrays=records,
                             units=dict(vm='mV', time='ms'), cobiveco_order=['ab', 'rt', 'tm', 'tv']))
    load_raw(manifest, require_vm)
    return manifest


def load_prepared(manifest):
    manifest = Path(manifest).resolve()
    meta = json.loads(manifest.read_text())
    if meta.get('kind') != 'prepared_fold' or meta.get('version') != 1:
        raise ValueError('Expected a prepared fold manifest; use --cv for a multi-fold preparation')
    verify_records(manifest.parent, meta['artifacts'])
    raw_path = resolve(manifest, meta['raw_manifest'])
    if digest(raw_path) != meta['raw_manifest_sha256']:
        raise ValueError('Raw dataset manifest changed since preparation')
    data = load_raw(raw_path)
    data['targets'] = np.load(manifest.parent / 'targets.npy', mmap_mode='r', allow_pickle=False)
    split = json.loads((manifest.parent / 'splits.json').read_text())
    if data['targets'].shape != (len(data['theta']), len(data['coords']), meta['n_components'] + 1):
        raise ValueError('Prepared target dimensions disagree with raw data')
    return meta, data, split


def fold_manifests(manifest):
    manifest = Path(manifest).resolve()
    meta = json.loads(manifest.read_text())
    if meta.get('kind') == 'prepared_fold':
        return [manifest]
    if meta.get('kind') != 'prepared_dataset' or meta.get('status') != 'complete':
        raise ValueError('Expected a complete preparation manifest')
    paths = [resolve(manifest, item['path']) for item in meta['folds']]
    for path, item in zip(paths, meta['folds']):
        if digest(path) != item['sha256']:
            raise ValueError(f'Fold manifest changed: {path}')
    return paths
