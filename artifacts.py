"""Artifact identities, dated scratch outputs and run provenance."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo


class Tee:
    """Write progress to both the terminal and the run log."""

    def __init__(self, *handles):
        self.handles = handles

    def write(self, text):
        for handle in self.handles:
            handle.write(text)
            handle.flush()
        return len(text)

    def flush(self):
        for handle in self.handles:
            handle.flush()


@contextlib.contextmanager
def log_run(folder):
    """Keep stdout/stderr in logs/run.log while preserving console output."""
    log_folder = Path(folder) / 'logs'
    log_folder.mkdir()
    with (log_folder / 'run.log').open('w') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)):
            with contextlib.redirect_stderr(Tee(sys.stderr, log)):
                yield


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def artifact_records(folder, names):
    return {name: digest(Path(folder) / name) for name in names}


def verify_records(folder, records):
    for name, expected in records.items():
        if digest(Path(folder) / name) != expected:
            raise ValueError(f'Prepared artifact changed: {Path(folder) / name}')


def check_output_location(path):
    """Keep every generated array, checkpoint and figure outside home/source."""
    root = Path(path).expanduser().resolve()
    source = Path(__file__).resolve().parent
    if root == source or source in root.parents or str(root).startswith(('/home/', '/nfs/home/')):
        raise ValueError('Generated artifacts must be outside home/source; choose scratch storage')
    # Do not silently create a missing scratch mount or account directory.
    if str(root).startswith('/scratch/'):
        account = Path(*root.parts[:3])
        if not account.is_dir():
            raise ValueError(f'Scratch account directory is unavailable: {account}')
    return root


def new_directory(path):
    """Create one exact output folder; never silently overwrite earlier results."""
    path = check_output_location(path)
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(
            f'Output folder already exists: {path}. Choose another --name or --output.'
        ) from None
    return path


def experiment_directory(root, name=None):
    """Use a readable name when supplied; retain dated training/preparation otherwise."""
    if name is None:
        return dated_directory(root)
    return new_directory(Path(root) / name)


def checkpoint_directory(checkpoint):
    """Find the model's run/fold folder from checkpoints/best.pt."""
    parent = Path(checkpoint).expanduser().resolve().parent
    return parent.parent if parent.name == 'checkpoints' else parent


def resolve_prepared_manifest(saved_path, expected_sha256, replacement=None,
                              relative_path=None, anchor=None):
    """Locate the same fitted preparation after a move, checking its identity."""
    manifest = Path(replacement if replacement is not None else saved_path).expanduser().resolve()
    if replacement is None and not manifest.is_file() and relative_path and anchor:
        manifest = (Path(anchor) / relative_path).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(
            f'Preparation manifest not found: {manifest}\n'
            'If the preparation folder was moved, supply '
            '--prepared-manifest /new/location/fold_01/manifest.json. '
            'The original fitted decoder and raw dataset are required.'
        )
    if digest(manifest) != expected_sha256:
        raise ValueError('Saved features and preprocessing identity disagree; select the original preparation')
    return manifest


def dated_directory(root):
    root = check_output_location(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y-%m-%d_%H%M%S')
    index = 0
    while True:
        candidate = root / (stamp if index == 0 else f'{stamp}_{index:02d}')
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            index += 1


def record_invocation(folder, args):
    folder = Path(folder)
    write_json(folder / 'config.json', {k: v for k, v in vars(args).items() if k != 'mode'})
    try:
        revision = subprocess.check_output(
            ['git', '-C', str(Path(__file__).parent), 'rev-parse', '--verify', '--quiet', 'HEAD'],
            text=True).strip()
        dirty = bool(subprocess.check_output(
            ['git', '-C', str(Path(__file__).parent), 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    write_json(folder / 'provenance.json', dict(
        created_at=datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),
        command=sys.argv, cwd=os.getcwd(), mode=args.mode, git_revision=revision,
        git_dirty=dirty, python=sys.version, hostname=platform.node(),
        source_sha256={str(p.relative_to(Path(__file__).parent)): digest(p)
                       for p in sorted(Path(__file__).parent.rglob('*.py'))},
    ))
