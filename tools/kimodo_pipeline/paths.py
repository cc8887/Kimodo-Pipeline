"""Explicit production/reference/consumer roots, shared by CLI and subprocesses."""
from functools import lru_cache
import os
from pathlib import Path
import subprocess
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

PIPELINE_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def settings():
    configured = os.environ.get('KIMODO_PIPELINE_CONFIG')
    path = Path(configured) if configured else PIPELINE_ROOT / 'pipeline.local.toml'
    if configured and not path.is_file():
        raise ValueError('Pipeline config does not exist: ' + str(path))
    if not path.is_file():
        return {}
    return tomllib.loads(path.read_text(encoding='utf-8-sig')).get('paths', {})


def configured_path(key, default=None):
    value = settings().get(key)
    if value is None:
        return Path(default).resolve() if default is not None else None
    value = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(value)
    if not path.is_absolute():
        path = PIPELINE_ROOT / path
    return path.resolve()


def asset_root():
    override = os.environ.get('KIMODO_PIPELINE_ASSET_ROOT')
    return Path(override).resolve() if override else configured_path('asset_root', PIPELINE_ROOT / 'workspace/als')


def reference_root():
    override = os.environ.get('KIMODO_PIPELINE_REFERENCE_ROOT')
    if override:
        return Path(override).resolve()
    return configured_path('reference_root', PIPELINE_ROOT / 'workspace/als')


def project_root():
    override = os.environ.get('KIMODO_PIPELINE_PROJECT_ROOT')
    if override:
        return Path(override).resolve()
    return configured_path('project_root', os.environ.get('GODOT_ALS_ROOT') or PIPELINE_ROOT.parent / 'Godot-Locomotion')


def work_root():
    return configured_path('work_root', PIPELINE_ROOT / 'workspace/runs')


def contract_path():
    return PIPELINE_ROOT / 'docs/architecture/kimodo-als-asset-contracts.json'


def windows_path(path):
    """Translate shared WSL files at the Windows runtime boundary only."""
    path = str(Path(path).resolve())
    if os.name == 'nt':
        return path
    result = subprocess.run(['wslpath', '-w', path], check=True, capture_output=True, text=True)
    return result.stdout.strip()
