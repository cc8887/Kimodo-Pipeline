"""Copy checked production payloads to a fresh consumer version and run its adapter."""
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

from .paths import PIPELINE_ROOT, project_root, reference_root, configured_path, windows_path
from .runner import file_hash, write_json


def pack_root(pack_path):
    pack_path = Path(pack_path).resolve()
    if pack_path.parent.parent.name != 'generated' or pack_path.parent.parent.parent.name != 'assets':
        raise ValueError('Pack must be under an explicit assets/generated root')
    if not pack_path.parent.name.startswith('als_nonprop_'):
        raise ValueError('Unexpected ALS production pack name')
    return pack_path.parents[3]


def checked_payloads(pack_path):
    pack_path = Path(pack_path).resolve()
    root = pack_root(pack_path)
    pack = json.loads(pack_path.read_text(encoding='utf-8'))
    files = []
    for entry in pack['files']:
        uri = entry['path']
        if not uri.startswith('res://'):
            raise ValueError('Expected a resource URI in the compatibility manifest')
        path = (root / uri.removeprefix('res://')).resolve()
        if not path.is_relative_to(pack_path.parent):
            raise ValueError('Pack payload escapes its version directory: ' + uri)
        if not path.is_file() or file_hash(path) != entry['sha256']:
            raise ValueError('Pack payload changed: ' + uri)
        files.append(dict(path=path.relative_to(pack_path.parent).as_posix(), sha256=entry['sha256']))
    return pack, files


def write_portable_bundle(pack_path):
    pack_path = Path(pack_path).resolve()
    pack, files = checked_payloads(pack_path)
    bundle = dict(schemaVersion=1, target='als', status=pack['status'], productionReady=False,
                  compatibilityManifest=dict(path=pack_path.name, sha256=file_hash(pack_path)),
                  files=files, sourceClosureComplete=pack.get('sourceClosureComplete', False),
                  missingSources=pack.get('missingNonPropSources', []),
                  finalFbxReadback=pack.get('offlineFbxReadback'),
                  referenceSha256=file_hash(reference_root()/'reference.json'))
    write_json(pack_path.parent/'bundle.json', bundle)
    return bundle


def deploy_pack(pack_path, project, output):
    pack_path, project, output = Path(pack_path).resolve(), Path(project).resolve(), Path(output).resolve()
    checked_payloads(pack_path)
    relative = pack_path.relative_to(pack_root(pack_path))
    destination = project / relative
    if destination.parent.exists():
        raise ValueError('Consumer version already exists; preserve it and select a fresh version')
    if not (project/'project.godot').is_file():
        raise ValueError('Consumer project.godot is missing')
    shutil.copytree(pack_path.parent, destination.parent)
    checked_payloads(destination)
    if file_hash(destination) != file_hash(pack_path):
        raise ValueError('Deployment changed compatibility manifest bytes')
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(schemaVersion=1, sourcePack=str(pack_path), sourcePackSha256=file_hash(pack_path),
                   deployedPack=str(destination), deployedPackSha256=file_hash(destination),
                   payloadsVerified=True, activated=False)
    write_json(output/'deployment-receipt.json', receipt)
    return destination


def deploy_and_verify(pack, output, godot, dotnet, python, *, runtime_cases=None):
    output = Path(output).resolve()
    if (output/'runtime-result.json').exists():
        raise ValueError('Runtime evidence already exists; use a fresh output directory')
    destination = deploy_pack(pack, project_root(), output)
    runtime_python = configured_path('runtime_python', python or sys.executable)
    if not runtime_python.is_file():
        raise ValueError('Runtime Python is missing: ' + str(runtime_python))
    win_bridge = requires_windows_bridge(runtime_python)
    translate = windows_path if win_bridge else lambda p: str(Path(p).resolve())
    if win_bridge:
        adapt_windows_metadata(destination, output, translate)
    command = [str(runtime_python), translate(PIPELINE_ROOT/'tools/runtime_bridge.py'),
               '--project', translate(project_root()), '--reference', translate(reference_root()),
               '--pack', translate(destination), '--output', translate(output),
               '--godot', translate(godot), '--dotnet', translate(dotnet)]
    config = os.environ.get('KIMODO_PIPELINE_CONFIG')
    if config:
        command += ['--config', translate(config)]
    for case in runtime_cases or ():
        command += ['--runtime-case', case]
    log = output/'adapter.log'
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=PIPELINE_ROOT, stdout=stream, stderr=subprocess.STDOUT)
    report = output/'runtime-result.json'
    if not report.is_file():
        raise ValueError('Godot adapter failed without a runtime receipt; inspect ' + str(log))
    runtime = json.loads(report.read_text(encoding='utf-8'))
    if result.returncode and runtime.get('status') == 'runtime_passed':
        raise ValueError('Adapter exit status conflicts with its runtime receipt')
    runtime.update(sourcePackSha256=file_hash(pack), deploymentReceipt=str(output/'deployment-receipt.json'))
    write_json(report, runtime)
    return runtime


def requires_windows_bridge(runtime_python):
    return os.name != 'nt' and Path(runtime_python).suffix.lower() == '.exe'


def adapt_windows_metadata(destination, output, translate):
    """Adapt WSL absolute evidence paths without rewriting animation payloads."""
    destination = Path(destination).resolve()
    pack = json.loads(destination.read_text(encoding='utf-8'))
    source = pack.get('nativeRetargetBundle')
    # A Windows-authored diagnostic replay already contains usable Windows paths.
    if not source or not PurePosixPath(source).is_absolute():
        return
    native = json.loads(Path(source).read_text(encoding='utf-8'))

    def remap(value, key=''):
        if isinstance(value, dict):
            return {k: remap(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [remap(v, key) for v in value]
        path_field = key in ('file','path','raw','fbx','authoringBundle') or key.endswith(('Path','File','Directory','Bundle'))
        if isinstance(value, str) and path_field and PurePosixPath(value).is_absolute():
            return translate(value)
        return value

    metadata = destination.parent/'deployment/retarget-bundle.json'
    metadata.parent.mkdir(parents=True, exist_ok=True)
    write_json(metadata, remap(native))
    pack['nativeRetargetBundle'] = translate(metadata)
    uri = 'res://'+metadata.relative_to(pack_root(destination)).as_posix()
    pack['files'].append(dict(path=uri,sha256=file_hash(metadata)))
    write_json(destination, pack)
    checked_payloads(destination)
    receipt = Path(output)/'deployment-receipt.json'
    saved = json.loads(receipt.read_text(encoding='utf-8'))
    saved.update(deployedPackSha256=file_hash(destination),
                 nativeMetadataSourceSha256=file_hash(source),
                 nativeMetadataDeployedSha256=file_hash(metadata),
                 pathAdapter='wsl-to-windows-evidence-v1')
    write_json(receipt, saved)
