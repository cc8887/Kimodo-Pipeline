"""Export original ALS inputs byte-for-byte into an isolated, ignored reference root."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from tools.kimodo_pipeline.paths import project_root, reference_root

SOURCE_FILES = ('src/Als.Core/Locomotion/AlsRefactoredStopRule.cs',
                'src/Als.Import/Compilation/AlsRefactoredStopEvaluators.cs')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export_reference(project, output):
    project, output = Path(project).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Reference directory exists; preserve it and choose a new version')
    if output.is_relative_to(project) or project.is_relative_to(output):
        raise ValueError('Reference snapshot must be outside the consumer project')
    roots = ('assets/config', 'assets/generated/als_v4')
    for relative in (*roots, *SOURCE_FILES):
        if not (project / relative).exists():
            raise ValueError('Required original export is missing: ' + relative)
    output.mkdir(parents=True)
    for relative in roots:
        shutil.copytree(project / relative, output / relative,
                        ignore=shutil.ignore_patterns('als_resource_pack.json', 'compiled', '*.import', '__pycache__'))
    for relative in SOURCE_FILES:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / relative, destination)
    contracts = 'docs/architecture/kimodo-als-asset-contracts.json'
    destination = output / contracts
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(project / contracts, destination)
    files = []
    for path in sorted(p for p in output.rglob('*') if p.is_file()):
        relative = path.relative_to(output).as_posix()
        actual = sha256(path)
        if actual != sha256(project / relative):
            raise ValueError('Copy differs from original bytes: ' + relative)
        files.append(dict(path=relative, sha256=actual, size=path.stat().st_size))
    report = dict(schemaVersion=1, files=files, bytes=sum(f['size'] for f in files))
    (output / 'reference.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return report


def verify_reference(output):
    output = Path(output).resolve()
    report = json.loads((output / 'reference.json').read_text(encoding='utf-8'))
    for entry in report['files']:
        path = (output / entry['path']).resolve()
        if not path.is_relative_to(output) or not path.is_file() or sha256(path) != entry['sha256']:
            raise ValueError('Reference payload changed or escaped root: ' + entry['path'])
    return dict(passed=True, files=len(report['files']), bytes=report['bytes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=project_root())
    parser.add_argument('--output', type=Path, default=reference_root())
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    result = verify_reference(args.output) if args.verify else export_reference(args.project, args.output)
    print(json.dumps(dict(files=len(result['files']), bytes=result['bytes']) if isinstance(result.get('files'), list) else result))


if __name__ == '__main__':
    main()
