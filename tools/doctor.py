"""Check local paths and reference availability without accessing account credentials."""
import argparse
import importlib.util
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--verify-reference', action='store_true')
    args = parser.parse_args()
    if args.config:
        os.environ['KIMODO_PIPELINE_CONFIG'] = str(args.config.resolve())
    from tools.kimodo_pipeline import paths
    paths.settings.cache_clear()
    checks = {}
    for key in ('kimodo_repo', 'python', 'runtime_python', 'godot', 'dotnet'):
        path = paths.configured_path(key)
        checks[key] = dict(path=str(path) if path else None, passed=bool(path and path.exists()))
    checks['consumer'] = dict(path=str(paths.project_root()), passed=(paths.project_root()/'project.godot').is_file())
    checks['reference'] = dict(path=str(paths.reference_root()), passed=(paths.reference_root()/'reference.json').is_file())
    checks['production_root'] = dict(path=str(paths.asset_root()), passed=paths.asset_root().exists())
    for name in ('numpy', 'scipy', 'ufbx'):
        checks[name] = dict(passed=importlib.util.find_spec(name) is not None)
    if args.verify_reference:
        from tools.reference_snapshot import verify_reference
        checks['reference_bytes'] = verify_reference(paths.reference_root())
    passed = all(c['passed'] for c in checks.values())
    print(json.dumps(dict(passed=passed, checks=checks), ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
