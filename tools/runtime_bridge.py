"""Run the consumer adapter in its native OS, including WSL -> Windows handoff."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('project', 'reference', 'pack', 'output', 'godot', 'dotnet'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--runtime-case', action='append')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    if args.config:
        os.environ['KIMODO_PIPELINE_CONFIG'] = str(args.config.resolve())
    os.environ.update(KIMODO_PIPELINE_ASSET_ROOT=str(args.project.resolve()),
                      KIMODO_PIPELINE_PROJECT_ROOT=str(args.project.resolve()),
                      KIMODO_PIPELINE_REFERENCE_ROOT=str(args.reference.resolve()),
                      KIMODO_PIPELINE_HOME=str(root))
    from tools.kimodo_pipeline.godot_adapter import verify_runtime
    result = verify_runtime(args.pack, args.output, args.godot, args.dotnet, sys.executable,
                            runtime_cases=args.runtime_case)
    raise SystemExit(0 if result['status'] == 'runtime_passed' else 1)


if __name__ == '__main__':
    main()
