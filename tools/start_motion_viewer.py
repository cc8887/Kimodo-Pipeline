"""Start the read-only Viewer with independent production and consumer roots."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--port', type=int, default=10015)
    parser.add_argument('--bind', default='127.0.0.1')
    args = parser.parse_args()
    if args.config:
        os.environ['KIMODO_PIPELINE_CONFIG'] = str(args.config.resolve())
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from tools.kimodo_pipeline.paths import asset_root, project_root, settings
    settings.cache_clear()
    spec = importlib.util.spec_from_file_location('motion_viewer_server', root/'tools/motion-viewer/server.py')
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    server.PROJECT_ROOT = str(project_root())
    server.ASSETS_ROOT = os.path.realpath(project_root()/'assets/generated')
    server.REGISTRY_ROOT = os.path.realpath(asset_root()/'artifacts/kimodo_viewer')
    server.PRODUCTION_ASSETS_ROOT = os.path.realpath(asset_root()/'assets/generated')
    server.PORT, server.BIND = args.port, args.bind
    server.main()


if __name__ == '__main__':
    main()
