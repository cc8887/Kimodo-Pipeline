"""Generate and refine a Mantle style, then run conversion, FBX, import and Godot gates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--rules',type=Path,help='Existing modular Mantle rules; required for a new generation')
    parser.add_argument('--semantic',choices=('Mantle_High','Mantle_Low_Left','Mantle_Low_Right'))
    parser.add_argument('--output',type=Path,required=True,help='Fresh run root, or existing authoring folder with --resume')
    parser.add_argument('--base-pack',type=Path)
    parser.add_argument('--max-feedback-rounds',type=int,default=8,help='Combined source and target feedback budget')
    parser.add_argument('--device',choices=('cpu','cuda:0'))
    parser.add_argument('--runtime-case',action='append')
    parser.add_argument('--resume',action='store_true',help='Verify existing author receipts and continue the full pipeline')
    args=parser.parse_args()
    if args.config:
        os.environ['KIMODO_PIPELINE_CONFIG']=str(args.config.resolve())
    from tools.kimodo_pipeline.mantle_refinement import refine,resume
    if args.resume:
        if args.rules or args.semantic or args.device:
            parser.error('--resume uses the recorded graph; omit rules/semantic/device overrides')
        result=resume(args.output,base_pack=args.base_pack,runtime_cases=args.runtime_case)
    else:
        if not args.rules or not args.semantic:
            parser.error('New generation requires --rules and --semantic')
        result=refine(rules_path=args.rules,semantic=args.semantic,output=args.output,
            max_feedback_rounds=args.max_feedback_rounds,base_pack=args.base_pack,
            runtime_cases=args.runtime_case,device=args.device)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['status']!='pipeline_complete':
        raise SystemExit(1)


if __name__=='__main__':
    main()
