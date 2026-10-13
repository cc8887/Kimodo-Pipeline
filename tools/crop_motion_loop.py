"""Save one diagnostic locomotion loop without overwriting or approving assets."""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.kimodo_pipeline import motion,gait
from tools.kimodo_pipeline.graph import read_json
from tools.kimodo_pipeline.loop_segment import crop_loop,tile_loop
from tools.kimodo_pipeline.loop_seam import repair_loop_seam
from tools.kimodo_pipeline.runner import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion',type=Path,required=True)
    parser.add_argument('--rig',type=Path,required=True)
    parser.add_argument('--start-frame',type=int,required=True)
    parser.add_argument('--end-frame',type=int,required=True,help='Inclusive endpoint')
    parser.add_argument('--speed',type=float,required=True)
    parser.add_argument('--seam-frames',type=int,default=3)
    parser.add_argument('--pose-only',action='store_true',help='Close pose; retain existing tangents and validate them')
    parser.add_argument('--seam-anchor',choices=('first','last'),default='last')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output exists; choose a new evidence directory')
    rig=read_json(args.rig);source=motion.load_motion(args.motion,rig)
    source_sha=hashlib.sha256(args.motion.read_bytes()).hexdigest()
    loop,report=crop_loop(source,rig,args.start_frame,args.end_frame,seam_frames=args.seam_frames,
                          close=args.pose_only,match_tangents=False,include_correction_paths=True)
    if not args.pose_only:
        loop,seam=repair_loop_seam(loop,rig,window_frames=args.seam_frames,anchor=args.seam_anchor,
                                  prior_paths=report['correctionPaths'],
                                  playback_displacement=report['removedPlanarTrajectory']['displacement'],
                                  expected_speed_mps=args.speed)
        report['finalSeamRepair']=seam
    tiled=tile_loop(loop,rig,displacement=report['removedPlanarTrajectory']['displacement'])
    audit=gait.audit_candidate(tiled,rig,args.speed)
    args.output.mkdir(parents=True,exist_ok=False)
    motion.save_motion(args.output/'single-loop.npz',loop,rig)
    motion.save_motion(args.output/'playback.npz',tiled,rig)
    report.update(inputFile=str(args.motion.resolve()),inputSha256=source_sha,
        playbackAuthoringAudit=audit,status='source_checks_passed' if audit['pass'] else 'source_checks_rejected',
        remainingStages='target retarget, root/pelvis, curves, FBX and Godot not executed by this diagnostic command')
    write_json(args.output/'loop-crop.json',report)
    assert hashlib.sha256(args.motion.read_bytes()).hexdigest()==source_sha
    print(report['status'],str(args.output/'single-loop.npz'))
    return 0 if audit['pass'] else 2


if __name__=='__main__':raise SystemExit(main())
