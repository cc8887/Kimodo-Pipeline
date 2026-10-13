"""Run the unchanged target Mantle gates before committing to export/runtime work."""
from __future__ import annotations
import sys

from .graph import digest,read_json
from .paths import reference_root,configured_path
from .runner import file_hash

POLICY='mantle-target-preflight-v1'


def check_targets(runner,semantic):
    # Generation and offline FBX tooling may live in separate configured
    # environments. Use the same offline dependency path as complete_run.
    dependencies=configured_path('offline_deps')
    if dependencies is not None and str(dependencies) not in sys.path:
        sys.path.insert(0,str(dependencies))
    from .mantle import reference_input,target
    from .offline_retarget import convert
    from .mantle_hand_repair import immutable_target_skin,repair as repair_hands
    native,reference_skeleton,reference_identity=reference_input(semantic)
    v4=read_json(reference_root()/'assets/config/v4_recovery_movement_source_inputs.json')['skeletons'][0]
    skeletons=dict(refactored=reference_skeleton,v4=v4)
    rig=runner.rig();checks=[]
    for key,receipt in runner.results.items():
        if receipt['op']!='deliver':
            continue
        contract=runner.graph['nodes'][key]['params']['contract']
        if contract['family']!='mantle' or contract['semanticName']!=semantic:
            continue
        motion_entry=receipt['outputs']['motion.npz']
        if file_hash(motion_entry['path'])!=motion_entry['sha256']:
            raise ValueError('Delivered source motion changed before target preflight')
        skeleton=skeletons[contract['edition']]
        raw,retarget=convert(runner.motion(key),rig,skeleton,contract['source'],contract['originalTiming'])
        repaired,_,_,assessment=target(raw,skeleton,semantic)
        skin_timing={}
        _,hand_skin=repair_hands(repaired,skeleton,semantic,immutable_target_skin(skeleton),reconstruct_twist=True,telemetry=skin_timing)
        assessment['handSkinRepair']=hand_skin
        assessment['checks'].append(dict(check='boundedTargetHandSkinRepair',passed=hand_skin['passed']))
        assessment['passed']=assessment['passed'] and hand_skin['passed']
        checks.append(dict(node=key,source=contract['source'],edition=contract['edition'],passed=assessment['passed'],
            sourceMotionSha256=motion_entry['sha256'],targetSkeletonSha256=digest(skeleton),
            angularRetargetPolicy=retarget['policy'],assessment=assessment,skinRepairTelemetry=skin_timing))
    return dict(schemaVersion=1,policy=POLICY,semanticName=semantic,reference=reference_identity,
        passed=bool(checks) and all(check['passed'] for check in checks),
        checks=checks,acceptedTargets=sum(check['passed'] for check in checks),
        scope='target contact/body and bounded hand-skin repair gates after actual retarget; final Root/FBX and runtime still required',
        deliverable=False,productionReady=False)


def failure_score(preflight):
    ratios=[]
    for target in preflight['checks']:
        for check in target['assessment']['checks']:
            if check['passed']:
                continue
            if check.get('limit',0)>0:
                ratios.append(float(check['value'])/float(check['limit']))
            elif check.get('minimum',0)>0:
                ratios.append(float(check['minimum'])/max(float(check['value']),1e-12))
            else:
                # Skin reports provide physical finite values and unchanged
                # limits; an aggregate boolean is not an infinite JSON score.
                skin=target['assessment'].get('handSkinRepair',{})
                if check['check']=='boundedTargetHandSkinRepair' and (skin.get('records') or skin.get('temporalSkin',{}).get('hands')):
                    ratios.extend(max(row['afterDepthCm'],row['transitionAnchorDepthCm'])/skin['maximumDepthCm']
                                  for row in skin.get('records',[]))
                    for row in skin.get('temporalSkin',{}).get('hands',[]):
                        ratios.extend((row['maximumDepthCm']/skin['maximumDepthCm'],
                            row['maximumCumulativeWristCorrectionCm']/skin['maximumArmCorrectionCm'],
                            row['maximumCumulativeFingerCorrectionDegrees']/skin['maximumFingerCorrectionDegrees'],
                            row['maximumFullSupportReachResidualCm']/.001))
                else:
                    ratios.append(2.)
    return max(ratios,default=0.)
