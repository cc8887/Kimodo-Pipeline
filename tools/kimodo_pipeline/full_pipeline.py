"""Default production orchestration: delivered sources -> assets -> Godot checks."""
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from .graph import read_json
from .runner import write_json, file_hash
from .publish import ROOT, finalize_requests, validate_mantle_curve_delivery


def verify_runtime(pack, output, godot, dotnet, python, *, runtime_cases=None):
    from .godot_delivery import deploy_and_verify
    return deploy_and_verify(pack, output, godot, dotnet, python, runtime_cases=runtime_cases)


def complete_run(runner, authoring_result, *, pack=None, godot=None, dotnet=None, deps=None, runtime_cases=None, base_pack=None):
    from .offline_pipeline import convert_bundle
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    pipeline_scope=getattr(runner,'graph',{}).get('pipeline','non-prop')
    output=runner.output/'full-pipeline'
    if output.exists():raise ValueError('Full pipeline output exists; preserve evidence and choose a fresh run output')
    output.mkdir()
    receipt_path=output/'receipts.json';write_json(receipt_path,runner.results)
    source_receipts=output/'source-receipts.json';write_json(source_receipts,runner.results)
    assets=[dict(node=k,identity=r['identity'],outputs=r['outputs']) for k,r in runner.results.items() if r['op']=='deliver']
    for asset in assets:
        asset['sourceAuthoringReceipts']=dict(file=str(source_receipts),sha256=file_hash(source_receipts))
    new_deliveries=len(assets)
    bundle_path=output/'authoring-bundle.json'
    write_json(bundle_path,dict(assets=assets,productionReady=False,status=authoring_result['status']))
    if not assets:
        # With no accepted generated sources, a native-only fallback pack
        # cannot supply evidence about the failed generation. Keep the source
        # receipts and explicitly block export/runtime instead of reporting
        # vacuous zero-asset FBX success and running native fallback scenes.
        result=dict(authoring=authoring_result,status='pipeline_complete_rejected',
            activated=False,productionReady=False,failedPhase='source-quality',
            sourceDeliveries=0,evidence=str(output),
            remainingStageStatus='blocked: no accepted generated source deliveries')
        write_json(output/'result.json',result)
        return result
    deps=Path(deps or ROOT/'artifacts/kimodo_als/offline-python-deps')
    sys.path.insert(0,str(deps));sys.path.insert(0,str(runner.repo))
    pack=Path(pack or ROOT/'assets/generated'/('als_nonprop_'+('air_' if pipeline_scope=='air' else '')+stamp))
    bid=(authoring_result.get('viewerBatch') or {}).get('id')
    phase='conversion'
    try:
        if base_pack:
            phase='base-merge'
            from .derived import retain_base_deliveries
            assets,combined_receipts=retain_base_deliveries(base_pack,assets,dict(runner.results))
            write_json(receipt_path,combined_receipts)
            write_json(bundle_path,dict(assets=assets,productionReady=False,status=authoring_result['status'],
                newSourceDeliveries=new_deliveries,basePack=str(Path(base_pack).resolve())))
        phase='conversion'
        conversion=convert_bundle(bundle_path,receipt_path,output/'conversion',pack,batch_id=None if base_pack else bid)
        target_bid=conversion['viewer']['id'] if base_pack else bid
        write_json(pack/'pipeline-provenance.json',dict(batchId=None if base_pack else target_bid,
            targetViewerBatchId=target_bid if base_pack else None,authoringBatchId=bid,
            authoringBundle=str(bundle_path),pipeline='pure-python-full-v1',productionLane=pipeline_scope))
        phase='runtime'
        runtime=verify_runtime(pack/'resource-pack.json',output/'runtime',godot,dotnet,runner.python,
            runtime_cases=runtime_cases)
    except Exception as error:
        result=dict(authoring=authoring_result,status='pipeline_complete_rejected',activated=False,
            failedPhase=phase,error=str(error),evidence=str(output),
            remainingStageStatus='blocked by '+phase+' failure')
        write_json(output/'result.json',result)
        return result
    accepted=(authoring_result['status']=='authoring_complete' and runtime['status']=='runtime_passed' and
        conversion['sourceClosureComplete'] and not conversion['failedTargetGaits'] and
        not conversion['failedTargetMantles'] and not conversion.get('failedTargetAir',[]) and conversion['finalFbxReadback']['passed'])
    result=dict(authoring=authoring_result,conversion=conversion,runtime=runtime,
        status='pipeline_complete' if accepted else 'pipeline_complete_rejected',activated=False,
        newSourceDeliveries=new_deliveries,retainedSourceDeliveries=len(assets)-new_deliveries)
    if pipeline_scope=='air':
        expected={k for k,n in runner.graph['nodes'].items() if n['op']=='deliver'}
        result.update(productionLane='air',sourceScopeComplete={a['node'] for a in assets}==expected,
                      sourceScopeAssets=len(expected),productionReady=False)
    write_json(output/'result.json',result)
    return result
