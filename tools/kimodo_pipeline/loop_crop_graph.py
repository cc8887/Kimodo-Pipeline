"""Optional single-cycle authoring in the ordinary production DAG."""
from __future__ import annotations

import copy


def configure(nodes, rules):
    crops = rules.get('loopCrops', {})
    if not isinstance(crops, dict):
        raise ValueError('loopCrops must map semantic motion names to crop parameters')
    allowed = {'source', 'contactReference', 'startFrame', 'endFrameInclusive', 'seamFrames',
        'harmonics', 'contactPaddingFrames', 'contactRampFrames', 'contactMinimumKneeBendDegrees',
        'jointContactSeam', 'jointMaximumEvaluations', 'jointSeamWeight', 'jointSupportWeight',
        'jointOptimizeSharedAnchor', 'jointOptimizePelvis', 'jointNearGroundWeight', 'seamAnchor'}
    for semantic, spec in crops.items():
        key = 'motion:'+semantic
        master = nodes.get(key)
        if not master or master['op'] != 'gait_cycle' or not isinstance(spec, dict) or set(spec)-allowed:
            raise ValueError('Loop crop needs a gait-cycle master and known parameters: '+semantic)
        params = copy.deepcopy(spec)
        source = params.setdefault('source', master['params']['source'])
        if source not in nodes or source == key:
            raise ValueError('Unknown or cyclic loop crop source: '+source)
        start, end = params.get('startFrame'), params.get('endFrameInclusive')
        period = master['params']['targetCount']-1
        if type(start) is not int or type(end) is not int or start < 0 or end-start != period:
            raise ValueError('Loop crop span must match the native cycle time domain')
        dependencies = ['rig:soma', source]
        reference = params.get('contactReference')
        if reference:
            node = nodes.get(reference)
            if not node or node['op'] != 'native_gait_reference' or not node['params'].get('nativePhaseEvidence'):
                raise ValueError('Loop contact reference needs independent native phase evidence')
            dependencies.append(reference)
        if params.get('jointContactSeam') and not reference:
            raise ValueError('Joint loop repair requires a verified contact reference')
        params.update(expectedSpeedMps=master['params']['expectedSpeedMps'], finalSeamRepair=True)
        for field in ('postureValidationIdle', 'postureProfile'):
            if field in master['params']:
                params[field] = copy.deepcopy(master['params'][field])
        if params.get('postureValidationIdle'):
            dependencies.append(params['postureValidationIdle'])
        crop = 'loop-crop:'+semantic
        if crop in nodes:
            raise ValueError('Loop crop node already exists: '+crop)
        nodes[crop] = dict(id=crop, op='loop_crop', deps=list(dict.fromkeys(dependencies)), params=params)
        old_source = master['params']['source']
        master['deps'] = list(dict.fromkeys([crop if dependency == old_source else dependency for dependency in master['deps']]))
        master['params'].update(source=crop, postureAlreadyFinal=True)
