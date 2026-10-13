"""Runtime additive baseline sampling used by final twist validation."""
import copy
import numpy as np


def sample_runtime_base(raw, frame, *, sequence_play_length=None, interpolation='Linear'):
    """Match Catalog ABPT_AnimFrame time and raw-pose shortest-path nlerp.

    UE computes base time as length * frame / KEY COUNT, not frame/FPS.
    Return a one-key absolute raw pose plus sampling evidence for the receipt.
    """
    count = raw['sampledKeyCount']
    if not isinstance(frame, int) or isinstance(frame, bool) or not 0 <= frame < count:
        raise ValueError('Runtime additive base frame outside source range')
    length = raw['playLength'] if sequence_play_length is None else sequence_play_length
    if not np.isfinite(length) or length < 0 or interpolation not in ('Linear', 'Step'):
        raise ValueError('Invalid runtime additive base evaluation policy')
    time = float(length)*frame/count
    position = time*raw['frameRateNumerator']/raw['frameRateDenominator']
    first = min(count-1, int(position))
    second = min(count-1, first+1)
    alpha = float(np.float32(position-int(position))) if interpolation == 'Linear' else 0.
    result = copy.deepcopy(raw)
    result.update(sampledKeyCount=1, playLength=0.)
    for track in result['tracks']:
        for field in ('positions', 'rotations', 'scales'):
            values = np.asarray(track[field], dtype=float)
            if len(values) <= 1:
                continue
            if len(values) != count:
                raise ValueError('Runtime base track sample count differs')
            a, b = values[first], values[second]
            if field == 'rotations':
                if a@b < 0:
                    b = -b
                value = a*(1-alpha)+b*alpha
                value /= np.linalg.norm(value)
            else:
                value = a*(1-alpha)+b*alpha
            track[field] = [value.tolist()]
    return result, dict(policy='UE-AnimFrame-key-count-time-shortest-path-nlerp', frame=frame,
        sequencePlayLengthSeconds=float(length), sourceSampledKeyCount=count, timeSeconds=time,
        firstKey=first, secondKey=second, alpha=alpha, interpolation=interpolation)
