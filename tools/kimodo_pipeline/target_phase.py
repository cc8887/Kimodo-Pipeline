"""Infer periodic target gait phases before grounded-contact validation."""
import numpy as np
from scipy.signal import find_peaks

from .target_gait import PILOTS, DIRECTIONS, asset_name, _sole_geometry

POLICY = 'target-vertical-extrema-phase-v1'


def extend_trailing_low_support(raw, skeleton, source, contacts):
    """At most one extra IK-intent key, corroborated by measured toe height."""
    name = asset_name(source)
    mask = np.asarray(contacts, dtype=bool)
    count = raw['sampledKeyCount']
    if name not in PILOTS or mask.shape != (count, 2) or not np.array_equal(mask[0], mask[-1]):
        raise ValueError('Invalid periodic support intent for extension')
    points, _, _ = _sole_geometry(raw, skeleton, PILOTS[name][0], DIRECTIONS[name])
    heights = points[:-1, :, 1, 2]
    result = mask[:-1].copy()
    for side in range(2):
        stable = mask[:-1, side]
        if not stable.any() or not np.isfinite(heights[:, side]).all():
            raise ValueError('Missing measurable low support for extension')
        adjacent = np.roll(stable, 1) & ~stable
        adjacent &= heights[:, side] <= np.max(heights[stable, side])+1.
        if stable.sum()+adjacent.sum() <= count-4:
            result[:, side] |= adjacent
    return np.concatenate([result, result[:1]])


def infer_verified_source_phases(raw, skeleton, source, source_gate, *, contact_mode):
    """Project independently measured source FK phases, solely as IK intent.

    Callers must validate the receipt's source bytes before entering here.
    Target geometry must corroborate swing/low bands; final target gates remain
    independent and do not consume this mask as a contact approval.
    """
    name = asset_name(source)
    count = raw['sampledKeyCount']; period = count-1
    fps = source_gate.get('fps', 0)
    if (name not in PILOTS or period < 8 or not source_gate.get('pass')
            or source_gate.get('outputFrameCount') != count
            or not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0
            or abs(period/fps-raw['playLength']) > 1e-6
            or abs(source_gate.get('expectedSpeedMps', -1)*100-PILOTS[name][0]) > 1e-6):
        raise ValueError('Source phase repair needs an accepted same-time/speed FK gate')
    if contact_mode not in ('toe', 'flat_sole'):
        raise ValueError('Unknown source phase contact mode')
    samples = np.asarray(source_gate.get('transformedDetection', {}).get('sampleContacts'))
    if samples.shape != (count, 2, 2) or not np.isin(samples, (0, 1)).all():
        raise ValueError('Invalid measured source FK contacts')
    contacts = samples[:, :, 1].astype(bool) if contact_mode == 'toe' else samples.astype(bool).all(axis=-1)
    if not np.array_equal(contacts[0], contacts[-1]):
        raise ValueError('Source phase endpoints differ')
    points, _, _ = _sole_geometry(raw, skeleton, PILOTS[name][0], DIRECTIONS[name])
    heights = points[:-1, :, 1, 2] if contact_mode == 'toe' else points[:-1, :, :, 2].mean(axis=2)
    starts, sides = [], []
    for side, label in enumerate(('left', 'right')):
        mask = contacts[:-1, side]
        edges = np.flatnonzero(mask & ~np.roll(mask, 1))
        if len(edges) != 1 or mask.sum() < 3 or mask.sum() > period-3:
            raise ValueError('Source support must have one bounded stable window: '+label)
        signal = heights[:, side]
        excursion = float(np.ptp(signal))
        if (not np.isfinite(signal).all() or excursion < 1.
                or np.max(signal[mask]) > np.min(signal)+.3*excursion):
            raise ValueError('Target vertical geometry contradicts source support: '+label)
        starts.append(int(edges[0]))
        sides.append(dict(side=label, windowFrames=np.flatnonzero(mask).tolist(), excursionCm=excursion,
                          measuredHeightCm=signal.tolist()))
    gap = abs(starts[0]-starts[1])/period
    if not .35 <= gap <= .65:
        raise ValueError('Source support windows do not alternate')
    return dict(policy='verified-source-fk-target-phase-v1', passed=True, source=source,
        purpose='bounded IK intent only; target acceptance remeasures geometry',
        sourceMotionSha256=source_gate.get('outputMotionSha256'),
        independentOfAuthoredContactCurves=True, targetVerticalGeometryCorroborated=True,
        sampledKeyCount=count, playLengthSeconds=raw['playLength'], contactMode=contact_mode,
        sides=sides, troughPhaseGap=gap, candidateContacts=contacts.astype(int).tolist())


def infer_phases(raw, skeleton, source, *, harmonics=3, low_fraction=.1,
                 minimum_excursion_cm=1., minimum_window_seconds=.06, contact_mode='flat_sole'):
    name = asset_name(source)
    if name not in PILOTS:
        raise ValueError('No periodic phase contract: ' + name)
    count = raw['sampledKeyCount']; period = count - 1
    length = float(raw['playLength'])
    if period < 8 or length <= 0:
        raise ValueError('Insufficient periodic phase samples')
    if (not 0 < low_fraction < .5 or not isinstance(harmonics, int)
            or isinstance(harmonics, bool) or not 1 <= harmonics <= period // 2):
        raise ValueError('Invalid phase smoothing/band parameters')
    points, velocity, _ = _sole_geometry(raw, skeleton, PILOTS[name][0], DIRECTIONS[name])
    if contact_mode not in ('flat_sole','toe'):
        raise ValueError('Unknown phase contact mode: '+contact_mode)
    heights = points[:-1, :, 1, 2] if contact_mode=='toe' else points[:-1, :, :, 2].mean(axis=2)
    if not np.isfinite(heights).all():
        raise ValueError('Non-finite vertical phase signal')
    contacts = np.zeros((count, 2), dtype=bool); sides = []; offsets = []
    for side, label in enumerate(('left', 'right')):
        signal = heights[:, side]
        spectrum = np.fft.rfft(signal); spectrum[harmonics+1:] = 0
        smooth = np.fft.irfft(spectrum, n=period)
        amplitude = float(np.ptp(smooth))
        if amplitude < minimum_excursion_cm:
            raise ValueError('No measurable swing excursion: ' + label)
        extrema = {}
        for kind, values in (('maximum', smooth), ('minimum', -smooth)):
            indices, props = find_peaks(np.tile(values, 3),
                prominence=max(.2 * amplitude, .25), distance=max(1, period//4))
            ranked = sorted([(float(prom), int(i-period)) for i, prom in
                             zip(indices, props['prominences']) if period <= i < 2*period], reverse=True)
            if not ranked or len(ranked) > 1 and ranked[1][0] >= .6 * ranked[0][0]:
                raise ValueError('Ambiguous vertical ' + kind + ': ' + label)
            extrema[kind] = ranked[0][1]
        trough = extrema['minimum']
        low = smooth <= smooth[trough] + low_fraction * amplitude
        if contact_mode=='toe':
            # A forefoot may stay low through a fast toe-off. Restrict its
            # repair window by travel-corrected speed, without a ground plane.
            low &= np.linalg.norm(velocity[:-1,side,1,:2],axis=-1)<=25.
            if not low[trough]:
                raise ValueError('Forefoot trough has no slow support candidate: '+label)
        first = last = trough
        while low[(first-1) % period] and trough-first < period-1:
            first -= 1
        while low[(last+1) % period] and last-first < period-1:
            last += 1
        if (last-first) * length/period < minimum_window_seconds or last-first >= period-1:
            raise ValueError('Unresolved vertical low phase: ' + label)
        frames = np.arange(first, last+1) % period
        offsets.append((first-trough, last-trough))
        sides.append(dict(side=label, troughFrame=trough, peakFrame=extrema['maximum'],
            excursionCm=amplitude, windowStartFrame=first % period,
            windowEndFrame=last % period, windowFrames=frames.tolist(),
            windowSeconds=(last-first)*length/period, smoothedHeightCm=smooth.tolist(),
            measuredHeightCm=signal.tolist()))
    gap = abs(sides[0]['troughFrame']-sides[1]['troughFrame'])/period
    if not .35 <= gap <= .65:
        raise ValueError('Vertical extrema do not alternate: ' + str(gap))
    # Use the overlap of the two measured low bands relative to each trough.
    # Different plateau widths must not shift one foot's inferred landing by
    # several frames when the extrema themselves alternate consistently.
    first = max(v[0] for v in offsets); last = min(v[1] for v in offsets)
    if (last-first)*length/period < minimum_window_seconds:
        raise ValueError('No common reliable trough-relative phase window')
    for side, row in enumerate(sides):
        row['measuredLowBandFrames'] = row['windowFrames']
        frames = (row['troughFrame']+np.arange(first, last+1)) % period
        contacts[frames, side] = True
        row.update(windowStartFrame=int(frames[0]), windowEndFrame=int(frames[-1]),
                   windowFrames=frames.tolist(), windowSeconds=(last-first)*length/period)
    contacts[-1] = contacts[0]
    return dict(policy=POLICY, passed=True, purpose='phase candidates for IK, not grounded-contact acceptance',
        source=source, sampledKeyCount=count, playLengthSeconds=length,
        independentOfGroundHeight=True, independentOfAuthoredContactCurves=True,
        smoothingHarmonics=harmonics, lowBandFraction=low_fraction,contactMode=contact_mode,
        minimumExcursionCm=minimum_excursion_cm, minimumWindowSeconds=minimum_window_seconds,
        troughPhaseGap=gap, windowPolicy='common-trough-relative-measured-low-band',
        sides=sides, candidateContacts=contacts.astype(int).tolist())
