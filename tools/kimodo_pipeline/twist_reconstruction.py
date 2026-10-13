"""Reconstruct ALS leaf twist bones from immutable segment axes.

The source SOMA rig has no twist joints. Proximal branches counter half of
their parent's axial roll; distal branches share half of the end joint's
axial roll. Only the eight leaf rotations change. This is a skinning policy,
not an attempt to copy the reference animation's authored twist tracks.
"""
import copy
import numpy as np
from scipy.spatial.transform import Rotation

POLICY = 'als-segment-swing-twist-half-v1'
ADDITIVE_POLICY = 'als-segment-swing-twist-half-relative-base-v1'
FRACTION = .5
ERROR_DEGREES = 1e-5
SEGMENTS = [('upperarm', 'lowerarm', True), ('lowerarm', 'hand', False),
            ('thigh', 'calf', True), ('calf', 'foot', False)]


class TwistReconstruction:
    def __init__(self, skeleton):
        names = skeleton['rawBoneNames']; parents = skeleton['rawParents']
        reference = skeleton['referencePose']; lookup = {n.lower(): i for i, n in enumerate(names)}
        # Native reference arrays can also contain evaluated virtual bones;
        # rawBoneNames/rawParents describe the physical prefix only.
        if len(lookup) != len(names) or len(parents) != len(names) or len(reference) < len(names):
            raise ValueError('Incomplete twist skeleton')
        if any(p < -1 or p >= i for i, p in enumerate(parents)):
            raise ValueError('Twist reconstruction requires parent-first bones')
        self.names = names; self.parents=parents; self.bones = []
        for side in ('l', 'r'):
            for main, end, proximal in SEGMENTS:
                bone = main + '_twist_01_' + side
                if any(n not in lookup for n in (bone, main + '_' + side, end + '_' + side)):
                    raise ValueError('Missing ALS twist segment: ' + bone)
                twist, parent, endpoint = (lookup[n] for n in (bone, main + '_' + side, end + '_' + side))
                if parents[twist] != parent or parents[endpoint] != parent or twist in parents:
                    raise ValueError('Twist must be a leaf sibling of its segment endpoint: ' + bone)
                local_axis = np.asarray(reference[endpoint]['position'], dtype=float)
                length = np.linalg.norm(local_axis)
                if not np.isfinite(length) or length < 1e-8:
                    raise ValueError('Zero or nonfinite twist segment: ' + bone)
                local_axis = local_axis / length
                driver = parent if proximal else endpoint
                rest = Rotation.from_quat(reference[driver]['rotation'])
                # Parent roll is expressed in its parent's space. Hand/foot
                # roll is already expressed in the segment parent's space.
                axis = rest.apply(local_axis) if proximal else local_axis
                self.bones.append(dict(bone=bone, index=twist, driver=driver, parent=parent,
                    proximal=proximal, axis=axis, localAxis=local_axis,
                    driverRest=rest.as_matrix(), rest=Rotation.from_quat(reference[twist]['rotation']).as_matrix()))

    def angles(self, local, *, unwrap=True, branches=None):
        local = np.asarray(local, dtype=float)
        if local.ndim != 4 or local.shape[1:] != (len(self.names), 3, 3) or not np.isfinite(local).all():
            raise ValueError('Invalid local rotations for twist reconstruction')
        angles = []
        for segment in self.bones:
            delta = local[:, segment['driver']] @ segment['driverRest'].T
            quaternion = Rotation.from_matrix(delta).as_quat()
            projection = quaternion[:, :3] @ segment['axis']
            if np.any(projection ** 2 + quaternion[:, 3] ** 2 < 1e-16):
                # A 180-degree swing has no defined axial twist. Do not hide
                # this with identity rotation or certify an arbitrary branch.
                raise ValueError('Undefined 180-degree swing at ' + segment['bone'])
            angle = 2 * np.arctan2(projection, quaternion[:, 3])
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            if unwrap: angle = np.unwrap(angle)
            angles.append(angle)
        angles = np.stack(angles, axis=1)
        if branches is not None:
            angles += 2 * np.pi * np.round((np.asarray(branches) - angles) / (2 * np.pi))
        return angles

    def matrices(self, local, *, unwrap=True, branches=None):
        local = np.asarray(local, dtype=float)
        angles = self.angles(local, unwrap=unwrap, branches=branches)
        result = local.copy()
        for i, segment in enumerate(self.bones):
            sign = -1 if segment['proximal'] else 1
            rotation = Rotation.from_rotvec(angles[:, i, None] * (sign * FRACTION) * segment['localAxis'])
            result[:, segment['index']] = rotation.as_matrix() @ segment['rest']
        return result

    def raw_matrices(self, raw):
        count = raw['sampledKeyCount']; tracks = {t['bone'].lower(): t for t in raw['tracks']}
        if count < 1 or len(tracks) != len(raw['tracks']): raise ValueError('Invalid twist tracks')
        return np.stack([Rotation.from_quat(np.broadcast_to(tracks[n.lower()]['rotations'], (count, 4))).as_matrix()
                         for n in self.names], axis=1)

    def assess(self, raw):
        local = self.raw_matrices(raw); expected = self.matrices(local); angles = self.angles(local)
        records = []
        for i, segment in enumerate(self.bones):
            actual = local[:, segment['index']]
            error = Rotation.from_matrix(actual.transpose(0, 2, 1) @ expected[:, segment['index']]).magnitude()
            relative = Rotation.from_matrix(actual @ segment['rest'].T).magnitude()
            steps = Rotation.from_matrix(actual[:-1].transpose(0, 2, 1) @ actual[1:]).magnitude()
            records.append(dict(bone=segment['bone'], driver=self.names[segment['driver']],
                role='proximal-counter-roll' if segment['proximal'] else 'distal-end-roll',
                fraction=FRACTION, segmentAxis=segment['localAxis'].tolist(),
                maximumDriverRollDegrees=float(np.max(np.abs(angles[:, i])) * 180 / np.pi),
                maximumLocalDeltaDegrees=float(np.max(relative) * 180 / np.pi),
                maximumStepDegrees=float(np.max(steps, initial=0) * 180 / np.pi),
                maximumConformanceErrorDegrees=float(np.max(error) * 180 / np.pi)))
        return dict(schemaVersion=1, policy=POLICY, bones=records, physicalTwistBones=len(records),
            sampledKeyCount=raw['sampledKeyCount'], playLength=raw['playLength'],
            passed=all(r['maximumConformanceErrorDegrees'] <= ERROR_DEGREES and
                       r['maximumStepDegrees'] <= 90 + ERROR_DEGREES for r in records),
            scope='eight physical leaf rotations; main joints, bind offsets, scale and timing preserved',
            productionReady=False)

    def relative_matrices(self, raw, runtime_base, correction_bone_roots=None):
        """Apply only reconstructed twist DELTAS to the actual sampled base.

        The runtime base can contain authored legacy twist that does not obey
        this reconstructor. Replacing it would create a nonidentity additive
        even when every main-joint correction is zero.
        """
        if runtime_base['sampledKeyCount']!=1:
            raise ValueError('Relative twist requires the actual one-key runtime base sample')
        local=self.raw_matrices(raw);base=self.raw_matrices(runtime_base)
        lookup={name.lower():index for index,name in enumerate(self.names)}
        owned=np.ones(len(self.names),dtype=bool)
        if correction_bone_roots is not None:
            if not isinstance(correction_bone_roots,(list,tuple)) or any(
                    not isinstance(name,str) or name.lower() not in lookup for name in correction_bone_roots):
                raise ValueError('Unknown relative twist ownership root')
            owned[:]=False;roots={lookup[name.lower()] for name in correction_bone_roots}
            for index,parent in enumerate(self.parents):
                owned[index]=index in roots or parent>=0 and owned[parent]
        base_angles=self.angles(base)
        angles=self.angles(local)
        # Pick the current sequence's initial half-twist branch relative to
        # its baseline, then keep the whole sequence unwrapped consistently.
        angles+=2*np.pi*np.round((base_angles-angles[:1])/(2*np.pi))
        expected=local.copy()
        for index,segment in enumerate(self.bones):
            bone=segment['index']
            if owned[bone]:
                sign=-1 if segment['proximal'] else 1
                difference=(angles[:,index]-base_angles[0,index])*(sign*FRACTION)
                delta=Rotation.from_rotvec(difference[:,None]*segment['localAxis']).as_matrix()
                expected[:,bone]=delta@base[0,bone]
            else:
                expected[:,bone]=base[0,bone]
        return expected,owned,angles-base_angles

    def assess_relative(self, raw, runtime_base, correction_bone_roots=None):
        from .root_ownership import semantic_hash
        local=self.raw_matrices(raw)
        expected,owned,driver_delta=self.relative_matrices(raw,runtime_base,correction_bone_roots)
        records=[]
        for index,segment in enumerate(self.bones):
            actual=local[:,segment['index']]
            error=Rotation.from_matrix(actual.transpose(0,2,1)@expected[:,segment['index']]).magnitude()
            steps=Rotation.from_matrix(actual[:-1].transpose(0,2,1)@actual[1:]).magnitude()
            records.append(dict(bone=segment['bone'],driver=self.names[segment['driver']],
                owned=bool(owned[segment['index']]),fraction=FRACTION,
                maximumDriverDeltaDegrees=float(np.max(np.abs(driver_delta[:,index]))*180/np.pi),
                maximumStepDegrees=float(np.max(steps,initial=0)*180/np.pi),
                maximumConformanceErrorDegrees=float(np.max(error)*180/np.pi)))
        return dict(schemaVersion=1,policy=ADDITIVE_POLICY,reconstructionPolicy=POLICY,
            evaluationDomain='relative-to-actual-runtime-additive-base',
            baselineSampleSemanticSha256=semantic_hash(runtime_base),
            baselineAbsoluteConformancePassed=self.assess(runtime_base)['passed'],
            correctionBoneRoots=None if correction_bone_roots is None else list(correction_bone_roots),
            bones=records,physicalTwistBones=len(records),sampledKeyCount=raw['sampledKeyCount'],playLength=raw['playLength'],
            passed=all(row['maximumConformanceErrorDegrees']<=ERROR_DEGREES and
                       row['maximumStepDegrees']<=90+ERROR_DEGREES for row in records),
            scope='owned twist uses reconstructed delta times actual base; unowned twist preserves actual base; no absolute baseline repair claimed',
            productionReady=False)


def reconstruct(raw, skeleton):
    plan = TwistReconstruction(skeleton); local = plan.matrices(plan.raw_matrices(raw))
    result = copy.deepcopy(raw); tracks = {t['bone'].lower(): t for t in result['tracks']}
    for segment in plan.bones:
        quaternion = Rotation.from_matrix(local[:, segment['index']]).as_quat()
        # Keep adjacent keys in one quaternion hemisphere as well as unwrap
        # axial angles before multiplying by the half-twist fraction.
        for frame in range(1, len(quaternion)):
            if quaternion[frame] @ quaternion[frame - 1] < 0: quaternion[frame] *= -1
        tracks[segment['bone']]['rotations'] = quaternion.tolist()
    return result, plan.assess(result)


def reconstruct_relative(raw,skeleton,runtime_base,correction_bone_roots=None):
    plan=TwistReconstruction(skeleton)
    local,_,_=plan.relative_matrices(raw,runtime_base,correction_bone_roots)
    result=copy.deepcopy(raw);tracks={track['bone'].lower():track for track in result['tracks']}
    for segment in plan.bones:
        quaternion=Rotation.from_matrix(local[:,segment['index']]).as_quat()
        for frame in range(1,len(quaternion)):
            if quaternion[frame]@quaternion[frame-1]<0:quaternion[frame]*=-1
        tracks[segment['bone']]['rotations']=quaternion.tolist()
    return result,plan.assess_relative(result,runtime_base,correction_bone_roots)
