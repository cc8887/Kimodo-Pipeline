"""Bounded neighbouring-key repair with explicit sampled interpolation evidence.

This is finite sampling of actual local quaternion interpolation and target LBS,
not a continuous collision proof. All budgets refer to the original raw input.
"""
from __future__ import annotations
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from .mantle import _arm_rig, _hand_frames
from .gait_constraints import solve_foot_ik
from .mantle_curve_contract import TRANSITION_LIFT_CM

POLICY='mantle-local-nlerp-skin-samples-v2'
SAMPLE_STEP=.025
SOLVE_STEP=.1
FINGERS=('thumb','index','middle','ring','pinky')


def nlerp(first,last,weight):
    shape=first.shape
    q0=Rotation.from_matrix(first.reshape(-1,3,3)).as_quat().reshape(*shape[:-2],4)
    q1=Rotation.from_matrix(last.reshape(-1,3,3)).as_quat().reshape(*shape[:-2],4)
    bias=np.where(np.sum(q0*q1,axis=-1)<0.,-1.,1.)
    q=q0*(1-weight[...,None])+q1*(bias*weight)[...,None]
    q/=np.linalg.norm(q,axis=-1)[...,None]
    return Rotation.from_quat(q.reshape(-1,4)).as_matrix().reshape(shape)


def sampled_local(rotations,frames):
    frames=np.asarray(frames);lo=np.floor(frames+1e-9).astype(int);hi=np.minimum(lo+1,len(rotations)-1)
    return nlerp(rotations[lo],rotations[hi],np.broadcast_to((frames-lo)[:,None],(len(frames),rotations.shape[1])))


def pareto_escape(points,height,clearance=.2):
    """Minimum translation in the front/top union for fixed hand geometry."""
    front=points[:,1]-50.+clearance;top=np.maximum(0.,height-points[:,2]+clearance)
    candidates=[]
    for outward in np.unique(np.r_[0.,front[front>0]]):
        upward=float(top[front>outward+1e-9].max(initial=0.))
        candidates.append(np.array([0.,-outward,upward]))
    return min(candidates,key=lambda vector:float(vector@vector))


class SampledSkin:
    def __init__(self,skeleton,offsets,skin,height):
        self.names=[n.lower() for n in skeleton['rawBoneNames']]
        self.parents=skeleton['rawParents'];self.offsets=offsets;self.skin=skin;self.height=height
        self.lookup={n:i for i,n in enumerate(self.names)}
        self.contacts={};self.control_side=None;self.last_control_residuals={}

    def world(self,rotations,frames,*,controlled=False):
        frames=np.asarray(frames);lo=np.floor(frames+1e-9).astype(int);hi=np.minimum(lo+1,len(rotations)-1)
        alpha=frames-lo
        # Actual AlsRawSequencePoseSampler.Blend uses shortest-arc nlerp.
        local=sampled_local(rotations,frames)
        offset=self.offsets[lo]*(1-alpha[:,None,None])+self.offsets[hi]*alpha[:,None,None]
        gr=np.empty_like(local);gp=np.empty(offset.shape)
        for i,p in enumerate(self.parents):
            gr[:,i]=local[:,i] if p<0 else gr[:,p]@local[:,i]
            gp[:,i]=offset[:,i] if p<0 else gp[:,p]+np.einsum('fij,fj->fi',gr[:,p],offset[:,i])
        if controlled and self.control_side is not None:
            start,end,anchor_r,anchor_p=self.contacts[self.control_side]
            weight=np.minimum(np.clip(frames-start+1,0,1),np.clip(end+1-frames,0,1))
            active=np.flatnonzero(weight>0)
            if len(active):
                suffix=self.control_side[0]
                upper,lower,hand=[self.lookup[b+'_'+suffix] for b in ('upperarm','lowerarm','hand')]
                hip,knee,wrist=[gp[active,i] for i in (upper,lower,hand)]
                target=wrist+(anchor_p-wrist)*weight[active,None]
                target[:,2]+=4.*TRANSITION_LIFT_CM*weight[active]*(1.-weight[active])
                # AlsMantlingHandSolver uses the same normalized linear blend.
                palm=nlerp(gr[active,hand],np.broadcast_to(anchor_r,gr[active,hand].shape),weight[active])
                a=np.linalg.norm(knee-hip,axis=1);b=np.linalg.norm(wrist-knee,axis=1)
                direction=target-hip;distance=np.linalg.norm(direction,axis=1)
                direction/=distance[:,None]
                # Match Core AlsTwoBoneIk's no-stretch outer reach clamp.
                physical_distance=np.minimum(distance,a+b)
                residual=np.abs(distance-physical_distance)
                self.last_control_residuals=dict(maximumPartialReachResidualCm=float(residual[weight[active]<1.-1e-7].max(initial=0.)),
                    maximumFullSupportReachResidualCm=float(residual[weight[active]>=1.-1e-7].max(initial=0.)))
                distance=physical_distance;target=hip+direction*distance[:,None]
                pole=knee-hip-direction*np.sum((knee-hip)*direction,axis=1)[:,None]
                lengths=np.linalg.norm(pole,axis=1)
                for at in np.flatnonzero(lengths<1e-8):
                    basis=np.eye(3)[np.argmin(np.abs(direction[at]))]
                    pole[at]=basis-direction[at]*np.dot(basis,direction[at])
                pole/=np.linalg.norm(pole,axis=1)[:,None]
                along=np.clip((a*a-b*b+distance*distance)/(2*distance),-a,a)
                bend=np.sqrt(np.maximum(0,a*a-along*along))
                solved=hip+direction*along[:,None]+pole*bend[:,None]
                def between(source,target):
                    source=source/np.linalg.norm(source,axis=1)[:,None];target=target/np.linalg.norm(target,axis=1)[:,None]
                    cross=np.cross(source,target);sine=np.linalg.norm(cross,axis=1);cosine=np.clip(np.sum(source*target,axis=1),-1,1)
                    vectors=cross*np.divide(np.arctan2(sine,cosine),sine,out=np.ones_like(sine),where=sine>1e-12)[:,None]
                    for at in np.flatnonzero((sine<1e-12)&(cosine<0)):
                        axis=np.cross(source[at],np.eye(3)[np.argmin(np.abs(source[at]))]);vectors[at]=axis/np.linalg.norm(axis)*np.pi
                    return Rotation.from_rotvec(vectors).as_matrix()
                arm_r=between(knee-hip,solved-hip)@gr[active,upper]
                forearm_r=between(wrist-knee,target-solved)@gr[active,lower]
                local[active,upper]=gr[active,self.parents[upper]].swapaxes(-1,-2)@arm_r
                local[active,lower]=arm_r.swapaxes(-1,-2)@forearm_r
                local[active,hand]=forearm_r.swapaxes(-1,-2)@palm
                for i,p in enumerate(self.parents):
                    gr[:,i]=local[:,i] if p<0 else gr[:,p]@local[:,i]
                    gp[:,i]=offset[:,i] if p<0 else gp[:,p]+np.einsum('fij,fj->fi',gr[:,p],offset[:,i])
        return gr,gp

    def points(self,rotations,frames,selected,*,controlled=True):
        gr,gp=self.world(rotations,frames,controlled=controlled);points=np.zeros((len(frames),len(selected),3))
        for bone,indices,local,weights in self.skin.subset(selected):
            i=self.lookup[bone]
            points[:,indices]+=(np.einsum('fij,vj->fvi',gr[:,i],local)+gp[:,i,None,:])*weights[None,:,None]
        return points

    def clearances(self,rotations,frames,selected,*,controlled=True):
        points=self.points(rotations,frames,selected,controlled=controlled)
        return np.maximum(points[:,:,2]-self.height,50.-points[:,:,1])


def refine(rotations,original,offsets,skeleton,spec,play_length,skin,reference_hash,*,twist_plan=None,telemetry=None):
    started=perf_counter();rotations=rotations.copy();count=len(rotations)
    sampler=SampledSkin(skeleton,offsets,skin,spec['heightCm']);lookup=sampler.lookup
    sample_frames=np.linspace(0,count-1,round((count-1)/SAMPLE_STEP)+1)
    original_world=sampler.world(original,np.arange(count));limit=np.deg2rad(45.)
    for side_name in spec['hands']:
        keys=_hand_frames(spec,side_name,count,play_length);hand=lookup['hand_'+side_name[0]]
        sampler.contacts[side_name]=(int(keys[0]),int(keys[-1]),
            Rotation.from_matrix(original_world[0][keys,hand]).mean().as_matrix(),original_world[1][keys,hand].mean(axis=0))
    aliases=[]
    for name in sampler.names:
        alias=name
        for side,suffix in (('Left','l'),('Right','r')):
            for bone,end in (('upperarm','Arm'),('lowerarm','ForeArm'),('hand','Hand')):
                if name==bone+'_'+suffix:alias=side+end
        aliases.append(alias)
    attempts=[];before={};solver_calls=0;solver_iterations=0
    for side_name in spec['hands']:
        sampler.control_side=side_name
        side=side_name[0];hand='hand_'+side;selected=skin.sides[side]
        contact=set(_hand_frames(spec,side_name,count,play_length).tolist())
        values=sampler.clearances(rotations,sample_frames,selected).min(axis=1)
        before[side_name]=max(0.,-float(values.min()))
        # At most two independent failed intervals per hand and three arm
        # choices each. Failure is returned to the authoring feedback loop.
        for iteration in range(2):
            values=sampler.clearances(rotations,sample_frames,selected).min(axis=1)
            worst=int(np.argmin(values))
            if values[worst]>=-2.-1e-7:break
            interval=min(int(sample_frames[worst]),count-2)
            possible=[key for key in (interval+1,interval) if key not in contact]
            best=rotations;best_depth=max(0.,-float(values.min()))
            choices=[(None,np.zeros(3))] if not possible else [(possible[0],np.array([0.,-1.,0.])),
                                  (possible[0],np.array([0.,0.,1.])),
                                  (possible[-1],np.array([0.,0.,1.]))]
            if possible:
                key=possible[0];baseline_pose=rotations.copy()
                arm=[lookup[stem+'_'+side] for stem in ('upperarm','lowerarm','hand')]
                baseline_pose[key,arm]=original[key,arm]
                if twist_plan is not None:baseline_pose=twist_plan.matrices(baseline_pose)
                escape=pareto_escape(sampler.points(baseline_pose,[key],selected,controlled=False)[0],spec['heightCm'])
                norm=float(np.linalg.norm(escape))
                if escape[1]<-1e-6 and escape[2]>1e-6:
                    # Use geometric corner escape before the axial fallbacks.
                    choices.insert(0,(key,escape/max(10.,norm)))
                    choices=choices[:3]
            for key,direction in choices:
                keys=sorted(set([interval,interval+1]+([key] if key is not None else [])))
                first=max(0,min(keys)-1);last=min(count-1,max(keys)+1)
                solve_frames=np.linspace(first,last,round((last-first)/SOLVE_STEP)+1)
                check_frames=np.linspace(first,last,round((last-first)/SAMPLE_STEP)+1)
                candidate=rotations.copy()
                if key is not None:
                    rig=dict(jointNames=aliases,parents=sampler.parents,offsets=(offsets[key]/100).tolist())
                    wi=lookup[hand];wrist=original_world[1][key,wi];orientation=original_world[0][key,wi]
                    try:
                        candidate[key]=solve_foot_ik(candidate[key],offsets[key,0]/100,_arm_rig(rig),side_name,
                            (wrist+direction*10.)/100,orientation,{})
                    except ValueError:continue
                if twist_plan is not None:candidate=twist_plan.matrices(candidate)
                row=dict(side=side_name,failedInterval=[interval,interval+1],armKey=key,
                    escapeDirection=direction.tolist(),fingerKeys=keys,solvers=[])
                # Minimize original-relative finger change while constraining
                # the adjacent intervals together, including their neighbours.
                for finger in FINGERS:
                    finger_vertices=skin.fingers[finger+'_'+side]
                    if sampler.clearances(candidate,check_frames,finger_vertices).min()>=-1.8:continue
                    indices=[lookup[f'{finger}_{part:02}_{side}'] for part in (1,2,3)]
                    baseline=original[np.ix_(keys,indices)].copy()
                    initial=Rotation.from_matrix((baseline.swapaxes(-1,-2)@candidate[np.ix_(keys,indices)]).reshape(-1,3,3)).as_rotvec().reshape(-1)
                    def proposed(x):
                        pose=candidate.copy()
                        pose[np.ix_(keys,indices)]=baseline@Rotation.from_rotvec(x.reshape(-1,3)).as_matrix().reshape(len(keys),3,3,3)
                        return pose
                    fixed_clearance=sampler.clearances(candidate,[first,last],finger_vertices)
                    def constraints(x):
                        required=np.full((len(solve_frames),len(finger_vertices)),goal)
                        # Neighbour keys are immutable during this local solve.
                        # Preserve their already-accepted depth instead of
                        # demanding a stronger margin they cannot change.
                        if first not in keys:
                            a=np.clip(solve_frames-first,0.,1.)
                            endpoint=np.maximum(-2.,np.minimum(goal,fixed_clearance[0]))
                            required=np.minimum(required,endpoint[None,:]*(1-a[:,None])+goal*a[:,None])
                        if last not in keys:
                            a=np.clip(last-solve_frames,0.,1.)
                            endpoint=np.maximum(-2.,np.minimum(goal,fixed_clearance[1]))
                            required=np.minimum(required,endpoint[None,:]*(1-a[:,None])+goal*a[:,None])
                        return np.r_[limit**2-np.sum(x.reshape(-1,3)**2,axis=1),
                            (sampler.clearances(proposed(x),solve_frames,finger_vertices)-required).reshape(-1)]
                    before_whole=max(0.,-float(sampler.clearances(candidate,check_frames,selected).min()))
                    for goal,maxiter in ((.2,25),(-1.5,45)):
                        fitted=minimize(lambda x:float(x@x),initial,method='SLSQP',constraints=[dict(type='ineq',fun=constraints)],
                            bounds=[(-limit,limit)]*len(initial),options=dict(ftol=1e-8,maxiter=maxiter))
                        solver_calls+=1;solver_iterations+=int(fitted.nit)
                        admissible=bool(np.isfinite(fitted.x).all() and np.min(constraints(fitted.x))>=-1e-5)
                        if admissible:
                            proposal=proposed(fitted.x)
                            after_whole=max(0.,-float(sampler.clearances(proposal,check_frames,selected).min()))
                            admissible=after_whole<=before_whole+1e-7
                            if admissible:candidate=proposal
                        row['solvers'].append(dict(finger=finger,requestedClearanceCm=goal,accepted=admissible,iterations=int(fitted.nit)))
                        if admissible:break
                checked=sampler.clearances(candidate,check_frames,selected).min(axis=1)
                depth=max(0.,-float(checked.min()));row['maximumSampledDepthCm']=depth
                row['accepted']=bool(depth<best_depth-1e-6)
                attempts.append(row)
                if row['accepted']:best=candidate;best_depth=depth
                if depth<=2.-1e-5:break
            if best is rotations:break
            rotations=best
    # Integer-safe wrist shifts can overshoot by a fraction of a millimetre
    # under local nlerp. Contract their original-relative target vectors by
    # the measured overshoot, then recheck actual FK. Never reset the budget.
    original_sample_world=sampler.world(original,sample_frames);budget_projections=[]
    for side_name in spec['hands']:
        wi=lookup['hand_'+side_name[0]];contact=set(_hand_frames(spec,side_name,count,play_length).tolist())
        for iteration in range(2):
            observed=sampler.world(rotations,sample_frames)
            displacement=np.linalg.norm(observed[1][:,wi]-original_sample_world[1][:,wi],axis=1)
            maximum=float(displacement.max())
            if maximum<=10.000001:break
            scale=(10.-.002)/maximum;at_keys=sampler.world(rotations,np.arange(count));affected=[]
            for key in range(count):
                delta=at_keys[1][key,wi]-original_world[1][key,wi]
                if key in contact or np.linalg.norm(delta)<1e-6:continue
                rig=dict(jointNames=aliases,parents=sampler.parents,offsets=(offsets[key]/100).tolist())
                rotations[key]=solve_foot_ik(rotations[key],offsets[key,0]/100,_arm_rig(rig),side_name,
                    (original_world[1][key,wi]+delta*scale)/100,original_world[0][key,wi],{})
                affected.append(key)
            if twist_plan is not None:rotations=twist_plan.matrices(rotations)
            budget_projections.append(dict(side=side_name,observedSubframeWristCorrectionCm=maximum,
                originalRelativeTargetScale=scale,interiorReserveCm=.002,affectedKeys=affected))
    # Independent whole-clip verification includes all unchanged neighbours.
    final_world=sampler.world(rotations,np.arange(count));rows=[]
    final_sample_world=sampler.world(rotations,sample_frames)
    for side_name in spec['hands']:
        sampler.control_side=side_name
        side=side_name[0];hand=lookup['hand_'+side]
        c=sampler.clearances(rotations,sample_frames,skin.sides[side]).min(axis=1);worst=int(np.argmin(c))
        control_residuals=dict(sampler.last_control_residuals)
        fingers=[lookup[f'{f}_{p:02}_{side}'] for f in FINGERS for p in (1,2,3)]
        angles=Rotation.from_matrix((sampled_local(original[:,fingers],sample_frames).swapaxes(-1,-2)@
            sampled_local(rotations[:,fingers],sample_frames)).reshape(-1,3,3)).magnitude()
        displacement=np.linalg.norm(final_sample_world[1][:,hand]-original_sample_world[1][:,hand],axis=1)
        key_displacement=np.linalg.norm(final_world[1][:,hand]-original_world[1][:,hand],axis=1)
        orientation=Rotation.from_matrix(original_world[0][:,hand].swapaxes(-1,-2)@final_world[0][:,hand]).magnitude()
        support=_hand_frames(spec,side_name,count,play_length)
        rows.append(dict(side=side_name,beforeDepthCm=before[side_name],maximumDepthCm=max(0.,-float(c[worst])),**control_residuals,
            rawUncontrolledMaximumDepthCm=max(0.,-float(sampler.clearances(rotations,sample_frames,skin.sides[side],controlled=False).min())),
            worstFrame=float(sample_frames[worst]),maximumCumulativeWristCorrectionCm=float(displacement.max()),
            maximumSupportWristCorrectionCm=float(key_displacement[support].max(initial=0.)),
            maximumWristOrientationChangeDegrees=float(np.rad2deg(orientation.max())),
            maximumCumulativeFingerCorrectionDegrees=float(np.rad2deg(angles.max()))))
    passed=all(r['maximumDepthCm']<=2.+1e-7 and r['maximumCumulativeWristCorrectionCm']<=10.00001
        and r['maximumFullSupportReachResidualCm']<=.001
        and r['maximumSupportWristCorrectionCm']<=1e-5 and r['maximumWristOrientationChangeDegrees']<=1e-5
        and r['maximumCumulativeFingerCorrectionDegrees']<=45.00001 for r in rows)
    report=dict(policy=POLICY,originalReferenceSha256=reference_hash,sampleStepFrames=SAMPLE_STEP,
        sampleCount=len(sample_frames),intervals=count-1,interpolation='shortest-arc normalized quaternion lerp; linear local translation; actual target LBS',
        contactEvaluation='same one-key acquisition/release weights; lerp wrist target and nlerp palm; fixed-length two-bone IK with explicit reach residual; raw uncontrolled depth retained',
        transitionLiftCm=TRANSITION_LIFT_CM,
        previewLimitations='nominal geometry before actor height/progress transport and consumer twist adjustment; mixed arm/hand skin may differ',
        mandatoryRuntimeGates=['full-skin-depth-2cm','full-and-partial-contact-reach-0.1cm'],
        scope='finite subframe samples; not a continuous collision proof',maximumDepthCm=2.,passed=passed,
        cumulativeBudgetSampling='original vs final uncontrolled raw at every sampled subframe; runtime IK excluded',
        assetBudgetSampling=dict(reference='original-raw-subframes',sampleStepFrames=SAMPLE_STEP,runtimeIKIncluded=False),
        hands=rows,attempts=attempts,subframeBudgetProjections=budget_projections,solverCalls=solver_calls,solverIterations=solver_iterations)
    if telemetry is not None:telemetry.update(temporalSeconds=perf_counter()-started,temporalSolverCalls=solver_calls,temporalSolverIterations=solver_iterations)
    return rotations,report
