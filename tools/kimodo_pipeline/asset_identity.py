"""Original inventory direction mapping, without Unreal dependencies."""
def direction(n):
    # SOMA native zero heading: LeftLeg has +X, RightLeg has -X.
    for label,vec in [('Left_Backward',[.707107,-.707107]),('Right_Backward',[-.707107,-.707107]),
                      ('Left_Forward',[.707107,.707107]),('Right_Forward',[-.707107,.707107]),
                      ('Forward',[0,1]),('Backward',[0,-1]),('Left',[1,0]),('Right',[-1,0])]:
        if label in n:return label,vec
    return 'Forward',[0,1]
