"""Controlled gate/threshold sensitivity; no video, provider or accuracy labels."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neck_v1_scoring import DIMENSIONS, ModelOutput, NeckEffortConfig, aggregate


def run():
    cases = [('supported_zero',0,True,1.,1.,.95,0.),
             ('clear_independent_change',2,True,1.,1.,.95,0.),
             ('head_jaw_ambiguous_change',1,False,1.,1.,.95,0.),
             ('low_quality_short_speech_change',1,True,.5,.498,.6,0.),
             ('anterior_jaw_fit_lateral_independent',1,True,1.,1.,.17,.95)]
    results=[]
    for case,score,independent,quality,temporal,confidence,fraction in cases:
        output=ModelOutput.model_validate(dict(dimensions={key:dict(observable=True,
            score=score,continuous_score=float(score),confidence=confidence,
            evidence=['תיאור מבוקר לבדיקת תוכנה'],frame_indices=[0,1],speech_related=True,
            independent_local_activity=independent,confounds=[]) for key in DIMENSIONS},
            summary='בדיקת תוכנה מבוקרת',limitations=[]))
        ctx=dict(selected_indices=[0,1],dimension_quality={key:quality for key in DIMENSIONS},
            temporal_quality=temporal,speech_status='active',jaw_explained_fraction=fraction,
            quality={'overall':quality,'warnings':[]},versions={})
        for threshold in (.15,.10,.05):
            result=aggregate(output,ctx,NeckEffortConfig(min_dimension_confidence=threshold))
            results.append(dict(case=case,threshold=threshold,
                dimensions={name:dict(score=result['dimensions'][name]['score'],
                    reason=result['diagnostics']['dimensions'][name]['rejection_reason'],
                    adjusted_confidence=result['diagnostics']['dimensions'][name].get('adjusted_confidence'))
                    for name in DIMENSIONS[1:4]}))
    return dict(method='Controlled model outputs, not provider predictions or accuracy validation',
                production_confidence_threshold=.15,results=results)


if __name__=='__main__':
    destination=Path(__file__).with_name('threshold_audit_2026_10_08.json')
    destination.write_text(json.dumps(run(),ensure_ascii=False,indent=2))
    print('Controlled threshold audit written to',destination)
