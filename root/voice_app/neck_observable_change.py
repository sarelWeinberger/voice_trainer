"""Within-recording descriptive changes. No force, strain or effort mapping."""
import math
import numpy as np

VERSION = 'neck-observable-change-v1'
SIGNALS = {
    'head_pitch_deg': ('degrees', ('face',)),
    'head_roll_deg': ('degrees', ('face',)),
    'jaw_opening': ('eye_distance_ratio', ('face', 'jaw')),
    'shoulder_height': ('eye_distance_ratio', ('shoulders',)),
    'anterior_motion': ('eye_distances_per_second', ('anterior', 'submental')),
    'lateral_motion': ('eye_distances_per_second', ('left_lateral', 'right_lateral')),
}


def reference_duration(span):
    """Keep five seconds of reference for >=10s; retain comparison in short clips."""
    return 5.0 if span >= 10 else min(2.0, span / 3)


def compare_observable_windows(rows, audio=None, timing='unverified'):
    if len(rows) < 2:
        return dict(version=VERSION, status='insufficient_data', reference=None, windows=[],
                    limitations=['At least two timestamped frames required; no effort score'])
    first,last=float(rows[0]['t']),float(rows[-1]['t'])
    intervals=np.diff([row['t'] for row in rows])
    step=float(np.median(intervals[intervals>0])) if np.any(intervals>0) else 0
    end=last+min(step,.2)
    cut=first+reference_duration(end-first)

    def describe(start,stop):
        part=[row for row in rows if start<=row['t']<stop]
        estimates={}
        for key,(unit,regions) in SIGNALS.items():
            values=[]
            for row in part:
                value=row['features'].get(key)
                visibility=[row['visibility'].get(region,0) for region in regions]
                visible=max(visibility) if key=='lateral_motion' else min(visibility)
                if (isinstance(value,(int,float)) and math.isfinite(value) and
                        visible>=.25 and row.get('camera_stability',0)>=.5 and row.get('lighting',0)>=.3):
                    values.append(value)
            expected=max(1,round((stop-start)*5))
            coverage=min(1,len(values)/expected)
            # Missing samples never become zero; broad camera/visibility loss prevents comparison.
            usable=len(values)>=3 and coverage>=.5
            estimates[key]=dict(value=float(np.median(values)) if usable else None,
                iqr=float(np.percentile(values,75)-np.percentile(values,25)) if usable else None,
                unit=unit, valid_frames=len(values), expected_frames=expected,
                coverage=round(coverage,3), reason=None if usable else 'insufficient_observable_samples')
        active=None
        if audio is not None and timing!='unverified':
            overlapping=[a for a in audio if a['t']<stop and a['t']+a['duration']>start]
            if overlapping:
                active=any(a.get('speech_active',False) for a in overlapping)
        return dict(start=round(start,3),end=round(stop,3),frame_count=len(part),
                    speech_active=active,measurements=estimates)

    reference=describe(first,cut)
    reference.update(assumed_relaxed=False,kind='initial_recording_reference')
    windows=[];start=cut
    while start<end-1e-6:
        window=describe(start,min(start+2,end))
        for key,measurement in window['measurements'].items():
            baseline=reference['measurements'][key]['value']
            value=measurement['value']
            measurement['delta_from_reference']=value-baseline if value is not None and baseline is not None else None
            if baseline is None:measurement['reason']='reference_not_assessable'
            for field in ('value','iqr','delta_from_reference'):
                if measurement.get(field) is not None:measurement[field]=round(measurement[field],4)
        windows.append(window);start+=2
    for measurement in reference['measurements'].values():
        for field in ('value','iqr'):
            if measurement[field] is not None:measurement[field]=round(measurement[field],4)
    assessed=sum(m['delta_from_reference'] is not None for w in windows for m in w['measurements'].values())
    return dict(version=VERSION,status='complete' if assessed else 'insufficient_data',
        reference=reference,window_seconds=2,windows=windows,audio_timing=timing,
        effort_score=None,limitations=[
            'Initial reference is not known to be relaxed or physiologically normal',
            'Changes include ordinary articulation, head movement, camera and lighting confounds',
            'Local optical-flow speed is not contour prominence, muscle activation or effort',
            'Values and dispersion describe samples, not calibrated confidence or clinical thresholds',
            'Contour or static muscle shape changes are not directly measured by this module'])
