"""Descriptive temporal image motion, not an effort grade or tremor diagnosis."""
import json
import math
import numpy as np

VERSION='head-movement-stability-v3'

def parse_trace(payload):
    if not isinstance(payload,str) or len(payload.encode())>128*1024:
        raise ValueError('Invalid motion payload size')
    data=json.loads(payload)
    if not isinstance(data,dict) or data.get('version') not in ('head-motion-trace-v1','head-motion-trace-v2') or data.get('source')!='browser_patch_tracking':
        raise ValueError('Invalid motion trace')
    rows=data.get('samples')
    if not isinstance(rows,list) or len(rows)>900:raise ValueError('Invalid motion samples')
    previous=-1;previous_segment=0;columns=7 if data['version']=='head-motion-trace-v2' else 6
    for row in rows:
        if not isinstance(row,list) or len(row)!=columns or any(type(v) not in (int,float) or not math.isfinite(v) for v in row):
            raise ValueError('Invalid motion sample')
        if not previous<row[0]<=25 or row[0]<0 or any(abs(v)>2 for v in row[1:5]) or not 0<=row[5]<=1:
            raise ValueError('Invalid motion sample range')
        previous=row[0]
        if columns==7:
            if type(row[6]) is not int or not previous_segment<=row[6]<=900:
                raise ValueError('Invalid motion segment')
            previous_segment=row[6]
    return rows

def analyze_payload(payload):
    """Keep bounded acquisition diagnostics alongside derived measurements."""
    rows=parse_trace(payload)
    metadata=json.loads(payload)
    result=analyze_trace(rows)
    reason=metadata.get('capture_reason')
    if reason in ('video_frame_callback_unavailable','face_reference_unavailable','image_tracking_failed'):
        result['capture_reason']=reason
    diagnostics=metadata.get('tracking_diagnostics')
    if isinstance(diagnostics,dict):
        cleaned={}
        for field in ('head_missing_samples','background_missing_samples','background_rejected_samples'):
            value=diagnostics.get(field)
            if type(value) is int and 0<=value<=len(rows):cleaned[field]=value
        for field,limit in (('remaining_head_points',16),('remaining_background_points',12),('reinitializations',900),('reference_checks',900),('reference_corrections',900)):
            value=diagnostics.get(field)
            if type(value) is int and 0<=value<=limit:cleaned[field]=value
        result['tracking_diagnostics']=cleaned
    return result

def _detrend(t, y, duration=None):
    vectors=[np.ones(len(t)),t-np.mean(t)]
    if duration is not None:
        vectors += [wave(2*np.pi*k*t/duration) for k in (1,2) for wave in (np.sin,np.cos)]
    orthogonal=[];clean=y.copy()
    for vector in vectors:
        for q in orthogonal:vector=vector-q*float(q@vector)
        norm=float(np.linalg.norm(vector))
        if norm<1e-9:continue
        q=vector/norm;orthogonal.append(q);clean=clean-q[:,None]*(q@clean)
    return clean


def _fit_frequency(t, y, frequency, duration=None):
    phase=2*np.pi*frequency*t
    basis=_detrend(t,np.column_stack((np.sin(phase),np.cos(phase))),duration)
    gram=basis.T@basis
    if np.linalg.det(gram)<1e-12:return None
    coefficients=np.linalg.solve(gram,basis.T@y)
    gain=float(np.sum(coefficients[:,:2]*(basis.T@y)[:,:2]))
    return gain,coefficients


def analyze_trace(rows):
    result=dict(version=VERSION,status='insufficient_data',source='browser_patch_tracking',
        unit='percent_image_width',amplitude_kind='supported_sinusoid_rms',effort_score=None,windows=[],sample_count=len(rows),
        limitations=['Image periodicity, not muscle tension or diagnosis',
                    'Background consensus cannot establish that scenery is stationary',
                    'Periodicity gates are engineering controls, not calibrated clinical confidence'])
    if len(rows)<24:return dict(result,reason='too_few_motion_samples')
    a=np.array(rows,float);fps=1/np.median(np.diff(a[:,0]))
    result['effective_fps']=round(float(fps),3)
    if fps>60:return dict(result,reason='motion_sampling_invalid')
    if fps<12:return dict(result,reason='motion_sampling_too_slow')
    start=a[0,0];end=a[-1,0]
    # Three observed cycles are required: 2s windows cover >=1.5Hz, 6s
    # windows >=0.5Hz and 10s windows >=0.3Hz. Never infer above sampling support.
    for duration in (2,6,10):
      if end-start<duration-.1:continue
      for lower in np.arange(start,end-duration+.1,1 if duration==2 else 2):
        part=a[(a[:,0]>=lower)&(a[:,0]<lower+duration)]
        window=dict(start=round(float(lower),3),end=round(float(lower+duration),3),status='insufficient_data',
                    amplitude_percent_image_width=None,dominant_frequency_hz=None,background_amplitude_percent_image_width=None)
        result['windows'].append(window)
        usable=part[part[:,5]>=.5];coverage=len(usable)/max(1,len(part))
        span=float(usable[-1,0]-usable[0,0]) if len(usable)>1 else 0
        max_gap=float(np.max(np.diff(usable[:,0]))) if len(usable)>1 else None
        temporal=len(np.unique(np.floor((usable[:,0]-lower)/.1)))/round(duration*10) if len(usable) else 0
        window.update(received_samples=len(part),usable_samples=len(usable),tracking_coverage=round(coverage,3),
            tracked_span_seconds=round(span,3),max_gap_seconds=round(max_gap,3) if max_gap is not None else None,temporal_coverage=round(temporal,3))
        if a.shape[1]==7 and len(usable) and len(np.unique(usable[:,6]))>1:window['reason']='tracking_reinitialized';continue
        if len(usable)<20:window['reason']='too_few_usable_samples';continue
        if coverage<.8:window['reason']='tracking_coverage_low';continue
        if span<duration*.85:window['reason']='tracked_span_too_short';continue
        if temporal<.8:window['reason']='temporal_coverage_low';continue
        if max_gap>.4:window['reason']='sampling_gap_too_large';continue
        density=(len(usable)-1)/span
        if density<12:window['reason']='motion_sampling_too_slow';continue
        t=usable[:,0]-usable[0,0];clean=_detrend(t,usable[:,1:5],duration);energy=float(np.sum(clean[:,:2]**2))
        low=3/duration;high=min(12 if duration==2 else 1.5 if duration==6 else .5,.4*min(fps,density));best=None
        search_low=2/duration;search_high=min(high+1/duration,.48*min(fps,density))
        search_first=np.ceil((search_low-1e-9)/.05)*.05
        for frequency in np.arange(search_first,search_high+.001,.05):
            fit=_fit_frequency(t,clean,frequency,duration)
            if fit is not None and (best is None or fit[0]>best[0]):best=(fit[0],frequency,fit[1])
        if best is None:window['reason']='singular_motion_fit';continue
        gain,frequency,coefficients=best
        fraction=max(0,min(1,gain/max(energy,1e-20)))
        halves=[]
        for mask in (t<span/2,t>=span/2):
            fit=_fit_frequency(t[mask],_detrend(t[mask],usable[mask,1:5],duration),frequency,duration)
            halves.append(fit[1][:,:2].ravel() if fit is not None else np.zeros(4))
        norms=[float(np.linalg.norm(v)) for v in halves]
        coherence=float(halves[0]@halves[1]/max(norms[0]*norms[1],1e-20))
        ratio=min(norms)/max(max(norms),1e-20)
        head_rms=float(np.linalg.norm(coefficients[:,:2]))/np.sqrt(2)
        background_rms=float(np.linalg.norm(coefficients[:,2:]))/np.sqrt(2)
        candidate=low-1e-7<=frequency<=high+1e-7 and energy>1e-14 and head_rms>=3e-6 and fraction>=.55 and coherence>=.8 and ratio>=.5
        camera_confounded=candidate and background_rms>head_rms*.5
        supported=candidate and not camera_confounded
        stationary=energy<=1e-14
        window.update(status='measured',method='observed_timestamp_periodicity_check',
            periodicity_status='supported' if supported else 'camera_confounded' if camera_confounded else 'stationary' if stationary else 'not_established',
            amplitude_percent_image_width=round(100*head_rms,5) if supported else 0.0 if stationary else None,
            background_amplitude_percent_image_width=round(100*background_rms,5) if candidate else None,
            dominant_frequency_hz=round(float(frequency),2) if supported else None,
            motion_rms_percent_image_width=round(100*np.sqrt(energy/len(t)),5),
            periodic_fraction=round(fraction,4) if not stationary else 0.0,half_coherence=round(coherence,4) if not stationary else 0.0,half_amplitude_ratio=round(ratio,4) if not stationary else 0.0,
            band_hz=[round(low,2),round(high,2)],effective_sample_density=round(density,3))
    if any(w['status']=='measured' for w in result['windows']):result.update(status='measured',reason=None)
    else:result['reason']='no_usable_motion_windows'
    return result
