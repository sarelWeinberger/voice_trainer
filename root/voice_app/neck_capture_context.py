"""Bounded capture/evidence metadata; never converts image motion into effort."""
import json
import math
from numbers import Real


def parse_transport(payload):
    if payload is None:return None
    if not isinstance(payload,str) or len(payload)>2048:raise ValueError('Invalid capture receipt summary')
    data=json.loads(payload)
    fields=('captured_frames','acknowledged_frames','encoding_failed_frames','overflow_frames',
            'retry_count','failed_batches','unconfirmed_frames')
    if not isinstance(data,dict) or data.get('version')!='neck-capture-transport-v1' or set(data)!=set(fields)|{'version'}:
        raise ValueError('Invalid capture receipt summary')
    if any(type(data[k]) is not int or not 0<=data[k]<=900 for k in fields):raise ValueError('Invalid capture counters')
    if data['acknowledged_frames']>data['captured_frames'] or data['unconfirmed_frames']!=data['captured_frames']-data['acknowledged_frames']:
        raise ValueError('Inconsistent capture counters')
    return data


def transport_evidence(reported,received):
    if reported is None:return None
    return dict(reported,source='browser_counters_server_receipt',server_received_frames=received,
        missing_frames=max(0,reported['captured_frames']-received),
        complete=received>=reported['captured_frames'] and reported['encoding_failed_frames']==0)


def movement_evidence(movement,first,last):
    """Summarize server-recomputed windows overlapping this image recording.

    Group RMS by the same duration/band: nuisance removal differs by window.
    Missing/unsupported periodicity stays missing. No score or clinical cutoff.
    """
    if not movement:return None
    windows=[w for w in movement.get('windows',[]) if w.get('status')=='measured'
        and w['start']>=first-.25 and w['end']<=last+.25
        and isinstance(w.get('motion_rms_percent_image_width'),Real)
        and not isinstance(w['motion_rms_percent_image_width'],bool)
        and math.isfinite(w['motion_rms_percent_image_width'])]
    groups={}
    for window in windows:
        key=(round(window['end']-window['start']),tuple(window.get('band_hz',[])))
        groups.setdefault(key,[]).append(window)
    comparisons=[]
    for (duration,band),items in sorted(groups.items()):
        ordered=sorted(items,key=lambda w:w['start'])
        def describe(w):
            keys=('start','end','motion_rms_percent_image_width','dominant_frequency_hz',
                  'periodicity_status','tracking_coverage','temporal_coverage')
            return {k:w.get(k) for k in keys}
        comparisons.append(dict(window_seconds=duration,band_hz=list(band),measured_windows=len(ordered),
            first=describe(ordered[0]),last=describe(ordered[-1]),
            largest=describe(max(ordered,key=lambda w:w['motion_rms_percent_image_width']))))
    return dict(version='neck-head-motion-context-v1',source='server_recomputed_browser_trace',
        status='measured' if comparisons else 'unavailable',sample_count=movement.get('sample_count'),
        effective_fps=movement.get('effective_fps'),comparisons=comparisons[:3],
        interpretation='Image head movement relative to scenery; includes irregular motion and possible tracking error. '
          'RMS is after nuisance removal, not a muscle-effort grade. A missing frequency does not mean no movement. '
          'Motion magnitude alone cannot establish compensation, speech causation or neck muscle activation.')
