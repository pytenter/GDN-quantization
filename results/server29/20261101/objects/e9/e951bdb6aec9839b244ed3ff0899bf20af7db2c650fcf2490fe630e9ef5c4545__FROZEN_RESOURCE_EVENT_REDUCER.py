"""Pure CPU event validation for a future authorized run. Never queries a device.

Events must be persisted append-only by a future controller. This reducer does
not overwrite first observations and reports observation bounds, not invented
physical exit/release timestamps. Historical task events are never imported.
"""
import math

def reduce_events(events, as_of_monotonic=None):
    units = {}
    last_global = -math.inf
    for e in events:
        t = e['monotonic_seconds']
        if not isinstance(t, (float, int)) or not math.isfinite(t) or t < last_global:
            raise ValueError('nonfinite or unordered monotonic event')
        last_global = t
        key = e['unit_id']
        kind = e['kind']
        if kind == 'dispatch_before_spawn':
            if key in units:
                raise ValueError('duplicate unit dispatch; no restart admitted')
            units[key] = {'dispatch_before_spawn':t, 'GPU_UUID':e['GPU_UUID']}
            continue
        if key not in units:
            raise ValueError('event without dispatch')
        u = units[key]
        if e['GPU_UUID'] != u['GPU_UUID']:
            raise ValueError('device identity changed')
        if kind == 'spawn_observed':
            if kind in u:
                raise ValueError('duplicate spawn observation')
            u[kind]=t
        elif kind == 'busy_observed':
            if 'first_exit_and_release_observed' in u:
                raise ValueError('busy after terminal observation')
            u['last_busy_observed']=t
        elif kind == 'first_process_exit_observed':
            if kind in u:
                raise ValueError('first process exit overwritten')
            u[kind]=t
        elif kind == 'first_exit_and_release_observed':
            if kind in u:
                raise ValueError('first exit/release overwritten')
            if 'first_process_exit_observed' not in u:
                raise ValueError('release receipt without process-exit receipt')
            u[kind]=t
        else:
            raise ValueError('unknown event kind')
    completed=[]
    for key,u in units.items():
        if 'first_exit_and_release_observed' not in u:
            if as_of_monotonic is not None and (not math.isfinite(as_of_monotonic) or as_of_monotonic < last_global):
                raise ValueError('invalid live accounting time')
            completed.append({'unit_id':key,'status':'INCOMPLETE_OBSERVATION','exact_GPU_hours':'UNKNOWN',
                'occupied_seconds_upper':None if as_of_monotonic is None else as_of_monotonic-u['dispatch_before_spawn']})
            continue
        if 'spawn_observed' not in u:
            raise ValueError('terminal receipt missing spawn bracket')
        lower=max(0,u.get('last_busy_observed',u['spawn_observed'])-u['spawn_observed'])
        upper=u['first_exit_and_release_observed']-u['dispatch_before_spawn']
        if lower > upper:
            raise ValueError('invalid observation interval')
        completed.append({'unit_id':key,'status':'OBSERVATION_BOUNDS_ONLY',
            'exact_GPU_hours':'UNKNOWN','occupied_seconds_lower':lower,
            'occupied_seconds_upper':upper,'GPU_UUID':u['GPU_UUID'],
            'first_exit_observation':u['first_process_exit_observed'],
            'first_release_observation':u['first_exit_and_release_observed']})
    return {'units':completed,'GPU_seconds_upper_bound':('UNKNOWN' if any(x['occupied_seconds_upper'] is None for x in completed) else sum(x['occupied_seconds_upper'] for x in completed)),
            'has_incomplete_units':any(x['status']=='INCOMPLETE_OBSERVATION' for x in completed)}
