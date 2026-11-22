"""Independent stdlib verification of the saved timing samples and completion gates."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics

p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();r=Path(a.out)
read=lambda name:json.loads((r/name).read_text())
result=read('timing-results.json');protocol=read('timing-protocol.json');process=read('process-result.json')
assert result['status']=='COMPLETED' and process['exit_code']==0 and not process['timeout']
with (r/'timing-raw.csv').open() as f:rows=list(csv.DictReader(f))
groups={}
for row in rows:
    t=float(row['seconds']);assert math.isfinite(t) and t>0
    groups.setdefault((row['case'],row['scope'],row['group']),[]).append(row)
checks=[]
for (case,scope,group),sample in groups.items():
    expected=protocol['repeats'][group]
    assert [int(x['repeat']) for x in sample]==list(range(expected))
    values=[float(x['seconds']) for x in sample]
    if group=='preparation':
        key='Spline_fit' if scope=='fit' else 'Spline_load' if case=='Spline_CPU' else case+'_load'
        saved=result['preparation'][key]['warm_repeated']
    else:saved=result['timings'][case][group][scope]
    for key,value in [('median_seconds',statistics.median(values)),('mean_seconds',statistics.mean(values)),('standard_deviation_seconds',statistics.stdev(values))]:
        assert math.isclose(saved[key],value,rel_tol=1e-12,abs_tol=1e-15)
    ordered=sorted(values);position=.95*(len(ordered)-1);low=math.floor(position);high=math.ceil(position)
    p95=ordered[low]+(ordered[high]-ordered[low])*(position-low)
    assert math.isclose(saved['p95_seconds'],p95,rel_tol=1e-12,abs_tol=1e-15)
    checks.append({'case':case,'scope':scope,'group':group,'count':len(values),'statistics':'PASS'})
assert set(result['timings'])==set(protocol['cases'])
for case in protocol['cases']:
    needed=['single','Q400'] if case=='RK4_CPU' else ['single','batch32','Q400']
    for group in needed:
        scopes=['end_to_end'] if case=='RK4_CPU' else ['model_only','end_to_end']
        for scope in scopes:
            assert (case,scope,group) in groups
            v=result['validation'][case][group+'_'+scope]
            assert v['status']=='PASS' and v['finite'] and v['actual_compute_calls']>0
for path,info in protocol['sources'].items():
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    assert h.hexdigest()==info['sha256'],path
report={'status':'PASS','utc':dt.datetime.now(dt.timezone.utc).isoformat(),'raw_rows':len(rows),'groups':checks,
        'source_hashes_unchanged':True,'all_required_cases_and_counts_complete':True,
        'all_statistical_summaries_recomputed':True,'all_reported_output_checks_passed':True,
        'accuracy_results_not_recomputed_or_changed':True,'exit_code':process['exit_code'],
        'hami_context_error_count':process['hami_context_error_count'],'hami_only_after_atexit':process['hami_only_after_atexit'],
        'P1_acceptance':'PASS_FOR_APPROVED_SINGLE_SEED_T1200_SCOPE',
        'limitations':['Timing distributions from one shared-container session','Five full-field repetitions do not establish tail-latency stability','Cross-container timing repeatability not tested']}
with (r/'verification.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report))
