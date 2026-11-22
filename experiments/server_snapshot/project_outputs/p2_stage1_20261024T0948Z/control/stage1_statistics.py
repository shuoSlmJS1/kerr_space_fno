import itertools,json,math,statistics,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/'protocol.json').read_text())
SESSION=Path(CONFIG['session_root'])
SEEDS=[27,41,73]
METRICS=['mean_per_q_relative_l2','global_relative_l2','mse']
def summary(values):
 if len(values)!=3 or not all(math.isfinite(x) for x in values):raise ValueError('Expected three finite seed values')
 return {'sample_mean':statistics.mean(values),'sample_std':statistics.stdev(values),'ddof':1,'n':3}
def generate():
 rows=json.loads((HERE/'historical-cells.json').read_text());completed=[]
 for unit in CONFIG['run_units']:
  matrix=SESSION/'evaluations'/unit['id']/'matrix.json'
  if unit['family']!='track_a' or not matrix.exists():continue
  data=json.loads(matrix.read_text())
  train=json.loads((SESSION/'runs'/unit['id']/'summary.json').read_text())
  if train['training']['seed']!=unit['seed'] or train['completed_epochs']!=500 or data['checkpoint_sha256']!=train['best_checkpoint_sha256']:raise ValueError('Result identity mismatch')
  for resolution,metrics in data['metrics'].items():
   rows.append({'model':unit['model'],'train_T':unit['train_resolution'],'eval_T':int(resolution),'training_seed':unit['seed'],**{key:metrics[key] for key in METRICS}})
  completed.append(unit['id'])
 lookup={(x['model'],x['train_T'],x['eval_T'],x['training_seed']):x for x in rows}
 if len(lookup)!=len(rows):raise ValueError('Duplicate result cells')
 grouped=[];paired=[];transfer=[];training_grid=[]
 models=['fno1d','resnet','timesnet','bilstm','transformer','deeponet']
 for model,t,e in itertools.product(models,[1200,2399],[1200,2399,3598,4797]):
  available=[s for s in SEEDS if (model,t,e,s) in lookup]
  entry={'model':model,'train_T':t,'eval_T':e,'available_seeds':available,'missing_seeds':[s for s in SEEDS if s not in available],'status':'COMPLETE' if available==SEEDS else 'INCOMPLETE'}
  if available==SEEDS:
   entry['metrics']={k:summary([lookup[model,t,e,s][k] for s in SEEDS]) for k in METRICS}
   for k in METRICS:
    native=[lookup[model,t,t,s][k] for s in SEEDS];target=[lookup[model,t,e,s][k] for s in SEEDS]
    transfer.append({'model':model,'train_T':t,'eval_T':e,'metric':k,'paired_seed_differences':{str(s):v-n for s,v,n in zip(SEEDS,target,native)},'difference_summary':summary([v-n for v,n in zip(target,native)]),'ratio_summary':summary([v/n for v,n in zip(target,native)]) if all(native) else None})
  grouped.append(entry)
 for a,b in itertools.combinations(models,2):
  for t,e,k in itertools.product([1200,2399],[1200,2399,3598,4797],METRICS):
   if all((m,t,e,s) in lookup for m in [a,b] for s in SEEDS):
    values=[lookup[a,t,e,s][k]-lookup[b,t,e,s][k] for s in SEEDS]
    paired.append({'model_a':a,'model_b':b,'train_T':t,'eval_T':e,'metric':k,'direction':'A minus B; negative favors A for error metrics','per_seed_difference':dict(zip(map(str,SEEDS),values)),**summary(values)})
 for model,e,k in itertools.product(models,[1200,2399,3598,4797],METRICS):
  if all((model,t,e,s) in lookup for t in [1200,2399] for s in SEEDS):
   values=[lookup[model,2399,e,s][k]-lookup[model,1200,e,s][k] for s in SEEDS]
   training_grid.append({'model':model,'eval_T':e,'metric':k,'direction':'T2399 training minus T1200 training','per_seed_difference':dict(zip(map(str,SEEDS),values)),**summary(values)})
 return {'status':'PARTIAL_STAGE1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'required_seeds':SEEDS,'completed_new_training_and_evaluation_units':completed,'per_seed_metrics':rows,'three_seed_cells':grouped,'paired_model_comparisons':paired,'resolution_transfer_comparisons':transfer,'paired_training_resolution_comparisons':training_grid,'fno2d_status':'Three configurations retained; historical and execution verification pending; no fabricated metrics','stage2_status':'RESERVED_NOT_STARTED','inference_limit':'Descriptive paired comparisons; n=3 does not support strong significance claims'}
if __name__=='__main__':
 import os
 result=generate();path=SESSION/'stage1-statistics.json';tmp=path.with_suffix('.tmp.'+str(os.getpid()))
 with tmp.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);print(json.dumps({'completed_new_units':len(result['completed_new_training_and_evaluation_units']),'complete_three_seed_cells':sum(x['status']=='COMPLETE' for x in result['three_seed_cells'])}))