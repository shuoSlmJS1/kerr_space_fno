import sys,json,hashlib,os
from pathlib import Path
W=Path(__file__).resolve().parent
C=json.loads((W/'window.json').read_text());R=Path(C['session_root'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(R/'control/protocol.json')==C['frozen_protocol_sha256']
assert sha(R/'control/stage1_runner.py')==C['frozen_runner_sha256']
sys.path.insert(0,str(R/'control'))
import stage1_runner as a
assert a.CONFIG['launch_seeds']==[41,73]
a.CONFIG['safe_stop_utc']=C['safe_stop_utc']
if sys.argv[1:]==['--check']:
 assert a.provenance()['protocol_sha256']==C['frozen_protocol_sha256']
 print(json.dumps({'status':'PASS','original_hashes_unchanged':True,'deadline_override':a.CONFIG['safe_stop_utc'],'gpu':a.b.inspect_device('cuda:0',0)}))
else:
 assert sys.argv[1] in ['train','eval'] and sys.argv[2]=='--unit' and len(sys.argv)==4
 unit=next(u for u in C['units'] if u['id']==sys.argv[3])
 assert unit['status']=='PENDING' and unit['family']=='track_a' and unit['seed'] in [41,73]
 a.main()
