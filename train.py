from pathlib import Path
import json, urllib.request, hashlib, time
import numpy as np
import wfdb
from scipy.signal import resample_poly, welch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss, classification_report
from scipy.optimize import minimize_scalar
import torch
from torch import nn
ROOT=Path(__file__).parent
DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
OUT=ROOT/'web'; OUT.mkdir(exist_ok=True)
SEED=42
np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
SPLITS={'train':['100','101','103','105','106','107'],'validation':['108','109'],'test':['111','112','113']}
NOISE_RANGES={'train':(0,880),'validation':(910,1300),'test':(1330,1790)}
CLASSES=['Reference','Baseline wander','Muscle artifact','Electrode motion']

def download(db,rec):
 for ext in ['hea','dat']:
  p=DATA/f'{rec}.{ext}'
  if not p.exists():
   u=f'https://physionet.org/files/{db}/1.0.0/{rec}.{ext}'
   print('download',u,flush=True);urllib.request.urlretrieve(u,p)
 return resample_poly(wfdb.rdrecord(str(DATA/rec)).p_signal[:,0],16,45).astype(np.float32)

def normalize(x):
 x=x-x.mean(axis=-1,keepdims=True)
 return x/(np.sqrt(np.mean(x*x,axis=-1,keepdims=True))+1e-8)

def features(x):
 x=normalize(x)
 f,p=welch(x,fs=128,nperseg=256,axis=-1)
 total=p.sum(-1)+1e-12
 bands=[p[:,(f>=a)&(f<b)].sum(-1)/total for a,b in [(0,0.5),(0.5,3),(3,8),(8,20),(20,40),(40,65)]]
 d=np.diff(x,axis=-1)
 return np.stack(bands+[np.mean(abs(d),-1),np.std(d,-1),np.mean(d*d,-1),np.mean(x**4,-1),np.max(abs(x),-1),np.mean((x[:,1:]*x[:,:-1])<0,-1)],axis=-1)

class CNN(nn.Module):
 def __init__(self):
  super().__init__(); self.conv1=nn.Conv1d(1,8,9,stride=4,padding=4);self.conv2=nn.Conv1d(8,12,7,stride=4,padding=3);self.fc=nn.Linear(12,4)
 def forward(self,x):
  return self.fc(torch.relu(self.conv2(torch.relu(self.conv1(x)))).mean(-1))

def probs(z,t=1):
 z=z/t;z=z-z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)

def metrics(y,p):
 pred=p.argmax(1);conf=p.max(1);correct=pred==y
 ece=0
 for lo in np.linspace(0,.9,10):
  m=(conf>=lo)&(conf<lo+.1+1e-8)
  if m.any():ece+=m.mean()*abs(correct[m].mean()-conf[m].mean())
 m=conf>=.7
 return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'nll':float(log_loss(y,p,labels=list(range(4)))),'ece_10_bins':float(ece),'brier_multiclass':float(np.mean(np.sum((p-np.eye(4)[y])**2,axis=1))),'confusion_matrix':confusion_matrix(y,pred).tolist(),'per_class':classification_report(y,pred,target_names=CLASSES,output_dict=True,zero_division=0),'coverage_at_0_7':float(m.mean()),'accuracy_at_0_7':float(correct[m].mean()) if m.any() else None}

def main():
 start=time.time();rng=np.random.default_rng(SEED)
 noises={r:download('nstdb',r) for r in ['bw','ma','em']}
 sets={};meta={};samples=[]
 for split,recs in SPLITS.items():
  xs=[];ys=[];rows=[]
  for rec in recs:
   sig=download('mitdb',rec)
   # Non-overlapping 8 second reference windows, stride 10 seconds.
   for i in range(150):
    a=i*1280;clean=sig[a:a+1024].copy();clean-=clean.mean()
    if len(clean)!=1024 or np.std(clean)<1e-6:continue
    for label in range(4):
     snr=None;noise_start=None;x=clean.copy()
     if label:
      snr=float(rng.choice([-6,0,6,12]));lo,hi=NOISE_RANGES[split]
      noise_start=int(rng.integers(lo*128,(hi-8)*128))
      n=noises[['bw','ma','em'][label-1]][noise_start:noise_start+1024].copy();n-=n.mean()
      n=n/(np.sqrt(np.mean(n*n))+1e-8)*np.sqrt(np.mean(clean*clean))*10**(-snr/20)
      x+=n
     xs.append(x);ys.append(label);rows.append({'record':rec,'reference_start_sample_128hz':a,'label':label,'snr_db':snr,'noise_start_sample_128hz':noise_start})
  sets[split]=(normalize(np.array(xs,dtype=np.float32)),np.array(ys));meta[split]=rows
  print(split,sets[split][0].shape,flush=True)
 np.savez_compressed(DATA/'prepared.npz',**{f'{s}_{t}':v for s,pair in sets.items() for t,v in zip(['x','y'],pair)})
 (ROOT/'split_manifest.json').write_text(json.dumps({'seed':SEED,'records':SPLITS,'noise_time_seconds':NOISE_RANGES,'windows':meta},indent=2))
 x,y=sets['train'];vx,vy=sets['validation'];tx,ty=sets['test']
 scaler=StandardScaler();xf=scaler.fit_transform(features(x));vf=scaler.transform(features(vx));tf=scaler.transform(features(tx))
 lr=LogisticRegression(C=1,max_iter=1000,random_state=SEED).fit(xf,y)
 model=CNN();optim=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001)
 xt=torch.tensor(x[:,None]);yt=torch.tensor(y);vxt=torch.tensor(vx[:,None]);txt=torch.tensor(tx[:,None])
 best=1e9;history=[]
 for epoch in range(32):
  model.train();order=torch.randperm(len(x));losses=[]
  for idx in order.split(128):
   optim.zero_grad();loss=nn.functional.cross_entropy(model(xt[idx]),yt[idx]);loss.backward();optim.step();losses.append(loss.item())
  model.eval()
  with torch.no_grad():
   vz=model(vxt).numpy();vl=log_loss(vy,probs(vz),labels=[0,1,2,3]);va=accuracy_score(vy,vz.argmax(1))
  history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation_nll':float(vl),'validation_accuracy':float(va)})
  if vl<best:best=vl;torch.save(model.state_dict(),ROOT/'cnn.pt')
  print('epoch',epoch+1,history[-1],flush=True)
 model.load_state_dict(torch.load(ROOT/'cnn.pt',weights_only=True));model.eval()
 with torch.no_grad():vz=model(vxt).numpy();tz=model(txt).numpy()
 lrv=lr.decision_function(vf);lrt=lr.decision_function(tf)
 temps={k:float(minimize_scalar(lambda t:log_loss(vy,probs(z,t),labels=[0,1,2,3]),bounds=(.2,5),method='bounded').x) for k,z in [('cnn',vz),('baseline',lrv)]}
 result={'seed':SEED,'classes':CLASSES,'input':{'sample_rate':128,'samples':1024,'seconds':8},'records':SPLITS,'noise_time_seconds':NOISE_RANGES,'counts':{s:len(p[1]) for s,p in sets.items()},'temperatures':temps,'training_history':history,'test':{},'by_snr':{},'per_record':{},'latency_ms':{},'protocol':'Record-disjoint reference ECGs; disjoint time blocks of the same three noise recordings; additive semi-synthetic labels, not clinical quality labels.'}
 for name,z in [('baseline',lrt),('cnn',tz)]:
  p=probs(z,temps[name]);result['test'][name]=metrics(ty,p)
  result['by_snr'][name]={};result['per_record'][name]={}
  for snr in [-6,0,6,12]:
   mask=np.array([r['snr_db']==snr or r['label']==0 for r in meta['test']]);result['by_snr'][name][str(snr)]=metrics(ty[mask],p[mask])
  for rec in SPLITS['test']:
   mask=np.array([r['record']==rec for r in meta['test']]);result['per_record'][name][rec]=metrics(ty[mask],p[mask])
 # CPU batch-of-one latency, 10 warmups + 100 measurements. Feature extraction included in baseline.
 for name in ['baseline','cnn']:
  timings=[]
  for i in range(110):
   q=tx[i%len(tx):i%len(tx)+1];t=time.perf_counter()
   if name=='cnn':
    with torch.no_grad():model(torch.tensor(q[:,None]))
   else:lr.predict_proba(scaler.transform(features(q)))
   if i>=10:timings.append((time.perf_counter()-t)*1000)
  result['latency_ms'][name]={'median':float(np.median(timings)),'p95':float(np.quantile(timings,.95)),'environment':'CPU Python, four torch threads; not browser latency'}
 weights={'classes':CLASSES,'sample_rate':128,'samples':1024,'temperatures':temps,'cnn':{k:v.detach().numpy().tolist() for k,v in model.state_dict().items()},'baseline':{'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coef':lr.coef_.tolist(),'intercept':lr.intercept_.tolist()}}
 (OUT/'weights.json').write_text(json.dumps(weights,separators=(',',':')))
 # Small research examples, attributed with record/offset and label/SNR. No identifying metadata.
 for label in range(4):
  candidates=[i for i,r in enumerate(meta['test']) if r['label']==label and (label==0 or r['snr_db']==0)]
  for j,idx in enumerate(candidates[:3]):
   samples.append({'id':f'{label}-{j}','name':f'{CLASSES[label]} / record {meta["test"][idx]["record"]} / {j+1}','truth':label,'source':meta['test'][idx],'signal':np.round(tx[idx],6).tolist()})
 (OUT/'samples.json').write_text(json.dumps(samples,separators=(',',':')))
 result['elapsed_seconds']=time.time()-start
 (OUT/'results.json').write_text(json.dumps(result,indent=2))
 (ROOT/'results.json').write_text(json.dumps(result,indent=2))
 print('FINAL',json.dumps({k:{m:round(v,4) for m,v in d.items() if isinstance(v,float)} for k,d in result['test'].items()}),flush=True)
 # Full test predictions enable independent checking and error explorer without rerunning training.
 rows=[]
 for i,r in enumerate(meta['test']):
  rows.append({**r,'baseline_probs':probs(lrt[i:i+1],temps['baseline'])[0].tolist(),'cnn_probs':probs(tz[i:i+1],temps['cnn'])[0].tolist()})
 (OUT/'test_predictions.json').write_text(json.dumps(rows,separators=(',',':')))
 # Prediction parity fixtures for JS baseline/CNN.
 fixtures=[]
 for i in [0,1,2,3,100,301,777]:
  fixtures.append({'signal':tx[i].tolist(),'features':features(tx[i:i+1])[0].tolist(),'cnn':probs(tz[i:i+1],temps['cnn'])[0].tolist(),'baseline':probs(lrt[i:i+1],temps['baseline'])[0].tolist()})
 (OUT/'parity.json').write_text(json.dumps(fixtures))
if __name__=='__main__':main()
