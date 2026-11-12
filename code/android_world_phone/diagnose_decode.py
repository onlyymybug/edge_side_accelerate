"""Measure actual executed CL graphs; force CL within the unchanged old bundle."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from run_tasks import ROOT, REMOTE, BUNDLE, ResidentVLM
from phone_env import PhoneEnv, PhoneController
from timing import Timeline


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--control',action='store_true',help='Repeat new/old without the preload hooks')
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--source',type=Path,default=ROOT/'results/m3a_preflight_20261106_135645')
    p.add_argument('--max-tokens',type=int,default=192)
    p.add_argument('--long-context',action='store_true',help='A/B/A comparison with automatic CL selection')
    args=p.parse_args()
    controller=PhoneController('189f0fad')
    if controller.run(['shell','pidof vlm_server geniex-bench || true']).decode().strip():
        raise RuntimeError('Another benchmark/service is active')
    digest=hashlib.sha256((ROOT/'native/vlm_server').read_bytes()).hexdigest()
    if controller.run(['shell','sha256sum',REMOTE+'/vlm_server']).decode().split()[0]!=digest:
        raise RuntimeError('Runner hash mismatch')
    controller.run(['push',str(ROOT/'native/libaw_decode_diag.so'),REMOTE+'/libaw_decode_diag.so'])
    output=ROOT/'results'/('decode_diagnosis_'+time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir()
    source=args.source
    for name in ['prompt.txt','image_0.png','image_1.png']:
        (output/name).write_bytes((source/name).read_bytes())
    sessions=[('old_auto',BUNDLE,0)] if args.pilot else [
        ('old_auto',BUNDLE,0),('old_2k',BUNDLE,2048),('old_4k',BUNDLE,4096),
        ('new_16k','/data/local/tmp/qwen3_vl_geniex_ctx16384',0),
        ('old_1k_after',BUNDLE,1024)]
    if args.control:
        sessions=[('new_16k','/data/local/tmp/qwen3_vl_geniex_ctx16384',0),('old_auto',BUNDLE,0)]
    if args.long_context and not args.pilot and not args.control:
        sessions=[('old_4k_before',BUNDLE,0),
                  ('new_16k','/data/local/tmp/qwen3_vl_geniex_ctx16384',0),
                  ('old_4k_after',BUNDLE,0)]
    old_stay=controller.shell('settings get global stay_on_while_plugged_in').decode().strip()
    records=[]
    try:
        controller.shell('svc power stayon usb')
        controller.shell('input keyevent KEYCODE_WAKEUP')
        controller.shell('wm dismiss-keyguard')
        for label,bundle,force in sessions:
            folder=output/label;folder.mkdir()
            trace=Timeline(folder/'host_timeline.json');env=PhoneEnv('189f0fad',trace)
            exports={'LD_PRELOAD':REMOTE+'/libaw_decode_diag.so',
                     'AW_DIAG_CORE_PATH':bundle+'/lib/qairt/libgeniex_core.so',
                     'AW_DIAG_API_PATH':bundle+'/lib/libgeniex.so',
                     'AW_FORCE_CL':str(force)}
            (folder/'battery_before.txt').write_bytes(controller.shell('dumpsys battery'))
            model=None
            try:
                model=ResidentVLM(env,folder,bundle=bundle,request_timeout=300,
                                  extra_exports={} if args.control else exports)
                (folder/'ready.json').write_text(json.dumps(model.ready,indent=2))
                (folder/'meminfo_after_load.txt').write_bytes(controller.run(['shell','cat','/proc/meminfo']))
                for i in range(1 if args.pilot else args.repeats):
                    response=model.generate_images(f'diag-{label}-{i}',output/'prompt.txt',
                            [output/'image_0.png',output/'image_1.png'],8 if args.pilot else args.max_tokens)
                    (folder/f'response_{i}.json').write_text(json.dumps(response,indent=2,ensure_ascii=False))
                    records.append({'label':label,'repeat':i,**response})
                    print(label,i,'TPS',response['decoding_speed'],'TOKENS',response['generated_tokens'],flush=True)
            finally:
                if model:model.close()
                trace.write()
                (folder/'battery_after.txt').write_bytes(controller.shell('dumpsys battery'))
            diag=[]
            for line in (folder/'phone_raw.log').read_text().splitlines():
                if '@@AWDIAG@@' in line:diag.append(json.loads(line.split('@@AWDIAG@@',1)[1]))
            (folder/'diagnostics.json').write_text(json.dumps(diag,indent=2))
            if not args.control and (not diag or not any(e['phase']==1 for e in diag[0]['graphs']) or not diag[0]['shards']):
                raise RuntimeError('Instrumentation hooks did not capture graph/shard calls')
            print(label,'DECODE GRAPHS',sorted({e['graph'] for d in diag for e in d['graphs'] if e['phase']==1}),flush=True)
        summary={'output':str(output),'source':str(source),'max_new_tokens':8 if args.pilot else args.max_tokens,
                 'input_sha256':{name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ['prompt.txt','image_0.png','image_1.png']},
                 'runner_sha256':digest,'instrumentation_enabled':not args.control,'responses':records,
                 'same_output_text':len({r['text'] for r in records})==1,'sessions':{}}
        for label,_,force in sessions:
            rows=[r for r in records if r['label']==label]
            diag=json.loads((output/label/'diagnostics.json').read_text())
            runs=[]
            for d in diag:
                graphs=[e for e in d['graphs'] if e['phase']==1]
                shards=[e for e in d['shards'] if e['phase']==1]
                kvs=[e for e in d['kv_updates'] if e['phase']==1]
                times=d['token_seconds']
                runs.append({'decode_graph_names':sorted({e['graph'] for e in graphs}),
                    'decode_cl':sorted({e['cl'] for e in graphs}),
                    'decode_graph_calls':len(graphs),'decode_graph_wall_sum':sum(e['seconds'] for e in graphs),
                    'decode_shard_wall_sum':sum(e['seconds'] for e in shards),
                    'decode_kv_worker_wall_sum':sum(e['seconds'] for e in kvs),
                    'token_callbacks':len(times),
                    'token_interval_median':statistics.median(b-a for a,b in zip(times,times[1:])),
                    'graph_mean_seconds_by_shard':{str(s):statistics.mean(e['seconds'] for e in graphs if e['shard']==s) for s in sorted({e['shard'] for e in graphs})}})
            summary['sessions'][label]={'forced_cl':force,'decode_tps_pooled':sum(r['generated_tokens'] for r in rows)/sum(r['decode_time_us']/1e6 for r in rows),
                'ttft_seconds':[r['ttft_us']/1e6 for r in rows],'runs':runs}
        (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
        print('RESULTS',output,flush=True)
    finally:
        controller.shell('settings delete global stay_on_while_plugged_in' if old_stay=='null' else
                         'settings put global stay_on_while_plugged_in '+str(int(old_stay)))


if __name__=='__main__':main()
