"""Non-destructive phone acceptance of independently exported Multi-CL bundles."""
import hashlib
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time

from tokenizers import Tokenizer
from run_tasks import ROOT, WORKSPACE, REMOTE, BUNDLE, ResidentVLM
from phone_env import PhoneController, PhoneEnv
from timing import Timeline

NEW='/data/local/tmp/qwen3_vl_geniex_multicl8k'
HOST=WORKSPACE/'phone_bundle'


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False))


def prepare(output):
    source=ROOT/'results/m3a_preflight_20261106_135645'
    cases={}
    for label,folder in [('short',source),('near4k',ROOT/'results/near4k_input_20261109_142415')]:
        target=output/'inputs'/label;target.mkdir(parents=True)
        for name in ['prompt.txt','image_0.png','image_1.png']:
            (target/name).write_bytes((folder/name).read_bytes())
        cases[label]=(target,192)
    target=output/'inputs'/'official_action';target.mkdir()
    original=ROOT/'results/official_m3a_20261108_010622/rep01_folder/00/action'
    (target/'prompt.txt').write_bytes((original/'prompt.txt').read_bytes())
    for i in range(2):(target/f'image_{i}.png').write_bytes((original/f'model_{i}.png').read_bytes())
    cases['official_action']=(target,192)
    target=output/'inputs'/'long7k';target.mkdir()
    (target/'prompt.txt').write_bytes((HOST/'prompts_long.txt').read_bytes())
    for i in range(2):(target/f'image_{i}.png').write_bytes((source/f'image_{i}.png').read_bytes())
    cases['long7k']=(target,128)
    target=output/'inputs'/'boundary';target.mkdir()
    tokenizer=Tokenizer.from_file(str(HOST/'model/tokenizer.json'))
    count=lambda x:len(tokenizer.encode(x,add_special_tokens=False).ids)
    suffix='\nDescribe both screenshots in detail, in at least 400 words. Include visible controls, labels, and layout. Continue until you have fully described both images.'
    unit='Reference note: inspect visible labels, file names, navigation controls, and text fields carefully.\n'
    prefix='The following notes are background for visual inspection.\n';notes=''
    target_text=880-537
    while count(prefix+notes+unit+suffix)<target_text:notes+=unit
    candidates=[prefix+notes+tokenizer.decode(tokenizer.encode(unit,add_special_tokens=False).ids[:i])+suffix for i in range(count(unit)+1)]
    text=min(candidates,key=lambda x:abs(count(x)-target_text))
    (target/'prompt.txt').write_text(text)
    for i in range(2):(target/f'image_{i}.png').write_bytes((source/f'image_{i}.png').read_bytes())
    cases['boundary']=(target,192)
    write_json(output/'inputs.json',{key:{'text_tokens':count((folder/'prompt.txt').read_text()),
        'expected_runtime_tokens':count((folder/'prompt.txt').read_text())+537,'max_new_tokens':maximum,
        'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}}
        for key,(folder,maximum) in cases.items()})
    return cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-bundle',default=NEW)
    parser.add_argument('--baseline-bundle',default=BUNDLE)
    parser.add_argument('--context-size',type=int,choices=[8192,16384],default=8192)
    parser.add_argument('--official-long-request',type=Path)
    args=parser.parse_args()
    candidate=args.remote_bundle
    controller=PhoneController('189f0fad')
    if controller.shell('pidof vlm_server geniex-bench || true').decode().strip():
        raise RuntimeError('Another model service is active')
    output=ROOT/'results'/(f'multicl{args.context_size//1024}k_acceptance_'+time.strftime('%Y%m%d_%H%M%S'));output.mkdir()
    print('RESULT_DIR',output,flush=True)
    for name in ['README.md','BUILD_NOTES.md','BUILD_RECORD.json','CLOUD_GRAPH_SPECS.json','MANIFEST.sha256','LONG_PROMPT_INFO.json']:
        (output/name).write_bytes((HOST/name).read_bytes())
    check=subprocess.run(['sha256sum','-c','MANIFEST.sha256'],cwd=HOST,capture_output=True,text=True)
    (output/'host_integrity.log').write_text(check.stdout+check.stderr)
    check.check_returncode()
    configured=json.loads((HOST/'model/text-generator.json').read_text())['text-generator']['context']['size']
    if configured!=args.context_size:raise RuntimeError(f'Bundle context {configured} != requested {args.context_size}')
    exists=controller.shell('test -e '+shlex.quote(candidate)+' && echo exists || true').decode().strip()
    if exists:
        remote=controller.run(['shell','cat',candidate+'/MANIFEST.sha256'])
        if remote!=(HOST/'MANIFEST.sha256').read_bytes():
            raise RuntimeError('Existing independent phone directory has a different manifest; refusing to overwrite')
    else:
        print('DEPLOY',candidate,flush=True)
        command=controller.adb+['push',str(HOST),candidate]
        write_json(output/'deploy_command.json',command)
        result=subprocess.run(command,capture_output=True,text=True)
        (output/'deploy.log').write_text(result.stdout+result.stderr);result.check_returncode()
    verify=controller.shell('cd '+shlex.quote(candidate)+' && sha256sum -c MANIFEST.sha256',timeout=120)
    (output/'phone_integrity.log').write_bytes(verify)
    print('PHONE_MANIFEST_OK',flush=True)
    runner_hash=hashlib.sha256((ROOT/'native/vlm_server').read_bytes()).hexdigest()
    if controller.run(['shell','sha256sum',REMOTE+'/vlm_server']).decode().split()[0]!=runner_hash:
        raise RuntimeError('Runner differs from previous tests')
    controller.run(['push',str(ROOT/'native/libaw_decode_diag.so'),REMOTE+'/libaw_decode_diag.so'])
    cases=prepare(output)
    long_case='long7k' if args.context_size==8192 else 'long15k'
    if long_case!='long7k':
        source,maximum=cases.pop('long7k')
        target=source.with_name(long_case);source.rename(target)
        cases[long_case]=(target,maximum)
    long_official=[]
    if args.official_long_request:
        target=output/'inputs'/'official_over8k';target.mkdir()
        (target/'prompt.txt').write_bytes((args.official_long_request/'prompt.txt').read_bytes())
        for i in range(2):(target/f'image_{i}.png').write_bytes((args.official_long_request/f'model_{i}.png').read_bytes())
        cases['official_over8k']=(target,512)
        long_official=['official_over8k']
    tokenizer=Tokenizer.from_file(str(HOST/'model/tokenizer.json'))
    write_json(output/'inputs.json',{key:{'text_tokens':len(tokenizer.encode((folder/'prompt.txt').read_text(),add_special_tokens=False).ids),
        'max_new_tokens':maximum,'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}}
        for key,(folder,maximum) in cases.items()})
    graph_specs=json.loads((HOST/'CLOUD_GRAPH_SPECS.json').read_text())
    context_lengths=sorted({int(match.group(1)) for name in graph_specs
                           if (match:=re.search(r'_cl(\d+)_',name))})
    boundary_force_cl=next(cl for cl in context_lengths if cl>=2048)
    sessions=[('baseline_before',args.baseline_bundle,0,['short','near4k']),
              ('multicl_auto',candidate,0,['short','short','near4k','near4k','official_action']+long_official+[long_case,'boundary']),
              (f'multicl_force{boundary_force_cl//1024}k',candidate,boundary_force_cl,['boundary']),
              ('baseline_after',args.baseline_bundle,0,['short','near4k'])]
    old_stay=controller.shell('settings get global stay_on_while_plugged_in').decode().strip()
    records=[]
    try:
        controller.shell('svc power stayon usb');controller.shell('input keyevent KEYCODE_WAKEUP');controller.shell('wm dismiss-keyguard')
        for label,bundle,force,sequence in sessions:
            folder=output/label;folder.mkdir();trace=Timeline(folder/'host_timeline.json')
            env=PhoneEnv('189f0fad',trace);model=None
            (folder/'battery_before.txt').write_bytes(controller.shell('dumpsys battery'))
            monitor_stop=threading.Event()
            def monitor():
                with (folder/'memory_samples.jsonl').open('w',buffering=1) as stream:
                    while not monitor_stop.is_set():
                        try:
                            pid=controller.shell('pidof vlm_server || true').decode().strip()
                            command='cat /proc/meminfo; cat /proc/pressure/memory'
                            if pid.isdigit():command+='; cat /proc/'+pid+'/status'
                            result=controller.shell(command).decode()
                            stream.write(json.dumps({'host_perf_ns':time.perf_counter_ns(),'pid':pid,'text':result})+'\n')
                        except Exception as error:stream.write(json.dumps({'error':str(error)})+'\n')
                        monitor_stop.wait(2)
            thread=threading.Thread(target=monitor,daemon=True);thread.start()
            try:
                exports={'LD_PRELOAD':REMOTE+'/libaw_decode_diag.so','AW_DIAG_CORE_PATH':bundle+'/lib/qairt/libgeniex_core.so',
                    'AW_DIAG_API_PATH':bundle+'/lib/libgeniex.so','AW_FORCE_CL':str(force)}
                model=ResidentVLM(env,folder,bundle=bundle,request_timeout=600,extra_exports=exports)
                write_json(folder/'ready.json',model.ready)
                (folder/'meminfo_after_load.txt').write_bytes(controller.run(['shell','cat','/proc/meminfo']))
                for index,case in enumerate(sequence):
                    source,maximum=cases[case]
                    response=model.generate_images(f'accept-{label}-{index}-{case}',source/'prompt.txt',
                        [source/'image_0.png',source/'image_1.png'],maximum)
                    write_json(folder/f'response_{index}_{case}.json',response)
                    records.append({'session':label,'case':case,'index':index,**response})
                    write_json(output/'responses_partial.json',records)
                    print(label,case,'input',response['prompt_tokens'],'output',response['generated_tokens'],
                          'TPS',response['decoding_speed'],'TTFT',response['ttft_us']/1e6,flush=True)
            except Exception as error:
                write_json(folder/'failure.json',{'error':str(error)})
                raise
            finally:
                if model:model.close()
                monitor_stop.set();thread.join(timeout=10);trace.write()
                try:
                    (folder/'battery_after.txt').write_bytes(controller.shell('dumpsys battery'))
                except Exception as error:
                    write_json(folder/'cleanup_error.json',{'operation':'battery_snapshot','error':str(error)})
            diagnostics=[json.loads(line.split('@@AWDIAG@@',1)[1]) for line in (folder/'phone_raw.log').read_text().splitlines() if '@@AWDIAG@@' in line]
            write_json(folder/'diagnostics.json',diagnostics)
            assert len(diagnostics)==len(sequence)
            for row,d in zip([r for r in records if r['session']==label],diagnostics):
                row['actual_decode_cls']=sorted({e['cl'] for e in d['graphs'] if e['phase']==1})
                row['decode_graph_calls']=len([e for e in d['graphs'] if e['phase']==1])
                row['decode_graph_wall_sum']=sum(e['seconds'] for e in d['graphs'] if e['phase']==1)
                row['prefill_cls']=sorted({e['cl'] for e in d['graphs'] if e['phase']==0})
                print(label,row['case'],'ACTUAL_CL',row['actual_decode_cls'],flush=True)
        write_json(output/'summary.json',{'run':str(output),'remote_new_bundle':candidate,'baseline_bundle':args.baseline_bundle,
                                       'context_size':args.context_size,'context_lengths':context_lengths,'runner_sha256':runner_hash,
                                       'responses':records,'settings_restored_to':old_stay})
        print('RESULTS',output,flush=True)
    except Exception as error:
        write_json(output/'failure.json',{'error':str(error),'completed_responses':len(records),
                   'remote_bundle':candidate,'session':label})
        raise
    finally:
        try:
            controller.shell('settings delete global stay_on_while_plugged_in' if old_stay=='null' else
                             'settings put global stay_on_while_plugged_in '+str(int(old_stay)))
            write_json(output/'settings_restore.json',{'success':True,'value':old_stay})
        except Exception as error:
            write_json(output/'settings_restore.json',{'success':False,'value':old_stay,'error':str(error)})


if __name__=='__main__':main()
