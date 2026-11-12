"""Uninstrumented retry with boot records, a kernel log stream, and a load timeout."""
import json
import argparse
from pathlib import Path
import queue
import re
import shlex
import subprocess
import threading
import time

from run_tasks import ROOT, REMOTE
from phone_env import PhoneController

BUNDLE='/data/local/tmp/qwen3_vl_geniex_multicl16k_4tier'
SOURCE=ROOT/'results/multicl16k_acceptance_20261110_190148'


def save(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2))


def boot(controller):
    return controller.shell('date +%s; cat /proc/uptime; cat /proc/sys/kernel/random/boot_id; '
                            'getprop ro.serialno; getprop sys.boot.reason; settings get global boot_count').decode()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',default=BUNDLE)
    parser.add_argument('--manifest',type=Path,default=SOURCE/'MANIFEST.sha256')
    parser.add_argument('--source-guard',action='store_true')
    parser.add_argument('--result-prefix',default='multicl16k_retry')
    args=parser.parse_args()
    bundle=args.bundle
    c=PhoneController('189f0fad')
    if c.shell('pidof vlm_server geniex-bench || true').decode().strip():
        raise RuntimeError('Another model process is active')
    output=ROOT/'results'/(args.result_prefix+'_'+time.strftime('%Y%m%d_%H%M%S'));output.mkdir()
    print('RESULT_DIR',output,flush=True)
    (output/'boot_before.txt').write_text(boot(c))
    current=c.run(['shell','cat',bundle+'/MANIFEST.sha256'])
    if current!=args.manifest.read_bytes():raise RuntimeError('Phone bundle changed')
    (output/'MANIFEST.sha256').write_bytes(current)
    (output/'phone_integrity.log').write_bytes(c.shell('cd '+shlex.quote(bundle)+' && sha256sum -c MANIFEST.sha256',timeout=120))
    commands=[]
    for index,(case,maximum) in enumerate([('short',192),('near4k',192),('official_over8k',512),('long15k',128)]):
        folder=SOURCE/'inputs'/case;rid=f'retry16k-{output.name}-{index}'
        local=[folder/'prompt.txt',folder/'image_0.png',folder/'image_1.png']
        remote=[f'{REMOTE}/{rid}_{file.name}' for file in local]
        for file,target in zip(local,remote):c.run(['push',str(file),target])
        commands.append({'case':case,'id':rid,'source':str(folder),'max_new_tokens':maximum,
                         'line':'\t'.join([rid,*remote,str(maximum)])})
    save(output/'requests.json',commands)
    exports={'LD_LIBRARY_PATH':f'{bundle}/lib:{bundle}/lib/qairt:{bundle}/lib/qairt/htp-files',
             'GENIEX_PLUGIN_PATH':bundle+'/lib',
             'ADSP_LIBRARY_PATH':bundle+'/lib/qairt/htp-files;/vendor/lib/rfsa/adsp;/vendor/dsp;/system/lib/rfsa/adsp;/dsp'}
    shell='cd '+shlex.quote(bundle)+' && '+' && '.join('export '+k+'='+shlex.quote(v) for k,v in exports.items())
    # This phone's adb shell inherits -1000, which excludes it from OOM killing.
    # Make this diagnostic process a preferred victim before allocating model RAM.
    if args.source_guard:
        shell+=' && . ./oom_guard.sh'
    else:
        shell+=' && echo 1000 > /proc/$$/oom_score_adj'
        shell+=' && test "$(cat /proc/$$/oom_score_adj)" = 1000'
        shell+=' && echo @@AWOOM@@ && cat /proc/$$/oom_score_adj'
    shell+=' && echo @@AWPID@@$$ && exec '+REMOTE+'/vlm_server '+shlex.quote(bundle+'/model/tokenizer.json')
    argv=c.adb+['shell',shell];save(output/'command.json',argv)
    (output/'command.txt').write_text(shlex.join(argv)+'\n')
    save(output/'settings.json',{'load_timeout_seconds':35,'request_timeout_seconds':600,
                               'instrumentation':False,'model_configuration_changed':False,
                               'test_process_oom_score_adj':1000,'source_package_oom_guard':args.source_guard})
    events=queue.Queue();stop=threading.Event();proc=None;kernel=None;pid=None
    old_stay=c.shell('settings get global stay_on_while_plugged_in').decode().strip()
    summary={'status':'starting','ready':False,'responses':[]};started=None
    def monitor():
        with (output/'memory_samples.jsonl').open('w',buffering=1) as stream:
            while not stop.is_set():
                try:
                    text=c.shell('cat /proc/meminfo; cat /proc/pressure/memory',timeout=4).decode()
                    stream.write(json.dumps({'host_perf_ns':time.perf_counter_ns(),'text':text})+'\n')
                except Exception as error:
                    stream.write(json.dumps({'host_perf_ns':time.perf_counter_ns(),'error':str(error)})+'\n')
                stop.wait(1)
    thread=threading.Thread(target=monitor,daemon=True)
    log=(output/'phone_raw.log').open('w',buffering=1)
    kernel_log=(output/'kernel_stream.log').open('w',buffering=1)
    reader=None
    def read():
        try:
            for line in proc.stdout:
                log.write(line)
                if '@@AWPID@@' in line:
                    events.put({'event':'pid','pid':int(line.split('@@AWPID@@',1)[1])})
                if '@@AWJSON@@' in line:
                    events.put(json.loads(line.split('@@AWJSON@@',1)[1]))
        finally:events.put({'event':'eof','returncode':proc.poll()})
    try:
        c.shell('svc power stayon usb');c.shell('input keyevent KEYCODE_WAKEUP')
        kernel=subprocess.Popen(c.adb+['shell','su -c '+shlex.quote('dmesg -w')],stdout=kernel_log,stderr=subprocess.STDOUT)
        thread.start();started=time.perf_counter()
        proc=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        reader=threading.Thread(target=read,daemon=True);reader.start()
        deadline=started+35
        while True:
            event=events.get(timeout=max(.001,deadline-time.perf_counter()))
            if event.get('event')=='pid':
                pid=event['pid'];summary['phone_pid']=pid;continue
            save(output/'load_event.json',event)
            summary['load_attempt_host_seconds']=time.perf_counter()-started
            if event.get('event')!='ready':raise RuntimeError(event)
            summary['ready']=True;summary['ready_event']=event;save(output/'ready.json',event)
            print('MODEL_READY',event,flush=True);break
        for command in commands:
            start=time.perf_counter();proc.stdin.write(command['line']+'\n');proc.stdin.flush()
            event=events.get(timeout=600);save(output/(command['case']+'_response.json'),event)
            if event.get('event')!='response':raise RuntimeError(event)
            event['host_roundtrip_seconds']=time.perf_counter()-start
            save(output/(command['case']+'_response.json'),event)
            summary['responses'].append({'case':command['case'],**event});save(output/'summary_partial.json',summary)
            print(command['case'],'input',event['prompt_tokens'],'output',event['generated_tokens'],
                  'TTFT',event['ttft_us']/1e6,'TPS',event['decoding_speed'],flush=True)
        proc.stdin.write('shutdown\n');proc.stdin.flush();proc.wait(timeout=20)
        summary['status']='completed'
    except Exception as error:
        summary['status']='load_failed_or_interrupted' if not summary['ready'] else 'request_failed_or_interrupted'
        summary['error']=repr(error)
        summary['attempt_host_seconds']=time.perf_counter()-started if started else None
        save(output/'failure.json',summary);print('FAILURE',summary['error'],flush=True)
    finally:
        if proc and proc.poll() is None:
            if pid:
                try:
                    # Kill only this run's recorded process, after verifying its executable.
                    command=f'test "$(cat /proc/{pid}/comm 2>/dev/null)" = vlm_server && kill -KILL {pid}'
                    c.shell(command,timeout=5)
                    summary['owned_process_killed']=pid
                except Exception as error:summary['kill_error']=repr(error)
            proc.terminate()
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
        stop.set()
        if thread.is_alive():thread.join(timeout=6)
        if reader:reader.join(timeout=5)
        if kernel:
            kernel.terminate()
            try:kernel.wait(timeout=5)
            except subprocess.TimeoutExpired:kernel.kill();kernel.wait(timeout=5)
        log.close();kernel_log.close()
        try:
            (output/'boot_after.txt').write_text(boot(c))
            c.shell('settings delete global stay_on_while_plugged_in' if old_stay=='null' else
                    'settings put global stay_on_while_plugged_in '+str(int(old_stay)),timeout=5)
            summary['settings_restored']=True
        except Exception as error:summary['cleanup_error']=repr(error)
        save(output/'summary.json',summary);print('RESULTS',output,summary['status'],flush=True)


if __name__=='__main__':main()
