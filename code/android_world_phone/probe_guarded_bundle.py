"""Run a bundle's guarded smoke script once and preserve device/kernel evidence."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time

from run_tasks import ROOT
from phone_env import PhoneController
from retry_multicl16k import boot, save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();folder=args.output;folder.mkdir()
    c=PhoneController('189f0fad')
    if c.shell('pidof vlm_server geniex-bench || true').strip():raise RuntimeError('Model process active')
    (folder/'boot_before.txt').write_text(boot(c))
    old_stay=c.shell('settings get global stay_on_while_plugged_in').decode().strip()
    argv=c.adb+['shell','sh '+shlex.quote(args.bundle+'/run_smoke.sh')]
    save(folder/'command.json',argv);(folder/'command.txt').write_text(shlex.join(argv)+'\n')
    stop=threading.Event();kernel=None;proc=None;start=None
    outcome={'status':'starting','timeout_seconds':60,'automatic_retries':0}
    def monitor():
        with (folder/'memory_samples.jsonl').open('w',buffering=1) as stream:
            while not stop.is_set():
                try:
                    text=c.shell('cat /proc/meminfo; cat /proc/pressure/memory',timeout=4).decode()
                    stream.write(json.dumps({'host_ns':time.perf_counter_ns(),'text':text})+'\n')
                except Exception as error:stream.write(json.dumps({'error':repr(error)})+'\n')
                stop.wait(1)
    thread=threading.Thread(target=monitor,daemon=True)
    raw=(folder/'phone_raw.log').open('w',buffering=1)
    kernel_raw=(folder/'kernel_stream.log').open('w',buffering=1)
    try:
        c.shell('svc power stayon usb');c.shell('input keyevent KEYCODE_WAKEUP')
        kernel=subprocess.Popen(c.adb+['shell','su -c '+shlex.quote('dmesg -w')],stdout=kernel_raw,stderr=subprocess.STDOUT)
        thread.start();start=time.perf_counter()
        proc=subprocess.Popen(argv,stdout=raw,stderr=subprocess.STDOUT)
        proc.wait(timeout=60)
        outcome.update(status='completed' if proc.returncode==0 else 'process_failed',returncode=proc.returncode,
                       host_process_seconds=time.perf_counter()-start)
    except Exception as error:
        outcome.update(status='interrupted',error=repr(error),host_process_seconds=time.perf_counter()-start if start else None)
    finally:
        log=(folder/'phone_raw.log').read_text()
        guard=re.search(r'OOM_GUARD_OK pid=(\d+) oom_score_adj=(\d+)',log)
        outcome['oom_guard_confirmed']=bool(guard and guard.group(2)=='1000')
        if guard:outcome['phone_pid']=int(guard.group(1))
        if proc and proc.poll() is None:
            if guard:
                try:c.shell('kill -KILL '+guard.group(1),timeout=5)
                except Exception as error:outcome['kill_error']=repr(error)
            proc.terminate()
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
        stop.set()
        if thread.is_alive():thread.join(timeout=6)
        if kernel:
            kernel.terminate()
            try:kernel.wait(timeout=5)
            except subprocess.TimeoutExpired:kernel.kill();kernel.wait(timeout=5)
        raw.close();kernel_raw.close()
        try:
            (folder/'boot_after.txt').write_text(boot(c))
            c.shell('settings delete global stay_on_while_plugged_in' if old_stay=='null' else
                    'settings put global stay_on_while_plugged_in '+str(int(old_stay)),timeout=5)
            outcome['settings_restored']=True
            result=subprocess.run(c.adb+['pull',args.bundle+'/results',str(folder/'phone_results')],capture_output=True,text=True,timeout=15)
            (folder/'pull.log').write_text(result.stdout+result.stderr)
            outcome['results_pull_returncode']=result.returncode
        except Exception as error:outcome['cleanup_error']=repr(error)
        save(folder/'outcome.json',outcome);print('OUTCOME',json.dumps(outcome),flush=True)


if __name__=='__main__':main()
