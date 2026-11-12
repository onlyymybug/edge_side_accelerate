"""Sequential A/B: identical saved two-image request, runner and sampler."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from run_tasks import ROOT, REMOTE, BUNDLE, ResidentVLM
from phone_env import PhoneEnv, PhoneController
from timing import Timeline


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--source',type=Path,default=ROOT/'results/m3a_preflight_20261106_135645')
    args=parser.parse_args()
    controller=PhoneController('189f0fad')
    active=controller.run(['shell','pidof vlm_server geniex-bench || true']).decode().strip()
    if active:
        raise RuntimeError('Wait for the current GUI/benchmark service to exit: '+active)
    local_hash=hashlib.sha256((ROOT/'native/vlm_server').read_bytes()).hexdigest()
    remote_hash=controller.run(['shell','sha256sum',REMOTE+'/vlm_server']).decode().split()[0]
    if local_hash!=remote_hash:
        raise RuntimeError('Deployed runner hash differs from local native/vlm_server')
    output=ROOT/'results'/('decode_ab_'+time.strftime('%Y%m%d_%H%M%S'));output.mkdir()
    paths=[args.source/'prompt.txt',args.source/'image_0.png',args.source/'image_1.png']
    for p in paths: (output/p.name).write_bytes(p.read_bytes())
    records=[]; old_stay=None; env=None
    try:
        for label,bundle in [('4k_before',BUNDLE),('16k','/data/local/tmp/qwen3_vl_geniex_ctx16384'),
                             ('4k_after',BUNDLE)]:
            folder=output/label;folder.mkdir()
            trace=Timeline(folder/'host_timeline.json');env=PhoneEnv('189f0fad',trace)
            if old_stay is None:
                old_stay=env.controller.shell('settings get global stay_on_while_plugged_in').decode().strip()
                env.controller.shell('svc power stayon usb')
                env.controller.shell('input keyevent KEYCODE_WAKEUP')
                env.controller.shell('wm dismiss-keyguard')
            (folder/'battery_before.txt').write_bytes(env.controller.shell('dumpsys battery'))
            (folder/'meminfo_before_load.txt').write_bytes(env.controller.run(['shell','cat','/proc/meminfo']))
            (folder/'genie_config.json').write_bytes(env.controller.run(['shell','cat',bundle+'/model/genie_config.json']))
            model=None
            try:
                model=ResidentVLM(env,folder,bundle=bundle,request_timeout=300)
                (folder/'ready.json').write_text(json.dumps(model.ready,indent=2))
                (folder/'meminfo_after_load.txt').write_bytes(env.controller.run(['shell','cat','/proc/meminfo']))
                for i in range(args.repeats):
                    r=model.generate_images(f'{label}-{i}',output/'prompt.txt',
                        [output/'image_0.png',output/'image_1.png'],192)
                    (folder/f'response_{i}.json').write_text(json.dumps(r,indent=2,ensure_ascii=False))
                    records.append({'bundle':label,'repeat':i,**r})
                    print(label,i,r['decoding_speed'],r['prompt_tokens'],r['generated_tokens'],flush=True)
                (folder/'meminfo_after_generation.txt').write_bytes(env.controller.run(['shell','cat','/proc/meminfo']))
                (folder/'battery_after.txt').write_bytes(env.controller.shell('dumpsys battery'))
            finally:
                if model:model.close()
                trace.write()
        summary={'run':str(output),'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                 'runner_sha256':hashlib.sha256((ROOT/'native/vlm_server').read_bytes()).hexdigest(),
                 'power_mode':'BURST','max_new_tokens':192,'temperature':0,'seed':42,'repeats_per_session':args.repeats,
                 'order':['4k_before','16k','4k_after'],
                 'same_images':True,'same_prompt':True,'kv_reset_each_request':True,
                 'bundle_results':{},'responses':records}
        for label in ['4k_before','16k','4k_after']:
            rows=[r for r in records if r['bundle']==label]
            summary['bundle_results'][label]={'decode_tps_median':statistics.median(r['decoding_speed'] for r in rows),
                'decode_tps_pooled':sum(r['generated_tokens'] for r in rows)/sum(r['decode_time_us']/1e6 for r in rows),
                'generated_tokens':[r['generated_tokens'] for r in rows],
                'prompt_tokens':[r['prompt_tokens'] for r in rows],
                'ttft_seconds':[r['ttft_us']/1e6 for r in rows],
                'elapsed_seconds':[r['elapsed_seconds'] for r in rows]}
        baseline=[r for r in records if r['bundle'].startswith('4k_')]
        old=sum(r['generated_tokens'] for r in baseline)/sum(r['decode_time_us']/1e6 for r in baseline)
        new=summary['bundle_results']['16k']['decode_tps_pooled']
        summary['baseline_decode_tps_pooled']=old
        summary['decode_throughput_decrease_fraction']=1-new/old
        summary['decode_seconds_per_token_multiplier']=old/new
        summary['observed_prompt_token_counts']=sorted({r['prompt_tokens'] for r in records})
        summary['observed_generated_token_counts']=sorted({r['generated_tokens'] for r in records})
        summary['output_texts_identical']=len({r['text'] for r in records})==1
        summary['limitations']=['Small sequential A/B/A sample on one phone; not randomized for thermal/order effects.',
            'Both are W4A16 with same runtime; entire export bundles differ, including graph shapes/prefill AR.',
            'Token throughput uses SDK generated_tokens/decode_time; observed token counts and output equality are recorded.',
            'Does not by itself establish the internal kernel or memory mechanism causing a slowdown.']
        (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
        report=['# 相同双图输入的 decode 对照\n',
                '同一手机、runner、BURST、温度0、seed42、max_new_tokens192、同提示词/图片；每请求重置KV。\n',
                '| 包 | 两次 decode token/s | 合并 token/s |\n|---|---|---:|']
        for label in ['4k_before','16k','4k_after']:
            rates=', '.join(f"{r['decoding_speed']:.3f}" for r in records if r['bundle']==label)
            report.append(f"| {label} | {rates} | {summary['bundle_results'][label]['decode_tps_pooled']:.3f} |")
        report.append(f"\n吞吐下降 {1-new/old:.2%}，每 token 耗时为原来的 {old/new:.2f} 倍。\n")
        report.append(f"输入token数集合{summary['observed_prompt_token_counts']}，输出token数集合{summary['observed_generated_token_counts']}，输出文本完全相同：{summary['output_texts_identical']}。\n")
        report.extend('- '+s for s in summary['limitations'])
        (output/'REPORT.md').write_text('\n'.join(report))
        print('RESULTS',output,flush=True)
    finally:
        if env and old_stay is not None:
            command=('settings delete global stay_on_while_plugged_in' if old_stay=='null' else
                     'settings put global stay_on_while_plugged_in '+str(int(old_stay)))
            env.controller.shell(command)


if __name__=='__main__': main()
