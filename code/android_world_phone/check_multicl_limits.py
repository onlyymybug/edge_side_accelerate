"""Check recorded near-limit official request and explicit overflow rejection."""
import argparse
import json
from pathlib import Path
import time
from tokenizers import Tokenizer
from run_tasks import ROOT, REMOTE, ResidentVLM
from phone_env import PhoneController, PhoneEnv
from timing import Timeline
from validate_multicl_bundle import NEW, HOST, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=ROOT/'results/multicl8k_acceptance_20261110_141629')
    args=parser.parse_args()
    controller=PhoneController('189f0fad')
    if controller.shell('pidof vlm_server geniex-bench || true').decode().strip():
        raise RuntimeError('Another service is running')
    parent=args.run
    folder=parent/'limits'
    if folder.exists():folder=parent/('limits_'+time.strftime('%Y%m%d_%H%M%S'))
    folder.mkdir()
    original=ROOT/'results/official_m3a_20261108_010622/rep01_note/03/summary'
    (folder/'prompt.txt').write_bytes((original/'prompt.txt').read_bytes())
    for i in range(2):(folder/f'image_{i}.png').write_bytes((original/f'model_{i}.png').read_bytes())
    text=(folder/'prompt.txt').read_text()
    overflow='Additional reference context.\n'+('This additional note is background only.\n'*60)+text
    (folder/'overflow.txt').write_text(overflow)
    tokenizer=Tokenizer.from_file(str(HOST/'model/tokenizer.json'))
    write_json(folder/'input_counts.json',{'near_limit_text_tokens':len(tokenizer.encode(text,add_special_tokens=False).ids),
       'overflow_text_tokens':len(tokenizer.encode(overflow,add_special_tokens=False).ids)})
    old_stay=controller.shell('settings get global stay_on_while_plugged_in').decode().strip()
    model=None;trace=Timeline(folder/'host_timeline.json')
    try:
        controller.shell('svc power stayon usb');controller.shell('input keyevent KEYCODE_WAKEUP')
        env=PhoneEnv('189f0fad',trace)
        exports={'LD_PRELOAD':REMOTE+'/libaw_decode_diag.so','AW_DIAG_CORE_PATH':NEW+'/lib/qairt/libgeniex_core.so',
                 'AW_DIAG_API_PATH':NEW+'/lib/libgeniex.so','AW_FORCE_CL':'0'}
        model=ResidentVLM(env,folder,bundle=NEW,request_timeout=180,extra_exports=exports)
        write_json(folder/'ready.json',model.ready)
        images=[folder/'image_0.png',folder/'image_1.png']
        try:
            response=model.generate_images('limit-official8120',folder/'prompt.txt',images,64)
            write_json(folder/'response_near_limit.json',response)
            previous=json.loads((original/'response.json').read_text())['response']
            write_json(folder/'near_limit_comparison.json',{'same_text_as_recorded_16k':response['text']==previous['text'],
                'old_output':previous['text'],'new_output':response['text']})
            print('NEAR_LIMIT',response['prompt_tokens'],response['generated_tokens'],response['text'],flush=True)
        except Exception as error:
            write_json(folder/'near_limit_error.json',{'error':str(error)})
            print('NEAR_LIMIT_ERROR',str(error),flush=True)
        try:
            response=model.generate_images('limit-overflow',folder/'overflow.txt',images,64)
            write_json(folder/'overflow_response.json',response)
            print('OVERFLOW_UNEXPECTED_RESPONSE',response['prompt_tokens'],response['stop_reason'],flush=True)
        except Exception as error:
            write_json(folder/'overflow_error.json',{'error':str(error)})
            print('OVERFLOW_ERROR',str(error),flush=True)
    finally:
        if model:model.close()
        trace.write()
        controller.shell('settings delete global stay_on_while_plugged_in' if old_stay=='null' else
                         'settings put global stay_on_while_plugged_in '+str(int(old_stay)))
    print('RESULTS',folder,flush=True)


if __name__=='__main__':main()
