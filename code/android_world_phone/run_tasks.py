"""Real-phone pilot: original AndroidWorld goals/validators, isolated setup."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
UPSTREAM = ROOT.parent / 'android_world_upstream'
sys.path.insert(0, str(UPSTREAM))
from android_world.env import json_action, device_constants
from android_world.task_evals.single import markor
from android_world.utils import file_utils
from PIL import Image
from phone_env import PhoneEnv, MARKOR
from timing import Timeline

REMOTE = '/data/local/tmp/android_world_phone'
BUNDLE = '/data/local/tmp/qwen3_vl_geniex_sm8750'


class PersonalPhoneSetup:
    """Replace destructive emulator setup only; inherit upstream is_successful."""
    def initialize_task(self, env):
        if self.initialized:
            raise RuntimeError('Already initialized')
        name = self.params.get('folder_name', self.params.get('file_name'))
        if '/' in name or file_utils.check_file_or_folder_exists(name, device_constants.MARKOR_DATA, env.controller):
            raise RuntimeError('Test target must be a fresh basename')
        env.interaction_cache = ''
        self.initialized = True
        if hasattr(self, 'create_file_task'):
            self.create_file_task.initialized = True

    def tear_down(self, env):
        self.initialized = False


class PhoneCreateFolder(PersonalPhoneSetup, markor.MarkorCreateFolder):
    pass


class PhoneCreateNote(PersonalPhoneSetup, markor.MarkorCreateNote):
    pass


class ResidentVLM:
    def __init__(self, env, folder, bundle=BUNDLE, request_timeout=120, extra_exports=None):
        self.env = env
        self.folder = folder
        self.timeline = env.timeline
        self.request_timeout = request_timeout
        self.events = queue.Queue()
        exports = {
            'LD_LIBRARY_PATH': f'{bundle}/lib:{bundle}/lib/qairt:{bundle}/lib/qairt/htp-files',
            'GENIEX_PLUGIN_PATH': f'{bundle}/lib',
            'ADSP_LIBRARY_PATH': f'{bundle}/lib/qairt/htp-files;/vendor/lib/rfsa/adsp;/vendor/dsp;/system/lib/rfsa/adsp;/dsp',
        }
        exports.update(extra_exports or {})
        command = 'cd ' + shlex.quote(bundle) + ' && ' + ' && '.join('export '+k+'='+shlex.quote(v) for k,v in exports.items())
        # adb shell on this phone inherits -1000: do not protect model loading
        # from OOM at the expense of Android system and touch services.
        command += ' && echo 1000 > /proc/$$/oom_score_adj && test "$(cat /proc/$$/oom_score_adj)" = 1000'
        command += f' && exec {REMOTE}/vlm_server '+shlex.quote(bundle+'/model/tokenizer.json')
        self.command = env.controller.adb + ['shell', command]
        (folder/'command.json').write_text(json.dumps(self.command, indent=2))
        self.log = open(folder/'phone_raw.log', 'w', buffering=1)
        with self.timeline.span('model_load', kind='phase'):
            self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1)
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
            self.ready = self.receive(120)
        self.ready_host_ns = time.perf_counter_ns()
        if self.ready.get('event') != 'ready':
            raise RuntimeError(self.ready)
        print('MODEL_READY', json.dumps(self.ready), flush=True)

    def _read(self):
        try:
            for line in self.proc.stdout:
                self.log.write(line)
                prefix = '@@AWJSON@@'
                if prefix in line:
                    try:
                        self.events.put(json.loads(line.split(prefix,1)[1]))
                    except json.JSONDecodeError as error:
                        self.events.put({'event':'error','error':str(error)})
        finally:
            self.events.put({'event':'eof','returncode':self.proc.poll()})

    def receive(self, timeout=120):
        return self.events.get(timeout=timeout)

    def generate(self, request_id, prompt_path, image_path):
        return self.generate_images(request_id,prompt_path,[image_path],224)

    def generate_images(self, request_id, prompt_path, image_paths, max_tokens=512):
        start = time.perf_counter()
        remote_prompt=f'{request_id}_prompt.txt'
        remote_images=[f'{request_id}_screen_{i}.png' for i in range(len(image_paths))]
        with self.timeline.span('request_upload',kind='phase',request_id=request_id) as upload:
            for local, remote in [(prompt_path,remote_prompt), *zip(image_paths,remote_images)]:
                self.env.controller.run(['push', str(local), f'{REMOTE}/{remote}'])
        with self.timeline.span('inference_request',kind='phase',request_id=request_id) as request:
            fields=[request_id,f'{REMOTE}/{remote_prompt}',
                    *[f'{REMOTE}/{p}' for p in remote_images],str(max_tokens)]
            self.proc.stdin.write('\t'.join(fields)+'\n')
            self.proc.stdin.flush()
            response = self.receive(self.request_timeout)
        if response.get('event') != 'response' or response.get('id') != request_id:
            raise RuntimeError(response)
        response['host_request_seconds'] = time.perf_counter()-start
        response['host_upload_seconds'] = upload['seconds']
        response['host_roundtrip_seconds'] = request['seconds']
        response['host_send_ns'] = request['start_ns']
        response['host_response_ns'] = request['end_ns']
        response['model_image_sha256s']=[hashlib.sha256(p.read_bytes()).hexdigest() for p in image_paths]
        response['remote_image_paths']=[f'{REMOTE}/{p}' for p in remote_images]
        if len(image_paths)==1:
            response['model_image_sha256']=response['model_image_sha256s'][0]
            response['remote_image_path']=response['remote_image_paths'][0]
        return response

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write('shutdown\n'); self.proc.stdin.flush()
                self.proc.wait(timeout=20)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.proc.terminate()
                self.proc.wait(timeout=10)
        self.thread.join(timeout=5)
        self.log.close()
        self.session_stats = None
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event.get('event') == 'session_end':
                self.session_stats = event
        if self.session_stats:
            (self.folder/'service_stats.json').write_text(json.dumps(self.session_stats,indent=2))


def prompt_for(goal, state, history):
    elements = []
    for i,e in enumerate(state.ui_elements):
        box = e.bbox_pixels
        row = {'index':i,'text':e.text,'description':e.content_description,
               'bounds':[box.x_min,box.y_min,box.x_max,box.y_max]}
        if e.is_focused: row['focused'] = True
        if e.class_name and 'EditText' in e.class_name: row['editable'] = True
        if e.is_scrollable: row['scrollable'] = True
        elements.append({k:v for k,v in row.items() if v is not None and v != ''})
    w,h = state.pixels.shape[1],state.pixels.shape[0]
    past = '\n'.join(f"Step {i}: {r}" for i,r in enumerate(history[-4:])) or 'None'
    return f'''You are an Android GUI agent. Choose the NEXT action that makes progress on the task.
Task: {goal}
First explain in at most 20 words what to do next.
Then output exactly ONE JSON action. Format: Reason: ...\nAction: {{"action_type":...}}
Available actions:
{{"action_type":"click","index":<target_index>}}
{{"action_type":"input_text","index":<editable_index>,"text":"<text_to_type>","clear_text":true}}
{{"action_type":"navigate_back"}}
{{"action_type":"keyboard_enter"}}
{{"action_type":"scroll","direction":"down"}}
{{"action_type":"open_app","app_name":"markor"}}
{{"action_type":"wait"}}
{{"action_type":"status","goal_status":"complete"}}
Indices refer ONLY to the CURRENT UI list below. Prefer indices to coordinates.
Replace index placeholders with the current numeric index, and text placeholders with the requested text.
Input_text clicks its target field then types. Use clear_text to replace existing text.
For input_text choose a field marked editable:true. A toolbar, title or label is not editable.
If needed, click can use x,y in ORIGINAL {w}x{h} screen pixels.
Use only Markor. Do not delete any existing files. Finish the full requested task.
If repeated taps do not change the screen, try a different action or navigate_back.
Check the target type and content carefully; a file and a folder are different.
Before confirming a dialog, fill the requested exact name and other required fields.
When name and extension are separate fields, do not duplicate the extension in the name.
Dialogs may have different buttons for creating files and folders: choose the requested type.
After editing a note, use the app's Save button to persist the text before finishing.
Previous actions (past indices are obsolete):
{past}
Current UI elements: {json.dumps(elements,ensure_ascii=False,separators=(',',':'))}
Look at the current UI and task. Select the next useful action using a CURRENT index.
Reason and next Action:
'''


def parse_action(text):
    decoder = json.JSONDecoder()
    for i,c in enumerate(text):
        if c == '{':
            try:
                obj,end = decoder.raw_decode(text[i:])
                return json_action.JSONAction(**obj)
            except (ValueError,TypeError):
                continue
    raise ValueError('Model output has no valid JSONAction')


def run_task(task, env, model, folder, max_steps):
    trace=env.timeline
    trace.tags={'task':folder.name}
    with trace.span('task_setup',kind='phase'):
        folder.mkdir()
        task.initialize_task(env)
        result = {'task':task.__class__.__mro__[2].__name__, 'goal':task.goal,'params':task.params,
                  'validator':f'{type(task).__mro__[2].__module__}.{type(task).__mro__[2].__name__}.is_successful',
                  'initial_score':task.is_successful(env),'steps':[], 'success':False}
    if result['initial_score'] != 0:
        raise RuntimeError('Target already satisfied before agent')
    history=[]
    with trace.span('task_loop',kind='window') as loop:
        for step in range(max_steps):
            trace.tags={'task':folder.name,'step':step}
            with trace.span('step',kind='window'):
                stepdir=folder/f'{step:02d}'
                stepdir.mkdir()
                with trace.span('observe',kind='phase'):
                    state=env.get_state(wait_to_stabilize=True)
                with trace.span('prepare_inputs',kind='phase'):
                    Image.fromarray(state.pixels).save(stepdir/'screen.png')
                    # Preserve aspect ratio; the converted model expects 512x512.
                    im=Image.fromarray(state.pixels); im.thumbnail((512,512))
                    canvas=Image.new('RGB',(512,512),'white'); canvas.paste(im,((512-im.width)//2,0))
                    canvas.save(stepdir/'model_screen.png')
                    (stepdir/'ui.xml').write_text(env.last_xml)
                    prompt_path=stepdir/'prompt.txt'
                    prompt_path.write_text(prompt_for(task.goal,state,history))
                response=model.generate(f'{folder.name}-{step}',prompt_path,stepdir/'model_screen.png')
                record={'step':step,'response':response}
                try:
                    with trace.span('action_parse',kind='phase'):
                        action=parse_action(response['text'])
                        record['action']=action.as_dict()
                    with trace.span('actuate',kind='phase',action_type=action.action_type):
                        env.execute_action(action)
                    with trace.span('validate',kind='phase'):
                        record['score']=task.is_successful(env)
                    with trace.span('feedback',kind='phase'):
                        if action.action_type == 'status' and action.goal_status == 'complete' and record['score'] == 0:
                            record['completion_rejected'] = True
                        target = ''
                        if action.index is not None and 0 <= action.index < len(state.ui_elements):
                            e = state.ui_elements[action.index]
                            target = e.text or e.content_description or e.resource_id or ''
                        history.append(f'{action.action_type} on {target!r}' +
                                       (f' with text {action.text!r}' if action.text is not None else '') +
                                       '; task still incomplete' * (record['score'] == 0))
                        if record.get('completion_rejected'):
                            history.append('Completion rejected by the task verifier. Re-check the current screen, '
                                           'save or confirm pending changes, and choose a different action. '
                                           'Typing visible text again does not save it.')
                except (ValueError,TypeError) as error:
                    record['error']=str(error)
                    history.append(f'Action rejected: {error}. No action completed. Choose a different valid current target.')
                with trace.span('artifacts',kind='phase'):
                    (stepdir/'response.json').write_text(json.dumps(record,indent=2,ensure_ascii=False))
                    result['steps'].append(record)
                    result['elapsed_seconds']=(time.perf_counter_ns()-loop['start_ns'])/1e9
                    (folder/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
                print(folder.name,step,json.dumps(record.get('action',record.get('error')),ensure_ascii=False),
                      'score',record.get('score'),'vlm_seconds',response['elapsed_seconds'],flush=True)
            if record.get('score') == 1:
                result['success']=True
                break
            if record.get('action',{}).get('action_type') == 'status' and not record.get('completion_rejected'):
                break
    trace.tags={'task':folder.name}
    result['elapsed_seconds']=loop['seconds']
    result['loop_start_host_ns']=loop['start_ns']
    result['loop_end_host_ns']=loop['end_ns']
    with trace.span('final_observe',kind='phase'):
        final=env.get_state(True)
        Image.fromarray(final.pixels).save(folder/'final_screen.png')
    with trace.span('task_finalize',kind='phase'):
        (folder/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    trace.tags={}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--serial',default='189f0fad')
    parser.add_argument('--tasks',choices=['folder','note','both'],default='both')
    parser.add_argument('--max-steps',type=int,default=16)
    parser.add_argument('--note-name',default=None)
    parser.add_argument('--note-text',default='Hello AndroidWorld from Qwen3 VL on this phone.')
    parser.add_argument('--repeats',type=int,default=1)
    args=parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    stamp=time.strftime('%Y%m%d_%H%M%S')
    folder=ROOT/'results'/stamp; folder.mkdir(parents=True)
    trace=Timeline(folder/'host_timeline.json')
    with trace.span('device_setup',kind='phase'):
        env=PhoneEnv(args.serial,trace)
        env.controller.shell('input keyevent KEYCODE_WAKEUP')
        env.controller.shell('wm dismiss-keyguard')
        env.controller.run(['shell','mkdir -p '+REMOTE])
        env.controller.run(['push',str(ROOT/'native/vlm_server'),REMOTE+'/vlm_server'])
        env.controller.shell('chmod 755 '+REMOTE+'/vlm_server')
    manifest={'upstream_commit':subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip(),
              'argv':sys.argv,'serial':args.serial,'adaptations':['ADB/root transport','UIAutomator XML instead of emulator accessibility RPC',
              'No storage clearing, snapshots, clock change or noise files','Markor opened before each task; 512x512 letterboxed screenshot'],
              'screen_size':env.device_screen_size}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    model=None
    try:
        model=ResidentVLM(env,folder)
        manifest['model_ready']=model.ready
        outcomes=[]
        tasks=[]
        for repeat in range(args.repeats):
            suffix=f'_r{repeat+1:02d}' if args.repeats>1 else ''
            prefix=f'rep{repeat+1:02d}_' if args.repeats>1 else ''
            if args.tasks in ('folder','both'):
                tasks.append((prefix+'folder',PhoneCreateFolder({'folder_name':'aw_folder_'+stamp+suffix})))
            if args.tasks in ('note','both'):
                note_name=args.note_name or 'meeting_'+stamp+'.md'
                if suffix:
                    p=Path(note_name); note_name=p.stem+suffix+p.suffix
                tasks.append((prefix+'note',PhoneCreateNote({'file_name':note_name,'text':args.note_text})))
        for name,task in tasks:
            trace.tags={'task':name}
            with trace.span('app_setup',kind='phase'):
                env.controller.shell('am force-stop net.gsantner.markor')
                env.controller.shell('am start -n '+MARKOR)
                trace.sleep(1,'app_launch_fixed_wait')
            outcomes.append(run_task(task,env,model,folder/name,args.max_steps))
        manifest['outcomes']=[{k:r[k] for k in ('task','success','elapsed_seconds')} for r in outcomes]
        print('RESULTS',folder, json.dumps(manifest['outcomes']),flush=True)
    except Exception:
        manifest['error']=traceback.format_exc()
        raise
    finally:
        trace.tags={}
        if model:
            with trace.span('model_close',kind='phase'):
                model.close()
            manifest['service_stats_available']=model.session_stats is not None
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
        trace.write()


if __name__ == '__main__':
    main()
