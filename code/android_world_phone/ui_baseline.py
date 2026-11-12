"""Deterministic GUI transport/validator smoke test, NOT a VLM agent score."""
import json
from pathlib import Path
import time
from PIL import Image
from run_tasks import PhoneEnv, PhoneCreateFolder, PhoneCreateNote, MARKOR, ROOT
from android_world.env.json_action import JSONAction


def main():
    stamp=time.strftime('%Y%m%d_%H%M%S')
    folder=ROOT/'results'/('ui_baseline_'+stamp)
    folder.mkdir(parents=True)
    env=PhoneEnv('189f0fad')
    trace=[]

    def action(kind, match=None, **kwargs):
        state=env.get_state(True)
        step=folder/f'{len(trace):02d}'
        step.mkdir()
        Image.fromarray(state.pixels).save(step/'screen.png')
        (step/'ui.xml').write_text(env.last_xml)
        if match:
            matches=[i for i,e in enumerate(state.ui_elements) if match(e)]
            if not matches:
                raise RuntimeError('Baseline UI target not found')
            kwargs['index']=matches[0]
        a=JSONAction(action_type=kind,**kwargs)
        env.execute_action(a)
        trace.append(a.as_dict())
        (folder/'trace.json').write_text(json.dumps(trace,indent=2))
        print(a,flush=True)

    outcomes=[]
    for name,task in [
        ('folder',PhoneCreateFolder({'folder_name':'aw_baseline_folder_'+stamp})),
        ('note',PhoneCreateNote({'file_name':'aw_baseline_note_'+stamp+'.md',
                               'text':'AndroidWorld real phone transport works.'}))]:
        env.controller.shell('am force-stop net.gsantner.markor')
        env.controller.shell('am start -n '+MARKOR)
        task.initialize_task(env)
        initial=task.is_successful(env)
        action('click',lambda e:e.content_description=='Create a new file or folder')
        base=task.params.get('folder_name',task.params.get('file_name','').removesuffix('.md'))
        action('input_text',lambda e:e.resource_id=='net.gsantner.markor:id/new_file_dialog__name',
               text=base,clear_text=True)
        action('click',lambda e:e.text==('FOLDER' if name=='folder' else 'OK'))
        if name=='note':
            action('input_text',lambda e:'EditText' in (e.class_name or ''),
                   text=task.params['text'],clear_text=True)
            action('navigate_back')
            action('click',lambda e:e.content_description=='Save')
        score=task.is_successful(env)
        entry={'task':name,'params':task.params,'initial_score':initial,'final_score':score}
        outcomes.append(entry)
        (folder/'result.json').write_text(json.dumps({'mode':'scripted_ui_baseline_not_vlm',
                                                    'outcomes':outcomes},indent=2))
        print(entry,flush=True)
    final=env.get_state(True)
    Image.fromarray(final.pixels).save(folder/'final_screen.png')
    (folder/'result.json').write_text(json.dumps({'mode':'scripted_ui_baseline_not_vlm',
                                                'outcomes':outcomes},indent=2))
    print('RESULTS',folder,flush=True)


if __name__=='__main__': main()
