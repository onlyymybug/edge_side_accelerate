"""Execute unmodified upstream M3A and episode_runner against the real phone."""
import argparse
from contextlib import ExitStack
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
from unittest.mock import patch

from PIL import Image
import numpy as np
from tokenizers import Tokenizer
from run_tasks import ROOT, WORKSPACE, UPSTREAM, REMOTE, BUNDLE, ResidentVLM, PhoneCreateFolder, PhoneCreateNote
from official_m3a_env import OfficialPhoneEnv
from timing import Timeline
from android_world import episode_runner
from android_world.agents import m3a, base_agent, infer
from android_world.env import actuation


class TimedTime:
    """Instrument module-local time binding without patching global time.sleep."""
    def __init__(self, trace, name, kind):
        self.trace,self.name,self.kind=trace,name,kind
    def __getattr__(self,name): return getattr(time,name)
    def sleep(self,seconds):
        with self.trace.span(self.name,kind=self.kind,requested_seconds=seconds):
            time.sleep(seconds)


def serialize(value, directory, key='data'):
    if isinstance(value,np.ndarray):
        filename=key+'.png'
        Image.fromarray(value).save(directory/filename)
        return {'image_file':filename,'shape':list(value.shape)}
    if dataclasses.is_dataclass(value):
        return serialize(dataclasses.asdict(value),directory,key)
    if isinstance(value,dict):
        return {str(k):serialize(v,directory,key+'_'+str(k)) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [serialize(v,directory,key+'_'+str(i)) for i,v in enumerate(value)]
    if isinstance(value,np.generic): return value.item()
    return value


class ContextSpaceError(RuntimeError):
    def __init__(self,estimated,available,requested):
        self.estimated,self.available,self.requested=estimated,available,requested
        super().__init__(f'Full prompt needs {estimated} tokens; only {available} output tokens remain, requested {requested}')


class GenieXM3AWrapper(infer.MultimodalLlmWrapper):
    def __init__(self, model, trace, output, max_tokens, context_size=None, skip_tight=False):
        self.model,self.trace,self.output,self.max_tokens=model,trace,output,max_tokens
        self.context_size=context_size
        self.skip_tight=skip_tight
        self.tokenizer=Tokenizer.from_file(str(WORKSPACE/'phone_bundle/model/tokenizer.json'))
        self.task=None; self.step=0; self.calls=[]; self.all_calls=[]
    def begin_step(self,task,step):
        self.task,self.step,self.calls=task,step,[]
    def predict_mm(self,text_prompt,images):
        role='action' if not self.calls else 'summary'
        request_id=f'{self.task}-{self.step}-{role}'
        directory=self.output/self.task/f'{self.step:02d}'/role
        directory.mkdir(parents=True,exist_ok=True)
        self.trace.tags={'task':self.task,'step':self.step,'model_role':role}
        with self.trace.span('prepare_'+role,kind='phase'):
            prompt=directory/'prompt.txt'; prompt.write_text(text_prompt)
            paths=[]
            for i,array in enumerate(images):
                Image.fromarray(array).save(directory/f'original_{i}.png')
                image=Image.fromarray(array); image.thumbnail((512,512))
                canvas=Image.new('RGB',(512,512),'white')
                canvas.paste(image,((512-image.width)//2,0))
                path=directory/f'model_{i}.png'; canvas.save(path); paths.append(path)
            record={'id':request_id,'role':role,'image_count':len(images),
                    'text_tokens_without_chat_template':len(self.tokenizer.encode(text_prompt,add_special_tokens=False).ids),
                    'max_new_tokens':self.max_tokens,'prompt_sha256':hashlib.sha256(text_prompt.encode()).hexdigest(),
                    'status':'started'}
            actual_max_tokens=self.max_tokens
            if self.context_size is not None:
                # This fixed vision encoder emits 256 tokens/image. The C API
                # template adds 25 other tokens, checked against saved runs.
                estimated=record['text_tokens_without_chat_template']+256*len(images)+25
                available=self.context_size-estimated-1
                if self.skip_tight and available<self.max_tokens:
                    record.update(status='skipped_context',estimated_runtime_prompt_tokens=estimated,
                                  remaining_output_tokens=available,reason='Insufficient room for the unchanged output limit')
                    (directory/'context_skip.json').write_text(json.dumps(record,indent=2))
                    raise ContextSpaceError(estimated,available,self.max_tokens)
                actual_max_tokens=min(self.max_tokens,self.context_size-estimated-1)
                record.update(configured_max_new_tokens=self.max_tokens,
                              estimated_runtime_prompt_tokens=estimated,
                              max_new_tokens=actual_max_tokens,
                              output_budget_capped=actual_max_tokens<self.max_tokens)
                if actual_max_tokens<1:
                    raise RuntimeError('Full official prompt leaves no generation space in this bundle')
            (directory/'request.json').write_text(json.dumps(record,indent=2))
        print('REQUEST',request_id,'text_tokens',record['text_tokens_without_chat_template'],'images',len(images),flush=True)
        try:
            response=self.model.generate_images(request_id,prompt,paths,actual_max_tokens)
            record.update(status='ok',response=response)
        except Exception:
            record.update(status='error',error=traceback.format_exc())
            raise
        finally:
            self.calls.append(record);self.all_calls.append(record)
            with self.trace.span('request_artifacts',kind='phase'):
                (directory/'response.json').write_text(json.dumps(record,indent=2))
            self.trace.tags={'task':self.task,'step':self.step}
        print('RESPONSE',request_id,'seconds',response['elapsed_seconds'],
              'prompt_tokens',response['prompt_tokens'],'generated_tokens',response['generated_tokens'],flush=True)
        # None means no safety classifier is implemented in this local backend.
        return response['text'],None,response


class MeasuredM3A(m3a.M3A):
    """Only wrap step boundaries/artifacts; delegate every decision to upstream."""
    def __init__(self,env,llm,trace,folder):
        super().__init__(env,llm)
        self.trace,self.folder=trace,folder
        self.records=[]
    def step(self,goal):
        index=len(self.records)
        self.llm.begin_step(self.folder.name,index)
        self.trace.tags={'task':self.folder.name,'step':index}
        self.env.observation_role='observe_before'
        with self.trace.span('step',kind='window') as span:
            result=super().step(goal)
        directory=self.folder/f'{index:02d}';directory.mkdir(exist_ok=True)
        record={'step':index,'done':result.done,'seconds':span['seconds'],
                'requests':[r['id'] for r in self.llm.calls]}
        with self.trace.span('step_artifacts',kind='phase'):
            record['data']=serialize(result.data,directory)
            (directory/'step.json').write_text(json.dumps(record,indent=2,ensure_ascii=False))
        self.records.append(record)
        print('STEP',self.folder.name,index,'done',result.done,'calls',len(self.llm.calls),
              'summary',str(result.data.get('summary'))[:180],flush=True)
        return result


def run_task(task, env, model, trace, folder, max_tokens, context_size=None, skip_tight=False):
    folder.mkdir()
    trace.tags={'task':folder.name}
    with trace.span('task_setup',kind='phase'):
        task.initialize_task(env)
        initial=task.is_successful(env)
    if initial:
        raise RuntimeError('Task already satisfied before execution')
    wrapper=GenieXM3AWrapper(model,trace,folder.parent,max_tokens,context_size=context_size,skip_tight=skip_tight)
    agent=MeasuredM3A(env,wrapper,trace,folder)
    budget=int(10*task.complexity)  # upstream suite_utils._allocate_step_budget
    result={'task':type(task).__mro__[2].__name__,'goal':task.goal,'params':task.params,
            'initial_score':initial,'step_budget':budget,'success':False}
    try:
        with trace.span('task_loop',kind='window',task=folder.name) as window:
            episode=episode_runner.run_episode(goal=task.goal,agent=agent,max_n_steps=budget,
                start_on_home_screen=task.start_on_home_screen,
                print_fn=lambda message:print(folder.name,message,flush=True))
        result.update(agent_done=episode.done,elapsed_seconds=window['seconds'])
        trace.tags={'task':folder.name}
        with trace.span('validate_final',kind='phase'):
            score=task.is_successful(env)
        result.update(final_score=score,success=bool(episode.done and score==1))
    except BaseException as error:
        result['error']=traceback.format_exc()
        if isinstance(error,ContextSpaceError):
            result.update(skipped=True,skip_reason='context_budget',estimated_prompt_tokens=error.estimated,
                          remaining_output_tokens=error.available,requested_output_tokens=error.requested)
        result['elapsed_seconds']=(time.perf_counter_ns()-window['start_ns'])/1e9
        print(result['error'],flush=True)
        if not isinstance(error,Exception):
            raise
    finally:
        result['steps']=agent.records
        result['request_count']=len(wrapper.all_calls)
        result['action_requests']=sum(r['role']=='action' for r in wrapper.all_calls)
        result['summary_requests']=sum(r['role']=='summary' for r in wrapper.all_calls)
        with trace.span('task_artifacts',kind='phase'):
            (folder/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
        trace.tags={}
    print('TASK_RESULT',folder.name,json.dumps({k:v for k,v in result.items() if k!='steps'}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--serial',default='189f0fad')
    parser.add_argument('--tasks',choices=['folder','note','both','extended','header','footer','replace','rename'],default='both')
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--max-new-tokens',type=int,default=512)
    parser.add_argument('--remote-bundle',default=BUNDLE)
    parser.add_argument('--request-timeout',type=float,default=300)
    parser.add_argument('--cap-output-to-context',action='store_true',
                        help='Keep the full official prompt and cap generation to remaining context')
    parser.add_argument('--skip-context-tight-tasks',action='store_true',
                        help='Skip tasks that cannot fit the full prompt and unchanged generation limit')
    parser.add_argument('--extended-tasks',default=None,help='Comma-separated subset of header,footer,replace,rename')
    args=parser.parse_args()
    if args.cap_output_to_context and args.skip_context_tight_tasks:
        parser.error('Choose capping or skipping, not both')
    extended=args.tasks in ('extended','header','footer','replace','rename')
    stamp=time.strftime('%Y%m%d_%H%M%S')
    output=ROOT/'results'/('official_m3a_'+stamp);output.mkdir()
    trace=Timeline(output/'host_timeline.json')
    with trace.span('device_setup',kind='phase'):
        if extended:
            from extended_markor import FixturePhoneEnv
            env=FixturePhoneEnv(args.serial,trace)
        else:
            env=OfficialPhoneEnv(args.serial,trace)
        env.controller.shell('input keyevent KEYCODE_WAKEUP')
        env.controller.shell('wm dismiss-keyguard')
        env.controller.run(['push',str(ROOT/'native/vlm_server'),REMOTE+'/vlm_server'])
        env.controller.shell('chmod 755 '+REMOTE+'/vlm_server')
    sources={}
    for file in ['agents/m3a.py','agents/base_agent.py','episode_runner.py','env/actuation.py']:
        sources[file]=hashlib.sha256((UPSTREAM/'android_world'/file).read_bytes()).hexdigest()
    manifest={'argv':vars(args),'agent':'upstream M3A','upstream_commit':subprocess.check_output(
              ['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip(),
              'source_sha256':sources,'start_on_home_screen':True,'transition_pause_seconds':1,
              'wait_after_action_seconds':2,'score_feedback_to_agent':False,
              'task_evaluation':'only after episode, successful only if agent done and score == 1',
              'adaptations':['Root ADB/UIAutomator instead of emulator gRPC',
                'XML leaves/content-description/scrollable nodes, matching upstream forest selection rule',
                'GenieX wrapper letterboxes each of two separate images to 512x512',
                'Task setup uses fresh target names; no shared-storage wipe/snapshots/clock change/noise files',
                'Task app scope is Markor; model bundle and max generation configured in argv',
                'Timing wrappers replace module-local time binding without changing durations or agent decisions']}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if extended:
        manifest['adaptations'].append('Fresh per-task note fixtures replace destructive Markor setup; unrelated-file interactions are blocked by the real-phone adapter')
        manifest['fixture_adapter_sha256']=hashlib.sha256((ROOT/'extended_markor.py').read_bytes()).hexdigest()
    model=None
    prior_stay_on=None
    try:
        prior_stay_on=env.controller.shell('settings get global stay_on_while_plugged_in').decode().strip()
        manifest['prior_stay_on_while_plugged_in']=prior_stay_on
        env.controller.shell('svc power stayon usb')
        env.controller.shell('input keyevent KEYCODE_WAKEUP')
        env.controller.shell('wm dismiss-keyguard')
        manifest['remote_genie_config']=json.loads(env.controller.run(
            ['shell','cat',args.remote_bundle+'/model/genie_config.json']))
        model=ResidentVLM(env,output,bundle=args.remote_bundle,request_timeout=args.request_timeout)
        manifest['model_ready']=model.ready
        outcomes=[]
        with ExitStack() as stack:
            stack.enter_context(patch.object(m3a,'time',TimedTime(trace,'m3a_after_action_wait','phase')))
            stack.enter_context(patch.object(base_agent,'time',TimedTime(trace,'m3a_transition_wait','phase')))
            stack.enter_context(patch.object(actuation,'time',TimedTime(trace,'upstream_actuation_sleep','detail')))
            for repeat in range(args.repeats):
                cases=[]
                if args.tasks in ('folder','both'):
                    cases.append(('folder',PhoneCreateFolder({'folder_name':f'folder_{stamp}_r{repeat+1:02d}'})))
                if args.tasks in ('note','both'):
                    cases.append(('note',PhoneCreateNote({'file_name':f'notes_{stamp}_r{repeat+1:02d}.md',
                                                        'text':'Hello AndroidWorld from Qwen3 VL on this phone.'})))
                if extended:
                    from extended_markor import make_cases
                    cases=make_cases(stamp,repeat)
                    if args.tasks!='extended':cases=[case for case in cases if case[0]==args.tasks]
                    if args.extended_tasks:
                        selected=args.extended_tasks.split(',')
                        if not set(selected)<={'header','footer','replace','rename'}:
                            raise ValueError('Unknown extended task name')
                        cases=[case for case in cases if case[0] in selected]
                for kind,task in cases:
                    name=f'rep{repeat+1:02d}_{kind}'
                    if extended:
                        with trace.span('fixture_scope_setup',kind='phase',task=name):
                            env.set_fixture_scope(task)
                    context_size=(manifest['remote_genie_config']['dialog']['context']['size']
                                  if args.cap_output_to_context or args.skip_context_tight_tasks else None)
                    outcomes.append(run_task(task,env,model,trace,output/name,args.max_new_tokens,context_size,
                                             args.skip_context_tight_tasks))
                    # A model context/backend error is not a valid timing trial.
                    if outcomes[-1].get('error'):
                        if outcomes[-1].get('skipped') and args.skip_context_tight_tasks:continue
                        raise RuntimeError(f'Official episode failed; see {output/name}/result.json')
        manifest['outcomes']=[{k:r.get(k) for k in ['task','success','skipped','skip_reason','elapsed_seconds','request_count','action_requests','summary_requests']} for r in outcomes]
        if extended:manifest['fixture_scope_blocks']=env.scope_blocks
    except BaseException:
        manifest['error']=traceback.format_exc()
        raise
    finally:
        trace.tags={}
        if model:
            with trace.span('model_close',kind='phase'): model.close()
            manifest['service_stats_available']=model.session_stats is not None
        if prior_stay_on is not None:
            if prior_stay_on=='null':
                env.controller.shell('settings delete global stay_on_while_plugged_in')
            else:
                env.controller.shell('settings put global stay_on_while_plugged_in '+str(int(prior_stay_on)))
        trace.write()
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        print('RESULTS',output,flush=True)


if __name__=='__main__': main()
