"""Analyze complete upstream M3A episodes, including action and summary calls."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from analyze_timing import distribution, native_accounting, partition, write_csv, plt


def analyze(folder):
    folder=Path(folder).resolve()
    out=folder/'timing_analysis'; out.mkdir(exist_ok=True)
    host=json.loads((folder/'host_timeline.json').read_text())
    events=host['events']
    manifest=json.loads((folder/'manifest.json').read_text())
    stats=json.loads((folder/'service_stats.json').read_text())
    native_spans={r['id']:r for r in stats['requests']}
    requests=[]; steps=[]; tasks=[]; mapping={}
    for window in [e for e in events if e['kind']=='window' and e['name']=='task_loop']:
        task=window['task']
        result=json.loads((folder/task/'result.json').read_text())
        if result.get('error'):
            raise ValueError('Incomplete/backend-failed episode; do not report as valid timing: '+task)
        calls=[]
        for path in sorted((folder/task).glob('*/*/response.json')):
            record=json.loads(path.read_text())
            if record['status']!='ok':
                raise ValueError('Failed model request: '+str(path))
            response=record['response']; name=record['id']; role=record['role']
            span=native_spans[name]
            compute=(span['compute_end_ns']-span['compute_start_ns'])/1e9
            assert abs(compute-response['elapsed_seconds']) < .001
            mapping[name]=task
            row={'id':name,'task':task,'step':int(path.parent.parent.name),'role':role,
                 'compute_seconds':compute,'host_roundtrip_seconds':response['host_roundtrip_seconds'],
                 'host_send_ns':response['host_send_ns'],
                 'device_accepted_ns':response['request_accepted_steady_ns'],
                 'text_tokens_without_template':record['text_tokens_without_chat_template'],
                 'prompt_tokens':response['prompt_tokens'],'generated_tokens':response['generated_tokens'],
                 'image_count':response['image_count'],'stop_reason':response['stop_reason'],
                 'ttft_seconds':response['ttft_us']/1e6,
                 'callback_ttft_seconds':response['callback_ttft_seconds'],
                 'vision_seconds':response['media_time_us']/1e6,
                 'prefill_seconds':response['prompt_time_us']/1e6,
                 'decode_seconds':response['decode_time_us']/1e6,
                 'decoding_tokens_per_second':response['decoding_speed']}
            row['sdk_residual_seconds']=compute-sum(row[k] for k in ['vision_seconds','prefill_seconds','decode_seconds'])
            calls.append(row); requests.append(row)
        assert len(calls)==result['request_count']
        for record in result['steps']:
            index=record['step']
            sw=next(e for e in events if e['kind']=='window' and e['name']=='step'
                    and e.get('task')==task and e.get('step')==index)
            phase=partition(events,sw)
            sc=[c for c in calls if c['step']==index]
            action=record['data'].get('action_output_json') or {}
            steps.append({'task':task,'step':index,'done':record['done'],
                          'action_type':action.get('action_type','invalid'),
                          'step_seconds':sw['seconds'],'requests':len(sc),
                          'compute_seconds':sum(c['compute_seconds'] for c in sc),
                          **{'phase_'+k:v for k,v in phase.items()}})
        phases=partition(events,window)
        total=window['seconds']; compute=sum(c['compute_seconds'] for c in calls)
        # Host request and native compute durations use different clocks.
        # Their interval rates may differ; validate each clock internally.
        tasks.append({'task':task,'success':result['success'],'agent_done':result['agent_done'],
                      'final_score':result['final_score'],'steps':len(result['steps']),
                      'action_requests':result['action_requests'],'summary_requests':result['summary_requests'],
                      'loop_seconds':total,'native_compute_seconds':compute,
                      'host_request_seconds':phases.get('inference_request',0),
                      'host_request_idle_fraction':1-phases.get('inference_request',0)/total,
                      **{'phase_'+k:v for k,v in phases.items()}})
    assert len(requests)==len(stats['requests']), 'Unaccounted native requests'
    native=native_accounting(stats,mapping)
    roles={r['id']:r['role'] for r in requests}
    transitions=defaultdict(list)
    previous=None
    for request,gap in zip(stats['requests'],native['gaps']):
        name=request['id']
        transition=('initial' if previous is None else 'between_tasks' if mapping[previous]!=mapping[name]
                    else roles[previous]+'_to_'+roles[name])
        gap['transition']=transition
        gap['previous_request_id']=previous
        transitions[transition].append(gap['seconds'])
        previous=name
    transitions['terminal'].append(native['gaps'][-1]['seconds'])
    native['gap_by_transition_seconds']={k:distribution(v) for k,v in transitions.items()}
    total=sum(t['loop_seconds'] for t in tasks)
    compute=sum(r['compute_seconds'] for r in requests)
    phases=defaultdict(float)
    for task in tasks:
        for key,value in task.items():
            if key.startswith('phase_'): phases[key[6:]]+=value
    assert abs(sum(phases.values())-total)<1e-6
    by_role={}
    for role in ['action','summary']:
        rows=[r for r in requests if r['role']==role]
        by_role[role]={'count':len(rows),'compute_total_seconds':sum(r['compute_seconds'] for r in rows),
                       'host_request_total_seconds':sum(r['host_roundtrip_seconds'] for r in rows),
                       **{key:distribution([r[key] for r in rows]) for key in
                          ['compute_seconds','ttft_seconds','vision_seconds','prefill_seconds',
                           'decode_seconds','generated_tokens','prompt_tokens','decoding_tokens_per_second']}}
    details={name:sum(e['seconds'] for e in events if e['name']==name and any(
        w['start_ns']<=e['start_ns'] and e['end_ns']<=w['end_ns'] for w in events
        if w['kind']=='window' and w['name']=='task_loop')) for name in
        ['uiautomator','read_xml','screenshot_adb','parse_observation','observation_fixed_wait',
         'upstream_actuation_sleep','observation_retry_wait']}
    waits=phases.get('m3a_after_action_wait',0)+phases.get('m3a_transition_wait',0)
    observation=sum(value for key,value in phases.items() if 'observe' in key)
    ordered=sorted(requests,key=lambda r:r['host_send_ns'])
    x=np.array([r['device_accepted_ns']-ordered[0]['device_accepted_ns'] for r in ordered],float)/1e9
    y=np.array([r['host_send_ns']-ordered[0]['host_send_ns'] for r in ordered],float)/1e9
    if len(ordered)>1 and x[-1]>0:
        slope,offset=np.polyfit(x,y,1)
        residual=y-(slope*x+offset)
        clock_check={'host_seconds_per_device_second':float(slope),
                     'fit_residual_min_host_seconds':float(residual.min()),
                     'fit_residual_max_host_seconds':float(residual.max()),
                     'method':'Paired request-send/accepted intervals, each centered in its own clock; diagnostic only, no automatic rescaling.'}
    else:clock_check={'host_seconds_per_device_second':None}
    summary={'run':str(folder),'agent':'upstream M3A','remote_bundle':manifest['argv']['remote_bundle'],
             'counts':{'tasks':len(tasks),'successes':sum(t['success'] for t in tasks),'steps':len(steps),
                       'requests':len(requests),'action':by_role['action']['count'],'summary':by_role['summary']['count']},
             'startup':{'native_load_seconds':manifest['model_ready']['load_seconds']},
             'warm_loop':{'clock':'host perf_counter_ns','seconds':total,
                          'native_compute_seconds_unconverted':compute,
                          'no_request_seconds':total-phases.get('inference_request',0),
                          'host_request_seconds':phases.get('inference_request',0),
                          'host_request_idle_fraction':1-phases.get('inference_request',0)/total,
                          'observation_seconds':observation,'explicit_m3a_wait_seconds':waits,
                          'phase_seconds':dict(phases),'detail_seconds':details},
             'by_role':by_role,'native_service':native,'tasks':tasks,'clock_check':clock_check,
             'phase_errors':[{k:e[k] for k in ['id','name','kind','task','step','error_type','seconds','argv'] if k in e}
                             for e in events if e.get('error_type')],
             'limitations':['Real-phone environment adaptation; not the original API33 emulator benchmark.',
               'Model images are separately letterboxed to 512x512. Task setup avoids wiping personal storage.',
               'Compute is reset+template+VLM generate wall time; no-compute is not NPU utilization or whole-phone idle.',
               'Official M3A terminal/invalid actions may have no summary call.',
               'Host phase/idle fractions use host intervals only; native busy/idle fractions use device intervals only. Raw seconds are not subtracted across clocks.',
               'A detected host/device interval-rate difference is recorded in clock_check; no causal claim about its source.',
               'Model bundles, prompts and task trajectories may differ between runs; prior simplified-agent timings are not a controlled agent-only comparison.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    write_csv(out/'requests.csv',requests);write_csv(out/'steps.csv',steps);write_csv(out/'tasks.csv',tasks)
    groups={'Action inference (host request)':by_role['action']['host_request_total_seconds'],
            'Summary inference (host request)':by_role['summary']['host_request_total_seconds'],
            'Observation':observation,'GUI execution':phases.get('actuate',0),'M3A waits':waits}
    groups['Other']=total-sum(groups.values())
    assert groups['Other']>=-1e-6
    fig,ax=plt.subplots(figsize=(10,2.5)); left=0
    for label,value in groups.items():
        ax.barh(['Official M3A'],[value],left=left,label=f'{label}: {value:.1f}s ({value/total:.1%})')
        left+=value
    ax.set_xlabel('Host clock seconds (sum of task loops)');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.4),ncol=2)
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(out/('phase_breakdown.'+ext),bbox_inches='tight',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,2.5)); origin=stats['ready_ns']; seen=set()
    for r in stats['requests']:
        role=next(c['role'] for c in requests if c['id']==r['id'])
        ax.broken_barh([((r['compute_start_ns']-origin)/1e9,(r['compute_end_ns']-r['compute_start_ns'])/1e9)],
                      (0,1),facecolors='#476bb2' if role=='action' else '#b679b2',
                      label=role if role not in seen else None)
        seen.add(role)
    ax.set_xlim(0,native['window_seconds']);ax.set_ylim(0,1);ax.set_yticks([])
    ax.set_xlabel('Device steady clock seconds after ready; blank = no VLM compute');ax.legend()
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(out/('service_timeline.'+ext),dpi=160)
    plt.close(fig)
    chrome=[{'name':e['name'],'cat':e['kind'],'ph':'X','pid':1,'tid':1,
             'ts':(e['start_ns']-host['origin_ns'])/1000,'dur':(e['end_ns']-e['start_ns'])/1000,
             'args':{k:v for k,v in e.items() if k not in ['name','start_ns','end_ns']}} for e in events]
    (out/'host_chrome_trace.json').write_text(json.dumps({'traceEvents':chrome}))
    report=[f'# 官方 M3A 真机计时\n\n运行目录：`{folder}`。\n',
            f"{len(tasks)} 个任务，成功 {summary['counts']['successes']}；{len(steps)} 步；"
            f"动作请求 {by_role['action']['count']} 次，总结请求 {by_role['summary']['count']} 次。\n",
            f"冷加载（设备计时）{summary['startup']['native_load_seconds']:.3f} 秒。主机任务循环共 {total:.3f} 秒，"
            f"推理请求在途 {phases.get('inference_request',0):.3f} 秒，无请求在途 {total-phases.get('inference_request',0):.3f} 秒"
            f"（{1-phases.get('inference_request',0)/total:.2%}）。\n",
            f"服务 ready→shutdown {native['window_seconds']:.3f} 秒，handler 空闲率 "
            f"{native['handler_idle_fraction']:.2%}；该窗口包含任务初始化、跨任务及末尾间隙。\n",
            '| 阶段 | 秒 | 任务循环占比 |\n|---|---:|---:|']
    report.extend(f'| {key} | {value:.3f} | {value/total:.2%} |' for key,value in groups.items())
    report.extend(['\n设备handler空闲窗口（同一设备时钟）：\n',
                   '| 前后请求 | 次数 | 中位秒 | P95秒 |\n|---|---:|---:|---:|'])
    report.extend(f"| {key} | {d['n']} | {d['p50']:.3f} | {d['p95']:.3f} |"
                  for key,d in native['gap_by_transition_seconds'].items())
    report.extend(['\n主机无请求占比=(主机任务循环时间−主机请求在途时间)/主机任务循环时间；设备服务空闲率使用设备ready、handler和shutdown区间。',
                   f"\n本轮主机秒/设备秒的区间拟合斜率为 {clock_check['host_seconds_per_device_second']}；两种时间口径分别在各自时钟内计算。",
                   '\n不同计时源的秒数不直接相减。clock_check记录主机/设备区间速率拟合，用于发现计时口径差异；本报告不根据拟合自动重标时间。',
                   '\n执行上游 M3A 和 episode_runner；成功与失败任务均计入，模型后端异常会拒绝生成本报告。',
                   '\n'+ '\n'.join('- '+note for note in summary['limitations']),
                   '\n异常phase/detail见summary.json的phase_errors；计入实测，未删除失败步骤。',
                   '\n详见 summary.json、requests.csv、steps.csv、tasks.csv、host_chrome_trace.json 及两张图。'])
    (out/'REPORT.md').write_text('\n'.join(report))
    print(json.dumps({k:summary[k] for k in ['counts','startup','warm_loop']},indent=2))
    print(out/'REPORT.md')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder')
    analyze(parser.parse_args().folder)
