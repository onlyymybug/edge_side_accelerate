"""Account for service idle and serial GUI phases without aligning clock origins."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PHASES=['observe','prepare_inputs','request_upload','inference_request','actuate',
        'validate','action_parse','feedback','artifacts','other']
LABELS=['Observation','Prepare inputs','Upload','Inference request','Action execution',
        'Task validation','Parse action','Feedback','Artifacts','Other']
COLORS=['#46a7b7','#aacfcf','#e8b861','#476bb2','#e28e62',
        '#a998c0','#ccc0d9','#c2a5cf','#a2aaac','#d9dfe2']


def distribution(values):
    a=np.asarray(values,dtype=float)
    if not len(a):
        return {'n':0}
    return {'n':len(a),'sum':float(a.sum()),'mean':float(a.mean()),
            'p05':float(np.quantile(a,.05)), 'p10':float(np.quantile(a,.1)),
            'p50':float(np.quantile(a,.5)), 'p95':float(np.quantile(a,.95)),
            'min':float(a.min()),'max':float(a.max())}


def native_accounting(stats, task_by_request=None):
    start,end=stats['ready_ns'],stats['shutdown_received_ns']
    assert end > start
    previous=start
    compute_ns=handler_ns=0
    gaps=[]
    last_task=None
    for request in stats['requests']:
        a,b=request['accepted_ns'],request['handler_end_ns']
        c,d=request['compute_start_ns'],request['compute_end_ns']
        if not previous <= a <= b <= end:
            raise ValueError('Overlapping or out-of-window native handler spans')
        if not a <= c <= d <= b:
            raise ValueError('Compute span is not inside its request handler')
        task=(task_by_request[request['id']] if task_by_request is not None
              else request['id'].rsplit('-',1)[0])
        gaps.append({'start_ns':previous,'end_ns':a,'seconds':(a-previous)/1e9,
                     'type':'initial' if last_task is None else 'within_task' if task==last_task else 'between_tasks',
                     'next_request_id':request['id']})
        handler_ns+=b-a; compute_ns+=d-c
        previous=b; last_task=task
    gaps.append({'start_ns':previous,'end_ns':end,'seconds':(end-previous)/1e9,'type':'terminal'})
    idle_ns=sum(g['end_ns']-g['start_ns'] for g in gaps)
    assert idle_ns+handler_ns == end-start
    total=(end-start)/1e9
    handler=handler_ns/1e9; compute=compute_ns/1e9
    within=[g['seconds'] for g in gaps if g['type']=='within_task']
    all_gaps=[g['seconds'] for g in gaps]
    return {'window_seconds':total,'handler_busy_seconds':handler,
            'compute_busy_seconds':compute,'handler_idle_seconds':idle_ns/1e9,
            'handler_idle_fraction':idle_ns/(end-start),
            'no_compute_fraction':1-compute/total,
            'request_noncompute_seconds':handler-compute,
            'gaps':gaps,'within_task_gap_seconds':distribution(within),
            'all_idle_gap_seconds':distribution(all_gaps),
            'idle_windows_at_least_seconds':{
                str(t):{'count':sum(g>=t for g in all_gaps),
                        'total_seconds':sum(g for g in all_gaps if g>=t)}
                for t in [.1,.5,1,2,3,5,10]}}


def partition(events, window):
    """Only phase spans partition a window; nested detail spans never add time."""
    a,b=window['start_ns'],window['end_ns']
    phases=sorted((e for e in events if e['kind']=='phase' and
                   a <= e['start_ns'] and e['end_ns'] <= b),key=lambda e:e['start_ns'])
    previous=a; totals=defaultdict(float)
    for e in phases:
        if e['start_ns'] < previous:
            raise ValueError('Overlapping phase spans would double-count time')
        totals[e['name']]+=(e['end_ns']-e['start_ns'])/1e9
        previous=e['end_ns']
    total=(b-a)/1e9
    totals['other']=total-sum(totals.values())
    if totals['other'] < -1e-7:
        raise ValueError('Negative unaccounted time')
    return dict(totals)


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def analyze(folder):
    folder=Path(folder).resolve()
    out=folder/'timing_analysis'; out.mkdir(exist_ok=True)
    host=json.loads((folder/'host_timeline.json').read_text())
    events=host['events']
    stats=json.loads((folder/'service_stats.json').read_text())
    native=native_accounting(stats)
    span_by_id={s['id']:s for s in stats['requests']}
    manifest=json.loads((folder/'manifest.json').read_text())
    tasks=[]; steps=[]; all_responses=[]
    loops=[e for e in events if e['name']=='task_loop' and e['kind']=='window']
    for window in loops:
        task=window['task']; result=json.loads((folder/task/'result.json').read_text())
        responses=[s['response'] for s in result['steps']]
        all_responses.extend(responses)
        phases=partition(events,window)
        total=window['seconds']
        compute=sum((span_by_id[r['id']]['compute_end_ns']-span_by_id[r['id']]['compute_start_ns'])/1e9 for r in responses)
        assert abs(compute-sum(r['elapsed_seconds'] for r in responses)) < .001
        inflight=phases.get('inference_request',0)
        if inflight < compute-.01:
            raise ValueError('Native compute exceeds host request duration')
        row={'task':task,'success':result['success'],'steps':len(responses),'loop_seconds':total,
             'native_compute_seconds':compute,'compute_idle_fraction':1-compute/total,
             'host_request_busy_seconds':inflight,'host_request_idle_fraction':1-inflight/total,
             **{'phase_'+k:v for k,v in phases.items()}}
        tasks.append(row)
        for record in result['steps']:
            r=record['response']; index=record['step']
            sw=next(e for e in events if e['name']=='step' and e['kind']=='window' and
                    e.get('task')==task and e.get('step')==index)
            phase=partition(events,sw)
            details=[e for e in events if sw['start_ns']<=e['start_ns'] and e['end_ns']<=sw['end_ns']]
            step={'task':task,'step':index,'action':record.get('action',{}).get('action_type','invalid'),
                  'score':record.get('score'),'error':record.get('error'),
                  'step_seconds':sw['seconds'],'native_compute_seconds':r['elapsed_seconds'],
                  'prompt_tokens':r['prompt_tokens'],'generated_tokens':r['generated_tokens'],
                  'ttft_seconds':r['ttft_us']/1e6,'vision_seconds':r['media_time_us']/1e6,
                  'prefill_seconds':r['prompt_time_us']/1e6,'decode_seconds':r['decode_time_us']/1e6,
                  'sdk_residual_seconds':r['elapsed_seconds']-(r['media_time_us']+r['prompt_time_us']+r['decode_time_us'])/1e6,
                  **{'phase_'+k:v for k,v in phase.items()}}
            for name in ['observation_fixed_wait','action_fixed_wait','explicit_agent_wait',
                         'observation_retry_wait','uiautomator','read_xml','screenshot_adb','parse_observation']:
                step[name+'_seconds']=sum(e['seconds'] for e in details if e['name']==name)
            step['action_excluding_fixed_wait_seconds']=phase.get('actuate',0)-step['action_fixed_wait_seconds']-step['explicit_agent_wait_seconds']
            steps.append(step)
    assert len(steps)==len(stats['requests']), 'Missing/failed request results: analyze separately'
    total=sum(t['loop_seconds'] for t in tasks)
    compute=sum(t['native_compute_seconds'] for t in tasks)
    inflight=sum(t['host_request_busy_seconds'] for t in tasks)
    phase_totals={name:sum(t.get('phase_'+name,0) for t in tasks) for name in PHASES}
    assert abs(sum(phase_totals.values())-total) < 1e-6
    step_details={name:sum(s[name+'_seconds'] for s in steps) for name in
                  ['observation_fixed_wait','action_fixed_wait','explicit_agent_wait','observation_retry_wait',
                   'uiautomator','read_xml','screenshot_adb','parse_observation','action_excluding_fixed_wait']}
    fixed=step_details['observation_fixed_wait']+step_details['action_fixed_wait']
    sdk={key:sum(r[field]/1e6 for r in all_responses) for key,field in
         [('vision_seconds','media_time_us'),('prefill_seconds','prompt_time_us'),('decode_seconds','decode_time_us')]}
    sdk['residual_seconds']=compute-sum(sdk.values())
    sdk['generated_tokens']=sum(r['generated_tokens'] for r in all_responses)
    sdk['prompt_tokens']=sum(r['prompt_tokens'] for r in all_responses)
    summary={'run':str(folder),'counts':{'tasks':len(tasks),'successes':sum(t['success'] for t in tasks),'steps':len(steps)},
             'warm_loop':{'seconds':total,'compute_seconds':compute,'no_compute_seconds':total-compute,
                          'compute_idle_fraction':1-compute/total,'host_request_seconds':inflight,
                          'host_request_idle_fraction':1-inflight/total,
                          'phase_seconds':phase_totals,'detail_seconds':step_details,
                          'fixed_wait_seconds':fixed,'idle_without_explicit_fixed_wait_fraction':1-compute/(total-fixed),
                          'request_seconds':distribution([r['elapsed_seconds'] for r in all_responses]),
                          'step_seconds':distribution([s['step_seconds'] for s in steps]),
                          'sdk':sdk},
             'native_service':native,'tasks':tasks,
             'startup':{'native_load_seconds':manifest['model_ready']['load_seconds'],
                        'host_load_seconds':next(e['seconds'] for e in events if e['name']=='model_load'),
                        'host_close_seconds':next(e['seconds'] for e in events if e['name']=='model_close')},
             'scope_notes':['Current custom GUI agent: one action inference per step, no M3A summary inference.',
                            'All attempts included, not filtered on task success.',
                            'Compute means reset+chat template+VLM generation; not NPU utilization.',
                            'Host loop fractions combine durations, never absolute timestamps from different clocks.',
                            'Native service ready-to-shutdown fractions use only the device steady clock.',
                            'Removing fixed waits is an arithmetic sensitivity analysis, not an optimized deployment measurement.',
                            'p95 uses numpy quantile(method=linear), descriptive of this sample.']}
    summary['conditional_speedup']={
        'assumption':'Same action sequence and unchanged non-compute durations; algebra, not a rerun.',
        'model_2x':total/(total-compute+compute/2),
        'model_infinitely_fast_limit':total/(total-compute)}
    summary['cold_model_init_through_shutdown']={
        'seconds':summary['startup']['native_load_seconds']+native['window_seconds'],
        'initializing_unavailable_seconds':summary['startup']['native_load_seconds'],
        'ready_idle_fraction':native['handler_idle_seconds']/(summary['startup']['native_load_seconds']+native['window_seconds'])}
    summary['actions']={}
    for action in sorted({step['action'] for step in steps}):
        subset=[step for step in steps if step['action']==action]
        summary['actions'][action]={
            'count':len(subset),'errors':sum(bool(step.get('error')) for step in subset),
            'mean_model_seconds':statistics.mean(step['native_compute_seconds'] for step in subset),
            'mean_action_seconds':statistics.mean(step.get('phase_actuate',0) for step in subset),
            'mean_action_without_fixed_wait_seconds':statistics.mean(step['action_excluding_fixed_wait_seconds'] for step in subset)}
    summary['warm_loop']['ttft_seconds']=distribution([r['ttft_us']/1e6 for r in all_responses])
    summary['outcome_groups']={}
    for name,success in [('successful',True),('unsuccessful',False)]:
        subset=[task for task in tasks if task['success']==success]
        wall=sum(task['loop_seconds'] for task in subset)
        busy=sum(task['native_compute_seconds'] for task in subset)
        summary['outcome_groups'][name]={'tasks':len(subset),'loop_seconds':wall,
            'compute_seconds':busy,'compute_idle_fraction':1-busy/wall if wall else None}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    write_csv(out/'steps.csv',steps); write_csv(out/'tasks.csv',tasks)

    fig,ax=plt.subplots(figsize=(11,max(3,len(tasks)*.55+1.5)))
    left=np.zeros(len(tasks))
    for phase,label,color in zip(PHASES,LABELS,COLORS):
        values=np.array([t.get('phase_'+phase,0) for t in tasks])
        ax.barh([t['task'] for t in tasks],values,left=left,label=label,color=color)
        left+=values
    ax.set_xlabel('Wall-clock seconds (task loop, startup excluded)')
    ax.set_title('Accumulated phase time per task (not chronological)')
    ax.invert_yaxis(); ax.legend(loc='upper left',bbox_to_anchor=(0,-.18),ncol=4,fontsize=8)
    fig.tight_layout()
    for ext in ['png','svg']: fig.savefig(out/f'phase_breakdown.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(12,2.8))
    start=stats['ready_ns']
    ax.broken_barh([(0,native['window_seconds'])],(0,1),facecolors='#dfe9dd',label='Idle (no active request handler)')
    handler_ranges=[((r['accepted_ns']-start)/1e9,(r['handler_end_ns']-r['accepted_ns'])/1e9) for r in stats['requests']]
    ax.broken_barh(handler_ranges,(0,1),facecolors='#476bb2',label='Model request handler')
    ax.set_yticks([]); ax.set_xlabel('Seconds since model ready (device steady clock)')
    ax.set_title(f'Resident service: {native["handler_idle_fraction"]:.1%} idle; startup excluded')
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.3),ncol=2)
    fig.tight_layout()
    for ext in ['png','svg']: fig.savefig(out/f'service_timeline.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)

    chrome=[]
    for e in events:
        chrome.append({'name':e['name'],'cat':e['kind'],'ph':'X',
                       'ts':(e['start_ns']-host['origin_ns'])/1000,
                       'dur':(e['end_ns']-e['start_ns'])/1000,
                       'pid':1,'tid':1,'args':{k:v for k,v in e.items() if k not in ['start_ns','end_ns']}})
    (out/'host_chrome_trace.json').write_text(json.dumps({'traceEvents':chrome}))
    write_report(out,summary)
    print(json.dumps({'analysis':str(out),'counts':summary['counts'],'warm_loop':summary['warm_loop'],
                      'service_idle_fraction':native['handler_idle_fraction']},indent=2))
    return summary


def write_report(out,s):
    w=s['warm_loop']; n=s['native_service']; gaps=n['within_task_gap_seconds']
    rows='\n'.join(f'| {t["task"]} | {"成功" if t["success"] else "未完成"} | {t["steps"]} | {t["loop_seconds"]:.2f} | {t["native_compute_seconds"]:.2f} | {t["compute_idle_fraction"]:.1%} |' for t in s['tasks'])
    phases='\n'.join(f'| {label} | {w["phase_seconds"][p]:.3f} | {w["phase_seconds"][p]/w["seconds"]:.1%} |' for p,label in zip(PHASES,LABELS))
    actions='\n'.join(f'| {action} | {r["count"]} | {r["errors"]} | {r["mean_model_seconds"]:.3f} | {r["mean_action_seconds"]:.3f} | {r["mean_action_without_fixed_wait_seconds"]:.3f} |' for action,r in s['actions'].items())
    text=f'''# GUI Agent 时间分解与模型服务空闲率实测

数据目录：`{s['run']}`。REDMI K90 / SM8750 / Android 16，Qwen3-VL-4B，GenieX v0.10.0 + QAIRT，单模型常驻、串行 GUI 循环。主机负责编排和文件预处理，手机运行推理与 GUI，通过 USB/ADB 连接；不是所有组件都在手机进程内运行。每步 reset KV，无跨步前缀缓存。提示词、生成上限和动作策略沿用当前代理；本次增加计时，包含全部任务尝试，不按成功筛选。

## 主要结果

- {s['counts']['tasks']} 次任务、{s['counts']['steps']} 次模型请求，任务成功 {s['counts']['successes']} 次。
- 任务循环合计 **{w['seconds']:.3f} 秒**；模型计算调用合计 **{w['compute_seconds']:.3f} 秒**；未在计算 **{w['no_compute_seconds']:.3f} 秒 / {w['compute_idle_fraction']:.2%}**。
- 完全使用主机时钟、把提交请求到收到响应整个窗口都算繁忙时，空闲率为 **{w['host_request_idle_fraction']:.2%}**。该口径额外包含请求 IPC 和响应交付。
- 完全使用设备时钟，从模型 ready 到收到 shutdown：**{n['window_seconds']:.3f} 秒**；请求处理繁忙 **{n['handler_busy_seconds']:.3f} 秒**；无在途处理 **{n['handler_idle_seconds']:.3f} 秒 / {n['handler_idle_fraction']:.2%}**。这个窗口额外包括任务间初始化、终局截图等时间，不能与任务循环混为一谈。
- 同一任务内，相邻请求之间的真实服务空闲窗口：中位数 **{gaps.get('p50',0):.3f} 秒**，P95 **{gaps.get('p95',0):.3f} 秒**，最大 **{gaps.get('max',0):.3f} 秒**，样本数 {gaps['n']}。不含首次请求前、跨任务和退出前空档。
- 首次加载：设备 SDK **{s['startup']['native_load_seconds']:.3f} 秒**，主机进程启动到 ready **{s['startup']['host_load_seconds']:.3f} 秒**，单列，不算热态空闲。
- 退出释放：主机 close 调用 **{s['startup']['host_close_seconds']:.3f} 秒**，包括 shutdown 传输、SDK 释放和退出，另列；不把不可服务的释放阶段算作服务空闲。

连续自动运行，任务之间没有人工操作或等候；测量结束自动关闭服务，没有把后续写报告的时间加入空闲率。从模型初始化开始到 shutdown 的窗口为 {s['cold_model_init_through_shutdown']['seconds']:.3f} 秒，其中初始化为不可服务阶段；该冷启动窗口内已 ready 且无请求的占比为 {s['cold_model_init_through_shutdown']['ready_idle_fraction']:.2%}。

| 任务 | 结果 | 请求数 | 循环秒数 | 模型计算秒数 | 未计算比例 |
|---|---|---:|---:|---:|---:|
{rows}

## 时间去了哪里

以下各项互斥，相加等于任务循环总时间。Inference request 是客户端等待模型的完整窗口，不能再把 SDK prefill/decode 加到该表总计中。

| 阶段 | 秒数 | 循环占比 |
|---|---:|---:|
{phases}

观察阶段内：UIAutomator 共 {w['detail_seconds']['uiautomator']:.3f} 秒，显式观察等待共 {w['detail_seconds']['observation_fixed_wait']:.3f} 秒，XML 读取 {w['detail_seconds']['read_xml']:.3f} 秒，截图传输 {w['detail_seconds']['screenshot_adb']:.3f} 秒，解码/解析 {w['detail_seconds']['parse_observation']:.3f} 秒。这些是观察阶段的细分，不能再次加到总计。

动作执行内：扣除明确计时的固定等待与 wait 动作后，共 {w['detail_seconds']['action_excluding_fixed_wait']:.3f} 秒；动作后固定等待 {w['detail_seconds']['action_fixed_wait']:.3f} 秒；模型显式 wait 共 {w['detail_seconds']['explicit_agent_wait']:.3f} 秒。

动作注入时间衡量的是 ADB/input 命令返回时间，并非应用 UI 完成变化的精确时刻；UI 稳定相关开销混在显式等待和 UIAutomator 获取状态的等待中。未使用帧时间线判定应用渲染完成。

现有实现每步采集前固定等 0.5 秒、未抛出错误的动作后固定等 0.4 秒（包括 status 声明），本轮固定等待实测共 {w['fixed_wait_seconds']:.3f} 秒。如果只在算术上扣除此项、其他时间保持不变，未计算比例为 {w['idle_without_explicit_fixed_wait_fraction']:.2%}。这是敏感性分析，未实测移除等待后的 UI 稳定性或吞吐，不能当作优化后结果。

| 动作 | 次数 | 执行错误数 | 平均模型秒数 | 平均动作阶段秒数 | 扣除明确等待后动作秒数 |
|---|---:|---:|---:|---:|---:|
{actions}

最后一列包含命令调用及执行器内部开销；报错的动作也在平均值中，不表示每次动作都实际注入了事件。

单次模型计算中位数 {w['request_seconds']['p50']:.3f} 秒、P95 {w['request_seconds']['p95']:.3f} 秒；单步整体中位数 {w['step_seconds']['p50']:.3f} 秒、P95 {w['step_seconds']['p95']:.3f} 秒。

条件推导：假设动作序列和所有非模型耗时不变，仅将模型计算加速两倍，循环整体加速约 {s['conditional_speedup']['model_2x']:.2f} 倍；仅加速模型的理论极限约 {s['conditional_speedup']['model_infinitely_fast_limit']:.2f} 倍。这是时间分解的代数推导，未实测加速后的交互行为。

SDK 累计：视觉编码 {w['sdk']['vision_seconds']:.3f} 秒、prefill {w['sdk']['prefill_seconds']:.3f} 秒、decode {w['sdk']['decode_seconds']:.3f} 秒，其余 reset/template/SDK 内部及计时边界差 {w['sdk']['residual_seconds']:.3f} 秒。SDK 各字段来自 [GenieX C API](https://github.com/qualcomm/GenieX/blob/v0.10.0/sdk/include/geniex.h)；TTFT 另用于首 token 延迟诊断，不与视觉/prefill 重复相加。

## 定义与有效范围

任务循环空闲率 = `1 - sum(native reset/template/generate duration) / sum(task-loop wall duration)`。另给出全主机时钟的在途请求口径，避免把上传或 IPC 当作模型计算。

服务原生空闲率 = `sum(gaps between request handlers, including initial/tail) / (shutdown_received - model_ready)`。ready、handler 开始/结束、shutdown 均由同一设备 steady_clock 记录，间隙之和与繁忙之和严格闭合；没有直接相减主机与手机的绝对时间戳。

**这里量出的是服务没有处理请求的时间，不是 NPU 硬件空闲率、CPU idle、功耗或可无损共享算力比例。** GUI、ADB、图像预处理也会占用手机或主机资源；权重仍常驻内存。空档能否让另一个模型使用，还需调度/抢占、内存与竞争实验。

当前是一个动作推理调用/步、自定义代理，且每步运行文件校验器并向模型反馈分数。[上游 M3A](https://github.com/google-research/android_world/blob/55ddaf5edc6f6c25673aed343907d5b9dca66f84/android_world/agents/m3a.py) 在常规动作后默认等待 2 秒，重新采集状态，还会执行一次模型摘要调用；直接复用本次空闲率到 M3A 或其他 GUI Agent 不成立。

成功与失败循环另按结果分组存于 `summary.json/outcome_groups`。超过步数的失败循环也属于本次真实负载，但不能称为成功完成任务的耗时；成功率较低、样本少，不能推导通用任务分布。

任务仅覆盖 Markor 新建文件夹、笔记，两种任务各三次；UIAutomator 的启动与等待、主机传输、显式 sleep、调试截图和每步校验均包含在测量中。结果是当前原型系统的实测，不代表 GUI 应用本身固有的等待时间，也不是整个 AndroidWorld 的统计结论。失败步骤、重复输入、提前完成声明都保留。P95 为本次样本的线性插值经验分位数，不是总体置信区间。

## 复查材料

- [summary.json](summary.json)：完整数字和各阈值空闲窗口计数。
- [steps.csv](steps.csv)、[tasks.csv](tasks.csv)：逐步、逐任务表。
- [阶段分解图](phase_breakdown.png)、[SVG](phase_breakdown.svg)。
- [服务时间线](service_timeline.png)、[SVG](service_timeline.svg)。
- [主机 Chrome/Perfetto trace](host_chrome_trace.json)：可导入本地 trace viewer 交互检查嵌套阶段。
- [主机原始时间线](../host_timeline.json)、[服务原始时间线](../service_stats.json)、[启动命令](../command.json)、[设备日志](../phone_raw.log)、[参数](../manifest.json)。
'''
    (out/'REPORT.md').write_text(text)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run_directory')
    analyze(parser.parse_args().run_directory)
