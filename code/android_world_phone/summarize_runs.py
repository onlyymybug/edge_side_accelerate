"""Summarize every recorded attempt, including interrupted/failed runs."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
summary=[]
for folder in sorted((ROOT/'results').iterdir()):
    if not folder.is_dir():
        continue
    manifest=folder/'manifest.json'
    if manifest.exists():
        meta=json.loads(manifest.read_text())
        row={'run':folder.name,'mode':'vlm','load_seconds':meta.get('model_ready',{}).get('load_seconds'),
             'error':meta.get('error'),'tasks':[]}
        for taskdir in sorted(p for p in folder.iterdir() if p.is_dir()):
            result=taskdir/'result.json'
            responses=[json.loads(p.read_text())['response'] for p in sorted(taskdir.glob('*/response.json'))]
            seconds=[r['elapsed_seconds'] for r in responses]
            task={'task':taskdir.name,'steps':len(responses),'status':'interrupted_or_incomplete',
                  'median_vlm_seconds':statistics.median(seconds) if seconds else None,
                  'max_prompt_tokens':max((r['prompt_tokens'] for r in responses),default=None),
                  'length_stops':sum(r['stop_reason']=='length' for r in responses)}
            if result.exists():
                r=json.loads(result.read_text())
                task.update(success=r['success'],initial_score=r['initial_score'],elapsed_seconds=r['elapsed_seconds'])
                task['status']='complete' if meta.get('outcomes') else 'interrupted_or_incomplete'
            row['tasks'].append(task)
        summary.append(row)
    elif (folder/'result.json').exists():
        summary.append({'run':folder.name,**json.loads((folder/'result.json').read_text())})
    else:
        summary.append({'run':folder.name,'status':'no completed result'})
(ROOT/'results'/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
print(json.dumps(summary,indent=2,ensure_ascii=False))
