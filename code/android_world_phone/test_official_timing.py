"""Known-answer counterexample: host and device intervals differ by 2x."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from analyze_official_timing import analyze


class ClockAccountingTest(unittest.TestCase):
    def test_idle_fraction_never_subtracts_native_seconds_from_host_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            def save(path,data):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(data))
            def span(i,name,kind,a,b,**tags):
                return dict(id=i,name=name,kind=kind,start_ns=int(a*1e9),end_ns=int(b*1e9),
                            seconds=b-a,task='task',**tags)
            save(folder/'manifest.json',{'argv':{'remote_bundle':'fixture'},'model_ready':{'load_seconds':1}})
            save(folder/'host_timeline.json',{'origin_ns':0,'events':[
                span(0,'task_loop','window',0,20),span(1,'step','window',0,20,step=0),
                span(2,'inference_request','phase',2,8),span(3,'inference_request','phase',14,20)]})
            spans=[]
            for role,a,h in [('action',1,2),('summary',7,14)]:
                name='task-0-'+role
                spans.append(dict(id=name,accepted_ns=int(a*1e9),handler_end_ns=int((a+3)*1e9),
                                  compute_start_ns=int(a*1e9),compute_end_ns=int((a+3)*1e9)))
                response=dict(id=name,elapsed_seconds=3,host_roundtrip_seconds=6,host_send_ns=int(h*1e9),
                    request_accepted_steady_ns=int(a*1e9),prompt_tokens=10,generated_tokens=3,image_count=2,
                    stop_reason='eos',ttft_us=1_000_000,callback_ttft_seconds=1,media_time_us=0,
                    prompt_time_us=1_000_000,decode_time_us=2_000_000,decoding_speed=1.5)
                save(folder/'task/00'/role/'response.json',dict(id=name,role=role,status='ok',
                     text_tokens_without_chat_template=10,response=response))
            save(folder/'service_stats.json',dict(ready_ns=0,shutdown_received_ns=10_000_000_000,requests=spans))
            save(folder/'task/result.json',dict(success=False,agent_done=False,final_score=0,
                 request_count=2,action_requests=1,summary_requests=1,
                 steps=[dict(step=0,done=False,data={'action_output_json':{'action_type':'click'}})]))
            with redirect_stdout(io.StringIO()):result=analyze(folder)
            self.assertAlmostEqual(result['warm_loop']['host_request_idle_fraction'],.4)
            self.assertAlmostEqual(result['native_service']['handler_idle_fraction'],.4)
            self.assertAlmostEqual(result['clock_check']['host_seconds_per_device_second'],2)
            self.assertNotIn('compute_idle_fraction',result['warm_loop'])


if __name__=='__main__':unittest.main()
