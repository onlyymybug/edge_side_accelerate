"""Known-answer checks for clock boundaries and double-counting in the study."""
import unittest
from analyze_timing import native_accounting, partition


class AccountingTest(unittest.TestCase):
    def test_initial_and_terminal_idle_in_same_device_clock(self):
        # Large arbitrary clock origin must not affect durations or occupancy.
        origin=987654321000000
        ns=lambda seconds:origin+round(seconds*1e9)
        stats={'ready_ns':ns(0),'shutdown_received_ns':ns(10),'requests':[
            {'id':'task-0','accepted_ns':ns(2),'handler_end_ns':ns(4),
             'compute_start_ns':ns(2.25),'compute_end_ns':ns(3.75)},
            {'id':'task-1','accepted_ns':ns(7),'handler_end_ns':ns(9),
             'compute_start_ns':ns(7.25),'compute_end_ns':ns(8.75)}]}
        result=native_accounting(stats)
        self.assertEqual(result['handler_idle_seconds'],6)
        self.assertEqual(result['handler_idle_fraction'],.6)
        self.assertEqual(result['compute_busy_seconds'],3)
        self.assertEqual(result['within_task_gap_seconds']['p50'],3)
        self.assertEqual([g['seconds'] for g in result['gaps']],[2,3,1])

    def test_nested_detail_is_not_double_counted(self):
        window={'start_ns':1000000000,'end_ns':6000000000}
        phases=[{'name':'observe','kind':'phase','start_ns':1000000000,'end_ns':4000000000},
                {'name':'uiautomator','kind':'detail','start_ns':1500000000,'end_ns':3500000000},
                {'name':'inference_request','kind':'phase','start_ns':4000000000,'end_ns':5500000000}]
        self.assertEqual(partition(phases,window),{'observe':3,'inference_request':1.5,'other':.5})
        phases.append({'name':'incorrect_overlap','kind':'phase','start_ns':2000000000,'end_ns':3000000000})
        with self.assertRaises(ValueError): partition(phases,window)

    def test_m3a_action_summary_and_next_step_share_task(self):
        ids=['rep01_folder-0-action','rep01_folder-0-summary','rep01_folder-1-action',
             'rep01_note-0-action']
        stats={'ready_ns':0,'shutdown_received_ns':10_000_000_000,'requests':[
            {'id':name,'accepted_ns':(2*i+1)*1_000_000_000,
             'handler_end_ns':(2*i+2)*1_000_000_000,
             'compute_start_ns':(2*i+1)*1_000_000_000,
             'compute_end_ns':(2*i+2)*1_000_000_000} for i,name in enumerate(ids)]}
        mapping={name:name.rsplit('-',2)[0] for name in ids}
        result=native_accounting(stats,mapping)
        self.assertEqual([g['type'] for g in result['gaps']],
                         ['initial','within_task','within_task','between_tasks','terminal'])
        self.assertEqual(result['within_task_gap_seconds']['n'],2)


if __name__=='__main__': unittest.main()
