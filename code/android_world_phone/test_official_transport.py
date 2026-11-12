"""Regression: omitted upstream swipe duration must remain omitted on Android."""
import unittest
from unittest.mock import Mock
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'android_world_upstream'))
from android_env.proto import adb_pb2
from android_world.env import adb_utils
from official_m3a_env import OfficialPhoneController


class TransportTest(unittest.TestCase):
    def test_optional_swipe_duration(self):
        controller=OfficialPhoneController('unused')
        controller.shell=Mock(return_value=b'')
        for duration,expected in [(None,'input swipe 500 1000 500 100'),
                                  (200,'input swipe 500 1000 500 100 200')]:
            args=adb_utils.generate_swipe_command(500,1000,500,100,duration)
            controller.execute_adb_call(adb_pb2.AdbRequest(
                generic=adb_pb2.AdbRequest.GenericRequest(args=args)))
            self.assertEqual(controller.shell.call_args.args[0],expected)


if __name__=='__main__': unittest.main()
