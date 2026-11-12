"""ADB/root transport for a personal real device; no emulator gRPC dependency."""
import io
import re
import shlex
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image
from android_env.proto import adb_pb2
from android_world.env import interface, representation_utils

MARKOR = 'net.gsantner.markor/net.gsantner.markor.activity.MainActivity'


class PhoneController:
    def __init__(self, serial, timeline=None):
        self.adb = ['adb', '-s', serial]
        self.timeline = timeline

    def run(self, args, timeout=30):
        context = self.timeline.span('adb', argv=list(args)) if self.timeline else nullcontext()
        with context:
            return subprocess.check_output(self.adb + list(args), timeout=timeout)

    def shell(self, command, timeout=30):
        # adb concatenates remote arguments: quote the entire su -c argument.
        return self.run(['shell', 'su -c ' + shlex.quote(command)], timeout)

    def execute_adb_call(self, request):
        kind = request.WhichOneof('command')
        if kind != 'generic':
            raise NotImplementedError(f'Unsupported upstream ADB request: {kind}')
        args = list(request.generic.args)
        if not args or args[0] != 'shell':
            raise ValueError('Only shell validator requests supported')
        command = args[1] if len(args) == 2 else shlex.join(args[1:])
        output = self.shell(command, request.timeout_sec or 30)
        result = adb_pb2.AdbResponse(status=adb_pb2.AdbResponse.Status.OK)
        result.generic.output = output
        return result


class PhoneEnv(interface.AsyncEnv):
    interaction_cache = ''

    def __init__(self, serial, timeline=None):
        self.timeline = timeline
        self._controller = PhoneController(serial, timeline)
        self._state = None
        self.last_xml = ''
        dimensions = self._controller.shell('wm size').decode()
        self._size = tuple(map(int, re.findall(r'(\d+)x(\d+)', dimensions)[-1]))

    @property
    def controller(self):
        return self._controller

    @property
    def device_screen_size(self):
        return self._size

    logical_screen_size = device_screen_size

    @property
    def foreground_activity_name(self):
        data = self.controller.shell('dumpsys activity activities').decode()
        return next((l.strip() for l in data.splitlines() if 'mResumedActivity' in l), '')

    @property
    def orientation(self):
        return 0

    @property
    def physical_frame_boundary(self):
        return (0, 0, *self._size)

    def reset(self, go_home=False):
        if go_home:
            self.controller.shell('input keyevent KEYCODE_HOME')
        return self.get_state()

    def get_state(self, wait_to_stabilize=False):
        if wait_to_stabilize:
            self._sleep(.5, 'observation_fixed_wait')
        for attempt in range(3):
            try:
                # Bound the remote process too; killing only local adb can leave
                # a stalled UIAutomator session attached to accessibility.
                with self._span('uiautomator', attempt=attempt):
                    self.controller.shell('rm -f /data/local/tmp/aw_phone_window.xml; '
                                          'timeout 12 uiautomator dump /data/local/tmp/aw_phone_window.xml', timeout=15)
                with self._span('read_xml'):
                    self.last_xml = self.controller.shell('cat /data/local/tmp/aw_phone_window.xml').decode()
                if '<hierarchy' not in self.last_xml:
                    raise ValueError('UIAutomator did not return a fresh hierarchy')
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
                if attempt == 2:
                    raise
                self._sleep(.5, 'observation_retry_wait')
        with self._span('screenshot_adb'):
            raw = self.controller.run(['exec-out', 'screencap', '-p'])
        with self._span('parse_observation'):
            pixels = np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'))
            elements = representation_utils.xml_dump_to_ui_elements(self.last_xml)
        # Exclude duplicate noninteractive labels and layout containers. Clickable
        # Markor file rows already expose their names through content descriptions.
        elements = [e for e in elements if e.bbox_pixels and e.is_enabled and
                    (e.is_clickable or e.is_long_clickable or e.is_scrollable
                     or 'EditText' in (e.class_name or ''))]
        self._state = interface.State(pixels, self.last_xml, elements)
        return self._state

    def execute_action(self, action):
        kind = action.action_type
        shell = self.controller.shell
        if kind in ('click', 'long_press', 'input_text') and action.index is not None:
            if self._state is None or not 0 <= action.index < len(self._state.ui_elements):
                raise ValueError('UI index outside current observation')
            element = self._state.ui_elements[action.index]
            if kind == 'input_text' and 'EditText' not in (element.class_name or ''):
                raise ValueError('input_text requires an editable UI element')
            x, y = element.bbox_pixels.center
            if kind != 'long_press':
                shell(f'input tap {int(x)} {int(y)}')
        elif kind in ('click', 'long_press'):
            x, y = action.x, action.y
            w, h = self._size
            if x is None or y is None or not 0 <= x < w or not 0 <= y < h:
                raise ValueError('Click coordinates outside display')
            if kind != 'long_press':
                shell(f'input tap {int(x)} {int(y)}')
        if kind == 'long_press':
            shell(f'input swipe {int(x)} {int(y)} {int(x)} {int(y)} 900')
        elif kind == 'input_text':
            text = action.text
            if text is None or not text.isascii() or len(text) > 2000 or '%s' in text:
                raise ValueError('This initial transport supports ASCII text up to 2000 characters')
            if action.clear_text:
                shell('input keycombination 113 29')  # CTRL+A
                shell('input keyevent KEYCODE_DEL')
            for i, line in enumerate(text.split('\n')):
                if i:
                    shell('input keyevent KEYCODE_ENTER')
                if line:
                    shell('input text ' + shlex.quote(line.replace(' ', '%s')))
        elif kind in ('navigate_back', 'navigate_home', 'keyboard_enter'):
            key = {'navigate_back': 'BACK', 'navigate_home': 'HOME', 'keyboard_enter': 'ENTER'}[kind]
            shell('input keyevent KEYCODE_' + key)
        elif kind == 'open_app':
            if (action.app_name or '').lower() not in ('markor', 'net.gsantner.markor'):
                raise ValueError('Only the dedicated Markor test app is enabled')
            shell('am start -n ' + MARKOR)
        elif kind == 'scroll':
            w, h = self._size
            x1, y1, x2, y2 = {
                'down': (.5,.75,.5,.3), 'up': (.5,.3,.5,.75),
                'left': (.25,.5,.75,.5), 'right': (.75,.5,.25,.5)
            }[action.direction]
            shell(f'input swipe {int(x1*w)} {int(y1*h)} {int(x2*w)} {int(y2*h)} 400')
        elif kind == 'wait':
            self._sleep(1, 'explicit_agent_wait')
        elif kind not in ('click', 'long_press', 'status'):
            raise ValueError(f'Action not implemented: {kind}')
        self._sleep(.4, 'action_fixed_wait')

    def _span(self, name, **tags):
        return self.timeline.span(name, **tags) if self.timeline else nullcontext()

    def _sleep(self, seconds, name):
        if self.timeline:
            self.timeline.sleep(seconds, name)
        else:
            time.sleep(seconds)

    def ask_question(self, question, timeout_seconds=-1):
        raise NotImplementedError('Interactive questions are not enabled')

    def hide_automation_ui(self):
        pass

    def close(self):
        pass
