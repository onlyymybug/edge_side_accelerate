"""Real-device transport; execute the upstream actuation implementation unchanged."""
import shlex
import xml.etree.ElementTree as ET
from android_env.proto import adb_pb2
from android_world.env import actuation, interface, json_action, representation_utils
from phone_env import PhoneController, PhoneEnv, MARKOR


class OfficialPhoneController(PhoneController):
    def execute_adb_call(self, request):
        kind=request.WhichOneof('command')
        if kind=='generic':
            args=list(request.generic.args)
            if not args or args[0]!='shell':
                raise ValueError('Only shell generic requests supported')
            # Upstream swipe omits duration by appending an empty argv item.
            # adb shell normally drops it while joining arguments. Quoting it
            # as '' would instead pass an invalid numeric duration on-device.
            if args[1:3]==['input','swipe'] and args[-1]=='':
                args.pop()
            if len(args)==2:
                command=args[1]
            else:
                # Preserve shell operators emitted by upstream actuation, while
                # quoting ordinary values. Model text uses the typed API below.
                operators={'&&','||',';','|','>','>>'}
                command=' '.join(a if a in operators else shlex.quote(a) for a in args[1:])
        elif kind=='tap':
            command=f'input tap {request.tap.x} {request.tap.y}'
        elif kind=='press_button':
            name=adb_pb2.AdbRequest.PressButton.Button.Name(request.press_button.button)
            command='input keyevent KEYCODE_'+name
        elif kind=='input_text':
            words=shlex.split(request.input_text.text)
            text=' '.join(words)
            command='input text '+shlex.quote(text)
        elif kind=='start_activity':
            r=request.start_activity
            if r.full_activity!=MARKOR:
                raise ValueError('Pilot task app scope permits Markor only')
            command='am start '+('-S ' if r.force_stop else '')+'-n '+shlex.quote(r.full_activity)
            if r.extra_args: command+=' '+shlex.join(r.extra_args)
        else:
            raise NotImplementedError(f'Unsupported official ADB request: {kind}')
        output=self.shell(command,request.timeout_sec or 30)
        response=adb_pb2.AdbResponse(status=adb_pb2.AdbResponse.Status.OK)
        response.generic.output=output
        return response


class OfficialPhoneEnv(PhoneEnv):
    def __init__(self, serial, timeline):
        super().__init__(serial,timeline)
        self._controller=OfficialPhoneController(serial,timeline)
        self.observation_role='observe_before'

    def get_state(self, wait_to_stabilize=False):
        with self.timeline.span(self.observation_role,kind='phase'):
            state=super().get_state(wait_to_stabilize)
            # XML equivalent of upstream forest_to_ui_elements: include leaves,
            # content descriptions and scrollable nodes, in original order.
            nodes=list(ET.fromstring(self.last_xml).iter('node'))
            all_elements=representation_utils.xml_dump_to_ui_elements(self.last_xml)
            if len(nodes)!=len(all_elements):
                raise RuntimeError('XML nodes and upstream parsed UI elements diverged')
            elements=[]
            for node,e in zip(nodes,all_elements):
                e.is_editable='EditText' in (e.class_name or '')
                e.hint_text=node.get('hint') or None
                if not len(node) or e.content_description or e.is_scrollable:
                    elements.append(e)
            self._state=interface.State(state.pixels,state.forest,elements)
            return self._state

    def execute_action(self, action):
        if action.action_type==json_action.ANSWER:
            self.interaction_cache=action.text
            return
        if action.action_type==json_action.STATUS:
            return
        if action.action_type==json_action.OPEN_APP and (action.app_name or '').lower() not in ('markor','net.gsantner.markor'):
            raise ValueError('Pilot task app scope permits Markor only')
        previous=self.observation_role
        self.observation_role='actuation_observe'
        try:
            # Match AsyncAndroidEnv.execute_action: re-read state before input.
            state=self.get_state(wait_to_stabilize=False)
        finally:
            self.observation_role=previous
        with self.timeline.span('actuate',kind='phase',action_type=action.action_type):
            actuation.execute_adb_action(action,state.ui_elements,self.logical_screen_size,self.controller)
        self.observation_role='observe_after'

    def reset(self, go_home=False):
        if go_home:
            self.controller.execute_adb_call(adb_pb2.AdbRequest(
                press_button=adb_pb2.AdbRequest.PressButton(button=adb_pb2.AdbRequest.PressButton.HOME)))
        self.interaction_cache=''
        previous=self.observation_role
        self.observation_role='reset_observe'
        try: return self.get_state(False)
        finally: self.observation_role=previous
