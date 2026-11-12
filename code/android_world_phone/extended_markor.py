"""Fresh-file setup and file-scope guards; upstream goals/validators stay intact."""
import hashlib
from pathlib import PurePosixPath
import shlex

from run_tasks import markor, device_constants, file_utils
from official_m3a_env import OfficialPhoneEnv


class FreshNoteSetup:
    fixture_content='The garden has green trees.'

    def initialize_task(self,env):
        if self.initialized:raise RuntimeError('Task already initialized')
        source=self.params.get('file_name',self.params.get('original_name'))
        names=[source]+([self.params['new_name']] if 'new_name' in self.params else [])
        for name in names:
            if PurePosixPath(name).name!=name or not name:
                raise ValueError('Only fresh basenames are permitted')
            if file_utils.check_file_or_folder_exists(name,device_constants.MARKOR_DATA,env.controller):
                raise RuntimeError('Refusing to overwrite existing target: '+name)
        path=device_constants.MARKOR_DATA+'/'+source
        # Noclobber also protects against a file appearing after the existence check.
        env.controller.shell("set -C\nprintf '%s\\n' "+shlex.quote(self.fixture_content)+' > '+shlex.quote(path))
        self.original_content=self.fixture_content
        self.initialized=True
        env.interaction_cache=''

    def tear_down(self,env):
        self.initialized=False  # Preserve owned evidence; never wipe the notebook.


class PhoneEditNote(FreshNoteSetup,markor.MarkorEditNote):
    pass


class PhoneChangeNote(FreshNoteSetup,markor.MarkorChangeNoteContent):
    pass


def make_cases(stamp,repeat):
    prefix='aw'+hashlib.sha256(f'{stamp}-{repeat}'.encode()).hexdigest()[:4]
    return [
        ('header',PhoneEditNote({'file_name':prefix+'h.md','edit_type':'header','header':'Daily update'})),
        ('footer',PhoneEditNote({'file_name':prefix+'f.md','edit_type':'footer','footer':'End of note.'})),
        ('replace',PhoneEditNote({'file_name':prefix+'r.md','edit_type':'replace','replace_text':'The meeting starts at noon.'})),
        ('rename',PhoneChangeNote({'original_name':prefix+'c.md','new_name':prefix+'n.md','updated_content':'Buy apples and bread.'})),
    ]


class FixturePhoneEnv(OfficialPhoneEnv):
    """Block unrelated-file interactions without changing safe M3A actions."""
    def __init__(self,serial,timeline):
        super().__init__(serial,timeline)
        self.allowed_names=set();self.protected_names=set();self.pending_action=None
        self.scope_blocks=[]

    def set_fixture_scope(self,task):
        self.allowed_names={task.params[k] for k in ['file_name','original_name','new_name'] if k in task.params}
        root=shlex.quote(device_constants.MARKOR_DATA)
        paths=self.controller.shell('find '+root+' -maxdepth 1 -type f').decode().splitlines()
        self.protected_names={PurePosixPath(path).name for path in paths}-self.allowed_names

    def execute_action(self,action):
        self.pending_action=action
        try:return super().execute_action(action)
        finally:self.pending_action=None

    def get_state(self,wait_to_stabilize=False):
        state=super().get_state(wait_to_stabilize)
        action=self.pending_action
        if action is None or self.observation_role!='actuation_observe':return state
        elements=state.ui_elements
        names=self.allowed_names|{PurePosixPath(name).stem for name in self.allowed_names}
        title=(elements[0].text or '') if elements else ''
        descriptions={(e.content_description or '').casefold() for e in elements}
        editor=bool({'undo','save','view mode'}&descriptions)
        owned_editor=title in names
        index=action.index
        target=elements[index] if index is not None and 0<=index<len(elements) else None
        label=((target.text or '')+' '+(target.content_description or '')).strip() if target else ''
        reason=None
        if action.action_type in ('click','long_press','input_text'):
            if any(name in label for name in self.protected_names):
                reason='Target belongs to a file outside this task fixture'
        navigation=label.casefold() in ('back','navigate up','navigate back','go back')
        if editor and not owned_editor and action.action_type in ('input_text','keyboard_enter','click','long_press') and not navigation:
            reason='Editing an unrelated note is blocked'
        if action.action_type in ('click','long_press') and label.casefold() in ('delete all','empty trash','clear all'):
            reason='Bulk deletion is outside the task'
        if not editor and action.action_type in ('click','long_press') and label.casefold()=='select all':
            reason='Bulk file selection is outside the task'
        if action.action_type=='input_text' and action.text in self.protected_names:
            reason='An existing unrelated filename cannot be used as a destination'
        # Unindexed coordinate taps could bypass filename checks in a file list.
        if not editor and action.action_type in ('click','long_press') and index is None:
            reason='File-list coordinate taps cannot be scoped to this fixture'
        if reason:
            self.scope_blocks.append({'reason':reason,'action_type':action.action_type,'index':index,'label':label,'title':title})
            raise ValueError(reason)
        return state
