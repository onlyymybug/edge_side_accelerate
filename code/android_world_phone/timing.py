"""Monotonic host spans, including nested detail; no cross-device clock mixing."""
from contextlib import contextmanager
import itertools
import json
from pathlib import Path
import time


class Timeline:
    def __init__(self, path):
        self.path = Path(path)
        self.origin_ns = time.perf_counter_ns()
        self.events = []
        self.stack = []
        self.ids = itertools.count()
        self.tags = {}

    @contextmanager
    def span(self, name, kind='detail', **tags):
        event = {'id':next(self.ids), 'name':name, 'kind':kind,
                 'parent_id':self.stack[-1] if self.stack else None,
                 'start_ns':time.perf_counter_ns(), **self.tags, **tags}
        self.stack.append(event['id'])
        try:
            yield event
        except BaseException as error:
            event['error_type'] = type(error).__name__
            raise
        finally:
            event['end_ns'] = time.perf_counter_ns()
            event['seconds'] = (event['end_ns']-event['start_ns'])/1e9
            self.stack.pop()
            self.events.append(event)

    def sleep(self, seconds, name):
        with self.span(name, requested_seconds=seconds):
            time.sleep(seconds)

    def write(self):
        # Written once after measurement, so tracing does not add per-span I/O.
        self.path.write_text(json.dumps({'clock':'host perf_counter_ns',
            'origin_ns':self.origin_ns, 'events':sorted(self.events,key=lambda e:e['id'])},indent=2))
