"""Fault injection peer for actual pipe/subprocess integration tests."""
import json
import os
from pathlib import Path
import sys
import time

mode = sys.argv[1]
record = Path(sys.argv[2]) if len(sys.argv) > 2 else None
if record:
    with record.open('a') as stream:
        stream.write('start\n')
if mode == 'startup_eof' or (mode == 'first_eof' and record.read_text().count('start') == 1):
    sys.exit(1)
if mode == 'startup_hang':
    time.sleep(60)
if mode == 'startup_error':
    print(json.dumps({'protocol': 1, 'event': 'error', 'error': 'bad assets'}), flush=True)
    sys.exit(1)
print(json.dumps({'protocol': 1, 'event': 'ready', 'pid': os.getpid()}), flush=True)
if mode == 'spawn_child':
    import subprocess
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    record.write_text(record.read_text() + 'child=' + str(child.pid) + '\n')
for line in sys.stdin:
    request = json.loads(line)
    command = request['command']
    if command == 'close':
        if mode == 'hang_close':
            time.sleep(60)
        break
    if record:
        with record.open('a') as stream:
            stream.write(command + '\n')
    if mode == 'lost_response':
        sys.exit(1)
    if mode == 'delay':
        time.sleep(.2)
    if mode == 'giant_response':
        print('x' * 200000, flush=True)
        continue
    if mode == 'broken_json':
        print('a handler print on stdout', flush=True)
        continue
    if mode == 'stderr_flood':
        sys.stderr.write('diagnostic' * 100000)
        sys.stderr.flush()
    reply = {'protocol': 1, 'request_id': request['request_id'], 'ok': True,
             'result': {'pid': os.getpid(), 'env': os.environ.get('TOY_ENV'),
                        'payload': request['payload']}}
    if mode == 'wrong_id':
        reply['request_id'] += 1
    if mode == 'nonfinite':
        reply['result'] = float('nan')
    if mode == 'remote_error':
        reply.update(ok=False, error='observe partially applied')
    print(json.dumps(reply, ensure_ascii=False), flush=True)
