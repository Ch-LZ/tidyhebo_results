"""Worker-owned immutable toy checkpoints. Standalone stdlib, Python 3.7+."""
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import uuid


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(',', ':')).encode('utf-8')


def identity(configuration):
    digest = hashlib.sha256()
    for name in ('toy_worker.py', 'native_state.py'):
        digest.update(name.encode('ascii'))
        digest.update(Path(__file__).with_name(name).read_bytes().replace(b'\r\n', b'\n'))
    return {'python': platform.python_version(), 'code': digest.hexdigest(),
            'configuration': hashlib.sha256(json_bytes(configuration)).hexdigest()}


def save(root, role, compatibility, batch, state):
    directory = Path(root).resolve() / role
    directory.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex + '.json'
    data = json_bytes({'schema_version': 1, 'compatibility': compatibility,
                       'batch': batch, 'state': state})
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=str(directory))
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(directory / name))
        if os.name == 'posix':
            dirfd = os.open(str(directory), os.O_RDONLY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {'kind': 'worker-native', 'schema_version': 1, 'format': 'toy-json-v1',
            'path': role + '/' + name, 'sha256': hashlib.sha256(data).hexdigest(),
            'compatibility': compatibility, 'batch': batch}


def load(root, role, compatibility, descriptor):
    if (descriptor.get('kind') != 'worker-native' or descriptor.get('schema_version') != 1
            or descriptor.get('format') != 'toy-json-v1'
            or descriptor.get('compatibility') != compatibility):
        raise ValueError('Incompatible worker checkpoint')
    relative = Path(descriptor['path'])
    directory = (Path(root).resolve() / role).resolve()
    path = (Path(root).resolve() / relative).resolve()
    if relative.is_absolute() or path.parent != directory:
        raise ValueError('Checkpoint path outside worker directory')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != descriptor.get('sha256'):
        raise ValueError('Checkpoint checksum mismatch')
    record = json.loads(data.decode('utf-8'))
    if (record.get('schema_version') != 1 or record.get('compatibility') != compatibility
            or record.get('batch') != descriptor.get('batch')):
        raise ValueError('Checkpoint identity/batch mismatch')
    return record['batch'], record['state']


def tuples(value):
    return tuple(tuples(x) for x in value) if isinstance(value, list) else value
