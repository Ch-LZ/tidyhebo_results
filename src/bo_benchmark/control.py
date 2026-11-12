"""Live execution controls, independent of scientific RunSpec."""
import json
from pathlib import Path
import warnings
from .domain import positive_int


class ControlFile:
    """Only workers is live. Partial/invalid edits retain the last valid limit."""
    def __init__(self, path: Path):
        self.path = Path(path)
        self._last_error = None

    def read_limit(self, current: int) -> int:
        try:
            data = json.loads(self.path.read_text(encoding='utf-8-sig'))
            if not isinstance(data, dict):
                raise ValueError('Control config must be a JSON object')
            value = data.get('workers', current)
            positive_int(value)
        except FileNotFoundError:
            return current
        except (OSError, ValueError, TypeError) as error:
            detail = f'{type(error).__name__}: {error}'
            if detail != self._last_error:
                warnings.warn(f'Ignoring invalid control file {self.path}: {detail}; '
                              'retaining the last valid workers limit', RuntimeWarning, stacklevel=2)
            self._last_error = detail
            return current
        self._last_error = None
        return value
