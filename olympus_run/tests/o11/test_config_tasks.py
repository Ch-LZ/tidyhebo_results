import os
import glob
import yaml
import pytest

from pathlib import Path

# Path to the task-config directory.
ROOT_PATH = Path(__file__).parent.parent.parent.parent
CONFIG_DIR = ROOT_PATH / 'olympus_run' / "configs" / "tasks"

pytestmark = pytest.mark.env_o11


def test_config_files():
    """
    Validate CONFIG_DIR: files must be YAML and contain the required task keys.
    """
    # Collect files directly in this directory.
    all_files = [f for f in os.listdir(CONFIG_DIR) if os.path.isfile(os.path.join(CONFIG_DIR, f))]
    
    # Check that every file has a supported extension.
    yaml_extensions = ('.yaml', '.yml')
    non_yaml = [f for f in all_files if not f.lower().endswith(yaml_extensions)]
    assert not non_yaml, f"Found non-YAML files: {non_yaml}"
    
    # Validate every YAML file.
    yaml_files = [f for f in all_files if f.lower().endswith(yaml_extensions)]
    for filename in yaml_files:
        filepath = os.path.join(CONFIG_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                pytest.fail(f"File {filename} is not valid YAML: {e}")
        
        # Check required keys.
        required_keys = {'task_name', 'less_is_better', 'params', 'on_simplex'}

        missing = required_keys - set(data.keys())
        assert not missing, f"File {filename} is missing keys: {missing}"

        # task_name must match the filename stem.
        expected_name = os.path.splitext(filename)[0]  # Remove the final extension.
        assert data['task_name'] == expected_name, \
            f"In {filename}, task_name='{data['task_name']}' does not match filename '{expected_name}'"
