
"""
Validate that inference completed successfully for every result file.
"""
import os
import sys
import numpy as np

def check_file(filepath, rows=30, cols=100):
    try:
        data = np.loadtxt(filepath)
        rv = data.shape == (rows, cols)
        return rv
    except Exception:
        return False

def main():
    from pathlib import Path
    root = sys.argv[1] if len(sys.argv) > 1 else '.'  # Relative to the repository root.
    root = Path(__file__).parent.parent / root

    failed = [
        os.path.join(dirpath, f) for dirpath, _, filenames in os.walk(root) 
        for f in filenames if not check_file(os.path.join(dirpath, f))]
    print('\n'.join(failed) if failed else "All files passed validation.")

if __name__ == '__main__':
    main()
