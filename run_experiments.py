"""Repository CLI. Installed command: bo-benchmark."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from bo_benchmark_app.publication import main

if __name__=='__main__':
    raise SystemExit(main())
