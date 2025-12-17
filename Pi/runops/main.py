# main.py
import sys
import traceback

from config import Config
from orchestrator import Orchestrator

if __name__ == "__main__":
    try:
        cfg = Config.load()   # Load from config.json
        orch = Orchestrator(cfg)
        orch.run()
    except Exception as e:
        print(f"ERROR: ", repr(e))
        traceback.print_exc()
        sys.exit(1)
