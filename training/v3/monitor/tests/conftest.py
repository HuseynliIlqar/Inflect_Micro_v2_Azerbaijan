"""Make the monitor package importable when the suite runs from the repository root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
