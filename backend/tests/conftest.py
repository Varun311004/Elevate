from __future__ import annotations

import sys
from pathlib import Path


# Ensure `import backend.*` works when pytest is invoked from the workspace root
# or when tools use a different working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

