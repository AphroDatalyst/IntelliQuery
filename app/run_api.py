#!/usr/bin/env python3
"""
Launch FastAPI server.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch FastAPI server."""
    # Ensure project root is in Python path
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    subprocess.run([sys.executable, "-m", "uvicorn", "src.api:app", "--reload", "--host", "0.0.0.0", "--port", "8000"])

if __name__ == "__main__":
    main()

