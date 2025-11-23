#!/usr/bin/env python3
"""
Launch Streamlit web UI.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch Streamlit web UI."""
    # Ensure project root is in Python path
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Run streamlit with the module path
    # This ensures proper package resolution
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        str(project_root / "src" / "web_ui.py"),
        "--server.headless", "true"
    ])

if __name__ == "__main__":
    main()

