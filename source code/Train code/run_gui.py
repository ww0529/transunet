#!/usr/bin/env python
"""
Launch script for V4 Gravity Inversion GUI
Run this script to start the GUI application
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from jgui import main
    main()
