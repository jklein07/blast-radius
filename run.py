"""Entry point of the image: run selector.py as the main module from its
compiled bytecode. `python selector.py` compiles the 3000-line file on every
start (a script is never read from __pycache__); a module run through runpy
is. Same code, same `__main__`, same arguments."""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
runpy.run_module("selector", run_name="__main__", alter_sys=True)
