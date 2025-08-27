import os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def get_project_root():
    return PROJECT_ROOT

def cd_to_project_root():
    os.chdir(PROJECT_ROOT)

def add_project_root_to_path(index=0):
    # if index is not None or int, throw error
    if index is not None and not isinstance(index, int):
        raise ValueError("Index must be an integer")
    
    print(f"Project root: {PROJECT_ROOT}")

    if index is None:
        index = 0

    if index == -1:
        sys.path.append(str(PROJECT_ROOT)); return

    sys.path.insert(index, str(PROJECT_ROOT)); return