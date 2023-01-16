"""
__author__ = Radim Musalek
"""

import os
import shutil


def delete_dir_content(*, path_to_dir: str):
    '''
    Delete content of a directory
    '''
    with os.scandir(path_to_dir) as entries:
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)

def model():
    '''
    Model
    '''
    return 0
