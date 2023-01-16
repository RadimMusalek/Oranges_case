"""
__author__ = Radim Musalek
"""

from pathlib import Path


class Constants(object):
    # Paths
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATASETS_DIR = ROOT_DIR / "datasets"
    OUTPUTS_DIR = ROOT_DIR / "outputs_charts"

    # Other constants come below


constants = Constants()
