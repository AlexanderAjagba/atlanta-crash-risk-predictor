"""Holdout checks require an explicit opt-in; routine tests never evaluate 2025."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def pytest_addoption(parser):
    parser.addoption('--run-holdout',action='store_true',default=False,help='Explicitly allow tests reading holdout data or holdout reports')
def pytest_configure(config):
    config.addinivalue_line('markers','holdout: reads the reserved holdout or its evaluation reports')
def pytest_collection_modifyitems(config,items):
    if config.getoption('--run-holdout'): return
    for item in items:
        if 'holdout' in item.keywords:
            item.add_marker(pytest.mark.skip(reason='Holdout preserved; requires --run-holdout'))
