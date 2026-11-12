"""Compatibility import for recovery and the unified paper analysis command."""
import hashlib
from .configuration import load


def sha(data):
    return hashlib.sha256(data).hexdigest()


def analyze(config, workspace=None, snapshot=False):
    from .paper_analysis import analyze as paper_analysis
    name = {'olympus':'olympus','niah':'niah','bayesmark_full':'bayesmark'}[load(config,workspace)['benchmark']]
    return paper_analysis(name)
