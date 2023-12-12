"""Auxiliar module to implement Poetry commands."""
import subprocess as sp
from pathlib import Path
from shutil import which


def lint():
    """Run Black as linter."""
    print("====| LINT: RUNNING BLACK")
    if which("black"):
        sp.check_call("black --check .", shell=True)
    else:
        _missing_command("black")


def format_code():
    """Format code."""
    print("====| FORMAT: FORMAT PROJECT CODE")
    print("====| FORMAT: RUNNING ISORT")
    # No need to check if installed, it is included as dep of Pylint
    sp.check_call("isort .", shell=True)

    print("====| FORMAT: RUNNING BLACK")
    if which("black"):
        sp.check_call("black .", shell=True)
    else:
        _missing_command("black")


def run_scrapy():
    """Launch scrapy command."""
    print("====| EXECUTE: SCRAPY CRAWL")
    if which("scrapy"):
        sp.check_call("scrapy crawl good_reads", shell=True)
    else:
        _missing_command("scrapy")


def _missing_command(command):
    """Print error with command"""
    print("====| ERROR: WRONG COMMAND: " + command)
