"""The documentation stays in step with the CLI: every option has help text, the guide's command reference is
generated from the parser as it is now, and every relative link in the docs points at a file that exists."""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_every_option_has_help_text():
    from make_guide import _commands

    from wsr.cli import build_parser
    missing = [(name, a.dest) for name, _, sub in _commands(build_parser(), "wsr") for a in sub._actions
               if not isinstance(a, (argparse._HelpAction, argparse._SubParsersAction)) and not a.help]
    assert not missing, missing


def test_the_guides_command_reference_is_up_to_date():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_guide.py"), "--check"], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr


def test_relative_links_in_the_docs_resolve():
    broken = []
    for md in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]:
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", md.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("mailto:"):
                continue
            if not (md.parent / target).exists():
                broken.append(f"{md.name}: {target}")
    assert not broken, broken
