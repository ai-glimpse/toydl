"""
code from: https://github.com/pydantic/pydantic/blob/main/docs/plugins/main.py
"""

from __future__ import annotations as _annotations

import json
import logging
import re

from pathlib import Path
from textwrap import indent

import autoflake
import pyupgrade._main as pyupgrade_main  # type: ignore

from mkdocs.config import Config
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page

logger = logging.getLogger("mkdocs.plugin")
THIS_DIR = Path(__file__).parent
DOCS_DIR = THIS_DIR.parent
PROJECT_ROOT = DOCS_DIR.parent


def on_pre_build(config: Config) -> None:
    """
    Before the build starts.
    """
    add_mkdocs_run_deps()


def on_files(files: Files, config: Config) -> Files:
    """
    After the files are loaded, but before they are read.
    """
    return files


def on_page_markdown(markdown: str, page: Page, config: Config, files: Files) -> str:
    """
    Called on each file after it is read and before it is converted to HTML.
    """
    markdown = upgrade_python(markdown)
    markdown = insert_json_output(markdown)
    markdown = remove_code_fence_attributes(markdown)
    return markdown


def add_mkdocs_run_deps() -> None:
    # set the pydantic and pydantic-core versions to configure for running examples in the browser
    # pyproject_toml = (PROJECT_ROOT / 'pyproject.toml').read_text()
    # pydantic_core_version = re.search(r'pydantic-core==(.+?)["\']', pyproject_toml).group(1)

    # version_py = (PROJECT_ROOT / 'pydantic' / 'version.py').read_text()
    # pydantic_version = re.search(r'^VERSION ?= (["\'])(.+)\1', version_py, flags=re.M).group(2)

    toydl_version = "0.2.0"
    mkdocs_run_deps = json.dumps([f"toydl=={toydl_version}"])
    logger.info("Setting mkdocs_run_deps=%s", mkdocs_run_deps)

    html = f"""\
    <script>
    window.mkdocs_run_deps = {mkdocs_run_deps}
    </script>
"""
    path = DOCS_DIR / "theme/mkdocs_run_deps.html"
    path.write_text(html)


MIN_MINOR_VERSION = 7
MAX_MINOR_VERSION = 11


def upgrade_python(markdown: str) -> str:
    """
    Apply pyupgrade to all python code blocks, unless explicitly skipped, create a tab for each version.
    """

    def add_tabs(match: re.Match[str]) -> str:
        prefix = match.group(1)
        if 'upgrade="skip"' in prefix:
            return match.group(0)

        if m := re.search(r'requires="3.(\d+)"', prefix):
            min_minor_version = int(m.group(1))
        else:
            min_minor_version = MIN_MINOR_VERSION

        py_code = match.group(2)
        numbers = match.group(3)
        # import devtools
        # devtools.debug(numbers)
        output = []
        last_code = py_code
        for minor_version in range(min_minor_version, MAX_MINOR_VERSION + 1):
            if minor_version == min_minor_version:
                tab_code = py_code
            else:
                tab_code = _upgrade_code(py_code, minor_version)
                if tab_code == last_code:
                    continue
                last_code = tab_code

            content = indent(f"{prefix}\n{tab_code}```{numbers}", " " * 4)
            output.append(f'=== "Python 3.{minor_version} and above"\n\n{content}')

        if len(output) == 1:
            return match.group(0)
        else:
            return "\n\n".join(output)

    return re.sub(
        r"^(``` *py.*?)\n(.+?)^```(\s+(?:^\d+\. .+?\n)+)",
        add_tabs,
        markdown,
        flags=re.M | re.S,
    )


def _upgrade_code(code: str, min_version: int) -> str:
    upgraded = pyupgrade_main._fix_plugins(
        code,
        settings=pyupgrade_main.Settings(
            min_version=(3, min_version),
            keep_percent_format=True,
            keep_mock=False,
            keep_runtime_typing=True,
        ),
    )
    return autoflake.fix_code(upgraded, remove_all_unused_imports=True)


def insert_json_output(markdown: str) -> str:
    """
    Find `output="json"` code fence tags and replace with a separate JSON section
    """

    def replace_json(m: re.Match[str]) -> str:
        start, attrs, code = m.groups()

        def replace_last_print(m2: re.Match[str]) -> str:
            ind, json_text = m2.groups()
            json_text = indent(json.dumps(json.loads(json_text), indent=2), ind)
            # no trailing fence as that's not part of code
            return f"\n{ind}```\n\n{ind}JSON output:\n\n{ind}```json\n{json_text}\n"

        code = re.sub(r'\n( *)"""(.*?)\1"""\n$', replace_last_print, code, flags=re.S)
        return f"{start}{attrs}{code}{start}\n"

    return re.sub(
        r'(^ *```)([^\n]*?output="json"[^\n]*?\n)(.+?)\1',
        replace_json,
        markdown,
        flags=re.M | re.S,
    )


def remove_code_fence_attributes(markdown: str) -> str:
    """
    There's no way to add attributes to code fences that works with both pycharm and mkdocs, hence we use
    `py key="value"` to provide attributes to pytest-examples, then remove those attributes here.

    https://youtrack.jetbrains.com/issue/IDEA-297873 & https://python-markdown.github.io/extensions/fenced_code_blocks/
    """

    def remove_attrs(match: re.Match[str]) -> str:
        suffix = re.sub(
            r' (?:test|lint|upgrade|group|requires|output|rewrite_assert)=".+?"',
            "",
            match.group(2),
            flags=re.M,
        )
        return f"{match.group(1)}{suffix}"

    return re.sub(r"^( *``` *py)(.*)", remove_attrs, markdown, flags=re.M)
