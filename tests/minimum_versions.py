"""Print the lowest version that `requirements.txt` allows for each dependency that has one, as constraints for pip.

    python tests/minimum_versions.py > minimum.txt
    pip install -r requirements.txt -c minimum.txt

A job of the continuous integration runs the tests in this environment: the lowest versions
of `requirements.txt` are tested, and not only announced. The dependencies that have no
lowest version are installed in their newest one.
"""

import re
import sys
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"


def minimum_versions(requirements):
    """{dependency: lowest version} for the lines `dependency>=version` of the text of a requirements file."""
    versions = {}
    for line in requirements.splitlines():
        requirement = line.split("#")[0].strip()
        if not requirement:
            continue
        name, specifiers = re.fullmatch(r"([A-Za-z0-9._-]+)\s*(.*)", requirement).groups()
        for specifier in filter(None, (specifier.strip() for specifier in specifiers.split(","))):
            operator, version = re.fullmatch(r"(>=|<=|==|!=|~=|>|<)\s*(\S+)", specifier).groups()
            if operator in (">", "~=", "=="):
                # `>1.4` does not name the lowest version, and the two other ones are not written `>=`
                raise ValueError(f"{name}{specifier}: write the lowest version of a dependency as `{name}>=version`")
            if operator == ">=":
                versions[name] = version
    return versions


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else REQUIREMENTS
    for name, version in minimum_versions(path.read_text()).items():
        print(f"{name}=={version}")
