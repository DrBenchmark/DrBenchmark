"""Tests of the files that describe the dependencies: requirements.txt and constraints.txt."""

import re
from importlib import metadata

import pytest
from packaging.requirements import Requirement

from helpers import REPO


def read_requirements(name):
    """Requirements listed in a file, without its comments."""
    requirements = []
    for line in (REPO / name).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            requirements.append(Requirement(line))
    return requirements


def normalise(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_ids(requirement):
    return requirement.name


REQUIREMENTS = read_requirements("requirements.txt")
# Exact versions of a tested environment: {normalised name: requirement "name==version"}
TESTED = {normalise(requirement.name): requirement for requirement in read_requirements("constraints.txt")}


@pytest.mark.parametrize("requirement", REQUIREMENTS, ids=requirement_ids)
def test_requirement_has_an_upper_bound(requirement):
    """A new major release of a dependency is not installed before the tests pass with it."""
    assert any(specifier.operator in ("<", "<=", "==", "~=") for specifier in requirement.specifier), requirement


def test_constraints_are_exact_versions():
    for constraint in TESTED.values():
        (specifier,) = constraint.specifier
        assert specifier.operator == "==" and "*" not in specifier.version, constraint


@pytest.mark.parametrize("requirement", REQUIREMENTS, ids=requirement_ids)
def test_tested_version_is_allowed_by_the_requirements(requirement):
    """The two files can be used together: `pip install -r requirements.txt -c constraints.txt`."""
    assert normalise(requirement.name) in TESTED, f"{requirement.name} is missing from constraints.txt"
    constraint = TESTED[normalise(requirement.name)]
    (specifier,) = constraint.specifier
    assert requirement.specifier.contains(specifier.version), f"{constraint} is not allowed by {requirement}"


@pytest.mark.parametrize("requirement", REQUIREMENTS, ids=requirement_ids)
def test_installed_version_is_allowed_by_the_requirements(requirement):
    """The tests run in an environment that requirements.txt describes."""
    try:
        installed = metadata.version(requirement.name)
    except metadata.PackageNotFoundError:
        pytest.fail(f"{requirement.name} is not installed", pytrace=False)
    assert requirement.specifier.contains(installed, prereleases=True), (
        f"{requirement.name} {installed} is installed, but requirements.txt asks for {requirement.specifier}"
    )
