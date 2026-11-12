"""Select the regression tests that a change between two revisions adds or affects.

Usage: select_test_cases.py BASE_REV HEAD_REV

The script prints two lines in the GitHub Actions output format:

  cases=<input files of the selected test.py cases, separated by commas>
  coupled=<true or false>

The cases value is empty when no case is selected, and test.py accepts it in EXAMPLE_CASES. A case
is identified by its input file. HEAD_REV selects a case when one of the following is true:

- The case is new, or its TEST_CASES entry differs from the entry at BASE_REV.
- The case's input file changed.
- Another file changed in the directory that holds the case's input file, or in a subdirectory
  of it. The input file of another case does not count, so that editing one of several inputs
  in a directory selects only that case.
- A file changed in a directory above the input file, inside the same example. CASL_VERA_2D
  inputs, for example, import CASL_VERA_2D/casl_pin_power.py.

coupled is true when a file changed in the OpenSn-OpenFOAM coupling example, which the workflow
tests with its own scripts.

Changes to documentation and images (see DOCUMENTATION_SUFFIXES) select nothing. Cases removed at
HEAD_REV are not selected. The script must run inside the examples repository. It reads files with
git, so the revisions do not need to be checked out.
"""

import importlib.util
import os
import posixpath
import subprocess
import sys
import tempfile

# test.py reads these files while it builds TEST_CASES. Add a file here when test.py starts to
# read another one at import time.
CASE_DATA_FILES = ("test.py", "CASL_VERA_2D/casl_reference_k.csv")

COUPLED_TEST_DIR = "OpenFOAM_Electron_Converter"

DOCUMENTATION_SUFFIXES = (".md", ".png", ".gif", ".jpg", ".jpeg", ".svg")


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def verify_revision(rev):
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"], capture_output=True
    )
    if result.returncode != 0:
        raise ValueError(f"Revision {rev!r} is not a commit in this clone")


def path_exists(rev, path):
    result = subprocess.run(["git", "cat-file", "-e", f"{rev}:{path}"], capture_output=True)
    return result.returncode == 0


def normalized(value, root):
    """Replace absolute paths under root with relative ones, so that entries loaded from different
    directories compare equal when they describe the same test."""
    if isinstance(value, str):
        if value.startswith(root + os.sep):
            return os.path.relpath(value, root)
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(normalized(item, root) for item in value)
    if isinstance(value, dict):
        return {key: normalized(item, root) for key, item in value.items()}
    return value


def case_entries(rev, label, workdir):
    """Return the TEST_CASES entries at the given revision, keyed by input file."""
    root = os.path.join(workdir, label)
    for path in CASE_DATA_FILES:
        if not path_exists(rev, path):
            continue
        destination = os.path.join(root, *path.split("/"))
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "wb") as data_file:
            data_file.write(git("show", f"{rev}:{path}"))

    test_file = os.path.join(root, "test.py")
    if not os.path.isfile(test_file):
        return {}

    spec = importlib.util.spec_from_file_location(f"examples_test_{label}", test_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        os.path.normpath(case[0]).replace(os.sep, "/"): normalized(case, root)
        for case in module.TEST_CASES
    }


def changed_files(base_rev, head_rev):
    """Return the files added, modified, or removed between the revisions, ignoring documentation.

    Renames are reported as a removal and an addition, so both paths are returned.
    """
    output = git("diff", "--name-only", "--no-renames", base_rev, head_rev).decode()
    return [
        path for path in output.splitlines()
        if path and not path.lower().endswith(DOCUMENTATION_SUFFIXES)
    ]


def affects_case(path, input_file, input_files):
    """Return whether a changed file affects a case.

    input_files holds the input files of every case at either revision.
    """
    if path in input_files:
        return path == input_file
    file_dir = posixpath.dirname(path)
    input_dir = posixpath.dirname(input_file)
    if not file_dir:
        # Files at the repository root do not belong to an example.
        return False
    return (
        file_dir == input_dir
        or file_dir.startswith(input_dir + "/")
        or input_dir.startswith(file_dir + "/")
    )


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    base_rev, head_rev = sys.argv[1:]
    verify_revision(base_rev)
    verify_revision(head_rev)

    # Both variables restrict TEST_CASES, so remove them to see every case at each revision.
    os.environ.pop("CASL_CASES", None)
    os.environ.pop("EXAMPLE_CASES", None)

    with tempfile.TemporaryDirectory() as workdir:
        base_cases = case_entries(base_rev, "base", workdir)
        head_cases = case_entries(head_rev, "head", workdir)

    paths = changed_files(base_rev, head_rev)
    input_files = set(base_cases) | set(head_cases)
    selected_cases = sorted(
        input_file for input_file, entry in head_cases.items()
        if base_cases.get(input_file) != entry
        or any(affects_case(path, input_file, input_files) for path in paths)
    )
    for case in selected_cases:
        if "," in case or "\n" in case:
            raise ValueError(f"Input file name cannot contain a comma or newline: {case!r}")
    coupled = any(path.startswith(COUPLED_TEST_DIR + "/") for path in paths)

    print(f"cases={','.join(selected_cases)}")
    print(f"coupled={'true' if coupled else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
