import subprocess
import os
from pathlib import Path
from femtoscope import TEST_DIR

if __name__ == "__main__":

    try:
        import pytest
    except ImportError:
        raise RuntimeError(
            "pytest is required to run the femtoscope test suite. "
            "Install it with `pip install pytest` or `conda install -c conda-forge pytest`."
        )

    # Determine whether to use coverage
    use_cov = os.environ.get("FEMTOSCOPE_COVERAGE", "0") == "1"

    if use_cov:
        try:
            import pytest_cov
        except ImportError:
            print("pytest-cov not found, running tests without coverage.")
            use_cov = False

    if use_cov:
        # Clear previous coverage data
        subprocess.run(["coverage", "erase"])
    # Find all test files in the directory
    test_files = list(TEST_DIR.glob("test_*.py"))
    errors = []

    for test in test_files:
        print(f"Running {Path(test).name}...")

        cmd = ["pytest", str(test)]
        if use_cov:
            cmd += ["--cov=femtoscope", "--cov-append", "--cov-report=term-missing"]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            errors.append((test, result.stdout, result.stderr))

    if use_cov:
        subprocess.run(["coverage", "combine"])
        subprocess.run(["coverage", "report"])
        subprocess.run(["coverage", "xml"])

    # Print summary of failures
    if errors:
        print("\nSome tests failed:")
        for test, stdout, stderr in errors:
            print(f"\n--- {test} ---\n{stdout}\n{stderr}")
    else:
        print("\nAll tests passed!")

    # Exit with non-zero code if any test failed
    exit(1 if errors else 0)
