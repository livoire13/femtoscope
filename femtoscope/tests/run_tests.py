import subprocess
from pathlib import Path
from femtoscope import TEST_DIR

if __name__ == '__main__':

    # Ensure previous coverage data is cleared
    subprocess.run(["coverage", "erase"])

    # Find all test files in the directory
    test_files = list(TEST_DIR.glob("test_*.py"))

    errors = []
    for test in test_files:
        print(f"Running {Path(test).name}...")
        result = subprocess.run(
            ["pytest", str(test), "--cov=femtoscope", "--cov-append", "--cov-report=term-missing"],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            errors.append((test, result.stdout, result.stderr))

    # Combine coverage data from all test runs
    subprocess.run(["coverage", "combine"])
    subprocess.run(["coverage", "report"])
    subprocess.run(["coverage", "xml"])  # Generate XML for GitHub reporting

    # Print summary of failures
    if errors:
        print("\nSome tests failed:")
        for test, stdout, stderr in errors:
            print(f"\n--- {test} ---\n{stdout}\n{stderr}")
    else:
        print("\nAll tests passed!")

    # Exit with non-zero code if any test failed
    exit(1 if errors else 0)
