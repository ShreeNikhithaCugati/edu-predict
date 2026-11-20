"""Run all test suites."""
import subprocess
import sys

TESTS = [
    "tests.test_authentication",
    "tests.test_permissions",
    "tests.test_database",
    "tests.test_predictions",
]


def main():
    passed, failed = 0, 0
    for t in TESTS:
        print(f"\n▶ {t}")
        result = subprocess.run([sys.executable, "-m", t], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout.strip())
            passed += 1
        else:
            print("FAILED")
            print(result.stderr[-800:])
            failed += 1

    print(f"\n{'='*50}")
    print(f"Result: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()