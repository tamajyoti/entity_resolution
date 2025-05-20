"""Script to fetch results from a run of the integration tests."""
import requests
import urllib
import sys


def get_file(endpoint, filename):
    """Return a file from the result server."""
    result = requests.get(urllib.parse.urljoin(endpoint, filename))
    result.raise_for_status()
    return result.text


if __name__ == "__main__":
    endpoint = sys.argv[1]

    # Grab the test results and output into a file
    with open("tests.xml", "w") as outfile:
        outfile.write(get_file(endpoint, "tests.xml"))

    # Print the pytest log so its visible
    log_file = get_file(endpoint, "log.txt")
    print(log_file)

    # Finally - exit the script if the tests failed
    result_code = get_file(endpoint, "result_code.txt")
    result_code = int(result_code)
    exit(result_code)
