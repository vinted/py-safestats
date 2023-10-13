import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-rtest", action="store_true", default=False, help="run r comparison tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "rtest: mark test as the one which compares to R")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-rtest"):
        return
    skip_rtest = pytest.mark.skip(reason="need --run-rtest option to run")
    for item in items:
        if "rtest" in item.keywords:
            item.add_marker(skip_rtest)
