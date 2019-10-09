import pytest
import re, subprocess


def pytest_addoption(parser):
    cluster = re.sub('[0-9]+', '', subprocess.check_output(
        'hostname'.split()).decode('utf-8').strip())
    default_dirname = subprocess.check_output(
        'git rev-parse --show-toplevel'.split()).decode('utf-8').strip()
    parser.addoption('--cluster', action='store', default=cluster,
                     help='--cluster=<cluster> to specify the cluster being run on, for the purpose of determing which commands to use. Default the current cluster')
    parser.addoption('--dirname', action='store', default=default_dirname,
                     help='--dirname=<path_to_dir> specifies the top-level directory')


@pytest.fixture
def cluster(request):
    return request.config.getoption('--cluster')


@pytest.fixture
def dirname(request):
    return request.config.getoption('--dirname')
