import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os


def skeleton_mnist_conv_graph(cluster, executables, dir_name, compiler_name,
                              weekly, data_reader_percent):
    if compiler_name not in executables:
      e = 'skeleton_mnist_conv_graph: default_exes[%s] does not exist' % compiler_name
      print('Skip - ' + e)
      pytest.skip(e)
    output_file_name = '%s/bamboo/unit_tests/output/mnist_conv_graph_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/mnist_conv_graph_%s_error.txt' % (dir_name, compiler_name)
    if compiler_name == 'gcc7':
        tl = 240
    else:
        tl = None
    command = tools.get_command(
        cluster=cluster, executable=executables[compiler_name],
        num_nodes=1, time_limit=tl, num_processes=1,
        dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='mnist',
        data_reader_percent=data_reader_percent,
        model_folder='tests',
        model_name='mnist_conv_graph',
        optimizer_name='adam',
        output_file_name=output_file_name,
        error_file_name=error_file_name, weekly=weekly)
    return_code = os.system(command)
    tools.assert_success(return_code, error_file_name)


def test_unit_mnist_conv_graph_clang6(cluster, exes, dirname,
                                      weekly, data_reader_percent):
    skeleton_mnist_conv_graph(cluster, exes, dirname, 'clang6',
                              weekly, data_reader_percent)


def test_unit_mnist_conv_graph_gcc7(cluster, exes, dirname,
                                    weekly, data_reader_percent):
    skeleton_mnist_conv_graph(cluster, exes, dirname, 'gcc7',
                              weekly, data_reader_percent)


def test_unit_mnist_conv_graph_intel19(cluster, exes, dirname,
                                       weekly, data_reader_percent):
    skeleton_mnist_conv_graph(cluster, exes, dirname, 'intel19',
                              weekly, data_reader_percent)


# Run with python3 -m pytest -s test_unit_conv_graph.py -k 'test_unit_mnist_conv_graph_exe' --exe=<executable>
def test_unit_mnist_conv_graph_exe(cluster, dirname, exe, weekly, data_reader_percent):
    if exe is None:
        e = 'test_unit_mnist_conv_graph_exe: Non-local testing'
        print('Skip - ' + e)
        pytest.skip(e)
    exes = {'exe': exe}
    skeleton_mnist_conv_graph(cluster, exes, dirname, 'exe', weekly, data_reader_percent)
