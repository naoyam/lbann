import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os


def skeleton_mnist_softmax_classifier(cluster, executables, dir_name, compiler_name,
                                      weekly, data_reader_percent):
    if not weekly:
        e = 'test_unit_mnist_softmax_classifier: Not doing weekly testing'
        print('SKIP - ' + e)
        pytest.skip(e)

    if compiler_name not in executables:
      e = 'skeleton_mnist_softmax_classifier: default_exes[%s] does not exist' % compiler_name
      print('Skip - ' + e)
      pytest.skip(e)
    output_file_name = '%s/bamboo/unit_tests/output/mnist_softmax_classifier_%s_output.txt' % (dir_name, compiler_name)
    error_file_name  = '%s/bamboo/unit_tests/error/mnist_softmax_classifier_%s_error.txt' % (dir_name, compiler_name)
    command = tools.get_command(
        cluster=cluster, executable=executables[compiler_name], num_nodes=1,
        num_processes=1, dir_name=dir_name,
        data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
        data_reader_name='mnist',
        data_reader_percent=data_reader_percent,
        model_folder='tests', model_name='mnist_softmax_classifier',
        optimizer_name='adam',
        output_file_name=output_file_name, error_file_name=error_file_name, weekly=weekly)
    return_code = os.system(command)
    tools.assert_success(return_code, error_file_name)


def test_unit_mnist_softmax_classifier_clang6(cluster, exes, dirname,
                                              weekly, data_reader_percent):
    skeleton_mnist_softmax_classifier(cluster, exes, dirname, 'clang6',
                                      weekly, data_reader_percent)


def test_unit_mnist_softmax_classifier_gcc7(cluster, exes, dirname,
                                            weekly, data_reader_percent):
    skeleton_mnist_softmax_classifier(cluster, exes, dirname, 'gcc7',
                                      weekly, data_reader_percent)


def test_unit_mnist_softmax_classifier_intel19(cluster, exes, dirname,
                                               weekly, data_reader_percent):
    skeleton_mnist_softmax_classifier(cluster, exes, dirname, 'intel19',
                                      weekly, data_reader_percent)


# Run with python3 -m pytest -s test_unit_softmax_classifier.py -k 'test_unit_mnist_softmax_classifier_exe' --exe=<executable>
def test_unit_mnist_softmax_classifier_exe(cluster, dirname, exe,
                                           weekly, data_reader_percent):
    if exe is None:
        e = 'test_unit_mnist_softmax_classifier_exe: Non-local testing'
        print('Skip - ' + e)
        pytest.skip(e)
    exes = {'exe': exe}
    skeleton_mnist_softmax_classifier(cluster, exes, dirname, 'exe',
                                      weekly, data_reader_percent)
