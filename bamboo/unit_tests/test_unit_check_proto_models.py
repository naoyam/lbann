import sys
sys.path.insert(0, '../common_python')
import tools
import pytest
import os, re, subprocess, sys

def skeleton_models(cluster, executables, compiler_name):
    if compiler_name not in executables:
      pytest.skip('default_exes[%s] does not exist' % compiler_name)
    lbann_dir = subprocess.check_output('git rev-parse --show-toplevel'.split()).strip()
    hostname = subprocess.check_output('hostname'.split()).strip()
    host = re.sub("\d+", "", hostname)
    opt = 'adagrad'
    defective_models = []
    tell_Dylan = []
    for subdir, dirs, files in os.walk(lbann_dir + '/model_zoo/models/'):
        if 'greedy' in subdir:
            print('Skipping greedy_layerwise_autoencoder_mnist, kills bamboo agent')
            continue            
        for file_name in files:
            if file_name.endswith('.prototext') and "model" in file_name:
                model_path = subdir + '/' + file_name
                print('Attempting model setup for: ' + file_name )
                data_filedir_ray = None
                data_filedir_train_ray=None
                data_filename_train_ray=None
                data_filedir_test_ray=None
                data_filename_test_ray=None
                if 'mnist' in file_name:
                    data_filedir_ray = '/p/gscratchr/brainusr/datasets/MNIST'
                    data_reader_name = 'mnist'
                elif 'net' in file_name:
                    data_filedir_train_ray = '/p/gscratchr/brainusr/datasets/ILSVRC2012/original/train/'
                    data_filename_train_ray = '/p/gscratchr/brainusr/datasets/ILSVRC2012/labels/train.txt'
                    data_filedir_test_ray = '/p/lscratche/brainusr/datasets/ILSVRC2012/original/val/'
                    data_filename_test_ray = '/p/lscratche/brainusr/datasets/ILSVRC2012/original/labels/val.txt'
                    data_reader_name = 'imagenet'
                elif 'cifar' in file_name:
                    data_reader_name = 'cifar10'
                elif 'char' in file_name:
                    data_reader_name = 'ascii'
                else:
                    print("Tell Dylan which data reader this model needs")
                    tell_Dylan.append(file_name)
                if (cluster == 'ray') and (data_filedir_ray == None) and (data_filedir_train_ray == None):
                    print('Skipping %s because data is not available on ray' % model_path)
                else:
                    cmd = tools.get_command(
                        cluster=cluster, executable=executables[compiler_name], num_nodes=1,
                        partition='pdebug', time_limit=1, dir_name=lbann_dir,
                        data_filedir_ray=data_filedir_ray,
                        data_filedir_train_ray=data_filedir_train_ray,
                        data_filename_train_ray=data_filename_train_ray,
                        data_filedir_test_ray=data_filedir_test_ray,
                        data_filename_test_ray=data_filename_test_ray,
                        data_reader_name=data_reader_name, exit_after_setup=True,
                        model_path=model_path, optimizer_name=opt)
                    if os.system(cmd) != 0:
                        print("Error detected in " + model_path)
                        #defective_models.append(file_name)
                        defective_models.append(cmd)
    if len(defective_models) != 0:
        print("ERRORS: The following models exited with errors")
        for i in defective_models:
            print('ERRORS', i)
        print('ERRORS: tell Dylan: the following models have unknown data readers:')
        for i in tell_Dylan :
            print('ERRORS', i)
    assert len(defective_models) == 0

def test_unit_models_clang4(cluster, exes):
    skeleton_models(cluster, exes, 'clang4')

def test_unit_models_gcc4(cluster, exes):
    skeleton_models(cluster, exes, 'gcc4')

def test_unit_models_gcc7(cluster, exes):
    skeleton_models(cluster, exes, 'gcc7')

def test_unit_models_intel18(cluster, exes):
    skeleton_models(cluster, exes, 'intel18')
