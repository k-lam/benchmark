import os
import argparse
from os.path import join
from datetime import datetime
import logging
from tqdm import tqdm
import shlex
import subprocess
import gc
import torchvision.models as models
from utils import get_env_pytorch_examples, cmd_string, model_names
import torchvision

model_names = sorted(name for name in models.__dict__
                     if not (not (name.islower() and not name.startswith("__"))
                             or not callable(models.__dict__[name])))

parser = argparse.ArgumentParser(description="PyTorch model kl test.")
parser.add_argument('--arch', type=str, default='all', choices=model_names, nargs='+',
                    help='model architectures: ' + ' | '.join(model_names) + ' (default: all)')
parser.add_argument('--log-path', type=str, default='log',
                    help='the path on the file system to place the working log directory at')
parser.add_argument('--filename', type=str, default='perf_test',
                    help='name of the output file')
parser.add_argument('--data-dir', type=str, required=True,
                    help='path to imagenet dataset')

def log_init():
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # logging
    log_filename = args.filename + '.out'
    logging.basicConfig(filename=join(temp_dir, log_filename), level=20)


def config_models(model):
    if model == 'all':
        model = model_names
    return model


def execution(cmd, log_path):
    gc.collect()

    # logging
    log_file = open(log_path, "a+")
    log_file.write(cmd)
    log_file.write('\n')

    exec_command = shlex.split(cmd)
    proc = subprocess.Popen(exec_command, stdout=log_file, stderr=subprocess.STDOUT)
    proc.wait()
    return_code = proc.returncode
    log_file.close()


def main():
    global args, temp_dir
    args = parser.parse_args()

    log_init()

    log_path = args.log_path
    examples_home = get_env_pytorch_examples()
    imagenet = join(examples_home, 'imagenet', 'main.py')
    nets = config_models(args.arch)

    for i in tqdm(range(len(nets))):
        model = nets[i]
        # execution
        cmd = cmd_string(imagenet, model, args.data_dir)
        execution(cmd, log_path)
        logging.info('{}'.format(model))


if __name__ == '__main__':
    main()