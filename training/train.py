import argparse
from pathlib import Path

import yaml

from mae3d.main_pretrain import main 
from mae3d.util.io_manager import Args, copy_files, get_output_folder

import sequence as seq

def train_model(par_file):
    with open (par_file, 'r') as ff:
        par = ff.read()
        args = yaml.safe_load(par)

    train_data_args = Args({**args['data']['general'], **args['data']['train']})
    dataset_train = seq.CustomDataset(train_data_args)

    model_args = Args(args['model'])

    model_args.output_dir = get_output_folder(
        args['output']['output_dir'],
        args['output']['sub'],
    )
    model_args.log_dir = model_args.output_dir

    # sequence output size = model input size
    model_args.input_size = train_data_args.shape

    if model_args.output_dir:
        Path(model_args.output_dir).mkdir(parents=True, exist_ok=True)
    files_to_copy = [
        __file__,
        seq.__file__,
        [par_file, 'parameter.yml'],
    ]
    copy_files(files_to_copy, model_args.output_dir)
    main(model_args, dataset_train)

def get_param_files(path):
    path = Path(path)
    if path.is_dir():
        files = sorted([ff for ff in path.iterdir() if ff.suffix == '.yml'])
    elif path.suffix == '.yml':
        files = [path]
    else:
        raise ValueError('non valid parameter files (must be folder or .yml)')
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pdir', help='parameter directory')
    parser.add_argument('--gpu', help='choose gpu [0, 1]')
    pargs = parser.parse_args()
    
    for paramf in get_param_files(pargs.pdir):
        print('processing: '+str(paramf))
        try:
            train_model(paramf)
        except Exception as e:
            print(f'failed: {e}')
