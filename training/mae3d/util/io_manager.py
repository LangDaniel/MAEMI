from pathlib import Path

class Args:
    '''convert dict to class variables (yaml to parser like)'''
    def __init__(self, d):
        for key, value in d.items():
            if type(value) == dict:
                value = Args(value)
            if value == 'None':
                value = None
            setattr(self, key, value)
    def __str__(self):
        return str(self.__dict__)

def get_output_folder(output_dir, sub_pattern):
    output_dir = Path(output_dir)
    
    sub_pattern += '_' 
    
    if not output_dir.exists():
        return str(output_dir / (sub_pattern + '0'))
    
    counts = sorted(
        [int(ff.stem.replace(sub_pattern, ''))\
            for ff in output_dir.iterdir() if sub_pattern in ff.stem]
    )   
    c = int(counts[-1]) + 1 
    return str(output_dir / (sub_pattern + str(c)))

def copy_files(files, path):
    path = Path(path)
    for f in files:
        if isinstance(f, list):
            name = Path(f[1]).name
            content = Path(f[0]).read_text()
        else:
            name = Path(f).name
            content = Path(f).read_text()
        with open(path/name, 'w') as dest:
            dest.write(content)
        (path/name).chmod(0o444)

def get_param_files(path):
    path = Path(path)
    if path.is_dir():
        files = sorted([ff for ff in path.iterdir() if ff.suffix == '.yml'])
    elif path.suffix == '.yml':
        files = [path]
    else:
        raise ValueError('non valid parameter files (must be folder or .yml)')
    return files
