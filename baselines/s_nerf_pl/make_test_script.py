import os, stat
import json
from pathlib import Path
import argparse

def make_executable(file_path):
    # https: // stackoverflow.com/questions/12791997/how-do-you-do-a-simple-chmod-x-from-within-python
    st = os.stat(file_path)
    os.chmod(file_path, st.st_mode | stat.S_IEXEC)

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', required=True, type=str, help='pass a folder that contains all experiments')

args = parser.parse_args()
src_dir = args.input_path

sh_files = sorted([os.path.join(src_dir, f) for f in sorted(os.listdir(src_dir)) if f.endswith('.sh')])
sh_files = [f for f in sh_files if os.path.isfile(f)]
trg_dir = src_dir+'_test'
os.makedirs(trg_dir, exist_ok=True)
single_bash_file = trg_dir+'/run_test.sh'
with open(single_bash_file, 'w') as fid:
    for f in sorted(sh_files):
        f_base = Path(f).stem + f[-3:]
        fid.writelines(f'./{f_base}\n')
make_executable(single_bash_file)

# read the existing bash files
for f in sorted(sh_files):
    base_name = Path(f).stem + f[-3:]
    with open(f, 'r') as fid:
        data = fid.read()
    # print(data)
    d1, d2 = (data).split('python ')
    cmd = (data).replace('train.py', 'eval.py')
    cmd = cmd.replace(' --mem=24G ', ' --mem=48G ')
    cmd = cmd.replace('RTXA6000', 'A100')
    
    target_sh_file = f'{trg_dir}/{base_name}'
    with open(target_sh_file, 'w') as fid:
        fid.write(cmd)
    make_executable(target_sh_file)
    # print(cmd)
    # print(base_name)
    # exit()
# Modify the text within each test bash file


# remove srun command
