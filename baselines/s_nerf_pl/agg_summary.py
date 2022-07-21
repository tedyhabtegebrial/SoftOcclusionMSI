import os
import json
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', required=True, type=str, help='pass a folder that contains all experiments')

args = parser.parse_args()
src_dir = args.input_path

json_files = sorted([str(f) for f in Path(src_dir).rglob('*summary.json')])
print(len(json_files))
summary_txt = os.path.join(src_dir, 'summary.txt')
text_lines = []
for j in json_files:
    with open(j, 'r') as fid:
        data = json.load(fid)
    text_line = f'{j}   ssim: {data["ssim"]} psnr: {data["psnr"]} \n'
    text_lines.append(text_line)
    print(text_line)
with open(summary_txt, 'w') as fid:
    # for tl in text_lines:
    fid.writelines(text_lines)

