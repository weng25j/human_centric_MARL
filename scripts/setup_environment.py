"""Create or refresh the project's Python 3.12 environment without editing source."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', default=os.environ.get('MARL_VENV', str(ROOT / '.venv')))
    parser.add_argument('--torch-index-url', default='https://download.pytorch.org/whl/cpu',
                        help='Official PyTorch wheel index; CPU is the default')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error('Run this script with Python 3.12 (python3.12 or py -3.12).')
    destination = Path(args.venv).expanduser().resolve()
    if not (destination/'pyvenv.cfg').is_file():
        if destination.exists() and any(destination.iterdir()):
            parser.error(f'{destination} is not an empty directory or an existing virtual environment.')
        venv.EnvBuilder(with_pip=True, symlinks=os.name != 'nt').create(destination)
    bindir = destination / ('Scripts' if os.name == 'nt' else 'bin')
    python = str(bindir / ('python.exe' if os.name == 'nt' else 'python'))
    version = subprocess.check_output([python, '-c', 'import sys; print(".".join(map(str, sys.version_info[:2])))'], text=True).strip()
    if version != '3.12':
        parser.error(f'{destination} uses Python {version}; choose a Python 3.12 environment.')
    env = dict(os.environ)
    env['PATH'] = str(bindir) + os.pathsep + env.get('PATH', '')
    env.setdefault('MPLBACKEND', 'Agg')
    commands = [
        [python, '-m', 'pip', 'install', '--index-url', args.torch_index_url, 'torch==2.13.0'],
        [python, '-m', 'pip', 'install', '-r', str(ROOT/'requirements.txt'),
         '-r', str(ROOT/'requirements-tools.txt')],
        [python, '-m', 'pip', 'check'],
        [str(bindir / ('sumo.exe' if os.name == 'nt' else 'sumo')), '--version'],
        [python, '-c', 'import torch, traci, cv2, pandas, openpyxl; '
         'from algorithms.mappo import MAPPO; from algorithms.z_merge import ZMergeAgent; '
         'print("Training, simulation and spreadsheet imports passed")'],
    ]
    for command in commands:
        print(subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    print(f'Environment ready: {destination}')
    print(f'Linux/macOS: source "{destination}/bin/activate"')
    print(f'Windows PowerShell: & "{destination}/Scripts/Activate.ps1"')


if __name__ == '__main__':
    main()
