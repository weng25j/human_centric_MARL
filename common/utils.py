import cv2, os
import torch as th
import numpy as np
from shutil import copy
import torch.nn as nn

def entropy(p):
    # Adds a small epsilon to prevent log(0) which results in NaN
    p = th.clamp(p, 1e-8, 1.0)
    return -th.sum(p * th.log(p), 1)

def kl_log_probs(log_p1, log_p2):
    return -th.sum(th.exp(log_p1) * (log_p2 - log_p1), 1)

class AddBias(nn.Module):
    """Useful for KFAC/ACKTR implementations"""
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias

def index_to_one_hot(index, dim):
    if isinstance(index, int) or isinstance(index, np.integer):
        one_hot = np.zeros(dim)
        one_hot[index] = 1.
    else:
        one_hot = np.zeros((len(index), dim))
        one_hot[np.arange(len(index)), index] = 1.
    return one_hot

def to_tensor_var(x, use_cuda=True, dtype="float"):
    """
    Modernized PyTorch tensor conversion. 
    Removes the deprecated Variable() and np.long.
    """
    device = th.device("cuda" if use_cuda and th.cuda.is_available() else "cpu")
    
    if dtype == "float":
        return th.tensor(np.array(x, dtype=np.float32), device=device, dtype=th.float32)
    elif dtype == "long":
        # np.long is deprecated, replaced with np.int64
        return th.tensor(np.array(x, dtype=np.int64), device=device, dtype=th.long)
    elif dtype == "byte":
        return th.tensor(np.array(x, dtype=np.int8), device=device, dtype=th.int8)
    else:
        return th.tensor(np.array(x, dtype=np.float32), device=device, dtype=th.float32)

def agg_double_list(l):
    # l: [ [...], [...], [...] ]
    # l_i: result of each step in the i-th episode
    s = [np.sum(np.array(l_i), 0) for l_i in l]
    s_mu = np.mean(np.array(s), 0)
    s_std = np.std(np.array(s), 0)
    return s_mu, s_std

class VideoRecorder:
    """This is used to record videos of evaluations if GUI rendering is converted to RGB"""
    def __init__(self, filename, frame_size, fps):
        self.video_writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*"mp4v"), int(fps),
            (frame_size[1], frame_size[0])
        )

    def add_frame(self, frame):
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def release(self):
        self.video_writer.release()

    def __del__(self):
        self.release()

def backup_project_files(tar_dir):
    """
    Adapted for Traffic Shepherd. 
    Backs up your SUMO config files and python scripts to the results directory.
    """
    files_to_copy = [
        'merging_network.net.xml',  # SUMO network
        'traffic_routes.rou.xml',   # SUMO routes and HDV mixtures
        'marl_on_ramp_env.py',      # Your PettingZoo environment
        'ma2c_agent.py',            # The MA2C Algorithm
        'main.py'                   # The training script
    ]
    
    for file in files_to_copy:
        if os.path.exists(file):
            copy(file, tar_dir)
        else:
            print(f"Backup warning: {file} not found.")

def init_dir(base_dir, pathes=['train_videos', 'configs', 'models', 'eval_videos', 'eval_logs']):
    """Initializes the results folder structure."""
    if not os.path.exists("./results/"):
        os.mkdir("./results/")
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
        
    dirs = {}
    for path in pathes:
        cur_dir = os.path.join(base_dir, path)
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
        
    return dirs