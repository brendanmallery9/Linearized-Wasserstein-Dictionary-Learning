import os
import subprocess
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SAE_analysis_functions import *
import torch
import shutil

# ---- config ----

import argparse
import json 

parser = argparse.ArgumentParser()
parser.add_argument("--out_root", type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--seed",type=int,default=0)
parser.add_argument(
    "--sae_params",
    type=json.loads,   # <-- magic line
    default={"JUMPRELUAE_10_5e5_mon":{"architecture":"JumpReLU_monotone","l1":"5e-5","lr":1e-4,"hidden_dim":10,"top_K":0}},
    help="JSON dictionary of SAE parameters"
)



args = parser.parse_args()

OUT_ROOT = Path(args.out_root)
DATA_DIR = Path(args.data_dir)
EPOCHS = args.epochs




'''
SAE PARAMETERS:

SAE_PARAMETERS={
    # 'TOPKAE_128_mon':{'architecture':'TopKAE_monotone','l1':'0', 'lr': 1e-7,'hidden_dim':128,'top_K':8},
   # 'TOPKAE_512_mon':{'architecture':'TopKAE_monotone','l1':'0','hidden_dim':512,'top_K':12},
   # 'TOPKAE_1024_mon':{'architecture':'TopKAE_monotone','l1':'0', 'lr': 1e-6,'hidden_dim':1024,'top_K':32},
  #  'TOPKAE_2048_mon':{'architecture':'TopKAE_monotone','l1':'0', 'lr': 1e-8,'hidden_dim':2048,'top_K':32},

#    'JUMPRELUAE_40_1e3_mon':{'architecture':'JumpReLU_monotone','l1':'1e-3','hidden_dim':40,'top_K':0},
  #  'JUMPRELUAE_128_1e3_mon':{'architecture':'JumpReLU_monotone','l1':'1e-3', 'lr': 1e-5,'hidden_dim':128,'top_K':0},

#    'JUMPRELUAE_1024_1e3_mon':{'architecture':'JumpReLU_monotone','l1':'1e-3', 'lr': 1e-5,'hidden_dim':1024,'top_K':0},
  #     'JUMPRELUAE_1024_1e5_mon':{'architecture':'JumpReLU_monotone','l1':'1e-5', 'lr': 1e-5,'hidden_dim':1024,'top_K':0},
  #    'JUMPRELUAE_10_1e5_mon_many_iters':{'architecture':'JumpReLU_monotone','l1':'1e-5','lr': 1e-5,'hidden_dim':10,'top_K':0},
  #          'JUMPRELUAE_10_1e4_mon':{'architecture':'JumpReLU_monotone','l1':'1e-4','lr': 1e-5,'hidden_dim':10,'top_K':0},

    #  'JUMPRELUAE_10_1e2_mon':{'architecture':'JumpReLU_monotone','l1':'1e-2','lr': 1e-5,'hidden_dim':10,'top_K':0},
     #   'JUMPRELUAE_10_1e1_mon':{'architecture':'JumpReLU_monotone','l1':'1e-1','lr': 1e-5,'hidden_dim':10,'top_K':0},
     #           'JUMPRELUAE_10_1e0_mon':{'architecture':'JumpReLU_monotone','l1':'1','lr': 1e-5,'hidden_dim':10,'top_K':0},

     'JUMPRELUAE_10_5e5_mon':{'architecture':'JumpReLU_monotone','l1':'5e-5','lr': 1e-4,'hidden_dim':10,'top_K':0},

  #  'JUMPRELUAE_20_1e8_mon':{'architecture':'JumpReLU_monotone','l1':'1e-8','lr': 1e-5,'hidden_dim':20,'top_K':0},
  #   'JUMPRELUAE_40_1e8_mon':{'architecture':'JumpReLU_monotone','l1':'1e-8','lr': 1e-5,'hidden_dim':20,'top_K':0},

   # 'JUMPRELUAE_40_1e4_mon':{'architecture':'JumpReLU_monotone','l1':'1e-4','lr': 1e-5,'hidden_dim':40,'top_K':0},
 #       'JUMPRELUAE_40_1e3_mon':{'architecture':'JumpReLU_monotone','l1':'1e-3','lr': 1e-5,'hidden_dim':40,'top_K':0},
 #   'JUMPRELUAE_40_1e2_mon':{'architecture':'JumpReLU_monotone','l1':'1e-2','lr': 1e-5,'hidden_dim':40,'top_K':0},
 #   'JUMPRELUAE_40_1e1_mon':{'architecture':'JumpReLU_monotone','l1':'1e-1','lr': 1e-5,'hidden_dim':40,'top_K':0},
  #  'JUMPRELUAE_40_1e0_mon':{'architecture':'JumpReLU_monotone','l1':'1','lr': 1e-5,'hidden_dim':40,'top_K':0},

 #   'JUMPRELUAE_1024_1e4_mon':{'architecture':'JumpReLU_monotone','l1':'1e-4','lr': 1e-5,'hidden_dim':1024,'top_K':0},
 #   'JUMPRELUAE_2048_1e3_mon':{'architecture':'JumpReLU_monotone','l1':'1e-3','lr': 1e-5,'hidden_dim':2048,'top_K':0},
 #   'JUMPRELUAE_40_1e8_mon':{'architecture':'JumpReLU_monotone','l1':'1e-8','lr': 1e-5,'hidden_dim':40,'top_K':0}

 #  'JUMPRELUAE_1024':{'architecture':'JumpReLU','l1':'1e-3','hidden_dim':1024,'top_K':0},
  #  'JUMPRELUAE_2048':{'architecture':'JumpReLU','l1':'5e-2','hidden_dim':2048,'top_K':0},
 #   'RELUAE_1024_1e3_mon':{'architecture':'ReLUAE','l1':'1e-3', 'lr': 1e-5,'hidden_dim':1024,'top_K':0},

    #'RELUAE_1E-3':{'architecture':'ReLUAE','l1':'1e-3','hidden_dim':1024,'top_K':0},
    #'RELUAE_1E-2':{'architecture':'ReLUAE','l1':'5e-2','hidden_dim':1024,'top_K':0},


    #GATEDSAE does not perform well
  #  'GATEDSAE_1E-3':{'architecture':'GatedSAE','l1':'1e-3'},
    #'GATEDSAE_1E-2':{'architecture':'GatedSAE','l1':'5e-3'},
    }



'''
SAE_PARAMETERS = args.sae_params


#SUBROOTS=['eps_0.001','eps_0.01','eps_0.1','unreg']
SUBROOTS=['unreg']
#BASE_TENSOR='datasets/noised_luther/activations/base/EleutherAI__pythia-70m-deduped/base/luther_L2_residual.pt'



SAE_SCRIPT = str(REPO_ROOT / "SAE.py")
#OUT_DIR     = SOURCE_ROOT / "potential_training_data" 
#OUT_DIR     = SOURCE_ROOT / "map_training_data"

#OUT_FILE=OUT_DIR/f"{}_training_data.pt"
SEED = args.seed
EXT         = ".pt"
# --------------
#SAE PARAMETERS
#HIDDEN_DIM=40
BATCH_SIZE=1024
#INPUT_DIM=1130 #<----Base Measure Supp Size

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)  


pt_files = list(DATA_DIR.glob("*.pt"))

if len(pt_files) == 0:
    raise FileNotFoundError(f"No .pt files found in {DATA_DIR}")
if len(pt_files) > 1:
    raise RuntimeError(f"Multiple .pt files found in {DATA_DIR}: {pt_files}")

data = torch.load(pt_files[0])
if type(data)==dict:
    cube = data["cube"]
else:
    cube=data
a, b, c = cube.shape
INPUT_DIM = c  # <----Base Measure Supp Size
flat_cube = cube.reshape(a * b, c)

n_samples = flat_cube.shape[0]

#HACKY FIX, KEEPING IT TO 1 VAL POINT
n_val = 1
#int(0.0 * n_samples)

perm = torch.randperm(n_samples)
val_indices = perm[:n_val]
train_indices = perm[n_val:]

train_data = flat_cube[train_indices]
val_data = flat_cube[val_indices]

# Save train data
TEMP_TRAIN_DIR = DATA_DIR / 'stacked_data'
TEMP_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
torch.save(train_data, TEMP_TRAIN_DIR / 'stacked_data.pt')

# Save validation data
TEMP_VAL_DIR = DATA_DIR / 'val_data'
TEMP_VAL_DIR.mkdir(parents=True, exist_ok=True)
torch.save(val_data, TEMP_VAL_DIR / 'val_data.pt')

print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

for subroot in SUBROOTS:
    #PREPROCESSING 
    SAE_DIR = OUT_ROOT / subroot/f"seed_{SEED}"
    SAE_DIR.mkdir(parents=True, exist_ok=True)

    #TRAINING SAE
    for trial_name, params in SAE_PARAMETERS.items():
        l1 = params["l1"]
        architecture = params["architecture"]
        lr=params['lr']
        HIDDEN_DIM=params["hidden_dim"]
        trial_top_K=params['top_K']
        run_dir = SAE_DIR / trial_name
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", SAE_SCRIPT,
            "--data_path", str(TEMP_TRAIN_DIR),
            "--hidden_dim", str(HIDDEN_DIM),
            "--architecture",str(architecture),
            "--input_dim",str(INPUT_DIM),
            "--batch_size", str(BATCH_SIZE),
            "--epochs", str(EPOCHS),
            "--lr", str(lr),
            "--l1", str(l1),
            "--top_k", str(trial_top_K),
            "--log_every", "50",
            "--device", "cuda",
            "--log_dir", str(run_dir),
            "--grad_clip","100",
            "--no-normalize",
            '--validation_path', str(TEMP_VAL_DIR),
            '--val_frequency', '0.9',
            ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        #storing save path for trained model

        model_save_path = None
        for line in proc.stdout:
            print(line, end="")          # live progress
            if line.startswith("CKPT_PATH="):
                model_save_path = Path(line.split("=", 1)[1].strip())

        ret = proc.wait()
        if ret != 0:
            raise RuntimeError("SAE process failed")
        if model_save_path is None:
            raise RuntimeError("CKPT_PATH not found")
#shutil.rmtree(TEMP_TRAIN_DIR)
#shutil.rmtree(TEMP_VAL_DIR) 



'''
EXAMPLE:

python hsi/pipeline/hyperspectral_potential_SAE_script.py \
  --out_root datasets/hsi_data/Pavia/SAE_params/transport_maps \
  --data_dir datasets/hsi_data/Pavia/transport_maps \
  --epochs 1000 \
  --seed 0 \
  --sae_params '{"JUMPRELUAE_10_5e5_mon":{"architecture":"JumpReLU_monotone","l1":"5e-5","lr":1e-4,"hidden_dim":10,"top_K":0}}'



'''
