import os

from matplotlib.pyplot import step
from SAE_analysis_functions import *
import subprocess
from pathlib import Path
import random
import torch
from torch.utils.tensorboard import SummaryWriter

# ---- config ----


REPO_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = REPO_ROOT / "datasets" / "noised_luther" / "potentials"          # change if needed (e.g., Path("/path/to/your/root"))
MODELS=['EleutherAI__pythia-70m-deduped','EleutherAI__pythia-160m-deduped','EleutherAI__pythia-410m-deduped']
SUBROOTS=['unreg']
SAE_PARAMETERS={
    'TOPKAE_1E-3':{'architecture':'TopKAE','l1':'0'},
  #  'GATEDSAE_1E-3':{'architecture':'GatedSAE','l1':'1e-3'},
  #  'JUMPRELUAE_1E-3':{'architecture':'JumpReLU','l1':'1e-3'},
  #  'RELUAE_1E-3':{'architecture':'ReLUAE','l1':'1e-3'},
    'RELUAE_1E-2':{'architecture':'ReLUAE','l1':'5e-3'},
    'JUMPRELUAE_1E-2':{'architecture':'JumpReLU','l1':'5e-3'},
    'GATEDSAE_1E-2':{'architecture':'GatedSAE','l1':'5e-3'},
    }
#l1_ranges=["1e-2","1e-3"]
#ARCHITECTURES=['ReLUAE','JumpReLUAE','GatedSAE','TopKAE']

SAE_SCRIPT = "SAE.py"
#OUT_DIR     = SOURCE_ROOT / "potential_training_data"
#OUT_DIR     = SOURCE_ROOT / "map_training_data"
N_PER_DIR   = 950
#OUT_FILE=OUT_DIR/f"{N_PER_DIR}_training_data.pt"
SEED        = 0
EXT         = ".pt"
# --------------
#SAE PARAMETERS
HIDDEN_DIM=40
BATCH_SIZE=1024
# BASE MEASURE
SUPP_SIZE=1130

random.seed(SEED)

for model in MODELS:
    for subroot in SUBROOTS:

        #PREPROCESSING 
        OUT_DIR = SOURCE_ROOT / model / subroot
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        DIR_NAMES = sorted([p.name for p in OUT_DIR.iterdir() if p.is_dir()])

        all_tensors = []
        labels = []

        name_to_id = {name: i for i, name in enumerate(DIR_NAMES)}
        id_to_name = {i: n for n, i in name_to_id.items()}

        for name in DIR_NAMES:
            d = OUT_DIR / name
            pt_files = sorted(d.rglob(f"*{EXT}"))
            if len(pt_files) < N_PER_DIR:
                raise ValueError(f"{name}: only {len(pt_files)} files")
            for p in pt_files[:N_PER_DIR]:
                print(p)
                all_tensors.append(torch.load(p)["object"])  # already a tensor in most cases
                labels.append(name_to_id[name])

        #tensor of kantorovich potentials
        stacked_tensor = torch.stack([t if torch.is_tensor(t) else torch.tensor(t) for t in all_tensors], dim=0)
        #shifting by squared norm of base support (brenier potential)
       # base_tensor=torch.load(BASE_TENSOR)["acts"]
       # norm_shift= 0.5 * (base_tensor**2).sum(dim=1)   # shape (1130,)

       # stacked_tensor=stacked_tensor-norm_shift

        int_labels = torch.tensor(labels, dtype=torch.long)
        labels = [id_to_name[i] for i in labels]  # list[str], length N


        data_path = OUT_DIR / "stacked_potential_tensor.pt"

        torch.save(stacked_tensor, data_path)

        OUT_DIR_RESULTS = SOURCE_ROOT / "normalized_results"

        #TRAINING SAE
        for trial_name, params in SAE_PARAMETERS.items():
            l1 = params["l1"]
            architecture = params["architecture"]
            run_dir = OUT_DIR_RESULTS / model / subroot / trial_name
            run_dir.mkdir(parents=True, exist_ok=True)

            writer = SummaryWriter(log_dir=str(run_dir))

            cmd = [
                "python", SAE_SCRIPT,
                "--data_dir", str(data_path),
                "--hidden_dim", str(HIDDEN_DIM),
                "--architecture",str(architecture),
                "--batch_size", str(BATCH_SIZE),
                "--epochs", "2000",
                "--lr", "5e-7",
                "--l1", str(l1),
                "--log_every", "50",
                "--device", "cuda",
                "--log_dir", str(run_dir),
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
            #GENERATING CODES
            '''
            fista_code_tensor=fista_batch(stacked_tensor,
                        model_save_path,
                        supp_size=SUPP_SIZE,
                        no_atoms=HIDDEN_DIM,
                        l1=0.01,
                        max_iter=200,
                        tol=1e-6)
            '''

            sae_code_tensor = sae_encode_batch(stacked_tensor, 
                                                model_save_path,
                                                architecture=architecture,
                                                m=SUPP_SIZE,hidden_dim=HIDDEN_DIM,
                                                top_k=4,
                                                device="cuda", batch_size=BATCH_SIZE)
            

            #EVALUATION: PLOTTING
            l2_hist=plot_blocked_l2_histograms(stacked_tensor, labels, bins=60)
            writer.add_figure("plots/l2_histogram", l2_hist, global_step=0)
            plt.close(l2_hist)

            coeff_matrix=plot_coeff_matrix(sae_code_tensor,labels)
            writer.add_figure("plots/coeff_matrix", coeff_matrix, global_step=0)
            plt.close(coeff_matrix)

            writer.flush()
            writer.close()



