#!/bin/bash
#SBATCH --job-name=BERTweet
#SBATCH --output=overnight_%j.log
#SBATCH --error=overnight_%j.err
#SBATCH --partition=ug-gpu-small
#SBATCH --gres=gpu:turing:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G

# 1. Activate Environment
source /home2/nchw73/venv312/bin/activate

# 2. Go to your folder
cd /home2/nchw73/Year4/NLP

# 3. Debug: Verify we got the GPU
echo "Job running on node: $(hostname)"
echo "------------------------------------------------------"
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
echo "------------------------------------------------------"

# 4. Run the experiments
python3 ncc_prompt.py  # run prompting experiments & save classification reports