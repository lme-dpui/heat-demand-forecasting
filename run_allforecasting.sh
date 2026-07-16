#!/bin/bash
#SBATCH --job-name=roll_time
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#Timelimit format: "hours:minutes:seconds" -- max is 24h
#SBATCH --time=24:00:00
#SBATCH -o /cluster/ramachandran/downstream_ai_tasks/forecasting/wavelet_scalogram/heat/losscurves.out
#SBATCH -e /cluster/ramachandran/downstream_ai_tasks/forecasting/wavelet_scalogram/heat/losscurves.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12000
#SBATCH --nodelist=lme221


export PATH=/cluster/ramachandran/anaconda/envs/pyt/bin:$PATH
source ~/.bashrc

eval "$(conda shell.bash hook)"
conda activate pyt

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:cluster/ramachandran/anaconda/envs/pyt/lib

conda env list

echo $CONDA_DEFAULT_ENV
echo $CONDA_PREFIX

echo "Your job is running on" $(hostname)

nvidia-smi

# Reproduces the paper's proposed model (Table 11 "Proposed" row) for all
# three Bronderslev zones. Architecture (conv/dense widths, dropout) and
# training hyperparameters (batch size, epochs, patience) are the repo
# defaults in conf/config.yaml and conf/setup/wavelet/vgg.yaml, which already
# match the paper's Table 2 / Table A.12 configuration; only the per-zone
# mother wavelet, holiday-handling strategy, and decomposed feature set
# (Table A.12, Table 9/10) are overridden below.
python main.py info.exp_no=dma_a paths.dma=dma_a features=dma_a variables.use_previous_holiday=True
python main.py info.exp_no=dma_b paths.dma=dma_b features=dma_b variables.use_previous_holiday=True setup.encoding_schema.wavelet_function=gaus8
python main.py info.exp_no=dma_c paths.dma=dma_c features=dma_c variables.use_previous_holiday=False
