#!/bin/bash -l
#SBATCH -p amdsmall
#SBATCH -t 0-24:00:00
#SBATCH -N 1
#SBATCH --ntasks=32
#SBATCH --mem=60G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lu000285@umn.edu
#SBATCH -A huws

module load texlive
module load python3/3.8.3_anaconda2020.07_mamba
source activate ChEn_5801_project #activate my conda environment myenv_py371 or myconda_env ChEn_5801_project
export MKL_DEBUG_CPU_TYPE=5
python synthetic_data.py -nexp 100000 -p 32 -ms 1