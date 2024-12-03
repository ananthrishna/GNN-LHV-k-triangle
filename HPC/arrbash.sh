#!/bin/bash
#SBATCH --partition=long
#SBATCH --job-name=my_array_job
#SBATCH --array=1-25
#SBATCH --nodes=1
#SBATCH --ntasks=1  


module load gcc/9.2.0
module load python/3.7.6
module load openmpi/4.0.1
export CC=/opt/apps/gcc/9.2/bin/gcc
export CXX=/opt/apps/gcc/9.2/bin/c++
#pyvenv py_dev
#source py_dev/bin/activate

#pip install pip --upgrade
#pip -vv install numpy 

PYTHON_SCRIPT="/home/shajigroup2/Anantha/Factory/WorkshopIV/Code_7f/Code/train.py"
TASK_ID=$SLURM_ARRAY_TASK_ID
  
python $PYTHON_SCRIPT



