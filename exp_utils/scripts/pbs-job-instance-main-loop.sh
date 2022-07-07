#!/bin/sh
#PBS -N dorz3py

# Send the e-mail messages from the server to a user address
# This line and the Technion address are mandatory!
#--------------------------------------------------------
#PBS -M dor.cohen@campus.technion.ac.il

#PBS -mbea
#
# running 1 process on the available CPU of the available node
#------------------------------------------------------------------------
#PBS -l select=1:ncpus=1
#PBS -l walltime=24:00:00
#PBS -l place=excl

PBS_O_WORKDIR=$HOME/nnsynth
cd $PBS_O_WORKDIR

#formula_name=$CURR_SUB_EXP_PATH
#Run command
#-----------------------
source /usr/local/epd/setup.sh
echo "mkdir sub-exp/$exp/logs"
mkdir -p sub-exp/$exp/logs
echo "python3 -u main_loop_instances_solver.py --experiment $exp --sub_exp_filename $formula > sub-exp/$exp/logs/$formula.log"
python3 -u main_loop_instances_solver.py --experiment $exp --sub_exp_filename $formula > sub-exp/$exp/logs/$formula.log
