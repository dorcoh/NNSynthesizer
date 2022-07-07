#!/bin/sh

# run z3 individually, gets as parameters experiment name (var 'exp') and sub-exp (var 'formula')

#PBS -N dorz3

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

PBS_O_WORKDIR=$HOME/nnsynth
cd $PBS_O_WORKDIR

#Run command
#-----------------------
mkdir -p sub-exp/$exp/z3_logs/
echo "/usr/local/z3_V4.8.7/bin/z3 -st -smt2 $formula > sub-exp/$exp/z3_logs/$formula.log"
/usr/local/z3_V4.8.7/bin/z3 -st -smt2 $formula > sub-exp/$exp/z3_logs/$formula.log
