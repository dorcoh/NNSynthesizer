#!/bin/sh

formula_name=$1

if [[ -n "$formula_name" ]]; then
    echo "Starting pbs job runner with formula: $formula_name"
else
    echo "Argument error, formula_name: $formula_name"
    exit 1
fi

#PBS -N redcsl

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
#PBS -o log.out
#PBS -e log.err

PBS_O_WORKDIR=$HOME/
cd $PBS_O_WORKDIR

#Run command
#-----------------------
source /usr/local/epd/setup.csh
python3 -u solver.py $formula_name > $formula_name-log.out
