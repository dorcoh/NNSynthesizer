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
#PBS -o log.out
#PBS -e log.err

PBS_O_WORKDIR=$HOME/nnsynth
cd $PBS_O_WORKDIR

#Run command
#-----------------------
source /usr/local/epd/setup.sh
python3 -u dummy.py
