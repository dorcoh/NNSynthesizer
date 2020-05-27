"""Iterates over sub experiments dir (with serialized instances of main loop),
for each instance invokes a PBS job with main_loop_instances_solver.py """
import os
from pathlib import Path
from subprocess import call

sub_exp_dir = Path('sub-exp')
pwd = os.getcwd()
sub_exps_list = os.listdir(str(sub_exp_dir.absolute()))

pbs_runner_script = "{}/pbs-job-instance-main-loop.sh".format(pwd)

queue = "all_l_p"

for sub_exp in sub_exps_list:
    print("Calling runner for exp: " + str(sub_exp))
    # os.environ['CURR_SUB_EXP_PATH'] = "sub_exp"
    call_args = ["qsub", "-q", queue, "-v", "formula={}".format(sub_exp), pbs_runner_script]
    print("Call args: {}".format(str(call_args)))
    call(call_args)
