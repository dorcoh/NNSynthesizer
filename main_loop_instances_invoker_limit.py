"""Iterates over soft constraints limit sizes and invokes PBS script for each"""
import os
import sys
from pathlib import Path
from subprocess import call

from nnsynth.common.arguments_handler import ArgumentsParser

args = ArgumentsParser.parser.parse_args()

sub_exp_dir = Path('sub-exp/{}'.format(args.experiment))
pwd = os.getcwd()
sub_exps_list = os.listdir(str(sub_exp_dir.absolute()))

if not args.invoke_z3_binary:
    pbs_runner_script = "{}/pbs-job-instance-main-loop-limit.sh".format(pwd)
else:
    pbs_runner_script = "{}/z3-pbs-job-runner.sh".format(pwd)

queue = "all_l_p"

SUFFIX = 'pkl'
# set by suffix
logs_dir = {'smt2': 'z3_logs', 'pkl': 'logs'}

# TODO: change
sub_exp = [i for i in sub_exps_list if '_param_w_1_1_1' in i][0]
print("Sub experiment:")
print(sub_exp)

for limit in list(reversed([i for i in range(1, 50+1)])):
    log_dir = sub_exp_dir / logs_dir[SUFFIX] / sub_exp.replace(SUFFIX, 'runner.log')
    err_dir = sub_exp_dir / logs_dir[SUFFIX] / sub_exp.replace(SUFFIX, 'runner.err')
    print("Calling runner for exp: " + str(sub_exp))
    call_args = ["qsub", "-q", queue, "-o", str(log_dir.absolute()), "-e", str(err_dir.absolute()),
                 "-v", "formula={},exp={},threshold={}".format(sub_exp, args.experiment, limit),
                 pbs_runner_script]
    print("Call args: {}".format(str(call_args)))
    if args.dev:
        print("Dev mode: skipping call")
    else:
        call(call_args)
