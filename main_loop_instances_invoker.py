"""Iterates over sub experiments dir (with serialized instances of main loop),
for each instance invokes a PBS job with main_loop_instances_solver.py """
import os
from pathlib import Path
from subprocess import call

from nnsynth.common.arguments_handler import ArgumentsParser

args = ArgumentsParser.parser.parse_args()

sub_exp_dir = Path('sub-exp/{}'.format(args.experiment))
pwd = os.getcwd()
sub_exps_list = os.listdir(str(sub_exp_dir.absolute()))

if not args.invoke_z3_binary:
    pbs_runner_script = "{}/pbs-job-instance-main-loop.sh".format(pwd)
else:
    pbs_runner_script = "{}/z3-pbs-job-runner.sh".format(pwd)

queue = "all_l_p"

SUFFIX = 'pkl'
# set by suffix
logs_dir = {'smt2': 'z3_logs', 'pkl': 'logs'}

# "cut" long experiments
if args.sub_exp_count:
    if len(sub_exps_list) < args.sub_exp_count:
        sub_exps_list = sub_exps_list[:args.sub_exp_count]

for sub_exp in sub_exps_list:
    # for Z3
    if args.invoke_z3_binary and not sub_exp.endswith(SUFFIX):
        continue

    # for Z3 wrapper
    if not args.invoke_z3_binary and not sub_exp.endswith(SUFFIX):
        continue

    if args.dev:
        # one formula
        if '_param_w_1_1_2' not in sub_exp:
            continue

    log_dir = sub_exp_dir / logs_dir[SUFFIX] / sub_exp.replace(SUFFIX, 'runner.log')
    err_dir = sub_exp_dir / logs_dir[SUFFIX] / sub_exp.replace(SUFFIX, 'runner.err')
    print("Calling runner for exp: " + str(sub_exp))
    call_args = ["qsub", "-q", queue, "-o", str(log_dir.absolute()), "-e", str(err_dir.absolute()),
                 "-v", "formula={},exp={}".format(sub_exp, args.experiment),
                 pbs_runner_script]
    print("Call args: {}".format(str(call_args)))
    call(call_args)
