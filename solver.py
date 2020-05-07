# TODO: the PBS job runs the code below
import pickle

from z3 import unsat, unknown
from z3 import parse_smt2_file, parse_smt2_string

from nnsynth.z3_context_manager import Z3ContextManager

z3_mgr = Z3ContextManager()
# TODO: add formula name/path as paramter for this script
z3_mgr.add_formula_from_disk('check.smt2')

z3_mgr.solve()

res = z3_mgr.get_result()

# exit if not sat
if (res == unsat or res == unknown):
    print("Stopped with result: " + str(res))
    exit(1)

# TODO: mark the current instance as sat/unsat/unknown on some csv (append results for each formula)
#  should list the configurations as well

model = z3_mgr.solver.model()
# model_mapping = z3_mgr.get_model_mapping(generator.get_z3_weight_variables(),
#                                          generator.get_original_weight_values())

# z3_mgr.model_mapping_sanity_check()


# TODO - assign the file name with the formula name (add some key as identifier)
with open('model.smt2', 'w') as handle:
    handle.write(model.sexpr())

# TODO: find a way to save the model mapping to disk, with the ability of loading it back into z3 (why?)
#  probably the easiest solution is to parse model from the logs files (or to somehow parse it inside script and save it
#  to disk during execution, then later collect it to set as actual parameters for some network if we'd like)
#  other way is to serialize the init variables for FormulaGenerator, including the property, weights_selector and
#  keep_context_property instances, this way the actual z3 variables could be directly evaluated from the model
#  (as done in get_model_mapping). To achieve this in easier way, we can only serialize get_z3_weight_variables and
#  get_original_weight_values, however get_z3_weight_variables cannot be serialized, then a workaround is to create
#  a class method which only generates this z3 variables
with open('model.smt2', 'r') as handle:
    content = handle.read()
print(model)
# it doesn't work to load it with parse_smt_string