from z3 import *

x = Real('x')
y = Real('y')
s = Solver()
s.add(x + y > 5, x > 1, y > 1)
# print(s.check())
# print(s.model())
with open('dummy.smt2', 'w') as handle:
    handle.write(s.sexpr())
    handle.writelines(["(check-sat)\n", "(get-model)\n"])
