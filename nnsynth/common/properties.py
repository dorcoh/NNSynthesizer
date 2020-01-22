import operator
from typing import Dict

from nnsynth.common.models import InputImpliesOutputProperty, OutputConstraint, PropertyElement, Property


class DeltaRobustnessProperty:
    def __init__(self, input_size, output_size, desired_output, coordinate=(10, 10),
                 delta: float = 1e-3, output_constraint_type: OutputConstraint = OutputConstraint.Min):
        self.input_size = input_size
        self.output_size = output_size
        self.desired_output = desired_output
        self.coordinate = coordinate
        self.delta = delta
        self.output_constraint_type = output_constraint_type
        self.variables_dict = {}

        self.input_property_placeholder = []
        self.output_property_placeholder = []

    def set_variables_dict(self, variables_dict: Dict):
        self.variables_dict = variables_dict

    def generate_property(self):
        if not self.variables_dict:
            raise Exception("Cannot generate property without setting variables dict")
        self._set_input_proeprty()
        self._set_output_propety()

    def get_property_constraint(self):
        self.generate_property()
        return InputImpliesOutputProperty(
            Property(self.input_property_placeholder).get_property(),
            Property(self.output_property_placeholder).get_property()
        ).get_property_constraint()

    def _set_input_proeprty(self):
        # fill input consraints
        for input_idx in range(self.input_size):
            for id, op in enumerate([operator.ge, operator.le]):
                delta_sign = 1
                if op == operator.ge:
                    delta_sign = -1
                elif op == operator.le:
                    delta_sign = 1
                key = 'input_%d' % (input_idx + 1)
                prop_elem = PropertyElement(self.variables_dict[key], self.coordinate[input_idx] + delta_sign*self.delta, op)
                self.input_property_placeholder.append(prop_elem.return_property_element())

    def _set_output_propety(self):
        # fill output costraints
        op_sign = None
        if self.output_constraint_type == OutputConstraint.Min:
            op_sign = operator.le
            op_sign = operator.lt
        elif self.output_constraint_type == OutputConstraint.Max:
            op_sign = operator.ge
            op_sign = operator.gt
        output_key = 'out_%d' % self.desired_output
        range_ = [i for i in range(1, self.output_size+1) if i is not self.desired_output]
        for output_idx in range_:
            curr_key = 'out_%d' % output_idx
            prop_elem = PropertyElement(self.variables_dict[output_key],
                                        self.variables_dict[curr_key], op_sign)
            self.output_property_placeholder.append(prop_elem.return_property_element())