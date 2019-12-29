from enum import Enum
from typing import List

from z3 import Implies, And


class PropertyElement:
    """Represents basic property"""
    def __init__(self, key, value, op):
        self.key = key
        self.value = value
        self.op = op

    def return_property_element(self):
        return self.op(self.key, self.value)


class Property:
    """Collection of property elements"""
    def __init__(self, properties_elements: List[PropertyElement]):
        self.property = properties_elements

    def get_property(self):
        return self.property


class InputImpliesOutputProperty:
    """Builds input -> output property"""
    def __init__(self, input_prop: Property, output_prop: Property):
        self.input_prop = input_prop
        self.output_prop = output_prop

    def get_property_constraint(self):
        prop_constraint = Implies(
            And(self.input_prop),
            And(self.output_prop)
        )

        return prop_constraint


class OutputConstraint(Enum):
    """Behavior of output constraints, Max corresponds to:
     chosen_output > output_i AND chosen_output > output_i+1 .."""
    Max = 1,
    Min = 2,


