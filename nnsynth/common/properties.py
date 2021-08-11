import logging
import operator
import random
import sys
from abc import ABC, abstractmethod
from collections import Counter
from enum import Enum
from typing import Dict, Tuple, List, Union

from scipy.spatial.qhull import Voronoi
from sympy import Polygon, Point, Symbol
from z3 import Implies, And, AtLeast, AtMost, Bool, BoolRef, PbGe, Const, Real, RealSort, Function, RealVal, ForAll
import numpy as np

from nnsynth.common.formats import Formats
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


class ConstraintType(Enum):
    HARD = 1
    SOFT = 2


class KeepContextType(Enum):
    SAMPLES = 1
    GRID = 2
    VORONOI = 3
    SCORE = 4


class KeepContextProperty(ABC):
    """An abstract class to hold the keeping context property which is constructed from hard or soft constraints"""
    def __init__(self):
        self.variables = None
        self.keep_context_constraints = []
        self.main_constraint = None
        self.kwargs = {}
        self.constraints_type = None
        self.keep_context_type = None

    def get_property_constraint(self):
        """Main function, returns the constraints"""
        self._validate()
        self._generate_property()
        self._validate_constraints()
        self._generate_main_constraint()
        return self.main_constraint

    @abstractmethod
    def _generate_property(self):
        """Generates the actual property by setting self.keep_context_constraint"""
        pass

    def _validate(self):
        """Validates required attributes were set for generating property"""
        if not (self.variables and (isinstance(self.variables, dict) or (isinstance(self.variables, list)))):
            raise Exception("Cannot generate property without setting variables dict")

    def _validate_constraints(self):
        """Validates soft constraints were set, for generating the main constraint"""
        if not self.keep_context_constraints:
            raise Exception("Cannot generate main constraint as self.keep_context_constraints must be set.")

    @abstractmethod
    def _generate_main_constraint(self):
        """Generates the main constraint which is being collected at a formula generator instance"""
        pass

    def set_variables(self, variables: Union[Dict, List[Dict]]):
        """In case of hard constraints it sets a dictionary with declared variables,
        while in soft constraints a list of declared variables, one for each soft constraint."""
        self.variables = variables

    def set_kwargs(self, **kwargs):
        self.kwargs = kwargs

    def get_constraints_type(self):
        return self.constraints_type

    def get_keep_context_type(self):
        return self.keep_context_type


class SoftConstraintsProperty(KeepContextProperty, ABC):
    """An abstract class to hold soft properties, should set threshold."""
    def __init__(self):
        super().__init__()
        self.threshold = None
        self.constraints_type = ConstraintType.SOFT

    def _validate(self):
        super(SoftConstraintsProperty, self)._validate()
        if not self.kwargs['threshold']:
            raise Exception("Cannot generate main constraint: threshold must be set")

    def _generate_main_constraint(self):
        self.threshold = self.kwargs['threshold']
        self.main_constraint = AtLeast(*self.keep_context_constraints, self.threshold)

    def set_threshold(self, threshold: int):
        # TODO: deprecate (as using kwargs)
        print("Setting threshold: {}".format(threshold))
        self.threshold = threshold

    def iter_soft_constraints_inputs(self):
        if self.keep_context_type in (KeepContextType.SAMPLES, KeepContextType.SCORE):
            return iter(self.kwargs['eval_set'][0])


class HardConstraintsProperty(KeepContextProperty, ABC):
    """An abstract class to hold hard properties"""
    def __init__(self):
        super().__init__()
        self.constraints_type = ConstraintType.HARD

    def _generate_main_constraint(self):
        self.main_constraint = self.keep_context_constraints


class EnforceSamplesProperty(KeepContextProperty, ABC):
    """Abstract class to enforce samples property - gets an evaluation set, and constraints of the form:
        e.g., X1=x1, X2=x2 -> y_1 > y_2 """
    def __init__(self, constraints_limit=None):
        super().__init__()
        self.keep_context_type = KeepContextType.SAMPLES
        self.constraints_limit = constraints_limit

    def _generate_property(self):
        self._generate()

    def _validate(self):
        super()._validate()
        if not self.kwargs['eval_set']:
            raise Exception("Cannot generate main constraint: evaluation set is required.")

    def _generate(self):
        """Generate the actual constraints by appending them into self.keep_context_constraints.
        In this case"""
        # add sample constraints
        self.test_set = self.kwargs['eval_set']
        self.X_test, self.y_test = self.test_set
        # use all test set (below) or inject the limit from solver (constraints_limit)
        if self.constraints_limit is None:
            self.constraints_limit = self.X_test.shape[0]

        # TODO: remove "constraints_limit"?
        i = 0
        for x, y in list(zip(self.X_test, self.y_test)):

            if self.constraints_type == ConstraintType.SOFT:
                variables = self.variables[i]
            else:
                variables = self.variables

            # note currently supports only binary classification
            if y == 0.0:
                out_constraint = variables['out_1'] > variables['out_2']
            else:
                out_constraint = variables['out_2'] > variables['out_1']

            input_constraint = And(variables['input_1'] == x[0], variables['input_2'] == x[1])
            if self.constraints_type == ConstraintType.HARD:
                curr_constraint = Implies(
                    input_constraint,
                    out_constraint)
            else:
                curr_constraint = out_constraint

            self.keep_context_constraints.append(curr_constraint)
            i += 1


class EnforceSamplesSoftProperty(EnforceSamplesProperty, SoftConstraintsProperty):
    """Enforces samples soft property"""
    def _validate_threshold(self):
        # workaround for setting the threshold - assuming the maximum length is at the size of test_set
        if self.kwargs['threshold'] > self.kwargs['eval_set'][0].shape[0]:
            raise Exception("Cannot set threshold which is bigger than test set length")

    def _validate(self):
        super()._validate()
        self._validate_threshold()


class EnforceSamplesHardProperty(EnforceSamplesProperty, HardConstraintsProperty):
    """Enforces samples hard property"""
    ...

# TODO: add abstract grid property, add hard grid property


class EnforceScoresProperty(KeepContextProperty, ABC):
    """Abstract class to enforce scores property"""
    def __init__(self):
        super().__init__()
        self.keep_context_type = KeepContextType.SCORE

    def _generate_property(self):
        self._generate()

    def _validate(self):
        super()._validate()
        if not self.kwargs['eval_set']:
            raise Exception("Cannot generate main constraint: evaluation set is required.")
        if not len(self.kwargs['eval_set']) == 3:
            raise Exception("Cannot generate main constraint: "
                            "Expecting 'predicted_score' type of evaluation set, a tuple of len 3")
        if not 'epsilon' in self.kwargs:
            raise Exception("Cannot generate main constraint: epsilon value is required.")

    def _generate(self):
        """Generate the actual constraints by appending them into self.keep_context_constraints.
        In this case"""
        # add sample constraints
        self.test_set = self.kwargs['eval_set']
        self.epsilon = self.kwargs['epsilon']
        self.X_test, self.y_test, self.y_score = self.test_set
        # use all test set (below) or inject the limit from solver (constraints_limit)

        i = 0
        for x, y, y_score in list(zip(self.X_test, self.y_test, self.y_score)):

            if self.constraints_type == ConstraintType.SOFT:
                variables = self.variables[i]
            else:
                variables = self.variables

            # note currently supports only binary classification
            variable_key = 'out_1' if y == 0.0 else 'out_2'
            out_constraint = And([RealVal(y_score[y] - self.epsilon) <= variables[variable_key],
                                  variables[variable_key] <= RealVal(y_score[y] + self.epsilon)])

            input_constraint = And(variables['input_1'] == x[0], variables['input_2'] == x[1])
            if self.constraints_type == ConstraintType.HARD:
                curr_constraint = Implies(
                    input_constraint,
                    out_constraint)
            else:
                curr_constraint = out_constraint

            self.keep_context_constraints.append(curr_constraint)
            i += 1


class EnforceScoresSoftProperty(EnforceScoresProperty, SoftConstraintsProperty):
    """Enforces scores soft property"""
    def _validate_threshold(self):
        # workaround for setting the threshold - assuming the maximum length is at the size of test_set
        if self.kwargs['threshold'] > self.kwargs['eval_set'][0].shape[0]:
            raise Exception("Cannot set threshold which is bigger than test set length")

    def _validate(self):
        super()._validate()
        self._validate_threshold()


class EnforceScoresHardProperty(EnforceScoresProperty, HardConstraintsProperty):
    """Enforces samples hard property"""
    ...


class EnforceGridSoftProperty(SoftConstraintsProperty):
    """Generates properties list of kind (x1 > a && x1 < a+th && x2 > b && x2 < b+th) -> y_i > y_j ,
    where (i,j) are decided by the network - for each such 'cell' we sample multiple points and take
    the majority. (an alternative: to decide by the total score each class recieved)

    - thought: this is actually multiple DeltaRobustnessProperty instances
    - thought: perhaps we can make the cells even smaller (and not adjacent) -> to allow smoothness"""

    def __init__(self, net, x_range: Tuple[float, float], y_range: Tuple[float, float], grid_delta: float,
                 samples_num: int, random_seed: int = 42):
        """

        :param net: network model
        :param x_range: minimum and maximum values (e.g., (-10.5, 10.5))
        :param y_range: min/max values for y
        :param grid_delta: the step size for dividing the range
        :param samples_num: number of points to sample for each cell
        *** example: p = EnforceGridSoftProperty(net, x_range=(1, 22), y_range=(-6, 1), grid_delta=5, samples_num=3)
        """
        super().__init__()
        self.net = net
        self.x_range = x_range
        self.y_range = y_range

        self.grid_delta = grid_delta
        # sampled points for each polygon
        self.samples_num = samples_num
        # to hold the actual patches for further investigation
        self.patches = []
        self.patches_labels = []
        # set random seed for reproducibility
        random.seed(random_seed)
        self.keep_context_type = KeepContextType.GRID

    def get_patches(self) -> Tuple[List, List]:
        return self.patches, self.patches_labels

    def _generate_property(self):
        # grid order: starting at top left corner and scanning the rows until reaching bottom right corner.
        xx = np.arange(self.x_range[0], self.x_range[1], self.grid_delta)
        yy = np.flip(np.arange(self.y_range[0], self.y_range[1], self.grid_delta))
        limit_cells = 4
        for i in range(len(yy)-1):
            for j in range(len(xx)-1):
                points_indices = [(j, i), (j+1, i), (j+1, i+1), (j, i+1)]
                points = [(xx[p[0]], yy[p[1]]) for p in points_indices]

                poly = Polygon(*points)
                X = self.get_multiple_random_points_in_poly(poly, self.samples_num)
                # print(X.shape)
                y_pred = self.net.predict(X)
                # take the majority to decide the cell classification,
                y_pred_majority = Counter(y_pred).most_common(1)[0][0]
                if y_pred_majority == 0.0:
                    out_constraint = self.variables['out_1'] > self.variables['out_2']
                else:
                    out_constraint = self.variables['out_2'] > self.variables['out_1']

                curr_constraint = ForAll(
                    [self.variables['input_1'], self.variables['input_2']],
                    Implies(
                        And(self.variables['input_1'] > points[0][0], self.variables['input_1'] <= points[1][0],
                            self.variables['input_2'] < points[0][1], self.variables['input_2'] >= points[3][1]),
                        out_constraint
                    )
                )

                # store the outcome
                if len(self.keep_context_constraints) < limit_cells:
                    # TODO: remove the custom cells
                    if -6 <= points[0][1] <= -1 and 1 <= points[0][0] <= 21: # y and x
                        self.keep_context_constraints.append(curr_constraint)
                        self.patches.append(points)
                        self.patches_labels.append(y_pred_majority)
                else:
                    break
            else:
                continue
            break

        print("Total cells: {}".format(len(self.keep_context_constraints)))
        print(self.patches)

    def get_multiple_random_points_in_poly(self, poly: Polygon, n: int):
        FLOAT_PRECISION = 7
        points = []
        for i in range(n):
            point = self.get_random_point_in_polygon(poly)
            cords = tuple([p.evalf(FLOAT_PRECISION) for p in point.coordinates])
            points.append(cords)

        _points = np.array(points).astype(np.float32)
        return _points

    def get_random_point_in_polygon(self, poly: Polygon):
        minx, miny, maxx, maxy = poly.bounds
        while True:
            p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if poly.encloses_point(p):
                return p

    def iter_soft_constraints_inputs(self):
        # TODO: deprecate, not required (at least in case of soft constraints)
        return iter(['x' for x in range(4)])


class EnforceVoronoiSoftProperty(SoftConstraintsProperty):
    def __init__(self):
        super().__init__()
        self.keep_context_type = KeepContextType.VORONOI
        self.vor = None
        self.polygons = []
        self.labels = []

    def _validate(self):
        super()._validate()
        if not self.kwargs['eval_set']:
            raise Exception("Cannot generate main constraint: evaluation set is required.")

    def get_patches(self):
        """Gets polygons and their labels"""
        return self.polygons, self.labels

    def _generate_property(self):
        self.test_set = self.kwargs['eval_set']
        self.X_test, self.y_test = self.test_set

        self.x_min_max, self.y_min_max = None, None

        self._generate_voronoi_cells()

        i = 0
        for x, y in list(zip(self.X_test, self.y_test)):
            # each point has its own Voronoi region

            variables = self.variables

            # note currently supports only binary classification
            if y == 0.0:
                out_constraint = variables['out_1'] > variables['out_2']
            else:
                out_constraint = variables['out_2'] > variables['out_1']

            # generate input constraints for each region. An example: And([x1 - 1.5 > 0 , x2 - 1.5 < 0 , etc.])
            # thought - what about the boundaries?
            inputs_constraints = self._generate_single_cell_input_constraints(i, variables)
            if inputs_constraints:
                curr_constraint = ForAll(
                    [variables['input_1'], variables['input_2']],
                    Implies(
                        And(inputs_constraints),
                        out_constraint
                    )
                )

                self.keep_context_constraints.append(curr_constraint)
            i += 1

    def _generate_voronoi_cells(self):
        # generate the actual voronoi regions (finite and infinite ones)
        # should return a mapping of each sample with its region
        self.vor = Voronoi(self.X_test)

        # will hold a tuples of the form (point_index: int, polygon: List)
        polygons = []
        labels = []
        # iterate over Voronoi regions
        for i, region in enumerate(self.vor.regions):
            reg_points = []
            # skip infinite regions
            if -1 in region:
                continue
            # collect vertices that construct current region
            for vert_idx in region:
                v = self.vor.vertices[vert_idx]
                reg_points.append(v)
            # extract the point index which the current region wraps
            if reg_points:
                curr_point_index = list(self.vor.point_region).index(i)
                polygons.append((curr_point_index, reg_points))
                self.polygons.append(reg_points)
                self.labels.append(self.y_test[curr_point_index])

        # keep the cells equations in the form List[Tuple[int, List]]
        self.cells_equations = []

        for sample_index, poly in polygons:
            print("Current sample", self.X_test[sample_index])
            equations = []

            for i in range(len(poly)):
                point_a = poly[i]
                if i < len(poly) - 1:
                    point_b = poly[i + 1]
                else:
                    point_b = poly[0]

                a, b, c = self._compute_eq_two_points(point_a, point_b)
                equations.append((i, a, b, c))

            self.cells_equations.append((sample_index, equations))

        self.voroni_cells_map = dict(self.cells_equations)

    def _calc_min_max(self, point):
        if point[0] < self.x_min_max[0]:
            self.x_min_max[0] = point[0]
            # TODO

    def _transform_equation_maps(self, equations_map, variables) -> List:
        """Turns the equations map into a list of z3 boolean constraints"""
        bool_constraints = []
        x, y = variables['input_1'], variables['input_2']
        for i, eq in equations_map:
            a, b, c = eq[0]
            sign = eq[1]

            # transform into z3 constraints
            if sign == 1:
                curr_constraint = a*x + b*y + c > 0
            elif sign == -1:
                curr_constraint = a*x + b*y + c < 0
            else:
                curr_constraint = a*x + b*y + c == 0

            bool_constraints.append(curr_constraint)

        return bool_constraints

    def _generate_single_cell_input_constraints(self, sample_index, variables) -> Union[list, None]:
        # should get an input, fetch its Voronoi cell, and return a list of its Z3 boolean constraints
        if sample_index in self.voroni_cells_map:
            equations = self.voroni_cells_map[sample_index]
            sample = self.X_test[sample_index]
            equations_map = self.calc_point_equations(equations, sample)
            return self._transform_equation_maps(equations_map, variables)
        else:
            return None

    def _compute_eq_two_points(self, point_a, point_b):
        """Compute a linear equation out of two points in 2d"""
        x1, y1 = point_a
        x2, y2 = point_b
        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        print("{} * x + {} * y + {} = 0".format(a, b, c))
        return a, b, c

    def calc_point_equations(self, equations, point) -> List[Tuple]:
        """Given equations and point, check the result of this point when evaluated on each of the equations, and return
        a list with tuples of the form (equation_index, sign) of each result: positive (+1), negative (-1), or zero (0)"""
        x, y = point
        results = []

        def transform_result(res):
            if res > 0:
                return 1
            elif res < 0:
                return -1
            return 0

        for eq in equations:
            i, a, b, c = eq
            res = a*x + b*y + c
            eq_params = (a, b, c)
            results.append((i, (eq_params, transform_result(res))))

        return results
