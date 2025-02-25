# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:47:24 2024

Definition of stopping criteria and 'NonLinearMonitor' class.
Instances of classes 'NonLinearSolver' and 'NonLinearMonitor' are interlinked.

"""

from __future__ import annotations
from typing import TYPE_CHECKING

import inspect
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable
from copy import deepcopy
from numbers import Number
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

import femtoscope.core.solvers as solvers  # to avoid circular import error
from femtoscope.core.weak_form import WeakForm

if TYPE_CHECKING:
    from femtoscope.core.solvers import NonLinearSolver  # Only for type hints


def get_criteria_dict() -> dict:
    """
    Retrieve all child classes of `Criterion` and put them in a dictionary
    (using their name as keys).

    Returns
    -------
    criteria_dict : dict
        Dictionary of all currently implemented criteria.
    """
    clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    criteria_dict = {}
    for member in clsmembers:
        if issubclass(member[1], Criterion) and member[1].__base__ != ABC:
            criteria_dict[member[0]] = member[1]
    return criteria_dict


class Decorators:
    """Class for defining decorators."""

    @staticmethod
    def check_wf_residual(func):
        """
        Check that the weak form 'wf_residual' has been correctly set in the
        nonlinear solver attribute `wf_dict` before calling a criterion that
        rely on it.
        """

        def wrapper(*args):
            crit_instance = args[0]
            nl_monitor = args[1]
            nl_solver = nl_monitor.nonlinear_solver
            if nl_solver is None:
                raise ValueError("'NonLinearMonitor' instance is not yet"
                                 "linked to a 'NonLinearSolver' instance!")
            if 'wf_residual' not in nl_solver.wf_dict:
                raise KeyError(f"Criterion {crit_instance.__class__.__name__}"
                               f"requires 'wf_residual'")
            if not isinstance(nl_solver.wf_dict['wf_residual'], WeakForm):
                raise TypeError("'wf_residual' is not set properly!")
            return func(*args)

        return wrapper

    @staticmethod
    def not_zeroth_iteration(func):
        """Check that at least one iteration has been completed."""

        def wrapper(*args):
            nl_monitor = args[1]
            if nl_monitor.current_iter_num == 0:
                return
            else:
                return func(*args)

        return wrapper


class Criterion(ABC):
    """
    Template for criteria implementation (abstract class).

    Attributes
    ----------
    threshold : Number
        Criterion's threshold value. The criterion is met when its `value`
        crosses the threshold. It is automatically set to '-1' when the
        criterion is not active (see below).
    status : dict
        Boolean dictionary containing information about the current status of
        the criterion, with keys:
        - 'stop' : bool
            Whether the criterion is met.
        - 'look' : bool
            Whether the criterion should be evaluated or not.
        - 'active' : bool
            Whether the criterion is taken into account for deciding to end
            the iterative algorithm.
    value : Union[Number, np.ndarray, dict]
        Current value associated with the criterion.
    history : dict
        Dictionary containing the criterion's value for all past iterations.
        Keys are integers representing the iteration number.

    """

    @property
    @abstractmethod
    def priority(self):
        pass

    def __init__(self, threshold: Number, look: bool, active: bool):
        look = active if active else look
        self.threshold = threshold if active else -1
        self.status = {'stop': False, 'look': look, 'active': active}
        self.value = None
        self.history = {}

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def evaluate(self, nonlinear_monitor: NonLinearMonitor):
        """
        Performs 3 tasks:
        (i) Evaluate the criterion's current value
        (ii) Update its status accordingly
        (iii) Fill in its history by calling method `update_history`

        Parameters
        ----------
        nonlinear_monitor : NonLinearMonitor
            Instance of class `NonLinearMonitor` used to evaluate the criterion.
        """
        raise NotImplementedError

    @abstractmethod
    def update_history(self, nonlinear_monitor: NonLinearMonitor):
        """Store current criterion value in attribute `history`."""
        raise NotImplementedError

    @abstractmethod
    def write_stop_message(self):
        """Output message is the present criterion is responsible for the
        algorithm termination."""
        raise NotImplementedError


class MaximumIterations(Criterion):
    r"""Maximum number of iterations $k < \text{threshold}$"""

    priority = False

    def evaluate(self, nonlinear_monitor: NonLinearMonitor):
        self.value = nonlinear_monitor.current_iter_num
        self.update_history(nonlinear_monitor)
        self.status['stop'] = (self.value == self.threshold)

    def update_history(self, nonlinear_monitor: NonLinearMonitor):
        self.history[nonlinear_monitor.current_iter_num] = self.value

    def write_stop_message(self):
        print(
            f"End of iterations: the maximum number of iterations "
            f"(= {self.threshold}) has been reached."
        )


class RelativeDeltaSolutionNorm2(Criterion):
    r"""
    Relative delta on the solution between two consecutive iterations (in
    2-norm), i.e.
    $$ \frac{\| u_k - u_{k-1} \|_2}{\| u_{k-1} \|_2} < \text{threshold} $$
    """

    priority = False

    @Decorators.not_zeroth_iteration
    def evaluate(self, nonlinear_monitor: NonLinearMonitor):
        nl_solver = nonlinear_monitor.nonlinear_solver
        old_int = nl_solver.old_int
        sol_int = nl_solver.sol_int
        value = np.linalg.norm(sol_int - old_int) / np.linalg.norm(old_int)
        sol_ext = nl_solver.sol_ext
        if sol_ext is not None:
            old_ext = nl_solver.old_ext
            value += np.linalg.norm(sol_ext - old_ext) / np.linalg.norm(old_ext)
        self.value = value
        self.update_history(nonlinear_monitor)
        self.status['stop'] = (self.value < self.threshold)

    def update_history(self, nonlinear_monitor: NonLinearMonitor):
        self.history[nonlinear_monitor.current_iter_num] = self.value

    def write_stop_message(self):
        print(
            "End of iterations: the relative variation of the solution vector "
            "(in 2-norm) between two consecutive iterations is smaller than "
            "the specified threshold = {:.1e}".format(self.threshold)
        )


class AbsoluteDeltaSolutionNorm2(Criterion):
    r"""
    Absolute delta on the solution between two consecutive iterations (in
    2-norm), i.e.
    $$ \| u_k - u_{k-1} \|_2 < \text{threshold} $$
    """

    priority = False

    @Decorators.not_zeroth_iteration
    def evaluate(self, nonlinear_monitor: NonLinearMonitor):
        nl_solver = nonlinear_monitor.nonlinear_solver
        old_int = nl_solver.old_int
        sol_int = nl_solver.sol_int
        value = np.linalg.norm(sol_int - old_int)
        sol_ext = nl_solver.sol_ext
        if sol_ext is not None:
            old_ext = nl_solver.old_ext
            value += np.linalg.norm(sol_ext - old_ext)
        self.value = value
        self.update_history(nonlinear_monitor)
        self.status['stop'] = (self.value < self.threshold)

    def update_history(self, nonlinear_monitor: NonLinearMonitor):
        self.history[nonlinear_monitor.current_iter_num] = self.value

    def write_stop_message(self):
        print(
            "End of iterations: the variation of the solution vector (in "
            "2-norm) between two consecutive iterations is smaller than the "
            "specified threshold = {:.1e}".format(self.threshold)
        )


class ResidualVector(Criterion):
    """Residual vector (look only, not active)."""

    priority = True

    @Decorators.check_wf_residual
    def evaluate(self, nonlinear_monitor: NonLinearMonitor):
        """See NonLinearSolver.evaluate_residual_vector for the implementation
        of the residual computation."""
        nl_solver = nonlinear_monitor.nonlinear_solver
        self.value = nl_solver.evaluate_residual_vector()
        self.update_history(nonlinear_monitor)

    def update_history(self, nonlinear_monitor: NonLinearMonitor):
        it = nonlinear_monitor.current_iter_num
        self.history[it] = self.value.copy()

    def write_stop_message(self):
        pass


class ResidualVectorParts(Criterion):
    """
    Residual vector parts (look only, not active). Decomposition of the
    residual vector $R$ into
    $$ R = A u_k - rhs_{\text{cst}} - rhs_{\text{mod}} $$
    The `value` attribute is a dictionary with keys
    ('mtx_term', 'rhs_cst_term', 'rhs_mod_term').
    """

    priority = False

    @Decorators.check_wf_residual
    def evaluate(self, nonlinear_monitor: NonLinearMonitor):
        """See NonLinearSolver.evaluate_residual_vector_parts for the
        implementation."""
        nl_solver = nonlinear_monitor.nonlinear_solver
        self.value = nl_solver.evaluate_residual_vector_parts()
        self.update_history(nonlinear_monitor)

    def update_history(self, nonlinear_monitor: NonLinearMonitor):
        it = nonlinear_monitor.current_iter_num
        self.history[it] = deepcopy(self.value)

    def write_stop_message(self):
        pass


class ResidualVectorNorm2(Criterion):
    """2-norm of the residual vector."""

    priority = False

    def evaluate(self, nonlinear_monitor: NonLinearMonitor):
        residual_vector = nonlinear_monitor.residual_vetor
        norm2 = np.linalg.norm(residual_vector)
        self.value = norm2
        self.update_history(nonlinear_monitor)
        self.status['stop'] = (self.value < self.threshold)

    def update_history(self, nonlinear_monitor: NonLinearMonitor):
        it = nonlinear_monitor.current_iter_num
        self.history[it] = self.value

    def write_stop_message(self):
        print(
            "End of iterations: the 2-norm of the residual vector is smaller "
            "than the specified threshold = {:.1e}".format(self.threshold)
        )


class RelativeDeltaResidualNorm2(Criterion):
    r"""
    Relative variation of the residual vector in 2-norm, i.e.
    $$ \frac{\| R_k - R_{k-1} \|_2}{\| R_{k-1} \|_2} < \text{threshold} $$
    """

    priority = False

    @Decorators.not_zeroth_iteration
    def evaluate(self, nonlinear_monitor: NonLinearMonitor):
        it = nonlinear_monitor.current_iter_num
        res_vec = nonlinear_monitor.residual_vetor
        res_vec_old = nonlinear_monitor._get_residual_vector_at_iteration(
            it - 1)
        value = np.linalg.norm(res_vec - res_vec_old) / np.linalg.norm(res_vec)
        self.value = value
        self.update_history(nonlinear_monitor)
        self.status['stop'] = (self.value < self.threshold)

    def update_history(self, nonlinear_monitor: NonLinearMonitor):
        it = nonlinear_monitor.current_iter_num
        self.history[it] = self.value

    def write_stop_message(self):
        print(
            "End of iterations: the relative variation of the residual vector "
            "(in 2-norm) between two consecutive iterations is smaller than "
            "the specified threshold = {:.1e}".format(self.threshold)
        )


class ResidualReductionFactor(Criterion):
    r"""
    Ratio $\| R_k \|_2 / \| R_0 \|_2$, i.e. reduction factor in the 2-norm of
    the residual vector.
    """

    priority = False

    def evaluate(self, nonlinear_monitor: NonLinearMonitor):
        res_vec = nonlinear_monitor.residual_vetor
        res_vec_initial = nonlinear_monitor._get_residual_vector_at_iteration(0)
        value = np.linalg.norm(res_vec) / np.linalg.norm(res_vec_initial)
        self.value = value
        self.update_history(nonlinear_monitor)
        self.status['stop'] = (self.value < self.threshold)

    def update_history(self, nonlinear_monitor: NonLinearMonitor):
        it = nonlinear_monitor.current_iter_num
        self.history[it] = self.value

    def write_stop_message(self):
        print(
            "End of iterations: the residual vector has decreased by {:.1f}% "
            "(in 2-norm) between the initial guess and the current iterate, "
            "which is smaller than the specified threshold = {:.1f}".format(
                self.value*100, self.threshold*100)
        )


class NonLinearMonitor:
    r"""
    Class for monitoring a `NonLinearSolver` object while solving a nonlinear
    problem.

    Attributes
    ----------
    criteria_class_dict : dict
        Class attribute. (criterion-name: criterion-class) key-value pairs.
    stop : bool
        Toggle for putting an end to the iterative algorithm.
    current_iter_num : int
        Current iteration number.
    minimum_iter_num : int
        Minimum number of iterations to be completed (lower threshold).
    criteria : dict
        Dictionary of criteria (child instances of parent abstract class
        `Criterion`) to be evaluated at the end of each iteration.
    nonlinear_solver : NonLinearSolver
        Instance of the `NonLinearSolver` class that is being monitored.

    """

    criteria_class_dict = get_criteria_dict()

    def __init__(self):
        self.stop = False
        self.current_iter_num = 0
        self.minimum_iter_num = 0
        self.criteria = None
        self.nonlinear_solver = None

    @property
    def maximum_iter_num(self):
        """Maximum number of iterations to be completed (upper threshold)."""
        return self.criteria['MaximumIterations'].threshold

    @property
    def residual_vetor(self):
        if 'ResidualVector' in self.criteria:
            return self.criteria['ResidualVector'].value
        return None

    def set_maximum_iter_num(self, maximum_iter_num: int):
        self.criteria['MaximumIterations'].threshold = maximum_iter_num

    @classmethod
    def from_scratch(cls, args_dict: dict) -> NonLinearMonitor:
        """
        Create an instance of `NonLinearMonitor` and set its attributes using
        the user provided `args_dict`, which contains the following items:

        Parameters
        ----------
        minimum_iter_num : int
            Minimum number of iterations to be completed (lower threshold).
        maximum_iter_num : int
            Maximum number of iterations to be completed (upper threshold).
        criteria : Tuple[dict]
            Tuple of dictionaries. Each dictionary corresponds to a given
            criterion, with keys
                - 'name': the name of the criterion
                - 'threshold': the criterion's threshold
                - 'look': Whether the criterion should be evaluated
                - 'active': Whether the criterion is involved in the
                decision-making process for stopping the algorithm

        Notes
        -----
        Sometimes, one may want to evaluate a given criterion to get more
        information about the current state of the iterations without making
        the stopping of the algorithm dependent on it. This 'information only'
        mode is achieved by specifying ('active' = False, 'look' = True).
        On the other hand, if a given criterion is set active, it will
        automatically be evaluated, regardless of the 'look' setting.

        Returns
        -------
        monitor : NonLinearMonitor
            The created NonLinearMonitor.
        """

        # Create a new NonLinearMonitor object
        monitor = NonLinearMonitor()

        # Fill in min/max number of iterations
        if 'minimum_iter_num' in args_dict:
            monitor.minimum_iter_num = args_dict['minimum_iter_num']
        if 'maximum_iter_num' in args_dict:
            maximum_iter_num = args_dict['maximum_iter_num']
        else:
            raise KeyError("'maximum_iter_num' must be specified by the user!")
        assert monitor.minimum_iter_num <= maximum_iter_num

        # Fill in MaximumIterations criterion
        monitor.criteria = {}
        max_iter_crit = MaximumIterations(maximum_iter_num, True, True)
        monitor.criteria['MaximumIterations'] = max_iter_crit

        # Fill in all other criteria
        criteria = args_dict.get('criteria', None)
        cls._check_criteria_input(criteria)
        for criterion in criteria:
            name = criterion['name']
            threshold = criterion['threshold']
            active = criterion['active']
            look = criterion.get('look', active)
            crit = cls.criteria_class_dict[name](threshold, look, active)
            monitor.criteria[name] = crit

        return monitor

    def display_criteria_info(self, *cols: Tuple[str]):
        """
        Display information regarding the registered criteria in
        `self.criteria`.

        Parameters
        ----------
        cols : Tuple[str]
            Keys for displayed information (see `valid_entries` in function
            definition).

        """

        # Preset
        valid_entries = (
            'criterion', 'stop', 'look', 'active', 'value', 'threshold')
        actual_entries = []
        cols = list(cols)
        if 'criterion' not in cols:
            cols.append('criterion')
        data_dict = {}

        criteria_copy = self.criteria.copy()  # shallow copy of dictionary
        if self.current_iter_num >= 1:
            for key, crit in self.criteria.items():
                if not isinstance(crit.value, Number):
                    criteria_copy.pop(key)

        # Fill in data_dict
        for entry in cols:
            assert entry in valid_entries, f"Entry '{entry}' is not valid!"
            actual_entries.append(entry)
            entry_data_list = []

            if entry == 'criterion':
                for criterion in criteria_copy.values():
                    entry_data_list.append(criterion.name)
                data_dict[entry] = entry_data_list
                continue

            for criterion in criteria_copy.values():
                value = criterion.value
                if hasattr(criterion, entry):
                    if entry == 'value' and not isinstance(value, Number):
                        entry_data_list.append(value.name)
                    else:
                        entry_data_list.append(getattr(criterion, entry))
                elif entry in criterion.status:
                    entry_data_list.append(criterion.status[entry])
                else:
                    raise ValueError(f"Cannot find entry '{entry}'")
                data_dict[entry] = entry_data_list

        # Create pandas DataFrame and sort the columns
        df = pd.DataFrame(data_dict)
        sort_key_list = [valid_entries.index(entry) for entry in actual_entries]
        sorted_zip = sorted(zip(sort_key_list, actual_entries))
        _, actual_entries = zip(*sorted_zip)
        df = df.reindex(columns=actual_entries)

        # Display information
        output_string = self._format_dataframe_string_iter(df)
        print(output_string + '\n')

    def display_iterations_report(self):
        """
        Display criteria's value for all iterations.

        Notes
        -----
        The criteria considered here are those for which the `value` field is a
        number.

        todo: remove the NaN when the iterations stop before 'maximum_iter_num'
        """

        data_dict = {
            'iter': list(range(self.maximum_iter_num + 1)) + ['threshold']
        }

        for crit in self.criteria.values():
            if not isinstance(crit.value, Number) or \
                    crit.name == 'MaximumIterations':
                continue
            data_dict[crit.name] = list(
                crit.history.values()) + [crit.threshold]

        df = pd.DataFrame(data_dict)
        df.set_index('iter')
        df.drop([*range(self.current_iter_num+1, self.maximum_iter_num+1)],
                inplace=True)
        df.rename(index={df.index[-1]: '__'}, inplace=True)
        output_string = self._format_dataframe_string_report(df)
        print(output_string + '\n')

    def link_monitor_to_solver(self, nl_solver: "NonLinearSolver"):
        """
        Link the current `NonLinearMonitor` instance to a given
        `NonLinearSolver` instance. This link is two-way in the sense that the
        two class instances hold themselves as attributes.

        Parameters
        ----------
        nl_solver : NonLinearSolver
            Instance of the class `NonLinearSolver` to be monitored.

        Notes
        -----
        One can either link with the present function, or equivalently with
        `NonLinearSolver.link_solver_to_monitor`.

        """
        self.nonlinear_solver = nl_solver
        nl_solver.nonlinear_monitor = self

    def update_status(self):
        """
        Update the `stop` attribute based on the status of all active criteria.
        todo: in the future, it might be nice to have the possibility to
            combine several criteria together for determining whether to stop
            the algorithm
        """
        # Check if one criterion has been met
        if self.current_iter_num >= self.minimum_iter_num:
            for criterion in self.criteria.values():
                if not criterion.status['active']:
                    continue
                if criterion.status['stop']:
                    self.stop = True
                    self._fill_history_holes_with_none()
                    return

    def write_termination_status(self):
        """Print the termination message associated with the criterion that has
        been met."""
        for criterion in self.criteria.values():
            if criterion.status['stop']:
                criterion.write_stop_message()

    def evaluate_all_criteria(self):
        """Evaluate all criteria registered in `self.criteria`."""
        for criterion in self._sorted_criteria_list():
            status = criterion.status
            if not status['look'] and not status['active']:
                continue
            criterion.evaluate(self)

    def disable_all_criteria(self):
        """Set `active=False` for all criteria but 'MaximumIterations'."""
        for criterion in self.criteria.values():
            criterion.status['active'] = False
        self.criteria['MaximumIterations'].status['active'] = True

    def _sorted_criteria_list(self) -> List[Criterion]:
        """
        Sort values of `criteria` dictionary according to their `priority`
        attribute.

        Returns
        -------
        List of sorted `Criterion` instances.

        """
        criteria = self.criteria
        priority_line = []
        waiting_line = []
        for criterion in criteria.values():
            if criterion.priority:
                priority_line.append(criterion)
            else:
                waiting_line.append(criterion)
        return priority_line + waiting_line

    def _get_residual_vector_at_iteration(
            self, iteration_number: int) -> Union[np.ndarray, None]:
        """Retrieve residual vector at a given iteration."""
        for criterion in self.criteria.values():
            if criterion.name == 'ResidualVector':
                return criterion.history[iteration_number]
        return None

    def _format_dataframe_string_iter(self, dataframe: pd.DataFrame) -> str:
        """Format dataframe string for subsequent printing of iteration info."""
        df_str = dataframe.to_string(col_space=15, index=False, justify='right',
                                     float_format='{:.2e}'.format)
        width = len(df_str.split('\n')[0])
        it = self.current_iter_num
        if it == 0:
            title = " MONITORING PARAMETERS ".center(width, '_')
        else:
            title = " ITERATION NO {:d} ".format(it).center(width, '_')
        bottom = "".center(width, '-')
        return title + '\n' + df_str + '\n' + bottom

    def _format_dataframe_string_report(self, dataframe: pd.DataFrame) -> str:
        """Format dataframe string for subsequent printing of report info."""
        df_str = dataframe.to_string(col_space=15, index=False,
                                     justify='center',
                                     float_format='{:.2e}'.format)
        width = len(df_str.split('\n')[0])
        title = " RECAP OF ALL ITERATIONS ".center(width, '_')
        bottom = "".center(width, '-')
        return title + '\n' + df_str + '\n' + bottom

    def _fill_history_holes_with_none(self):
        for k in range(self.maximum_iter_num + 1):
            for criterion in self.criteria.values():
                if k not in criterion.history:
                    criterion.history[k] = None
                criterion.history = dict(sorted(criterion.history.items()))

    @staticmethod
    def _check_criteria_input(criteria: List[Criterion]):
        """
        Check that the user-provided `criteria` tuple complies with
        `from_scratch` method (see docstring above).
        """

        if criteria is None: return
        mandatory_keys = ['name', 'threshold', 'active']

        # Check that criteria is iterable
        if not isinstance(criteria, Iterable):
            raise TypeError(f"'criteria' should be a tuple, not a "
                            f"{type(criteria)}")

        for criterion in criteria:
            # Check type of each item
            if not isinstance(criterion, dict):
                raise TypeError(f"'criteria' items should be dictionaries, "
                                f"but one item is of type {type(criterion)}")

            # Check keys of each item
            if not all(key in criterion for key in mandatory_keys):
                raise KeyError(
                    f"Missing (at least) one key from {mandatory_keys}")

            # Check name of the criterion
            if criterion['name'] not in NonLinearMonitor.criteria_class_dict:
                raise NotImplementedError(
                    f"Criterion '{criterion['name']}' is not implemented yet")

            # Check types
            if not isinstance(criterion['threshold'], Number):
                raise TypeError(f"'threshold' should be a float, but is of type"
                                f"{type(criterion['threshold'])} instead")
            if not isinstance(criterion['active'], bool):
                raise TypeError(f"'active' should a boolean, but is of type"
                                f"{type(criterion['active'])} instead")
