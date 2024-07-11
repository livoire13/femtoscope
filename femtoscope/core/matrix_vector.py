# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 13:34:04 2024

Utility of 'weak_form' module for storing stiffness matrices and rhs vectors
arising from the FE discretization.


"""

import warnings
from dataclasses import dataclass
from typing import Union

import numpy as np
from scipy.sparse import csr_matrix
from sfepy.discrete.problem import Problem


@dataclass
class MatrixVector:
    """
    Data class for storing sparse matrices and rhs vector arising from the
    finite element discretization. Matrices and vectors come in duplicate
    accounting for the various ways of representing boundary conditions:
    - 'no_bc' refers to pristine mtx/rhs with no bc imposed;
    - 'bc_full' refers to mtx/rhs with bc imposed but keeping all DOFs;
    - 'bc_reduced' refers to mtx/rhs with bc imposed by removing fixed DOFs.
    """

    # Matrices and vectors associated with 'pb_cst'
    mtx_no_bc_cst: csr_matrix = None
    vec_no_bc_cst: np.ndarray = None
    mtx_bc_full_cst: csr_matrix = None
    vec_bc_full_cst: np.ndarray = None
    mtx_bc_reduced_cst: csr_matrix = None
    vec_bc_reduced_cst: np.ndarray = None

    # Matrices and vectors associated with 'pb_mod'
    mtx_no_bc_mod: csr_matrix = None
    vec_no_bc_mod: np.ndarray = None
    mtx_bc_full_mod: csr_matrix = None
    vec_bc_full_mod: np.ndarray = None
    mtx_bc_reduced_mod: csr_matrix = None
    vec_bc_reduced_mod: np.ndarray = None

    class Decorators:
        """Inner class for defining decorators."""

        @staticmethod
        def check_pb_bc_full(func):
            """
            Decorator for checking that the implementation of bcs in the
            `Problem` instance allow for the computation of 'bc_full' objects.
            """
            def wrapper(*args):
                pb = args[-1]
                if not isinstance(pb, Problem):
                    raise TypeError(f"Broken 'check_pb_bc_full' decorator, "
                                    f"the last argument of {func} should be a "
                                    f"Problem instance.")
                if pb.epbcs.names:  # pb has epbcs
                    raise NotImplementedError("Cannot construct 'bc_full' "
                                              "mtx/vec if problem has epbcs.")
                if pb.lcbcs.names:  # pb has lcbcs
                    raise NotImplementedError("Cannot construct 'bc_full' "
                                              "mtx/vec if problem has lcbcs")
                if not pb.ebcs.names:  # pb has no ebcs
                    raise RuntimeError("'pb' has no registered essential "
                                       "boundary conditions. Consider calling "
                                       "`WeakForm.apply_bcs`")
                return func(*args)

            return wrapper

    # 'no_bc'
    @property
    def mtx_no_bc(self) -> Union[csr_matrix, None]:
        if self.mtx_no_bc_cst is None:
            # warnings.warn("'no_bc' matrices are not set, returning None.")
            return None
        elif self.mtx_no_bc_mod is None:
            return self.mtx_no_bc_cst
        else:  # 'no_bc' matrices can be safely added together
            return self.mtx_no_bc_cst + self.mtx_no_bc_mod

    def get_mtx_no_bc(self) -> Union[csr_matrix, None]:
        return self.mtx_no_bc

    @property
    def vec_no_bc(self) -> Union[np.ndarray, None]:
        if self.vec_no_bc_cst is None:
            return None  # 'no_bc' vectors are not set, returning None
        elif self.vec_no_bc_mod is None:
            return self.vec_no_bc_cst
        else:  # 'no_bc' vectors can be safely added together
            return self.vec_no_bc_cst + self.vec_no_bc_mod

    def get_vec_no_bc(self) -> Union[np.ndarray,  None]:
        return self.vec_no_bc

    # 'bc_reduced'
    @property
    def mtx_bc_reduced(self) -> Union[csr_matrix, None]:
        if self.mtx_bc_reduced_cst is None:
            return None  # 'bc_reduced' matrices are not set, returning None
        elif self.mtx_bc_reduced_mod is None:
            return self.mtx_bc_reduced_cst
        else:  # 'bc_reduced' matrices can be safely added together
            return self.mtx_bc_reduced_cst + self.mtx_bc_reduced_mod

    def get_mtx_bc_reduced(self) -> Union[csr_matrix, None]:
        return self.mtx_bc_reduced

    @property
    def vec_bc_reduced(self) -> Union[np.ndarray, None]:
        if self.vec_bc_reduced_cst is None:
            return None  # 'bc_reduced' vectors are not set, returning None
        elif self.vec_bc_reduced_mod is None:
            return self.vec_bc_reduced_cst
        else:  # 'bc_reduced' vectors can be safely added together
            return self.vec_bc_reduced_cst + self.vec_bc_reduced_mod

    def get_vec_bc_reduced(self) -> Union[np.ndarray, None]:
        return self.vec_bc_reduced

    # 'bc_full'
    def get_mtx_bc_full(self, pb: Problem) -> Union[csr_matrix, None]:
        if self.mtx_bc_full_cst is None:
            return None  # 'bc_full' matrices are not set, returning None
        elif self.mtx_bc_full_mod is None:
            return self.mtx_bc_full_cst
        else:
            if self.mtx_no_bc_cst is None and self.mtx_no_bc_mod is None:
                warnings.warn(
                    "'no_bc' matrices need to be set, returning None.")
            mtx_no_bc = self.mtx_no_bc
            return self._make_mtx_bc_full_from_mtx_no_bc(mtx_no_bc, pb)

    def get_vec_bc_full(self, pb: Problem) -> Union[np.ndarray, None]:
        if self.vec_bc_full_cst is None:
            return None  # 'bc_full' vectors are not set, returning None
        elif self.vec_bc_full_mod is None:
            return self.vec_bc_full_cst
        else:
            if self.vec_no_bc_cst is None and self.vec_no_bc_mod is None:
                warnings.warn("'no_bc' vectors need to be set, returning None.")
                return None
            vec_no_bc = self.vec_no_bc
            mtx_no_bc = self.mtx_no_bc
            return self._make_vec_bc_full_from_mtx_vec_no_bc(
                mtx_no_bc, vec_no_bc, pb)

    @Decorators.check_pb_bc_full
    def _make_mtx_bc_full_from_mtx_no_bc(self, mtx_no_bc: csr_matrix,
                                         pb: Problem) -> csr_matrix:
        """
        Construct 'mtx_bc_full' from 'mtx_no_bc'. The problem instance 'pb'
        must be passed as the last argument.
        """
        # Convert the matrix to the list of lists format in order to make
        # matrix editing much faster
        mtx = mtx_no_bc.copy().tolil()
        fixed_dofs = pb.get_ebc_indices()[0]  # Sfepy method to get fixed DOFs
        for idx in fixed_dofs:
            mtx[:, idx] = 0.0
            mtx[idx, :] = 0.0
            mtx[idx, idx] = 1.0
        return mtx.tocsr()

    @Decorators.check_pb_bc_full
    def _make_vec_bc_full_from_mtx_vec_no_bc(
            self, mtx_no_bc: csr_matrix,
            vec_no_bc: np.ndarray, pb: Problem) -> np.ndarray:
        """
        Construct 'vec_bc_full' from 'mtx_no_bc' and 'vec_no_bc'. The problem
        instance 'pb' must be passed as the last argument.
        """
        idx_bc = pb.get_ebc_indices()[0]
        vec_bc = pb.get_variables().get_state(reduced=False)
        vec_bc_full = vec_no_bc - mtx_no_bc.dot(vec_bc)
        vec_bc_full[idx_bc] = vec_bc[idx_bc]
        return vec_bc_full
