# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:39:20 2024

"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Union

import meshio
import numpy as np
from sfepy.discrete.fem import Field

from femtoscope import RESULT_DIR
from femtoscope.core.weak_form import WeakForm


class ResultsPostProcessor:
    """
    Class for post-processing FEM results offline.

    Attributes
    ----------
    wf_int : WeakForm
        Interior weak form.
    wf_ext : Union[None, WeakForm]
        Exterior weak form, left as None when not relevant.
    vars_dict : dict
        Dictionary containing the various data fields (such as 'sol_int',
        'sol_ext', 'residual', etc. present in the VTK file(s)).

    """

    def __init__(self):
        self.wf_int = None
        self.wf_ext = None
        self.vars_dict = {}

    @property
    def field_int(self) -> Field:
        """Field instance associated with the interior domain."""
        if self.wf_int is not None:
            return self.wf_int.field

    @property
    def coors_int(self) ->np.ndarray:
        """Coordinates of the DOFs in the interior domain."""
        if self.wf_int is not None:
            return self.field_int.coors

    @property
    def sol_int(self) -> np.ndarray:
        """Solution vector at interior DOFs."""
        return self.vars_dict.get('sol_int', None)

    @property
    def field_ext(self) -> Union[None, Field]:
        """Field instance associated with the exterior domain."""
        if self.wf_ext is not None:
            return self.wf_ext.field

    @property
    def coors_ext(self) -> Union[None, np.ndarray]:
        """Coordinates of the DOFs in the exterior domain."""
        if self.wf_ext is not None:
            return self.field_ext.coors

    @property
    def sol_ext(self) -> Union[None, np.ndarray]:
        """Solution vector at exterior DOFs."""
        return self.vars_dict.get('sol_ext', None)

    @classmethod
    def from_files(cls, name: str) -> ResultsPostProcessor:

        results_pp = ResultsPostProcessor()

        dir_path = Path(RESULT_DIR / name)

        # Read weak form metadata first
        for pkl_file in dir_path.glob('*.pkl'):
            with open(pkl_file, mode='rb') as f:
                args_dict = pickle.load(f)
            wf = WeakForm.from_scratch(args_dict)
            if pkl_file.stem.split('_')[-1] == 'ext':
                results_pp.wf_ext = wf
            else:
                results_pp.wf_int = wf

        # Then, read data from VTK
        for vtk_file in dir_path.glob('*.vtk'):
            data = meshio.read(vtk_file)
            point_data = data.point_data
            point_data.pop('node_groups')
            for key, val in point_data.items():
                results_pp.vars_dict[key] = _restore_1d_array(val)

        # Return ResultsPostProcessor instance
        return results_pp

    def evaluate_at(self, coors: np.ndarray, key: Union[None, str] = None,
                    mode='val', tag='int') -> np.ndarray:
        """
        Evaluate some source DOF-values at specified coordinates using the FEM
        basis functions (Sfepy).

        Parameters
        ----------
        coors : np.ndarray
            The coordinates the source values should be interpolated into.
        key : Union[None, str], optional
            Key of the source, should be a key of `self.vars_dict`. The default
            is None, is which case the solution values are used.
        mode : str, optional
            The evaluation mode: 'val' for the field values, 'grad' for the
            field gradient. The default is 'val'.
        tag : str, optional
            'int' if source values are defined over the interior domain;
            'ext' if source values are defined over the exterior domain.
            The default is 'int'.

        Returns
        -------
        ev : np.ndarray
            The interpolated values at specified coordinates.

        See Also
        --------
        Field.evaluate_at : Sfepy built-in f
        unction used in this method.
        """

        field = getattr(self, 'field_' + tag)  # get field instance

        if key is not None:  # get source DOF-values
            source = self.vars_dict[key]
        else:
            source = getattr(self, 'sol_' + tag)

        ev = field.evaluate_at(coors, source[:, np.newaxis], mode=mode)
        return ev.squeeze()


def _restore_1d_array(array: np.ndarray) -> np.ndarray:
    """Tweak data representation in a way that is suitable for Sfepy."""
    new_array = np.ascontiguousarray(array).byteswap().newbyteorder()
    return new_array
