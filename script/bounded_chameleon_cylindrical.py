from pathlib import Path

import numpy as np

from femtoscope.core.pre_term import PreTerm
from femtoscope.core.solvers import NonLinearSolver
from femtoscope.core.nonlinear_monitoring import NonLinearMonitor
from femtoscope.core.weak_form import WeakForm
from femtoscope.inout.meshfactory import MeshingTools
from femtoscope.inout.postprocess import ResultsPostProcessor
from femtoscope.physics.materials_library import LaplacianMaterials
from femtoscope import RESULT_DIR

rho_min = 1
rho_max = 1e2
alpha = 0.1
npot = 2
sa = 1.0
sc = 1.0
Rcut = 5.0
fem_order = 2

# Mesh creation
mt = MeshingTools(2)
s1 = mt.create_ellipse(rx=sa, ry=sc)
mt.create_subdomain(cell_size_min=0.03, cell_size_max=0.2,
                    dist_min=0.0, dist_max=4.0)
s2 = mt.create_ellipse(rx=Rcut, ry=Rcut)
mt.subtract_shapes(s2, s1, removeObject=True, removeTool=False)
mt.create_subdomain(cell_size_min=0.2, cell_size_max=0.2)
pre_mesh = mt.generate_mesh('mesh_cylindrical_chameleon.vtk',
                            cylindrical_symmetry=True,
                            show_mesh=False)


def create_wf_int():
    """Create the linearized weak form (instance of `WeakForm`)."""

    # Materials
    def mat1(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        x = coors[:, 0]
        val = abs(x)
        return {'val': val.reshape(-1, 1, 1)}

    t1 = PreTerm('dw_laplace', mat=mat1, tag='cst', prefactor=alpha)

    def mat2(ts, coors, mode=None, vec_qp=None, **kwargs):
        if mode != 'qp': return
        x = coors[:, 0]
        val = abs(x) * vec_qp ** (-(npot + 2))
        return {'val': val.reshape(-1, 1, 1)}

    t2 = PreTerm('dw_volume_dot', mat=mat2, tag='mod', prefactor=npot + 1)

    def mat3(ts, coors, mode=None, vec_qp=None, **kwargs):
        if mode != 'qp': return
        x = coors[:, 0]
        val = abs(x) * vec_qp ** (-(npot + 1))
        return {'val': val.reshape(-1, 1, 1)}

    t3 = PreTerm('dw_volume_integrate', mat=mat3, tag='mod',
                 prefactor=-(npot + 2))

    def mat4(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        x = coors[:, 0]
        val = abs(x) * rho_max
        return {'val': val.reshape(-1, 1, 1)}

    t4 = PreTerm('dw_volume_integrate', mat=mat4, tag='cst', prefactor=1,
                 region_key=('subomega', 300))

    def mat5(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        x = coors[:, 0]
        val = abs(x) * rho_min
        return {'val': val.reshape(-1, 1, 1)}

    t5 = PreTerm('dw_volume_integrate', mat=mat5, tag='cst', prefactor=1,
                 region_key=('subomega', 301))

    # WeakForm creation
    args_dict = {
        'name': 'wf_chameleon_2d',
        'dim': 2,
        'pre_mesh': pre_mesh,
        'pre_terms': [t1, t2, t3, t4, t5],
        'fem_order': fem_order,
        'pre_ebc_dict': {('facet', 201): rho_min ** (-1 / (npot + 1))}
    }
    wf = WeakForm.from_scratch(args_dict)
    return wf


def create_wf_res():
    """Create the residual weak form (instance of `WeakForm`)."""

    # Materials
    def mat1(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        x = coors[:, 0]
        val = abs(x)
        return {'val': val.reshape(-1, 1, 1)}

    t1 = PreTerm('dw_laplace', mat=mat1, tag='cst', prefactor=alpha)

    def mat2(ts, coors, mode=None, vec_qp=None, **kwargs):
        if mode != 'qp': return
        x = coors[:, 0]
        val = abs(x) * vec_qp ** (-(npot + 1))
        return {'val': val.reshape(-1, 1, 1)}

    t2 = PreTerm('dw_volume_integrate', mat=mat2, tag='mod', prefactor=-1)

    def mat3(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        x = coors[:, 0]
        val = abs(x) * rho_max
        return {'val': val.reshape(-1, 1, 1)}

    t3 = PreTerm('dw_volume_integrate', mat=mat3, tag='cst', prefactor=1,
                 region_key=('subomega', 300))

    def mat4(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        x = coors[:, 0]
        val = abs(x) * rho_min
        return {'val': val.reshape(-1, 1, 1)}

    t4 = PreTerm('dw_volume_integrate', mat=mat4, tag='cst', prefactor=1,
                 region_key=('subomega', 301))

    def mat5(ts, coors, mode=None, **kwargs):
        if mode != 'qp': return
        x = coors[:, 0]
        val = np.zeros((coors.shape[0], 2, 2))
        val[:, 0, 0] = abs(x)
        val[:, 1, 1] = abs(x)
        return {'val': val}

    t5 = PreTerm('dw_surface_flux', mat=mat5, tag='cst', prefactor=-alpha,
                 region_key=('facet', 201))

    args_dict = {
        'name': 'wf_residual_2d',
        'dim': 2,
        'pre_mesh': pre_mesh,
        'pre_terms': [t1, t2, t3, t4],
        'fem_order': fem_order
    }
    wf = WeakForm.from_scratch(args_dict)
    return wf


def create_nonlinear_solver(wf_int, wf_res):
    wf_dict = {'wf_int': wf_int, 'wf_residual': wf_res}
    phi_min = rho_max ** (-1 / (npot + 1))
    phi_max = rho_min ** (-1 / (npot + 1))
    dofs_int = wf_int.field.get_dofs_in_region(
        wf_int.region_dict[('subomega', 300)])
    dofs_ext = wf_int.field.get_dofs_in_region(
        wf_int.region_dict[('subomega', 301)])
    coors = wf_int.field.coors
    guess = np.empty(coors.shape[0])
    guess[dofs_int] = phi_min
    guess[dofs_ext] = phi_max
    initial_guess_dict = {'int': guess}
    solver = NonLinearSolver(wf_dict, initial_guess_dict,
                             sol_bounds=[phi_min, phi_max],
                             relax_method='constant')
    return solver


def create_nonlinear_monitor(nonlinear_solver):
    criteria = (
        {
            'name': 'RelativeDeltaSolutionNorm2',
            'threshold': 1e-6,
            'look': True,
            'active': False
        },
        {
            'name': 'ResidualVector',
            'threshold': -1,
            'look': True,
            'active': False
        },
        {
            'name': 'ResidualVectorNorm2',
            'threshold': -1,
            'look': True,
            'active': False
        },
        {
            'name': 'ResidualReductionFactor',
            'threshold': -1,
            'look': True,
            'active': False
        },
        {
            'name': 'ResidualVectorParts',
            'threshold': -1,
            'look': True,
            'active': False
        },
    )
    args_dict = {
        'minimum_iter_num': 5,
        'maximum_iter_num': 5,
        'criteria': criteria
    }
    monitor = NonLinearMonitor.from_scratch(args_dict)
    return monitor


wf_int = create_wf_int()
wf_res = create_wf_res()
solver = create_nonlinear_solver(wf_int, wf_res)
monitor = create_nonlinear_monitor(solver)
monitor.link_monitor_to_solver(solver)
solver.solve()

# solver.save_results("test")
# dir_name = str(Path(RESULT_DIR / 'test'))
# rpp = ResultsPostProcessor.from_files(dir_name)

