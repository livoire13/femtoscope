# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 18:02:02 2022

Set up a Chameleon problem on bounded or unbouded domains.

@author: hlevy
"""

from femtoscope.core.weak import WeakForm
from femtoscope.misc.util import merge_dicts
import numpy as np
from femtoscope.core.simu import Solver, StopCriteria
from femtoscope.inout.meshfactory import get_meshdim, get_rcut
from numpy import sin, sqrt


class ChameleonBounded():
    r"""
    Class for solving the Klein-Gordon equation
    $$ \alpha \Delta \phi = \rho - \phi^{-(n+1)} $$
    on bounded domains, where the field's value is supposedly known at the
    boundary.

    Attributes
    ----------
    alpha : float
        Physical parameter weighting the laplacian operator of the Klein-Gordon
        equation (dimensionless).
    npot : int
        Exponent (parameter of the chameleon model).
    solver : `Solver` instance
        The FEM solver instance to be run.

    """

    def __init__(self, alpha, npot, densities, dirichlet_bc, rho_bounds,
                 meshfile, coorsys='cartesian', **kwargs):
        """
        Construct a `ChameleonBounded` problem instance.

        Parameters
        ----------
        alpha : float
            Physical parameter weighting the laplacian operator of the
            Klein-Gordon equation (dimensionless).
        npot : int
            Exponent (parameter of the chameleon model).
        densities : list
            List of density functions or constants. The length of this list
            must match the number of subdomains in the mesh.
        dirichlet_bc : list
            List of Dirichlet boundary condition(s).
        rho_bounds : list
            List of length 2 containing the min & max values of the density in
            the whole simulation space.
        meshfile : str
            mesh file name.
        coorsys : str, optional
            The set of coordinates to be used. The default is 'cartesian'.

        Other Parameters
        ----------------
        mesh_dir : str
            Directory where the mesh files are located. The default is None and
            in which case the mesh file is sought in the `MESH_DIR` directory.
        fem_order : int
            The FE approximation order. The default is 2.
        func_init : func
            Function for initializing the chameleon field profile.
            The default is None.
        analytic_params : dict
            Dictionary containing the relevant arguements of function
            `chameleon_radial`. The default is None.
        entity_functions : list of 2-tuple
            List of tuples (dim, function) for manual entity selection.
            The default is [].
        verbose : bool
            Display user's information. The default is False.

        """

        if not isinstance(densities, list): densities = [densities]
        mesh_dir = kwargs.get('mesh_dir', None)
        if type(meshfile) == str:
            dim = get_meshdim(meshfile, mesh_dir=mesh_dir)
        else: # 1D meshes are not saved as VTK files
            dim = 1
        ent_func = kwargs.get('entity_functions', [])
        fem_order = kwargs.get('fem_order', 2)
        analytic_params = kwargs.get('analytic_params', None)
        if analytic_params is not None:
            analytic_params['alpha'] = alpha
            analytic_params['n'] = npot
        func_init = kwargs.get('func_init', None)
        relax = kwargs.get('relax', 0.9)
        verbose = kwargs.get('verbose', True)
        self.alpha = alpha
        self.npot = npot
        phi_bounds = [p**(-1/(npot+1)) for p in rho_bounds]
        phi_bounds.sort()

        name_weak = "wf"
        name_weak_nl = "wfnl"

        min_iter = kwargs.get('min_iter')
        max_iter = kwargs.get('max_iter')
        line_search_bool = kwargs.get('line_search_bool', False)

        # Cartesian coordinates
        if coorsys == 'cartesian':

            if dim == 2:

                # Linearized Weak Form
                def mat1(ts, coors, mode=None, **kwargs):
                    if mode != 'qp' : return
                    x = coors[:, 0]
                    val = abs(x)
                    return {'val' : val.reshape(-1, 1, 1)}

                def mat2(ts, coors, mode=None, phi=None, **kwargs):
                    if mode != 'qp' : return
                    x = coors[:, 0]
                    val = abs(x)* phi**(-(npot+2))
                    return {'val' : val.reshape(-1, 1, 1)}

                def mat3(ts, coors, mode=None, phi=None, **kwargs):
                    if mode != 'qp' : return
                    x = coors[:, 0]
                    val = abs(x)* phi**(-(npot+1))
                    return {'val' : val.reshape(-1, 1, 1)}

                def matrho(ts, coors, mode=None, rho=None, **kwargs):
                    if mode != 'qp' : return
                    if callable(rho): rho = rho(coors)
                    x = coors[:, 0]
                    val = abs(x) * rho
                    return {'val' : val.reshape(-1, 1, 1)}

                coeffs = [alpha, npot+1, -(npot+2)]
                terms = [[mat1, 'dw_laplace'],
                         [mat2, 'dw_volume_dot'],
                         [mat3, 'dw_volume_integrate']]
                _complete_terms(terms, densities, matrho)
                coeffs += [1.0]*len(densities)
                kwargswf = {'constcoeffs' : coeffs,
                            'unknown_name' : 'u1',
                            'test_name' : 'v1',
                            'order' : fem_order,
                            'integral_name' : 'i1',
                            'domain_name' : 'interior',
                            'densities' : densities,
                            'entity_functions' : ent_func,
                            'dirichlet_bc_facet' : dirichlet_bc,
                            'verbose' : verbose}
                wf = WeakForm('weakcham', meshfile, terms, **kwargswf)

                # Non-Linear Weak Form
                def matnl1(ts, coors, mode=None, **kwargs):
                    if mode != 'qp' : return
                    x = coors[:, 0]
                    val = abs(x).reshape(-1, 1, 1)
                    return {'val' : val}

                def matnl2(ts, coors, mode=None, phi=None, **kwargs):
                    if mode != 'qp' : return
                    x = coors[:, 0]
                    val = (abs(x) * phi**(-(npot+1))).reshape(-1, 1, 1)
                    return {'val' : val}

                def matrho_nl(ts, coors, mode=None, rho=None, **kwargs):
                    if mode != 'qp' : return
                    if callable(rho): rho = rho(coors)
                    x = coors[:, 0]
                    val = (abs(x) * rho).reshape(-1, 1, 1)
                    return {'val' : val}

                terms_nl = [[matnl1, 'dw_laplace'],
                            [matnl2, 'dw_volume_integrate']]
                _complete_terms(terms_nl, densities, matrho_nl)
                coeffs_nl = [alpha, -1.0] + [1.0]*len(densities)
                kwargs_nl = {'constcoeffs' : coeffs_nl,
                             'order' : fem_order,
                             'densities' : densities,
                             'dirichlet_bc_facet' : dirichlet_bc,
                             'verbose' : False}
                wf_nl = WeakForm('weakcham1_nl', meshfile, terms_nl,
                                 **kwargs_nl) # nonlinear weak form

                if line_search_bool:

                    def matls1(ts, coors, mode=None, **kwargs):
                        if mode != 'qp' : return
                        x = coors[:, 0]
                        val = abs(x).reshape(-1, 1, 1)
                        return {'val' : val}

                    def matls2(ts, coors, mode=None, phi=None, dphi=None,
                               **kwargs):
                        if mode != 'qp' : return
                        x = coors[:, 0]
                        val = (abs(x)*dphi*phi**(-(npot+2))).reshape(-1, 1, 1)
                        return {'val' : val}

                    terms_ls = [[matls1, 'dw_laplace'],
                                [matls2, 'dw_volume_integrate']]
                    coeffs_ls = [alpha, npot+1]
                    kwargs_ls = {'constcoeffs' : coeffs_ls,
                                 'order' : fem_order,
                                 'dirichlet_bc_facet' : dirichlet_bc,
                                 'verbose' : False}
                    wf_ls = WeakForm('weakcham1_ls', meshfile, terms_ls,
                                     **kwargs_ls) # line-search weak form

                # Stopping Criteria of Newton iterations
                criteria = StopCriteria(max_iter, res_delta_tol=1e-14,
                                        sol_delta_tol=1e-14, min_iter=min_iter)

                # Initial guess
                if kwargs.get('initial_guess') is not None:
                   if kwargs['initial_guess'] == 'min_pot':
                       phi0 = _initialize_minpot(wf, npot)
                   else:
                       phi0 = kwargs['initial_guess']
                elif func_init is not None:
                    phi0 = func_init(wf.field.coors)
                elif analytic_params is not None:
                    phi0 = _analytic_init(wf, analytic_params, coorsys)
                else:
                    phi_avg = np.mean(np.array(phi_bounds))
                    phi0 = phi_avg * np.ones(wf.field.n_nod)

                # Solver instance
                solver_kwargs = {'initial_guess' : phi0,
                                 'relax' : relax,
                                 'bounds' : phi_bounds,
                                 'criteria' : criteria,
                                 'line_search_bool' : line_search_bool,
                                 'verbose' : verbose}
                merge_dicts(solver_kwargs, kwargs)
                wfs_dic = {'weakforms' : [wf], 'wf_res' : wf_nl}
                if line_search_bool:
                    wfs_dic["wf_G"]  = wf_ls
                solver = Solver(wfs_dic, islinear=False, **solver_kwargs)

            elif dim == 3:

                # Linearized Weak Form
                def mat2(ts, coors, mode=None, phi=None, **kwargs):
                    if mode != 'qp' : return
                    val = phi**(-(npot+2))
                    return {'val' : val.reshape(-1, 1, 1)}

                def mat3(ts, coors, mode=None, phi=None, **kwargs):
                    if mode != 'qp' : return
                    val = phi**(-(npot+1))
                    return {'val' : val.reshape(-1, 1, 1)}

                def matrho(ts, coors, mode=None, rho=None, **kwargs):
                    if mode != 'qp' : return
                    if callable(rho): rho = rho(coors)
                    val = rho * np.ones_like(coors[:, 0])
                    return {'val' : val.reshape(-1, 1, 1)}

                coeffs = [alpha, npot+1, -(npot+2)]
                terms = [[None, 'dw_laplace'],
                         [mat2, 'dw_volume_dot'],
                         [mat3, 'dw_volume_integrate']]
                _complete_terms(terms, densities, matrho)
                coeffs += [1.0]*len(densities)
                kwargswf = {'constcoeffs' : coeffs,
                            'unknown_name' : 'u1',
                            'test_name' : 'v1',
                            'order' : fem_order,
                            'integral_name' : 'i1',
                            'domain_name' : 'interior',
                            'densities' : densities,
                            'entity_functions' : ent_func,
                            'dirichlet_bc_facet' : dirichlet_bc,
                            'verbose' : verbose}
                wf = WeakForm('weakcham', meshfile, terms, **kwargswf)

                # Non-Linear Weak Form
                def matnl2(ts, coors, mode=None, phi=None, **kwargs):
                    if mode != 'qp' : return
                    val = (phi**(-(npot+1))).reshape(-1, 1, 1)
                    return {'val' : val}

                def matrho_nl(ts, coors, mode=None, rho=None, **kwargs):
                    if mode != 'qp' : return
                    if callable(rho): rho = rho(coors)
                    val = (rho * np.ones(coors.shape[0])).reshape(-1, 1, 1)
                    return {'val' : val}

                terms_nl = [[None, 'dw_laplace'],
                            [matnl2, 'dw_volume_integrate']]
                _complete_terms(terms_nl, densities, matrho_nl)
                coeffs_nl = [alpha, -1.0] + [1.0]*len(densities)
                kwargs_nl = {'constcoeffs' : coeffs_nl,
                             'order' : fem_order,
                             'densities' : densities,
                             'dirichlet_bc_facet' : dirichlet_bc,
                             'verbose' : False}
                wf_nl = WeakForm('weakcham1_nl', meshfile, terms_nl,
                                 **kwargs_nl) # nonlinear weak form

                if line_search_bool:

                    def matls2(ts, coors, mode=None, phi=None, dphi=None,
                               **kwargs):
                        if mode != 'qp' : return
                        val = (dphi*phi**(-(npot+2))).reshape(-1, 1, 1)
                        return {'val' : val}

                    terms_ls = [[None, 'dw_laplace'],
                                [matls2, 'dw_volume_integrate']]
                    coeffs_ls = [alpha, npot+1]
                    kwargs_ls = {'constcoeffs' : coeffs_ls,
                                 'order' : fem_order,
                                 'dirichlet_bc_facet' : dirichlet_bc,
                                 'verbose' : False}
                    wf_ls = WeakForm('weakcham1_ls', meshfile, terms_ls,
                                     **kwargs_ls) # line-search weak form

                # Stopping Criteria of Newton iterations
                criteria = StopCriteria(max_iter, res_delta_tol=1e-14,
                                        sol_delta_tol=1e-14, min_iter=min_iter)

                # Initial guess
                if kwargs.get('initial_guess') is not None:
                   if kwargs['initial_guess'] == 'min_pot':
                       phi0 = _initialize_minpot(wf, npot)
                   else:
                       phi0 = kwargs['initial_guess']
                elif func_init is not None:
                    phi0 = func_init(wf.field.coors)
                elif analytic_params is not None:
                    phi0 = _analytic_init(wf, analytic_params, coorsys)
                else:
                    phi_avg = np.mean(np.array(phi_bounds))
                    phi0 = phi_avg * np.ones(wf.field.n_nod)

                # Solver instance
                solver_kwargs = {'initial_guess' : phi0,
                                 'relax' : relax,
                                 'bounds' : phi_bounds,
                                 'criteria' : criteria,
                                 'is_bounded' : True,
                                 'line_search_bool' : line_search_bool,
                                 'verbose' : verbose}
                merge_dicts(solver_kwargs, kwargs)
                wfs_dic = {'weakforms' : [wf], 'wf_res' : wf_nl}
                if line_search_bool:
                    wfs_dic["wf_G"]  = wf_ls
                solver = Solver(wfs_dic, islinear=False, **solver_kwargs)

        # Cartesian coordinates
        elif coorsys == 'polar' and dim == 1:

            def mat1(ts, r, mode=None, **kwargs):
                if mode != 'qp' : return
                r = r.squeeze()
                val = r**2
                return {'val' : val.reshape(r.shape[0], 1, 1)}

            def mat2(ts, r, mode=None, phi=None, **kwargs):
                if mode != 'qp' : return
                r = r.squeeze()
                val = r**2 * phi**(-(npot+2))
                return {'val' : val.reshape(r.shape[0], 1, 1)}

            def mat3(ts, r, mode=None, phi=None, **kwargs):
                if mode != 'qp' : return
                r = r.squeeze()
                val = r**2 * phi**(-(npot+1))
                return {'val' : val.reshape(r.shape[0], 1, 1)}

            def mat4(ts, r, mode=None, rho=densities[0], **kwargs):
                if mode != 'qp' : return
                r = r.squeeze()
                if callable(rho): rho = rho(r)
                val = r**2 * rho
                return {'val' : val.reshape(r.shape[0], 1, 1)}

            coeffs = [alpha, npot+1, -(npot+2), 1.0]
            terms = [[mat1, 'dw_laplace'],
                     [mat2, 'dw_volume_dot'],
                     [mat3, 'dw_volume_integrate'],
                     [mat4, 'dw_volume_integrate']]
            kwargswf = {'constcoeffs' : coeffs,
                        'unknown_name' : 'u1',
                        'test_name' : 'v1',
                        'order' : fem_order,
                        'integral_name' : 'i1',
                        'domain_name' : 'interior',
                        'dirichlet_bc_vertex' : dirichlet_bc,
                        'entity_functions' : ent_func,
                        'verbose' : verbose}
            wf = WeakForm(name_weak, meshfile, terms, **kwargswf)

            # Non-linearized equation for strong residual evaluation
            def matnl1(ts, r, mode=None, **kwargs):
                if mode != 'qp' : return
                r = r.squeeze()
                val = r**2
                return {'val' : val.reshape(r.shape[0], 1, 1)}

            def matnl2(ts, r, mode=None, phi=None, **kwargs):
                if mode != 'qp' : return
                r = r.squeeze()
                val = r**2 * phi**(-(npot+1))
                return {'val' : val.reshape(r.shape[0], 1, 1)}

            def matnl3(ts, r, mode=None, rho=densities[0], **kwargs):
                if mode != 'qp' : return
                r = r.squeeze()
                if callable(rho): rho = rho(r)
                val = r**2 * rho
                return {'val' : val.reshape(r.shape[0], 1, 1)}

            terms_nl = [[matnl1, 'dw_laplace'],
                        [matnl2, 'dw_volume_integrate'],
                        [matnl3, 'dw_volume_integrate']]
            coeffs_nl = [alpha, -1.0, 1.0]
            wf_nl_kwargs = {'constcoeffs' : coeffs_nl,
                            'order' : fem_order,
                            'dirichlet_bc_vertex' : dirichlet_bc,
                            'entity_functions' : ent_func,
                            'verbose' : False}
            wf_nl = WeakForm(name_weak_nl, meshfile, terms_nl,
                             **wf_nl_kwargs) # nonlinear weak form

            if line_search_bool:

                def matls1(ts, r, mode=None, **kwargs):
                    if mode != 'qp' : return
                    r = r.squeeze()
                    val = r**2
                    return {'val' : val.reshape(r.shape[0], 1, 1)}

                def matls2(ts, r, mode=None, phi=None, dphi=None, **kwargs):
                    if mode != 'qp' : return
                    r = r.squeeze()
                    val = r**2 * dphi * phi**(-(npot+2))
                    return {'val' : val.reshape(r.shape[0], 1, 1)}

                terms_ls = [[matls1, 'dw_laplace'],
                            [matls2, 'dw_volume_integrate']]
                coeffs_ls = [alpha, npot+1]
                kwargs_ls = {'constcoeffs' : coeffs_ls,
                             'order' : fem_order,
                             'dirichlet_bc_vertex' : dirichlet_bc,
                             'verbose' : False}
                wf_ls = WeakForm('weakcham1_ls', meshfile, terms_ls,
                                 **kwargs_ls) # line-search weak form

            # Stopping Criteria of Newton iterations
            criteria = StopCriteria(max_iter, res_delta_tol=1e-14,
                                    sol_delta_tol=1e-14, min_iter=min_iter)

            # Initial guess
            if kwargs.get('initial_guess') is not None:
               if kwargs['initial_guess'] == 'min_pot':
                   phi0 = _initialize_minpot(wf, npot, backup=densities[0])
               else:
                   phi0 = kwargs['initial_guess']
            elif func_init is not None:
                phi0 = func_init(wf.field.coors)
            elif analytic_params is not None:
                phi0 = _analytic_init(wf, analytic_params, coorsys)
            else:
                phi_avg = np.mean(np.array(phi_bounds))
                phi0 = phi_avg * np.ones(wf.field.n_nod)

            # Solver instance
            solver_kwargs = {'initial_guess' : phi0,
                             'relax' : relax,
                             'bounds' : phi_bounds,
                             'criteria' : criteria,
                             'is_bounded' : True,
                             'line_search_bool' : line_search_bool,
                             'verbose' : verbose}
            merge_dicts(solver_kwargs, kwargs)
            wfs_dic = {'weakforms' : [wf], 'wf_res' : wf_nl}
            if line_search_bool:
                wfs_dic["wf_G"]  = wf_ls
            solver = Solver(wfs_dic, islinear=False, **solver_kwargs)

        elif coorsys == 'polar' and dim == 2:
            def mat1(ts, coors, mode=None, **kwargs):
                if mode != 'qp' : return
                val = np.zeros((coors.shape[0], 2, 2))
                r, theta = coors[:, 0], coors[:, 1]
                val[:, 0, 0] = sin(theta)*r**2
                val[:, 1, 1] = sin(theta)
                return {'val' : val}

            def mat2(ts, coors, mode=None, phi=None, **kwargs):
                if mode != 'qp' : return
                r, theta = coors[:, 0], coors[:, 1]
                val = sin(theta) * r**2 * phi**(-(npot+2))
                return {'val' : val.reshape(-1, 1, 1)}

            def mat3(ts, coors, mode=None, phi=None, **kwargs):
                if mode != 'qp' : return
                r, theta = coors[:, 0], coors[:, 1]
                val = sin(theta) * r**2 * phi**(-(npot+1))
                return {'val' : val.reshape(-1, 1, 1)}

            def matrho(ts, coors, mode=None, rho=None, **kwargs):
                if mode != 'qp' : return
                if callable(rho) : rho = rho(coors)
                r, theta = coors[:, 0], coors[:, 1]
                val = sin(theta) * r**2 * rho
                return {'val' : val.reshape(-1, 1, 1)}

            coeffs = [alpha, npot+1, -(npot+2)]
            terms = [[mat1, 'dw_diffusion'],
                     [mat2, 'dw_volume_dot'],
                     [mat3, 'dw_volume_integrate']]
            _complete_terms(terms, densities, matrho)
            coeffs += [1.0]*len(densities)

            kwargswf = {'constcoeffs' : coeffs,
                        'unknown_name' : 'u1',
                        'test_name' : 'v1',
                        'order' : fem_order,
                        'integral_name' : 'i1',
                        'domain_name' : 'interior',
                        'densities' : densities,
                        'dirichlet_bc_facet' : dirichlet_bc,
                        'entity_functions' : ent_func,
                        'verbose' : verbose}
            wf = WeakForm(name_weak, meshfile, terms, **kwargswf)

            # Non-linearized equation for strong residual evaluation
            def matnl1(ts, coors, mode=None, **kwargs):
                if mode != 'qp' : return
                r, theta = coors[:, 0], coors[:, 1]
                val = np.zeros((coors.shape[0], 2, 2))
                val[:, 0, 0] = sin(theta) * r**2
                val[:, 1, 1] = sin(theta)
                return {'val' : val}

            def matnl2(ts, coors, mode=None, phi=None, **kwargs):
                if mode != 'qp' : return
                r, theta = coors[:, 0], coors[:, 1]
                val = (sin(theta) * r**2 * phi**(-(npot+1))).reshape(-1, 1, 1)
                return {'val' : val}

            def matnlrho(ts, coors, mode=None, rho=None, **kwargs):
                if mode != 'qp' : return
                if callable(rho): rho = rho(coors)
                r, theta = coors[:, 0], coors[:, 1]
                val = (sin(theta) * r**2 * rho).reshape(-1, 1, 1)
                return {'val' : val}

            terms_nl = [[matnl1, 'dw_diffusion'],
                        [matnl2, 'dw_volume_integrate']]
            coeffs_nl = [alpha, -1.0]
            _complete_terms(terms_nl, densities, matnlrho)
            coeffs_nl += [1.0]*len(densities)
            wf_nl_kwargs = {'constcoeffs' : coeffs_nl,
                            'order' : fem_order,
                            'densities' : densities,
                            'dirichlet_bc_facet' : dirichlet_bc,
                            'verbose' : False}
            wf_nl = WeakForm(name_weak_nl, meshfile, terms_nl,
                             **wf_nl_kwargs) # nonlinear weak form

            # Stopping Criteria of Newton iterations
            criteria = StopCriteria(max_iter, res_delta_tol=1e-14,
                                    sol_delta_tol=1e-14, min_iter=min_iter)

            # Initial guess
            if kwargs.get('initial_guess') is not None:
               if kwargs['initial_guess'] == 'min_pot':
                   phi0 = _initialize_minpot(wf, npot, backup=densities[0])
               else:
                   phi0 = kwargs['initial_guess']
            elif func_init is not None:
                phi0 = func_init(wf.field.coors)
            elif analytic_params is not None:
                phi0 = _analytic_init(wf, analytic_params, coorsys)
            else:
                phi_avg = np.mean(np.array(phi_bounds))
                phi0 = phi_avg * np.ones(wf.field.n_nod)

            # Solver instance
            solver_kwargs = {'initial_guess' : phi0,
                             'relax' : relax,
                             'bounds' : phi_bounds,
                             'criteria' : criteria,
                             'is_bounded' : True,
                             'line_search_bool' : line_search_bool,
                             'verbose' : verbose}
            merge_dicts(solver_kwargs, kwargs)
            wfs_dic = {'weakforms' : [wf], 'wf_res' : wf_nl}
            solver = Solver(wfs_dic, islinear=False, **solver_kwargs)

        elif coorsys == 'polar_mu' and dim == 2:
            def mat1(ts, coors, mode=None, **kwargs):
                if mode != 'qp' : return
                val = np.zeros((coors.shape[0], 2, 2))
                r, mu = coors[:, 0], coors[:, 1]
                val[:, 0, 0] = r**2
                val[:, 1, 1] = 1 - mu**2
                return {'val' : val}

            def mat2(ts, coors, mode=None, phi=None, **kwargs):
                if mode != 'qp' : return
                r = coors[:, 0]
                val = r**2 * phi**(-(npot+2))
                return {'val' : val.reshape(-1, 1, 1)}

            def mat3(ts, coors, mode=None, phi=None, **kwargs):
                if mode != 'qp' : return
                r = coors[:, 0]
                val = r**2 * phi**(-(npot+1))
                return {'val' : val.reshape(-1, 1, 1)}

            def matrho(ts, coors, mode=None, rho=None, **kwargs):
                if mode != 'qp' : return
                if callable(rho) : rho = rho(coors)
                r = coors[:, 0]
                val = r**2 * rho
                return {'val' : val.reshape(-1, 1, 1)}

            coeffs = [alpha, npot+1, -(npot+2)]
            terms = [[mat1, 'dw_diffusion'],
                     [mat2, 'dw_volume_dot'],
                     [mat3, 'dw_volume_integrate']]
            _complete_terms(terms, densities, matrho)
            coeffs += [1.0]*len(densities)

            kwargswf = {'constcoeffs' : coeffs,
                        'unknown_name' : 'u1',
                        'test_name' : 'v1',
                        'order' : fem_order,
                        'integral_name' : 'i1',
                        'domain_name' : 'interior',
                        'densities' : densities,
                        'dirichlet_bc_facet' : dirichlet_bc,
                        'entity_functions' : ent_func,
                        'verbose' : verbose}
            wf = WeakForm(name_weak, meshfile, terms, **kwargswf)

            # Non-linearized equation for strong residual evaluation
            def matnl1(ts, coors, mode=None, **kwargs):
                if mode != 'qp' : return
                r, mu = coors[:, 0], coors[:, 1]
                val = np.zeros((coors.shape[0], 2, 2))
                val[:, 0, 0] = r**2
                val[:, 1, 1] = 1 - mu**2
                return {'val' : val}

            def matnl2(ts, coors, mode=None, phi=None, **kwargs):
                if mode != 'qp' : return
                r = coors[:, 0]
                val = (r**2 * phi**(-(npot+1))).reshape(-1, 1, 1)
                return {'val' : val}

            def matnlrho(ts, coors, mode=None, rho=None, **kwargs):
                if mode != 'qp' : return
                if callable(rho): rho = rho(coors)
                r = coors[:, 0]
                val = (r**2 * rho).reshape(-1, 1, 1)
                return {'val' : val}

            terms_nl = [[matnl1, 'dw_diffusion'],
                        [matnl2, 'dw_volume_integrate']]
            coeffs_nl = [alpha, -1.0]
            _complete_terms(terms_nl, densities, matnlrho)
            coeffs_nl += [1.0]*len(densities)
            wf_nl_kwargs = {'constcoeffs' : coeffs_nl,
                            'order' : fem_order,
                            'densities' : densities,
                            'dirichlet_bc_facet' : dirichlet_bc,
                            'verbose' : False}
            wf_nl = WeakForm(name_weak_nl, meshfile, terms_nl,
                             **wf_nl_kwargs) # nonlinear weak form

            # Stopping Criteria of Newton iterations
            criteria = StopCriteria(max_iter, res_delta_tol=1e-14,
                                    sol_delta_tol=1e-14, min_iter=min_iter)

            # Initial guess
            if kwargs.get('initial_guess') is not None:
                if kwargs['initial_guess'] == 'min_pot':
                    phi0 = _initialize_minpot(wf, npot)
                else:
                    phi0 = kwargs['initial_guess']
            elif func_init is not None:
                phi0 = func_init(wf.field.coors)
            elif analytic_params is not None:
                phi0 = _analytic_init(wf, analytic_params, coorsys)
            else:
                phi_avg = np.mean(np.array(phi_bounds))
                phi0 = phi_avg * np.ones(wf.field.n_nod)

            # Solver instance
            solver_kwargs = {'initial_guess' : phi0,
                             'relax' : relax,
                             'bounds' : phi_bounds,
                             'criteria' : criteria,
                             'is_bounded' : True,
                             'line_search_bool' : line_search_bool,
                             'verbose' : verbose}
            merge_dicts(solver_kwargs, kwargs)
            wfs_dic = {'weakforms' : [wf], 'wf_res' : wf_nl}
            solver = Solver(wfs_dic, islinear=False, **solver_kwargs)

        else:
            raise Exception("Not implemented coordinates system: %s" %coorsys)

        self.solver = solver

    def solve(self, **kwargs):
        """Wrapper for Solver.solve method"""
        self.solver.solve(**kwargs)

    def save(self, grad=False, **kwargs):
        """Wrapper for Solver.save method"""
        self.solver.save(grad=grad, **kwargs)

    def plot(self, **kwargs):
        """Wrapper for pyvista_plot"""
        val_renorm = kwargs.get('val_renorm', 1.0)
        grad_renorm = kwargs.get('grad_renorm', 1.0)
        out = self.solver.save(grad=True, val_renorm=val_renorm,
                               grad_renorm=grad_renorm)
        from femtoscope.display.fea_plot import pyvista_plot
        for result_file in out:
            pyvista_plot(result_file, **kwargs)


class ChameleonSplit():
    r"""
    Class for solving the Klein-Gordon equation
    $$ \alpha \Delta \phi = \rho - \phi^{-(n+1)} $$
    on unbounded domains via domain splitting + kelvin inversion.

    The field's behaviour is only known infinitely far away from the sources
    (where the approaches the value that minimizes the effective potential in
    vacuum while its gradient vanishes).

    Attributes
    ----------
    Rcut : float
        Radius of the interior domain (disk in 2d / sphere in 3d).
    alpha : float
        Physical parameter weighting the laplacian operator of the Klein-Gordon
        equation (dimensionless).
    npot : int
        Exponent (parameter of the chameleon model).
    solver : Solver instance
        The FEM solver instance to be run.

    """

    def __init__(self, alpha, npot, densities_int, densities_ext, rho_bounds,
                 meshfiles, coorsys='cartesian', **kwargs):
        """
        Construct an ChameleonSplit problem instance.

        Parameters
        ----------
        alpha : float
            Physical parameter weighting the laplacian operator of the
            Klein-Gordon equation (dimensionless).
        npot : int
            Exponent (parameter of the chameleon model).
        densities_int : list
            List of density functions or constants for the interior domain.
            The length of this list must match the number of sub*domains in the
            mesh.
        densities_ext : list
            List of density functions or constants for the exterior domain.
            Thelength of this list must match the number of sub*domains in the
            mesh.
        rho_bounds : list
            List of length 2 containing the min & max values of the density in
            the whole simulation space.
        meshfiles : list
            List of files' name.
        coorsys : str, optional
            The set of coordinates to be used. The default is 'cartesian'.

        Other Parameters
        ----------------
        mesh_dir : str
            Directory where the mesh files are located. The default is None and
            in which case the mesh file is sought in the `MESH_DIR` directory.
        conn : str
            Method for linking the interior domain with the exterior domain.
            The default is 'connected'.
        fem_order : int
            The FE approximation order. The default is 2.
        func_init : func
            Function for initializing the chameleon field profile.
            The default is None.
        analytic_params : dict
            Dictionary containing the relevant arguements of function
            `chameleon_radial`. The default is None.
        entity_functions : list of 2-tuple
            List of tuples (dim, function) for manual entity selection.
            The default is [].
        verbose : bool
            Display user's information. The default is False.

        """

        if not isinstance(meshfiles, list): meshfiles = [meshfiles]
        if not isinstance(densities_int, list): densities_int = [densities_int]
        if not isinstance(densities_ext, list): densities_ext = [densities_ext]
        mesh_dir = kwargs.get('mesh_dir', None)
        if type(meshfiles[0]) == str:
            dim = get_meshdim(meshfiles[0], mesh_dir=mesh_dir)
        else: # 1D meshes are not saved as VTK files
            dim = 1
        if 'Rcut' in kwargs.keys():
            Rcut = kwargs['Rcut']
        else:
            Rcut = get_rcut(meshfiles[0], mesh_dir=mesh_dir, coorsys=coorsys)
        self.Rcut = Rcut
        conn = kwargs.get('conn', 'connected')
        ent_func_in = kwargs.get('entity_functions_in', [])
        ent_func_out = kwargs.get('entity_functions_out', [])
        fem_order = kwargs.get('fem_order', 2)
        relax = kwargs.get('relax', 0.9)
        analytic_params = kwargs.get('analytic_params', None)
        if analytic_params is not None:
            analytic_params['alpha'] = alpha
            analytic_params['n'] = npot
        func_init = kwargs.get('func_init', None)
        verbose = kwargs.get('verbose', True)
        self.alpha = alpha
        self.npot = npot
        rho_vac = densities_ext[-1]
        phi_bounds = [p**(-1/(npot+1)) for p in rho_bounds]
        phi_bounds.sort()
        phi_vac = phi_bounds[-1]

        name_weak_in = "wfin"
        name_weak_out = "wfout"
        name_weak_nl = "wfnl"

        min_iter = kwargs.get('min_iter')
        max_iter = kwargs.get('max_iter')
        line_search_bool = kwargs.get('line_search_bool', False)

        # Cartesian coordinates
        if coorsys == 'cartesian':

            if dim == 2:

                def matint1(ts, coors, mode=None, **kwargs):
                    if mode != 'qp' : return
                    x = coors[:, 0]
                    val = abs(x)
                    return {'val' : val.reshape(-1, 1, 1)}

                def matint2(ts, coors, mode=None, phi=None, **kwargs):
                    if mode != 'qp' : return
                    x = coors[:, 0]
                    val = abs(x) * phi**(-(npot+2))
                    return {'val' : val.reshape(-1, 1, 1)}

                def matint3(ts, coors, mode=None, phi=None, **kwargs):
                    if mode != 'qp' : return
                    x = coors[:, 0]
                    val = abs(x) * phi**(-(npot+1))
                    return {'val' : val.reshape(-1, 1, 1)}

                def matrho(ts, coors, mode=None, rho=None, **kwargs):
                    if mode != 'qp' : return
                    if callable(rho): rho = rho(coors)
                    x = coors[:, 0]
                    val = abs(x) * rho
                    return {'val' : val.reshape(-1, 1, 1)}

                coeffsint = [alpha, npot+1, -(npot+2)]
                termsint = [[matint1, 'dw_laplace'],
                            [matint2, 'dw_volume_dot'],
                            [matint3, 'dw_volume_integrate']]
                _complete_terms(termsint, densities_int, matrho)
                coeffsint += [1.0]*(len(densities_int)-densities_int.count(0))

                def matext1(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                    if mode != 'qp' : return
                    xi, eta = coors[:, 0], coors[:, 1]
                    norm2 = xi**2 + eta**2
                    norm = sqrt(norm2)
                    val = abs(xi) * norm2**2/Rcut**4 * (7 - 6*norm/Rcut)
                    return {'val' : val.reshape(-1, 1, 1)}

                def matext2(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                    if mode != 'qp' : return
                    val = np.zeros((coors.shape[0], 2, 1))
                    xi, eta = coors[:, 0], coors[:, 1]
                    norm2 = xi**2 + eta**2
                    norm = sqrt(norm2)
                    val[:, 0, 0] = abs(xi)*42*norm2/Rcut**4*(1-norm/Rcut) * xi
                    val[:, 1, 0] = abs(xi)*42*norm2/Rcut**4*(1-norm/Rcut) * eta
                    return {'val' : val}

                def matext3(ts, coors, mode=None, phi=None, Rcut=Rcut,
                            **kwargs):
                    if mode != 'qp' : return
                    xi, eta = coors[:, 0], coors[:, 1]
                    norm = sqrt(xi**2 + eta**2)
                    val = abs(xi) * (7 - 6*norm/Rcut) * phi**(-(npot+2))
                    return {'val' : val.reshape(-1, 1, 1)}

                def matext4(ts, coors, mode=None, phi=None, Rcut=Rcut,
                            **kwargs):
                    if mode != 'qp' : return
                    xi, eta = coors[:, 0], coors[:, 1]
                    norm = sqrt(xi**2 + eta**2)
                    val = abs(xi) * (7 - 6*norm/Rcut) * phi**(-(npot+1))
                    return {'val' : val.reshape(-1, 1, 1)}

                def matext5(ts, coors, mode=None, rho_vac=rho_vac, Rcut=Rcut,
                            **kwargs):
                    if mode != 'qp' : return
                    xi, eta = coors[:, 0], coors[:, 1]
                    norm = sqrt(xi**2 + eta**2)
                    val = abs(xi) * (7 - 6*norm/Rcut) * rho_vac
                    return {'val' : val.reshape(-1, 1, 1)}

                coeffsext = [alpha, alpha, npot+1, -(npot+2), 1.0]
                termsext = [[matext1, 'dw_laplace'],
                            [matext2, 'dw_s_dot_mgrad_s'],
                            [matext3, 'dw_volume_dot'],
                            [matext4, 'dw_volume_integrate'],
                            [matext5, 'dw_volume_integrate']]

                kwargsint = {'constcoeffs' : coeffsint,
                            'unknown_name' : 'u1',
                            'test_name' : 'v1',
                            'order' : fem_order,
                            'integral_name' : 'i1',
                            'domain_name' : 'interior',
                            'densities' : densities_int,
                            'entity_functions' : ent_func_in,
                            'verbose' : verbose}

                if conn == 'ping-pong': # add Dirichlet BC at Rcut
                    kwargsint['dirichlet_bc_facet'] = [phi_vac]

                # interior weak form
                wf1 = WeakForm('weakchamin', meshfiles[0], termsint,
                               **kwargsint) # interior weak form

                def matnl(ts, coors, mode=None, phi=None, **kwargs):
                    if mode != 'qp' : return
                    x = coors[:, 0]
                    val = (abs(x) * phi**(-(npot+1))).reshape(-1, 1, 1)
                    return {'val' : val}
                termsnl = [[matint1, 'dw_laplace'],
                           [matnl, 'dw_volume_integrate']]
                _complete_terms(termsnl, densities_int, matrho)
                coeffs_nl = [alpha, -1.0] \
                    + [1.0]*(len(densities_int)-densities_int.count(0))
                kwargsnl = {'constcoeffs' : coeffs_nl,
                             'order' : fem_order,
                             'densities' : densities_int,
                             'verbose' : False}
                wf_nl = WeakForm('weakcham1_nl', meshfiles[0], termsnl,
                                 **kwargsnl) # nonlinear weak form

                if line_search_bool:
                    def matls(ts, coors, mode=None, phi=None, dphi=None,
                               **kwargs):
                        if mode != 'qp' : return
                        x = coors[:, 0]
                        val = (abs(x)*dphi*phi**(-(npot+2))).reshape(-1, 1, 1)
                        return {'val' : val}
                    terms_ls = [[matint1, 'dw_laplace'],
                                [matls, 'dw_volume_integrate']]
                    coeffs_ls = [alpha, npot+1]
                    kwargs_ls = {'constcoeffs' : coeffs_ls,
                                 'order' : fem_order,
                                 'verbose' : False}
                    wf_ls = WeakForm('weakcham1_ls', meshfiles[0], terms_ls,
                                     **kwargs_ls) # line-search weak form

                vertex_bc = [phi_vac]
                kwargsext = {'constcoeffs' : coeffsext,
                             'unknown_name' : 'u2',
                             'test_name' : 'v2',
                             'order' : fem_order,
                             'integral_name' : 'i2',
                             'domain_name' : 'exterior',
                             'dirichlet_bc_vertex' : vertex_bc,
                             'entity_functions' : ent_func_out,
                             'verbose' : verbose}

                wf2 = WeakForm('weakchamext', meshfiles[1], termsext,
                               **kwargsext) # exterior weak form

                # Specify the region of dimension D-1 shared by the two meshes
                cogammas = [wf1.facets[-1], wf2.facets[0]]
                criteria = StopCriteria(max_iter, res_delta_tol=1e-14,
                                        sol_delta_tol=1e-14, min_iter=min_iter)

                if kwargs.get('initial_guess') is not None:
                   if kwargs['initial_guess'] == 'min_pot':
                       initial_guess = [_initialize_minpot(wf1, npot),
                                        _initialize_minpot(wf2, npot,
                                                           backup=rho_vac)]
                   else:
                       initial_guess = kwargs['initial_guess']
                elif func_init is not None:
                    phi0_in = func_init[0](wf1.field.coors)
                    phi0_out = func_init[1](wf2.field.coors)
                    initial_guess = [phi0_in, phi0_out]
                elif analytic_params is None:
                    phi0_in = np.mean(np.array(phi_bounds)) \
                        * np.ones(wf1.field.n_nod)
                    phi0_out = phi_vac * np.ones(wf2.field.n_nod)
                    initial_guess = [phi0_in, phi0_out]
                elif type(analytic_params==dict):
                    initial_guess = _analytic_init([wf1, wf2], analytic_params,
                                                   coorsys, Rcut=Rcut)
                else:
                    raise Exception("Cannot initialize the field!")

                solver_kwargs = {'initial_guess' : initial_guess,
                                 'relax' : relax,
                                 'bounds' : phi_bounds,
                                 'criteria' : criteria,
                                 'conn' : conn,
                                 'cogammas' : cogammas,
                                 'is_bounded' : False,
                                 'line_search_bool' : line_search_bool,
                                 'verbose' : verbose}
                merge_dicts(solver_kwargs, kwargs)
                wfs_dic = {'weakforms' : [wf1, wf2], 'wf_res' : wf_nl}
                if line_search_bool:
                    wfs_dic["wf_G"]  = wf_ls
                solver = Solver(wfs_dic, islinear=False, **solver_kwargs)

                if conn == 'ping-pong':
                    # Add the Neumann term to the exterior problem
                    def neumann(ts, coors, mode=None, solver=solver, **kwargs):
                        if mode != 'qp' : return
                        X = coors[:, 0]
                        wf1 = solver.weakforms[0]
                        sol1 = solver.sols[0]
                        if sol1 is None:
                            sol1 = np.zeros(coors.shape[0])
                        grad1 = wf1.field.evaluate_at(
                            coors, sol1[:, np.newaxis], mode='grad').squeeze()
                        n1 = coors/np.linalg.norm(coors, axis=1)[:, np.newaxis]
                        flux = np.empty_like(X)
                        for i in range(len(flux)):
                            flux[i] = np.dot(grad1[i], n1[i])
                        flux *= abs(X)
                        flux = flux.reshape(-1, 1, 1)
                        return {'val' : flux}
                    neumann_term = [neumann, 'dw_surface_integrate',
                                    cogammas[1].name]
                    wf2.add_term(neumann_term, update=True, newcoeff=alpha)

            # 3D FEM simulations only available in cartesian coordinates
            elif dim == 3:

                def matint2(ts, coors, mode=None, phi=None, **kwargs):
                    if mode != 'qp' : return
                    val = phi**(-(npot+2))
                    return {'val' : val.reshape(-1, 1, 1)}

                def matint3(ts, coors, mode=None, phi=None, **kwargs):
                    if mode != 'qp' : return
                    val = phi**(-(npot+1))
                    return {'val' : val.reshape(-1, 1, 1)}

                def matrho(ts, coors, mode=None, rho=None, **kwargs):
                    if mode != 'qp' : return
                    if callable(rho):
                        val = rho(coors)
                    else:
                        val = rho * np.ones(coors.shape[0])
                    return {'val' : val.reshape(-1, 1, 1)}

                coeffsint = [alpha, npot+1, -(npot+2)]
                termsint = [[None, 'dw_laplace'],
                            [matint2, 'dw_volume_dot'],
                            [matint3, 'dw_volume_integrate']]
                _complete_terms(termsint, densities_int, matrho)
                coeffsint += [1.0]*len(densities_int)

                def matext1(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                    if mode != 'qp' : return
                    xi, eta, zeta = coors[:, 0], coors[:, 1], coors[:, 2]
                    norm2 = xi**2 + eta**2 + zeta**2
                    norm = sqrt(norm2)
                    val = norm2**2/Rcut**4 * (7 - 6*norm/Rcut)
                    return {'val' : val.reshape(-1, 1, 1)}

                def matext2(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                    if mode != 'qp' : return
                    val = np.zeros((coors.shape[0], 3, 1))
                    xi, eta, zeta = coors[:, 0], coors[:, 1], coors[:, 2]
                    norm2 = xi**2 + eta**2 + zeta**2
                    norm = sqrt(norm2)
                    val[:, 0, 0] = 42*norm2/Rcut**4 * (1 - norm/Rcut) * xi
                    val[:, 1, 0] = 42*norm2/Rcut**4 * (1 - norm/Rcut) * eta
                    val[:, 2, 0] = 42*norm2/Rcut**4 * (1 - norm/Rcut) * zeta
                    return {'val' : val}

                def matext3(ts, coors, mode=None, phi=None, Rcut=Rcut,
                            **kwargs):
                    if mode != 'qp' : return
                    xi, eta, zeta = coors[:, 0], coors[:, 1], coors[:, 2]
                    norm = sqrt(xi**2 + eta**2 + zeta**2)
                    val = (7 - 6*norm/Rcut) * phi**(-(npot+2))
                    return {'val' : val.reshape(-1, 1, 1)}

                def matext4(ts, coors, mode=None, phi=None, Rcut=Rcut,
                            **kwargs):
                    if mode != 'qp' : return
                    xi, eta, zeta = coors[:, 0], coors[:, 1], coors[:, 2]
                    norm = sqrt(xi**2 + eta**2 + zeta**2)
                    val = (7 - 6*norm/Rcut) * phi**(-(npot+1))
                    return {'val' : val.reshape(-1, 1, 1)}

                def matext5(ts, coors, mode=None, rho_vac=rho_vac, Rcut=Rcut,
                            **kwargs):
                    if mode != 'qp' : return
                    xi, eta, zeta = coors[:, 0], coors[:, 1], coors[:, 2]
                    norm = sqrt(xi**2 + eta**2 + zeta**2)
                    val = (7 - 6*norm/Rcut) * rho_vac
                    return {'val' : val.reshape(-1, 1, 1)}

                coeffsext = [alpha, alpha, npot+1, -(npot+2), 1.0]
                termsext = [[matext1, 'dw_laplace'],
                            [matext2, 'dw_s_dot_mgrad_s'],
                            [matext3, 'dw_volume_dot'],
                            [matext4, 'dw_volume_integrate'],
                            [matext5, 'dw_volume_integrate']]

                kwargsint = {'constcoeffs' : coeffsint,
                            'unknown_name' : 'u1',
                            'test_name' : 'v1',
                            'order' : fem_order,
                            'integral_name' : 'i1',
                            'domain_name' : 'interior',
                            'densities' : densities_int,
                            'entity_functions' : ent_func_in,
                            'verbose' : verbose}

                def matnl(ts, coors, mode=None, phi=None, **kwargs):
                    if mode != 'qp' : return
                    val = (phi**(-(npot+1))).reshape(-1, 1, 1)
                    return {'val' : val}
                termsnl = [[None, 'dw_laplace'],
                           [matnl, 'dw_volume_integrate']]
                _complete_terms(termsnl, densities_int, matrho)
                coeffsnl = [alpha, -1.0] \
                    + [1.0]*(len(densities_int)-densities_int.count(0))
                kwargsnl = {'constcoeffs' : coeffsnl,
                             'order' : fem_order,
                             'densities' : densities_int,
                             'verbose' : False}
                wf_nl = WeakForm('weakcham1_nl', meshfiles[0], termsnl,
                                 **kwargsnl) # nonlinear weak form

                if conn == 'ping-pong': # add Dirichlet BC at Rcut
                    kwargsint['dirichlet_bc_facet'] = [phi_vac]

                wf1 = WeakForm('weakchamin', meshfiles[0], termsint,
                               **kwargsint) # interior weak form

                vertex_bc = [phi_vac]
                kwargsext = {'constcoeffs' : coeffsext,
                            'unknown_name' : 'u2',
                            'test_name' : 'v2',
                            'order' : fem_order,
                            'integral_name' : 'i2',
                            'domain_name' : 'exterior',
                            'dirichlet_bc_vertex' : vertex_bc,
                            'entity_functions' : ent_func_out,
                            'verbose' : verbose}
                wf2 = WeakForm('weakchamout', meshfiles[1], termsext,
                               **kwargsext) # exterior weak form

                # Specify the region of dimension D-1 shared by the two meshes
                cogammas = [wf1.facets[-1], wf2.facets[0]]
                criteria = StopCriteria(max_iter, res_delta_tol=1e-14,
                                        sol_delta_tol=1e-14, min_iter=min_iter)

                if kwargs.get('initial_guess') is not None:
                   if kwargs['initial_guess'] == 'min_pot':
                       initial_guess = [_initialize_minpot(wf1, npot),
                                        _initialize_minpot(wf2, npot,
                                                           backup=rho_vac)]
                   else:
                       initial_guess = kwargs['initial_guess']
                elif func_init is not None:
                    phi0_in = func_init[0](wf1.field.coors)
                    phi0_out = func_init[1](wf2.field.coors)
                    initial_guess = [phi0_in, phi0_out]
                elif analytic_params is None:
                    phi0_in = np.mean(np.array(phi_bounds)) \
                        * np.ones(wf1.field.n_nod)
                    phi0_out = phi_vac * np.ones(wf2.field.n_nod)
                    initial_guess = [phi0_in, phi0_out]
                elif type(analytic_params==dict):
                    initial_guess = _analytic_init([wf1, wf2], analytic_params,
                                                   coorsys, Rcut=Rcut)
                else:
                    raise Exception("Cannot initialize the field!")

                solver_kwargs = {'initial_guess' : initial_guess,
                                 'relax' : relax,
                                 'bounds' : phi_bounds,
                                 'criteria' : criteria,
                                 'conn' : conn,
                                 'cogammas' : cogammas,
                                 'is_bounded' : False,
                                 'line_search_bool' : line_search_bool,
                                 'verbose' : verbose}
                merge_dicts(solver_kwargs, kwargs)
                wfs_dic = {'weakforms' : [wf1, wf2], 'wf_res' : wf_nl}
                solver = Solver(wfs_dic, islinear=False, **solver_kwargs)

                if conn == 'ping-pong':
                    # Add the Neumann term to the exterior problem
                    def neumann(ts, coors, mode=None, solver=solver, **kwargs):
                        if mode != 'qp' : return
                        wf1 = solver.weakforms[0]
                        sol1 = solver.sols[0]
                        if sol1 is None:
                            sol1 = np.zeros(coors.shape[0])
                        grad1 = wf1.field.evaluate_at(
                            coors, sol1[:, np.newaxis], mode='grad').squeeze()
                        n1 = coors/np.linalg.norm(coors, axis=1)[:, np.newaxis]
                        flux = np.empty_like(coors[:, 0])
                        for i in range(len(flux)):
                            flux[i] = np.dot(grad1[i], n1[i])
                        flux = flux.reshape(-1, 1, 1)
                        return {'val' : flux}
                    neumann_term = [neumann, 'dw_surface_integrate',
                                    cogammas[1].name]
                    wf2.add_term(neumann_term, update=True, newcoeff=alpha)

            else:
                raise Exception("Dimension %d is not valid" %dim)

        # Polar coordinates
        elif coorsys == 'polar' and dim == 1:

            def matint1(ts, r, mode=None, **kwargs):
                if mode != 'qp' : return
                r = r.squeeze()
                val = r**2
                return {'val' : val.reshape(r.shape[0], 1, 1)}

            def matint2(ts, r, mode=None, phi=None, Rcut=Rcut, **kwargs):
                if mode != 'qp' : return
                r = r.squeeze()
                val = r**2 * phi**(-(npot+2))
                return {'val' : val.reshape(r.shape[0], 1, 1)}

            def matint3(ts, r, mode=None, phi=None, Rcut=Rcut, **kwargs):
                if mode != 'qp' : return
                r = r.squeeze()
                val = r**2 * phi**(-(npot+1))
                return {'val' : val.reshape(r.shape[0], 1, 1)}

            def matrho(ts, r, mode=None, Rcut=Rcut, rho=densities_int[0],
                       **kwargs):
                if mode != 'qp' : return
                r = r.squeeze()
                if callable(rho): rho = rho(r)
                val = r**2 * rho
                return {'val' : val.reshape(r.shape[0], 1, 1)}

            def matext1(ts, eta, mode=None, Rcut=Rcut, **kwargs):
                if mode != 'qp' : return
                eta = eta.squeeze()
                val = eta**4/Rcut**2 * (5 - 4*eta/Rcut)
                return {'val' : val.reshape(eta.shape[0], 1, 1)}

            def matext2(ts, eta, mode=None, Rcut=Rcut, **kwargs):
                if mode != 'qp' : return
                eta = eta.squeeze()
                val = 20*eta**3/Rcut**2 * (1 - eta/Rcut)
                return {'val' : val.reshape(eta.shape[0], 1, 1)}

            def matext3(ts, eta, mode=None, phi=None, **kwargs):
                if mode != 'qp' : return
                eta = eta.squeeze()
                val = Rcut**2 * (5 - 4*eta/Rcut) * phi**(-(npot+2))
                return {'val' : val.reshape(eta.shape[0], 1, 1)}

            def matext4(ts, eta, mode=None, phi=None, **kwargs):
                if mode != 'qp' : return
                eta = eta.squeeze()
                val = Rcut**2 * (5 - 4*eta/Rcut) * phi**(-(npot+1))
                return {'val' : val.reshape(eta.shape[0], 1, 1)}

            def matext5(ts, eta, mode=None, rho_vac=rho_vac, **kwargs):
                if mode != 'qp' : return
                eta = eta.squeeze()
                val = Rcut**2 * (5 - 4*eta/Rcut) * rho_vac
                return {'val' : val.reshape(eta.shape[0], 1, 1)}

            termsint = [[matint1, 'dw_laplace'],
                        [matint2, 'dw_volume_dot'],
                        [matint3, 'dw_volume_integrate'],
                        [matrho, 'dw_volume_integrate']]
            coeffsint = [alpha, (npot+1), -(npot+2), 1.0]
            termsext = [[matext1, 'dw_laplace'],
                        [matext2, 'dw_s_dot_mgrad_s'],
                        [matext3, 'dw_volume_dot'],
                        [matext4, 'dw_volume_integrate'],
                        [matext5, 'dw_volume_integrate']]
            coeffsext = [alpha, alpha, (npot+1), -(npot+2), 1.0]

            # Non-linearized equation for strong residual evaluation
            def matnl(ts, r, mode=None, phi=None, **kwargs):
                if mode != 'qp' : return
                r = r.squeeze()
                val = r**2 * phi**(-(npot+1))
                return {'val' : val.reshape(r.shape[0], 1, 1)}
            termsnl = [[matint1, 'dw_laplace'],
                       [matnl, 'dw_volume_integrate'],
                       [matrho, 'dw_volume_integrate']]
            coeffsnl = [alpha, -1.0, 1.0]
            kwargsnl = {'constcoeffs' : coeffsnl,
                        'order' : fem_order,
                        'verbose' : False}
            wf_nl = WeakForm(name_weak_nl, meshfiles[0], termsnl, **kwargsnl)

            if line_search_bool:
                def matls(ts, r, mode=None, phi=None, dphi=None, **kwargs):
                    if mode != 'qp' : return
                    r = r.squeeze()
                    val = r**2 * dphi * phi**(-(npot+2))
                    return {'val' : val.reshape(r.shape[0], 1, 1)}
                terms_ls = [[matint1, 'dw_laplace'],
                            [matls, 'dw_volume_integrate']]
                coeffs_ls = [alpha, npot+1]
                kwargs_ls = {'constcoeffs' : coeffs_ls,
                             'order' : fem_order,
                             'verbose' : False}
                wf_ls = WeakForm('weakcham1_ls', meshfiles[0], terms_ls,
                                 **kwargs_ls) # line-search weak form

            kwargsint = {'constcoeffs' : coeffsint,
                        'unknown_name' : 'u1',
                        'test_name' : 'v1',
                        'order' : fem_order,
                        'integral_name' : 'i1',
                        'domain_name' : 'interior',
                        'entity_functions' : ent_func_in,
                        'verbose' : verbose}

            # interior weak form
            wf1 = WeakForm(name_weak_in, meshfiles[0], termsint, **kwargsint)

            vertex_dbc2 = [phi_vac, None]
            kwargsext = {'constcoeffs' : coeffsext,
                        'unknown_name' : 'u2',
                        'test_name' : 'v2',
                        'order' : fem_order,
                        'integral_name' : 'i2',
                        'domain_name' : 'exterior',
                        'dirichlet_bc_vertex' : vertex_dbc2,
                        'entity_functions' : ent_func_out,
                        'verbose' : verbose}

            # exterior weak form
            wf2 = WeakForm(name_weak_out, meshfiles[-1], termsext, **kwargsext)

            # Specify the region of dimension D-1 shared by the two meshes
            cogammas = [wf1.vertices[0], wf2.vertices[1]]
            criteria = StopCriteria(max_iter, res_delta_tol=1e-14,
                                    sol_delta_tol=1e-14, min_iter=min_iter)

            if kwargs.get('initial_guess') is not None:
               if kwargs['initial_guess'] == 'min_pot':
                   initial_guess = [_initialize_minpot(wf1, npot,
                                                       backup=densities_int[0]),
                                    _initialize_minpot(wf2, npot,
                                                       backup=rho_vac)]
               else:
                   initial_guess = kwargs['initial_guess']
            elif func_init is not None:
                phi0_in = func_init[0](wf1.field.coors)
                phi0_out = func_init[1](wf2.field.coors)
                initial_guess = [phi0_in, phi0_out]
            elif analytic_params is None:
                phi0_in = np.mean(np.array(phi_bounds)) \
                    * np.ones(wf1.field.n_nod)
                phi0_out = phi_vac * np.ones(wf2.field.n_nod)
                initial_guess = [phi0_in, phi0_out]
            elif type(analytic_params==dict):
                initial_guess = _analytic_init([wf1, wf2], analytic_params,
                                               coorsys, Rcut=Rcut)
            else:
                raise Exception("Cannot initialize the field!")

            solver_kwargs = {'initial_guess' : initial_guess,
                             'relax' : relax,
                             'bounds' : phi_bounds,
                             'criteria' : criteria,
                             'conn' : conn,
                             'cogammas' : cogammas,
                             'is_bounded' : False,
                             'line_search_bool' : line_search_bool,
                             'verbose' : verbose}
            merge_dicts(solver_kwargs, kwargs)
            wfs_dic = {'weakforms' : [wf1, wf2], 'wf_res' : wf_nl}
            if line_search_bool:
                wfs_dic["wf_G"]  = wf_ls
            solver = Solver(wfs_dic, islinear=False, **solver_kwargs)

        # Polar coordinates
        elif coorsys == 'polar' and dim == 2:

            def matint1(ts, coors, mode=None, **kwargs):
                if mode != 'qp' : return
                val = np.zeros((coors.shape[0], 2, 2))
                r, theta = coors[:, 0], coors[:, 1]
                val[:, 0, 0] = sin(theta)*r**2
                val[:, 1, 1] = sin(theta)
                return {'val' : val}

            def matint2(ts, coors, mode=None, phi=None, **kwargs):
                if mode != 'qp' : return
                r, theta = coors[:, 0], coors[:, 1]
                val = sin(theta) * r**2 * phi**(-(npot+2))
                return {'val' : val.reshape(-1, 1, 1)}

            def matint3(ts, coors, mode=None, phi=None, **kwargs):
                if mode != 'qp' : return
                r, theta = coors[:, 0], coors[:, 1]
                val = sin(theta) * r**2 * phi**(-(npot+1))
                return {'val' : val.reshape(-1, 1, 1)}

            def matrho(ts, coors, mode=None, rho=None, **kwargs):
                if mode != 'qp' : return
                if callable(rho): rho = rho(coors)
                r, theta = coors[:, 0], coors[:, 1]
                val = sin(theta) * r**2 * rho
                return {'val' : val.reshape(-1, 1, 1)}

            def matext1(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                if mode != 'qp' : return
                val = np.zeros((coors.shape[0], 2, 2))
                eta, theta = coors[:, 0], coors[:, 1]
                val[:, 0, 0] = eta**4/Rcut**2 * (5 - 4*eta/Rcut) * sin(theta)
                val[:, 1, 1] = eta**2/Rcut**2 * (5 - 4*eta/Rcut) * sin(theta)
                return {'val' : val}

            def matext2(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                if mode != 'qp' : return
                val = np.zeros((coors.shape[0], 2, 1))
                eta, theta = coors[:, 0], coors[:, 1]
                val[:, 0, 0] = 20*eta**3/Rcut**2 * (1-eta/Rcut) * sin(theta)
                return {'val' : val}

            def matext3(ts, coors, mode=None, phi=None, **kwargs):
                if mode != 'qp' : return
                eta, theta = coors[:, 0], coors[:, 1]
                val = Rcut**2 * (5-4*eta/Rcut) * sin(theta) * phi**(-(npot+2))
                return {'val' : val.reshape(-1, 1, 1)}

            def matext4(ts, coors, mode=None, phi=None, **kwargs):
                if mode != 'qp' : return
                eta, theta = coors[:, 0], coors[:, 1]
                val = Rcut**2 * (5-4*eta/Rcut) * sin(theta) * phi**(-(npot+1))
                return {'val' : val.reshape(-1, 1, 1)}

            def matext5(ts, coors, mode=None, rho_vac=rho_vac, **kwargs):
                if mode != 'qp' : return
                eta, theta = coors[:, 0], coors[:, 1]
                val = Rcut**2 * (5-4*eta/Rcut) * sin(theta) * rho_vac
                return {'val' : val.reshape(-1, 1, 1)}

            coeffsint = [alpha, npot+1, -(npot+2)]
            termsint = [[matint1, 'dw_diffusion'],
                        [matint2, 'dw_volume_dot'],
                        [matint3, 'dw_volume_integrate']]
            _complete_terms(termsint, densities_int, matrho)
            coeffsint += [1.0]*(len(densities_int)-densities_int.count(0))
            coeffsext = [alpha, alpha, npot+1, -(npot+2), 1.0]
            termsext = [[matext1, 'dw_diffusion'],
                        [matext2, 'dw_s_dot_mgrad_s'],
                        [matext3, 'dw_volume_dot'],
                        [matext4, 'dw_volume_integrate'],
                        [matext5, 'dw_volume_integrate']]

            # Non-linearized equation for strong residual evaluation
            def matnl(ts, coors, mode=None, phi=None, **kwargs):
                if mode != 'qp' : return
                r, theta = coors[:, 0], coors[:, 1]
                val = (sin(theta) * r**2 * phi**(-(npot+1))).reshape(-1, 1, 1)
                return {'val' : val}

            termsnl = [[matint1, 'dw_diffusion'],
                       [matnl, 'dw_volume_integrate']]
            coeffsnl = [alpha, -1.0]
            _complete_terms(termsnl, densities_int, matrho)
            coeffsnl += [1.0]*(len(densities_int)-densities_int.count(0))
            kwargsnl = {'constcoeffs' : coeffsnl,
                        'order' : fem_order,
                        'densities' : densities_int,
                        'verbose' : False}
            wf_nl = WeakForm(name_weak_nl, meshfiles[0], termsnl, **kwargsnl)

            kwargsint = {'constcoeffs' : coeffsint,
                         'unknown_name' : 'u1',
                         'test_name' : 'v1',
                         'order' : fem_order,
                         'integral_name' : 'i1',
                         'domain_name' : 'interior',
                         'densities' : densities_int,
                         'entity_functions' : ent_func_in,
                         'verbose' : verbose}

            if conn == 'ping-pong': # add Dirichlet BC at Rcut
                kwargsint['dirichlet_bc_facet'] = [phi_vac, None]
            else:
                kwargsint['dirichlet_bc_facet'] = kwargs.get(
                    'dirichlet_bc_facet_int')

            # interior weak form
            wf1 = WeakForm(name_weak_in, meshfiles[0], termsint, **kwargsint)

            facet_dbc2 = [None, phi_vac]
            kwargsext = {'constcoeffs' : coeffsext,
                         'unknown_name' : 'u2',
                         'test_name' : 'v2',
                         'order' : fem_order,
                         'integral_name' : 'i2',
                         'domain_name' : 'exterior',
                         'dirichlet_bc_facet' : facet_dbc2,
                         'entity_functions' : ent_func_out,
                         'verbose' : verbose}

            # exterior weak form
            wf2 = WeakForm(name_weak_out, meshfiles[-1], termsext, **kwargsext)

            # Specify the region of dimension D-1 shared by the two meshes
            cogammas = [wf1.facets[0], wf2.facets[0]]
            criteria = StopCriteria(max_iter, res_delta_tol=1e-14,
                                    sol_delta_tol=1e-14, min_iter=min_iter)

            if kwargs.get('initial_guess') is not None:
               if kwargs['initial_guess'] == 'min_pot':
                   initial_guess = [_initialize_minpot(wf1, npot),
                                    _initialize_minpot(wf2, npot,
                                                       backup=rho_vac)]
               else:
                   initial_guess = kwargs['initial_guess']
            elif func_init is not None:
                phi0_in = func_init[0](wf1.field.coors)
                phi0_out = func_init[1](wf2.field.coors)
                initial_guess = [phi0_in, phi0_out]
            elif analytic_params is None:
                phi0_in = np.mean(np.array(phi_bounds)) \
                    * np.ones(wf1.field.n_nod)
                phi0_out = phi_vac * np.ones(wf2.field.n_nod)
                initial_guess = [phi0_in, phi0_out]
            elif type(analytic_params==dict):
                initial_guess = _analytic_init([wf1, wf2], analytic_params,
                                               coorsys, Rcut=Rcut)
            else:
                raise Exception("Cannot initialize the field!")

            solver_kwargs = {'initial_guess' : initial_guess,
                             'relax' : relax,
                             'bounds' : phi_bounds,
                             'criteria' : criteria,
                             'conn' : conn,
                             'cogammas' : cogammas,
                             'is_bounded' : False,
                             'line_search_bool' : line_search_bool,
                             'verbose' : verbose}
            merge_dicts(solver_kwargs, kwargs)
            wfs_dic = {'weakforms' : [wf1, wf2], 'wf_res' : wf_nl}
            solver = Solver(wfs_dic, islinear=False, **solver_kwargs)

            if conn == 'ping-pong':
                # Add the Neumann term to the exterior problem
                def neumann(ts, coors, mode=None, solver=solver, **kwargs):
                    if mode != 'qp' : return
                    theta = coors[:, 1]
                    wf1 = solver.weakforms[0]
                    sol1 = solver.sols[0]
                    if sol1 is None:
                        sol1 = np.zeros(coors.shape[0])
                    grad1 = wf1.field.evaluate_at(
                        coors, sol1[:, np.newaxis], mode='grad').squeeze()
                    val = (sin(theta)*grad1[:, 0]).reshape(coors.shape[0],1,1)
                    return {'val' : val}

                neumann_term = [neumann,
                                'dw_surface_integrate',
                                cogammas[1].name]
                wf2.add_term(neumann_term, update=True, newcoeff=alpha)

        # Polar-mu coordinates
        elif coorsys == 'polar_mu' and dim == 2:

            def matint1(ts, coors, mode=None, **kwargs):
                if mode != 'qp' : return
                val = np.zeros((coors.shape[0], 2, 2))
                r, mu = coors[:, 0], coors[:, 1]
                val[:, 0, 0] = r**2
                val[:, 1, 1] = 1 - mu**2
                return {'val' : val}

            def matint2(ts, coors, mode=None, phi=None, **kwargs):
                if mode != 'qp' : return
                r = coors[:, 0]
                val = r**2 * phi**(-(npot+2))
                return {'val' : val.reshape(-1, 1, 1)}

            def matint3(ts, coors, mode=None, phi=None, **kwargs):
                if mode != 'qp' : return
                r = coors[:, 0]
                val = r**2 * phi**(-(npot+1))
                return {'val' : val.reshape(-1, 1, 1)}

            def matrho(ts, coors, mode=None, rho=None, **kwargs):
                if mode != 'qp' : return
                if callable(rho): rho = rho(coors)
                r = coors[:, 0]
                val = r**2 * rho
                return {'val' : val.reshape(-1, 1, 1)}

            def matext1(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                if mode != 'qp' : return
                val = np.zeros((coors.shape[0], 2, 2))
                eta, mu = coors[:, 0], coors[:, 1]
                val[:, 0, 0] = eta**4/Rcut**2 * (5 - 4*eta/Rcut)
                val[:, 1, 1] = eta**2/Rcut**2 * (5 - 4*eta/Rcut) * (1-mu**2)
                return {'val' : val}

            def matext2(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                if mode != 'qp' : return
                val = np.zeros((coors.shape[0], 2, 1))
                eta = coors[:, 0]
                val[:, 0, 0] = 20*eta**3/Rcut**2 * (1-eta/Rcut)
                return {'val' : val}

            def matext3(ts, coors, mode=None, phi=None, **kwargs):
                if mode != 'qp' : return
                eta = coors[:, 0]
                val = Rcut**2 * (5-4*eta/Rcut) * phi**(-(npot+2))
                return {'val' : val.reshape(-1, 1, 1)}

            def matext4(ts, coors, mode=None, phi=None, **kwargs):
                if mode != 'qp' : return
                eta = coors[:, 0]
                val = Rcut**2 * (5-4*eta/Rcut) * phi**(-(npot+1))
                return {'val' : val.reshape(-1, 1, 1)}

            def matext5(ts, coors, mode=None, rho_vac=rho_vac, **kwargs):
                if mode != 'qp' : return
                eta = coors[:, 0]
                val = Rcut**2 * (5-4*eta/Rcut) * rho_vac
                return {'val' : val.reshape(-1, 1, 1)}

            coeffsint = [alpha, npot+1, -(npot+2)]
            termsint = [[matint1, 'dw_diffusion'],
                        [matint2, 'dw_volume_dot'],
                        [matint3, 'dw_volume_integrate']]
            _complete_terms(termsint, densities_int, matrho)
            coeffsint += [1.0]*(len(densities_int)-densities_int.count(0))
            coeffsext = [alpha, alpha, npot+1, -(npot+2), 1.0]
            termsext = [[matext1, 'dw_diffusion'],
                        [matext2, 'dw_s_dot_mgrad_s'],
                        [matext3, 'dw_volume_dot'],
                        [matext4, 'dw_volume_integrate'],
                        [matext5, 'dw_volume_integrate']]

            # Non-linearized equation for strong residual evaluation
            def matnl(ts, coors, mode=None, phi=None, **kwargs):
                if mode != 'qp' : return
                r = coors[:, 0]
                val = (r**2 * phi**(-(npot+1))).reshape(-1, 1, 1)
                return {'val' : val}

            termsnl = [[matint1, 'dw_diffusion'],
                       [matnl, 'dw_volume_integrate']]
            coeffsnl = [alpha, -1.0]
            _complete_terms(termsnl, densities_int, matrho)
            coeffsnl += [1.0]*(len(densities_int)-densities_int.count(0))
            kwargsnl = {'constcoeffs' : coeffsnl,
                        'order' : fem_order,
                        'densities' : densities_int,
                        'verbose' : False}
            wf_nl = WeakForm(name_weak_nl, meshfiles[0], termsnl, **kwargsnl)

            kwargsint = {'constcoeffs' : coeffsint,
                         'unknown_name' : 'u1',
                         'test_name' : 'v1',
                         'order' : fem_order,
                         'integral_name' : 'i1',
                         'domain_name' : 'interior',
                         'densities' : densities_int,
                         'entity_functions' : ent_func_in,
                         'verbose' : verbose}

            if conn == 'ping-pong': # add Dirichlet BC at Rcut
                kwargsint['dirichlet_bc_facet'] = [phi_vac, None]
            else:
                kwargsint['dirichlet_bc_facet'] = kwargs.get(
                    'dirichlet_bc_facet_int')

            # interior weak form
            wf1 = WeakForm(name_weak_in, meshfiles[0], termsint, **kwargsint)

            facet_dbc2 = [None, phi_vac]
            kwargsext = {'constcoeffs' : coeffsext,
                         'unknown_name' : 'u2',
                         'test_name' : 'v2',
                         'order' : fem_order,
                         'integral_name' : 'i2',
                         'domain_name' : 'exterior',
                         'dirichlet_bc_facet' : facet_dbc2,
                         'entity_functions' : ent_func_out,
                         'verbose' : verbose}

            # exterior weak form
            wf2 = WeakForm(name_weak_out, meshfiles[-1], termsext, **kwargsext)

            # Specify the region of dimension D-1 shared by the two meshes
            cogammas = [wf1.facets[0], wf2.facets[0]]
            criteria = StopCriteria(max_iter, res_delta_tol=1e-16,
                                    sol_delta_tol=1e-16, min_iter=min_iter)

            if kwargs.get('initial_guess') is not None:
                if kwargs['initial_guess'] == 'min_pot':
                    initial_guess = [_initialize_minpot(wf1, npot),
                                     _initialize_minpot(wf2, npot,
                                                        backup=rho_vac)]
                else:
                    initial_guess = kwargs['initial_guess']
            elif func_init is not None:
                phi0_in = func_init[0](wf1.field.coors)
                phi0_out = func_init[1](wf2.field.coors)
                initial_guess = [phi0_in, phi0_out]
            elif analytic_params is None:
                phi0_in = np.mean(np.array(phi_bounds)) \
                    * np.ones(wf1.field.n_nod)
                phi0_out = phi_vac * np.ones(wf2.field.n_nod)
                initial_guess = [phi0_in, phi0_out]
            elif type(analytic_params==dict):
                initial_guess = _analytic_init([wf1, wf2], analytic_params,
                                               coorsys, Rcut=Rcut)
            else:
                raise Exception("Cannot initialize the field!")

            solver_kwargs = {'initial_guess' : initial_guess,
                             'relax' : relax,
                             'bounds' : phi_bounds,
                             'criteria' : criteria,
                             'conn' : conn,
                             'cogammas' : cogammas,
                             'is_bounded' : False,
                             'line_search_bool' : line_search_bool,
                             'verbose' : verbose}
            merge_dicts(solver_kwargs, kwargs)
            wfs_dic = {'weakforms' : [wf1, wf2], 'wf_res' : wf_nl}
            solver = Solver(wfs_dic, islinear=False, **solver_kwargs)

            if conn == 'ping-pong':
                # Add the Neumann term to the exterior problem
                def neumann(ts, coors, mode=None, solver=solver, **kwargs):
                    if mode != 'qp' : return
                    wf1 = solver.weakforms[0]
                    sol1 = solver.sols[0]
                    if sol1 is None:
                        sol1 = np.zeros(coors.shape[0])
                    grad1 = wf1.field.evaluate_at(
                        coors, sol1[:, np.newaxis], mode='grad').squeeze()
                    val = (grad1[:, 0]).reshape(coors.shape[0], 1, 1)
                    return {'val' : val}

                neumann_term = [neumann,
                                'dw_surface_integrate',
                                cogammas[1].name]
                wf2.add_term(neumann_term, update=True, newcoeff=alpha)

        else:
            raise Exception("Not implemented coordinates system: %s" %coorsys)

        self.solver = solver

    def solve(self, **kwargs):
        """Wrapper for Solver.solve method"""
        self.solver.solve(**kwargs)

    def save(self, grad=False, **kwargs):
        """Wrapper for Solver.save method"""
        self.solver.save(grad=grad, **kwargs)

    def plot(self, **kwargs):
        """Wrapper for pyvista_plot"""
        val_renorm = kwargs.get('val_renorm', 1.0)
        grad_renorm = kwargs.get('grad_renorm', 1.0)
        out = self.solver.save(grad=True, val_renorm=val_renorm,
                               grad_renorm=grad_renorm)
        from femtoscope.display.fea_plot import pyvista_plot
        for result_file in out:
            pyvista_plot(result_file, **kwargs)

class ChameleonDiff():
    def __init__(self, alpha, npot, phi0s, densities_int, meshfiles, **kwargs):
        if not isinstance(meshfiles, list): meshfiles = [meshfiles]
        if not isinstance(densities_int, list): densities_int = [densities_int]
        mesh_dir = kwargs.get('mesh_dir', None)
        if type(meshfiles[0]) == str:
            dim = get_meshdim(meshfiles[0], mesh_dir=mesh_dir)
        else: # 1D meshes are not saved as VTK files
            dim = 1
        if 'Rcut' in kwargs.keys():
            Rcut = kwargs['Rcut']
        else:
            Rcut = get_rcut(meshfiles[0], mesh_dir=mesh_dir, coorsys='polar')
        self.Rcut = Rcut
        conn = kwargs.get('conn', 'connected')
        fem_order = kwargs.get('fem_order', 2)
        ent_func_int = kwargs.get('entity_functions_int', [])
        ent_func_ext = kwargs.get('entity_functions_ext', [])
        verbose = kwargs.get('verbose', True)
        phi0_int = phi0s[0]
        phi0_ext = phi0s[1]
        self.alpha = alpha
        self.Rcut = Rcut
        name_weak_int = "wfin"
        name_weak_out = "wfout"

        def matint1(ts, coors, mode=None, Rcut=Rcut, **kwargs):
            if mode != 'qp' : return
            val = np.zeros((coors.shape[0], 2, 2))
            r = coors[:, 0]
            theta = coors[:, 1]
            val[:, 0, 0] = sin(theta)*r**2
            val[:, 1, 1] = sin(theta)
            return {'val' : val}

        def matint2(ts, coors, mode=None, phi0=phi0_int, npot=npot, **kwargs):
            if mode != 'qp' : return
            r = coors[:, 0]
            theta = coors[:, 1]
            val = sin(theta)*r**2 * phi0(r)**(-(npot+2))
            val.shape = (coors.shape[0], 1, 1)
            return {'val' : val}

        def matrho(ts, coors, mode=None, rho=None, **kwargs):
            if mode != 'qp' : return
            if callable(rho): rho = rho(coors)
            r = coors[:, 0]
            theta = coors[:, 1]
            val = rho*sin(theta)*r**2
            val.shape = (coors.shape[0], 1, 1)
            return {'val' : val}

        coeffsint = [alpha, npot+1]
        termsint = [[matint1, 'dw_diffusion'],
                    [matint2, 'dw_volume_dot']]
        _complete_terms(termsint, densities_int, matrho)
        coeffsint += [1.0]*(len(densities_int)-densities_int.count(0))

        def matext1(ts, coors, mode=None, Rcut=Rcut, **kwargs):
            if mode != 'qp' : return
            val = np.zeros((coors.shape[0], 2, 2))
            eta = coors[:, 0]
            theta = coors[:, 1]
            val[:, 0, 0] = sin(theta)*eta**4/Rcut**2 * (5 - 4*eta/Rcut)
            val[:, 1, 1] = sin(theta)*eta**2/Rcut**2 * (5 - 4*eta/Rcut)
            return {'val' : val}

        def matext2(ts, coors, mode=None, Rcut=Rcut, **kwargs):
            if mode != 'qp' : return
            val = np.zeros((coors.shape[0], 2, 1))
            eta = coors[:, 0]
            theta = coors[:, 1]
            val[:, 0, 0] = 20*eta**3/Rcut**2 * (1-eta/Rcut) * sin(theta)
            return {'val' : val}

        def matext3(ts, coors, mode=None, phi0=phi0_ext, Rcut=Rcut, **kwargs):
            if mode != 'qp' : return
            eta = coors[:, 0]
            theta = coors[:, 1]
            val = Rcut**2 * sin(theta) * (5 - 4*eta/Rcut) * phi0(eta)**(-(npot+2))
            return {'val' : val.reshape(eta.shape[0], 1, 1)}

        coeffsext = [alpha, alpha, npot+1]
        termsext = [[matext1, 'dw_diffusion'],
                    [matext2, 'dw_s_dot_mgrad_s'],
                    [matext3, 'dw_volume_dot']]

        kwargsint = {'constcoeffs' : coeffsint,
                    'unknown_name' : 'u1',
                    'test_name' : 'v1',
                    'order' : fem_order,
                    'integral_name' : 'i1',
                    'domain_name' : 'interior',
                    'densities' : densities_int,
                    'entity_functions' : ent_func_int,
                    'verbose' : verbose}

        if conn == 'ping-pong': # add Dirichlet BC at Rcut
            kwargsint['dirichlet_bc_facet'] = [0.0, None]

        # interior weak form
        wf1 = WeakForm(name_weak_int, meshfiles[0], termsint, **kwargsint)

        facet_dbc2 = [None, 0.0]
        kwargsext = {'constcoeffs' : coeffsext,
                    'unknown_name' : 'u2',
                    'test_name' : 'v2',
                    'order' : fem_order,
                    'integral_name' : 'i2',
                    'domain_name' : 'exterior',
                    'dirichlet_bc_facet' : facet_dbc2,
                    'entity_functions' : ent_func_ext,
                    'verbose' : verbose}

        # exterior weak form
        wf2 = WeakForm(name_weak_out, meshfiles[-1], termsext, **kwargsext)

        # Specify the region of dimension D-1 shared by the two meshes
        cogammas = [wf1.facets[0], wf2.facets[0]]
        solver_kwargs = {'cogammas' : cogammas,
                         'conn' : conn,
                         'is_bounded'  : False,
                         'verbose' : verbose}
        merge_dicts(solver_kwargs, kwargs)
        wfs_dic = {'weakforms' : [wf1, wf2]}
        solver = Solver(wfs_dic, islinear=True, **solver_kwargs)

        if conn == 'ping-pong':
            # Add the Neumann term to the exterior problem
            def neumann(ts, coors, mode=None, solver=solver, **kwargs):
                if mode != 'qp' : return
                theta = coors[:, 1]
                wf1 = solver.weakforms[0]
                sol1 = solver.sols[0]
                if sol1 is None:
                    sol1 = np.zeros(coors.shape[0])
                grad1 = wf1.field.evaluate_at(
                    coors, sol1[:, np.newaxis], mode='grad').squeeze()
                val = sin(theta)*grad1[:, 0]
                return {'val' : val.reshape(coors.shape[0], 1, 1)}
            neumann_term = [neumann, 'dw_surface_integrate',
                            cogammas[1].name]
            wf2.add_term(neumann_term, update=True, newcoeff=Rcut**2)

        self.solver = solver

    def solve(self):
        """Wrapper for Solver.solve method"""
        self.solver.solve()

    def save(self, grad=False, **kwargs):
        """Wrapper for Solver.save method"""
        self.solver.save(grad=grad, **kwargs)

    def plot(self, **kwargs):
        """Wrapper for pyvista_plot"""
        val_renorm = kwargs.get('val_renorm', 1.0)
        grad_renorm = kwargs.get('grad_renorm', 1.0)
        out = self.solver.save(grad=True, val_renorm=val_renorm,
                               grad_renorm=grad_renorm)
        from femtoscope.display.fea_plot import pyvista_plot
        for result_file in out:
            pyvista_plot(result_file, **kwargs)

class ChameleonDiffBounded():
    def __init__(self, alpha, npot, phi0, densities_int, meshfile, **kwargs):
        if not isinstance(meshfile, list): meshfiles = [meshfile]
        if not isinstance(densities_int, list): densities_int = [densities_int]
        mesh_dir = kwargs.get('mesh_dir', None)
        if type(meshfiles[0]) == str:
            dim = get_meshdim(meshfiles[0], mesh_dir=mesh_dir)
        else: # 1D meshes are not saved as VTK files
            dim = 1
        if 'Rcut' in kwargs.keys():
            Rcut = kwargs['Rcut']
        else:
            Rcut = get_rcut(meshfiles[0], mesh_dir=mesh_dir, coorsys='polar')
        self.Rcut = Rcut
        conn = kwargs.get('conn', 'connected')
        fem_order = kwargs.get('fem_order', 2)
        ent_func_int = kwargs.get('entity_functions_int', [])
        verbose = kwargs.get('verbose', True)
        self.alpha = alpha
        self.Rcut = Rcut
        name_weak_int = "wfin"

        def matint1(ts, coors, mode=None, Rcut=Rcut, **kwargs):
            if mode != 'qp' : return
            val = np.zeros((coors.shape[0], 2, 2))
            r = coors[:, 0]
            theta = coors[:, 1]
            val[:, 0, 0] = sin(theta)*r**2
            val[:, 1, 1] = sin(theta)
            return {'val' : val}

        def matint2(ts, coors, mode=None, phi0=phi0, npot=npot, **kwargs):
            if mode != 'qp' : return
            r = coors[:, 0]
            theta = coors[:, 1]
            val = sin(theta)*r**2 * phi0(r)**(-(npot+2))
            val.shape = (coors.shape[0], 1, 1)
            return {'val' : val}

        def matrho(ts, coors, mode=None, rho=None, **kwargs):
            if mode != 'qp' : return
            if callable(rho): rho = rho(coors)
            r = coors[:, 0]
            theta = coors[:, 1]
            val = rho*sin(theta)*r**2
            val.shape = (coors.shape[0], 1, 1)
            return {'val' : val}

        coeffsint = [alpha, npot+1]
        termsint = [[matint1, 'dw_diffusion'],
                    [matint2, 'dw_volume_dot']]
        _complete_terms(termsint, densities_int, matrho)
        coeffsint += [1.0]*(len(densities_int)-densities_int.count(0))

        dirichlet_bc = [0.0, None]
        kwargsint = {'constcoeffs' : coeffsint,
                    'unknown_name' : 'u1',
                    'test_name' : 'v1',
                    'order' : fem_order,
                    'integral_name' : 'i1',
                    'domain_name' : 'interior',
                    'densities' : densities_int,
                    'dirichlet_bc_facet' : dirichlet_bc,
                    'entity_functions' : ent_func_int,
                    'verbose' : verbose}
        # interior weak form
        wf1 = WeakForm(name_weak_int, meshfiles[0], termsint, **kwargsint)

        solver_kwargs = {'is_bounded'  : True,
                         'verbose' : verbose}
        merge_dicts(solver_kwargs, kwargs)
        wf_dic = {'weakforms' : wf1}
        solver = Solver(wf_dic, islinear=True, **solver_kwargs)
        self.solver = solver

    def solve(self):
        """Wrapper for Solver.solve method"""
        self.solver.solve()

    def save(self, grad=False, **kwargs):
        """Wrapper for Solver.save method"""
        self.solver.save(grad=grad, **kwargs)

    def plot(self, **kwargs):
        """Wrapper for pyvista_plot"""
        val_renorm = kwargs.get('val_renorm', 1.0)
        grad_renorm = kwargs.get('grad_renorm', 1.0)
        out = self.solver.save(grad=True, val_renorm=val_renorm,
                                grad_renorm=grad_renorm)
        from femtoscope.display.fea_plot import pyvista_plot
        for result_file in out:
            pyvista_plot(result_file, **kwargs)


class ChameleonPicard():
    def __init__(self, alpha, npot, densities, dirichlet_bc, rho_bounds,
                 meshfile, func_phi0=None, **kwargs):

        if not isinstance(densities, list): densities = [densities]
        mesh_dir = kwargs.get('mesh_dir', None)
        if type(meshfile) == str:
            dim = get_meshdim(meshfile, mesh_dir=mesh_dir)
        else: # 1D meshes are not saved as VTK files
            dim = 1
        ent_func = kwargs.get('entity_functions', [])
        fem_order = kwargs.get('fem_order', 2)
        relax = kwargs.get('relax', 0.9)
        verbose = kwargs.get('verbose', True)
        self.alpha = alpha
        self.npot = npot
        phi_bounds = [p**(-1/(npot+1)) for p in rho_bounds]
        phi_bounds.sort()

        name_weak = "wf"
        name_weak_nl = "wfnl"

        min_iter = kwargs.get('min_iter')
        max_iter = kwargs.get('max_iter')
        line_search_bool = kwargs.get('line_search_bool', False)

        def mat1(ts, coors, mode=None, **kwargs):
            if mode != 'qp' : return
            val = np.zeros((coors.shape[0], 2, 2))
            r, theta = coors[:, 0], coors[:, 1]
            val[:, 0, 0] = sin(theta)*r**2
            val[:, 1, 1] = sin(theta)
            return {'val' : val}

        def mat2(ts, coors, mode=None, phi=None, **kwargs):
            if mode != 'qp' : return
            r, theta = coors[:, 0], coors[:, 1]
            val = sin(theta) * r**2 * phi**(-(npot+2))
            return {'val' : val.reshape(-1, 1, 1)}

        def matrho(ts, coors, mode=None, rho=None, **kwargs):
            if mode != 'qp' : return
            if callable(rho) : rho = rho(coors)
            r, theta = coors[:, 0], coors[:, 1]
            val = sin(theta) * r**2 * rho
            return {'val' : val.reshape(-1, 1, 1)}

        coeffs = [alpha, -1]
        terms = [[mat1, 'dw_diffusion'],
                 [mat2, 'dw_volume_dot']]
        _complete_terms(terms, densities, matrho)
        coeffs += [1.0]*len(densities)

        kwargswf = {'constcoeffs' : coeffs,
                    'unknown_name' : 'u1',
                    'test_name' : 'v1',
                    'order' : fem_order,
                    'integral_name' : 'i1',
                    'domain_name' : 'interior',
                    'densities' : densities,
                    'dirichlet_bc_facet' : dirichlet_bc,
                    'entity_functions' : ent_func,
                    'verbose' : verbose}
        wf = WeakForm(name_weak, meshfile, terms, **kwargswf)

        # Non-linearized equation for strong residual evaluation
        def matnl1(ts, coors, mode=None, **kwargs):
            if mode != 'qp' : return
            r, theta = coors[:, 0], coors[:, 1]
            val = np.zeros((coors.shape[0], 2, 2))
            val[:, 0, 0] = sin(theta) * r**2
            val[:, 1, 1] = sin(theta)
            return {'val' : val}

        def matnl2(ts, coors, mode=None, phi=None, **kwargs):
            if mode != 'qp' : return
            r, theta = coors[:, 0], coors[:, 1]
            val = (sin(theta) * r**2 * phi**(-(npot+1))).reshape(-1, 1, 1)
            return {'val' : val}

        def matnlrho(ts, coors, mode=None, rho=None, **kwargs):
            if mode != 'qp' : return
            if callable(rho): rho = rho(coors)
            r, theta = coors[:, 0], coors[:, 1]
            val = (sin(theta) * r**2 * rho).reshape(-1, 1, 1)
            return {'val' : val}

        terms_nl = [[matnl1, 'dw_diffusion'],
                    [matnl2, 'dw_volume_integrate']]
        coeffs_nl = [alpha, -1.0]
        _complete_terms(terms_nl, densities, matnlrho)
        coeffs_nl += [1.0]*len(densities)
        wf_nl_kwargs = {'constcoeffs' : coeffs_nl,
                        'order' : fem_order,
                        'densities' : densities,
                        'dirichlet_bc_facet' : dirichlet_bc,
                        'verbose' : False}
        wf_nl = WeakForm(name_weak_nl, meshfile, terms_nl,
                         **wf_nl_kwargs) # nonlinear weak form

        # Stopping Criteria of Newton iterations
        criteria = StopCriteria(max_iter, res_delta_tol=1e-14,
                                sol_delta_tol=1e-14, min_iter=min_iter)

        # Initial guess
        coors = wf.field.coors
        if callable(func_phi0):
            phi0 = func_phi0(coors)
        else:
            phi0 =  np.mean(np.array(phi_bounds)) * np.ones(coors.shape[0])

        # Solver instance
        solver_kwargs = {'initial_guess' : phi0,
                         'relax' : relax,
                         'bounds' : phi_bounds,
                         'criteria' : criteria,
                         'is_bounded' : True,
                         'line_search_bool' : line_search_bool,
                         'verbose' : verbose}
        merge_dicts(solver_kwargs, kwargs)
        wfs_dic = {'weakforms' : [wf], 'wf_res' : wf_nl}
        solver = Solver(wfs_dic, islinear=False, **solver_kwargs)


        self.solver = solver

    def solve(self):
        """Wrapper for Solver.solve method"""
        self.solver.solve()

    def save(self, grad, **kwargs):
        """Wrapper for Solver.save method"""
        self.solver.save(grad=grad, **kwargs)

    def plot(self, **kwargs):
        """Wrapper for pyvista_plot"""
        val_renorm = kwargs.get('val_renorm', 1.0)
        grad_renorm = kwargs.get('grad_renorm', 1.0)
        out = self.solver.save(grad=True, val_renorm=val_renorm,
                               grad_renorm=grad_renorm)
        from femtoscope.display.fea_plot import pyvista_plot
        for result_file in out:
            pyvista_plot(result_file, **kwargs)


def _analytic_init(wf, analytic_params, coorsys, Rcut=None):
    """Initialization of the chameleon's field using the analytical
    approximation for the field around a perfect solid sphere."""
    from femtoscope.misc.analytical import chameleon_radial
    R_A = analytic_params['R_A']
    rho_in = analytic_params['rho_in']
    rho_vac = analytic_params['rho_vac']
    alpha = analytic_params['alpha']
    n = analytic_params['n']
    if isinstance(wf, list):
        if Rcut is None:
            raise TypeError("Rcut must be specified!")
        wf1, wf2 = wf[0], wf[1]
        if coorsys == 'cartesian':
            r1 = np.linalg.norm(wf1.field.coors, axis=1)
            norm2 = np.linalg.norm(wf2.field.coors, axis=1)
            with np.errstate(divide='ignore'):
                r2 = np.where(norm2>0, Rcut**2/norm2, np.nan)
        elif coorsys in ['polar', 'polar_mu']:
            r1 = wf1.field.coors[:, 0]
            norm2 = wf2.field.coors[:, 0]
            with np.errstate(divide='ignore'):
                r2 = np.where(norm2>0, Rcut**2/norm2, np.nan)
        else:
            raise Exception("Not implemented coordinates system: %s" %coorsys)
        phi0_int = chameleon_radial(r1, R_A, rho_in, rho_vac, alpha, n)
        phi0_ext = chameleon_radial(r2, R_A, rho_in, rho_vac, alpha, n)
        phi0_ext[phi0_ext==0.0] = rho_vac**(-1/(n+1))
        return [phi0_int, phi0_ext]
    else:
        if coorsys == 'cartesian':
            r = np.linalg.norm(wf.field.coors, axis=1)
        elif coorsys in ['polar', 'polar_mu']:
            r = wf.field.coors[:, 0]
        else:
            raise Exception("Not implemented coordinates system: %s" %coorsys)
        phi0 = chameleon_radial(r, R_A, rho_in, rho_vac, alpha, n)
        return phi0


def _complete_terms(terms, densities, matrho):
    """Add the density integrals to the list of terms (one per subdomain)"""
    for k in range(len(densities)):
        terms.append([matrho, 'dw_volume_integrate', 'subomega30'+str(k)])

def _initialize_minpot(wf, npot, backup=None):
    """
    Initialize the chameleon field at the minimum of its effective potential
    in the domain (which is a function of density and npot, only in
    dimensionless variables).

    Parameters
    ----------
    wf : `WeakForm`
        Weakform associated with the initialization.
    npot : int
        Exponent parameter of the chameleon model.
    backup : float or callable, optional
        If `wf` has no 'densities' attribute, use the user-provided function or
        float to compute the initial guess. The default is None.

    Returns
    -------
    guess : 1d-array
        Initial guess of Newton's iterations.

    """
    guess = np.empty(max(wf.field.coors.shape))
    if (not hasattr(wf, 'densities')) and (backup is not None):
        if callable(backup):
            guess[:] = backup(wf.field.coors.squeeze())**(-1/(npot+1))
        else:
            guess[:] = backup**(-1/(npot+1))
        return guess
    guess[:] = np.nan
    for key in wf.densities:
        density = wf.densities[key]
        region = wf.region_dic[key]
        dofs = wf.field.get_dofs_in_region(region)
        if callable(density):
            coors = wf.field.coors[dofs]
            rho = density(coors)
        else:
            rho = density * np.ones(max(dofs.shape), dtype=np.float64)
        guess[dofs] = rho**(-1/(npot+1))
    assert (not np.isnan(guess).any()), "Initial guess is not set correctly!"
    return guess