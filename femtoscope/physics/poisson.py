# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 10:18:32 2022

Set up a Poisson problem on bounded or unbounded domains.

Note about Neumann terms:
    In neumann material, one needs to evaluate of the gradient of a given
    field at coordinates belonging to the domain's boundary. The best way
    to do so would be to evaluate a `ev_grad` term. It is however a painful
    task as `ev_grad` returns gradients at quadrature points disorderly with
    respect to the `coors` variable...
    Using `field.evaluate_at()` provides very close gradient values and is
    much less cumbersome!

@author: hlevy
"""

from femtoscope.core.weak import WeakForm
from femtoscope.misc.util import merge_dicts
import numpy as np
from femtoscope.core.simu import Solver
from femtoscope.inout.meshfactory import get_meshdim, get_rcut
from numpy import sin, sqrt


class PoissonBounded():
    r"""
    Class for solving the Poisson equation
    $$ \Delta \Phi = \alpha \times \rho $$
    on bounded domains.

    Attributes
    ----------
    alpha : float
        Physical parameter weighting the lhs of Poisson's equation
        (dimensionless).
    solver : Solver instance
        The FEM solver instance to be run.

    """

    def __init__(self, alpha, densities, dirichlet_bc, meshfile,
                 coorsys='cartesian', **kwargs):
        """
        Construct a Poisson problem instance.

        Parameters
        ----------
        alpha : float
            Physical parameter weighting the rhs of Poisson's equation
            (dimensionless).
        densities : list
            List of density functions or constants. The length of this list
            must match the number of sub*domains in the mesh.
        dirichlet_bc : list
            List of Dirichlet boundary condition(s).
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
        entity_functions : list of 2-tuple
            List of tuples (dim, function) for manual entity selection.
            The default is [].
        verbose : bool
            Display user's information. The default is True.

        """

        if not isinstance(densities, list): densities = [densities]
        mesh_dir = kwargs.get('mesh_dir', None)
        if type(meshfile) == str:
            dim = get_meshdim(meshfile, mesh_dir=mesh_dir)
        else: # 1D meshes are not saved as VTK files
            dim = 1
        fem_order = kwargs.get('fem_order', 2)
        ent_func = kwargs.get('entity_functions', [])
        verbose = kwargs.get('verbose', True)
        self.alpha = alpha

        name_weak = "wf"

        # Cartesian coordinates
        if coorsys == 'cartesian' and dim == 2:

            def matlap(ts, coors, mode=None, **kwargs):
                if mode != 'qp' : return
                x = coors[:, 0]
                val = abs(x).reshape(coors.shape[0],1,1)
                return {'val' : val}

            def matrho(ts, coors, mode=None, rho=None, **kwargs):
                if mode != 'qp' : return
                if callable(rho): rho = rho(coors)
                x = coors[:, 0]
                val = (rho * abs(x)).reshape(coors.shape[0],1,1)
                return {'val' : val}

            coeffs = [1.0]
            terms = [[matlap, 'dw_laplace']]
            _complete_terms(terms, densities, matrho)
            coeffs += [alpha]*(len(densities)-densities.count(0))
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
            wf = WeakForm(name_weak, meshfile, terms, **kwargswf)
            wfs_dic = {'weakforms' : wf}
            solver = Solver(wfs_dic, islinear=True, is_bounded=True, **kwargs)
            self.solver = solver

        elif coorsys == 'cartesian' and dim == 3:

            def matrho(ts, coors, mode=None, rho=None, **kwargs):
                if mode != 'qp': return
                if callable(rho): rho = rho(coors)
                val = rho*np.ones(coors.shape[0])
                return {'val' : val.reshape(coors.shape[0], 1, 1)}

            coeffs = [1.0]
            terms = [[None, 'dw_laplace']]
            _complete_terms(terms, densities, matrho)
            coeffs += [alpha]*(len(densities)-densities.count(0))
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
            wf = WeakForm(name_weak, meshfile, terms, **kwargswf)
            wfs_dic = {'weakforms' : wf}
            solver = Solver(wfs_dic, islinear=True, is_bounded=True, **kwargs)
            self.solver = solver

        # Polar coordinates
        elif coorsys == 'polar' and dim == 1:

            def mat1(ts, r, mode=None, **kwargs):
                if mode != 'qp' : return
                return {'val' : (r**2).reshape(r.shape[0], 1, 1)}

            def mat2(ts, r, mode=None, rho=densities[0], **kwargs):
                if mode != 'qp' : return
                if callable(rho): rho = rho(r)
                return {'val' : (r**2 * rho).reshape(r.shape[0], 1, 1)}

            coeffs = [1.0, alpha]
            terms = [[mat1, 'dw_laplace'],
                     [mat2, 'dw_volume_integrate']]
            kwargswf = {'constcoeffs' : coeffs,
                        'unknown_name' : 'u1',
                        'test_name' : 'v1',
                        'order' : fem_order,
                        'integral_name' : 'i1',
                        'domain_name' : 'interior',
                        'entity_functions' : ent_func,
                        'dirichlet_bc_vertex' : dirichlet_bc,
                        'verbose' : verbose}
            wf = WeakForm(name_weak, meshfile, terms, **kwargswf)
            wfs_dic = {'weakforms' : wf}
            solver = Solver(wfs_dic, islinear=True, is_bounded=True, **kwargs)
            self.solver = solver


        elif coorsys == 'polar' and dim == 2:

            def mat1(ts, coors, mode=None, **kwargs):
                if mode != 'qp' : return
                val = np.zeros((coors.shape[0], 2, 2))
                r, theta = coors[:, 0], coors[:, 1]
                val[:, 0, 0] = sin(theta)*r**2
                val[:, 1, 1] = sin(theta)
                return {'val' : val}

            def matrho(ts, coors, mode=None, rho=None, **kwargs):
                if mode != 'qp' : return
                if callable(rho): rho = rho(coors)
                r = coors[:, 0]
                theta = coors[:, 1]
                val = rho*sin(theta)*r**2
                val.shape = (coors.shape[0], 1, 1)
                return {'val' : val}

            coeffs = [1.0]
            terms = [[mat1, 'dw_diffusion']]
            _complete_terms(terms, densities, matrho)
            coeffs += [alpha]*(len(densities)-densities.count(0))

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
            wf = WeakForm(name_weak, meshfile, terms, **kwargswf)
            wfs_dic = {'weakforms' : wf}
            solver = Solver(wfs_dic, islinear=True, is_bounded=True, **kwargs)
            self.solver = solver
            
        elif coorsys == 'polar_mu' and dim == 2:

            def mat1(ts, coors, mode=None, **kwargs):
                if mode != 'qp' : return
                val = np.zeros((coors.shape[0], 2, 2))
                r, mu = coors[:, 0], coors[:, 1]
                val[:, 0, 0] = r**2
                val[:, 1, 1] = 1 - mu**2
                return {'val' : val}

            def matrho(ts, coors, mode=None, rho=None, **kwargs):
                if mode != 'qp' : return
                if callable(rho): rho = rho(coors)
                r = coors[:, 0]
                val = rho*r**2
                val.shape = (coors.shape[0], 1, 1)
                return {'val' : val}

            coeffs = [1.0]
            terms = [[mat1, 'dw_diffusion']]
            _complete_terms(terms, densities, matrho)
            coeffs += [alpha]*(len(densities)-densities.count(0))

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
            wf = WeakForm(name_weak, meshfile, terms, **kwargswf)
            wfs_dic = {'weakforms' : wf}
            solver = Solver(wfs_dic, islinear=True, is_bounded=True, **kwargs)
            self.solver = solver
            
        else:
            raise Exception("Not implemented coordinates system: %s" %coorsys)

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
                               grad_renorm=grad_renorm, **kwargs)
        from femtoscope.display.fea_plot import pyvista_plot
        for result_file in out:
            pyvista_plot(result_file, **kwargs)


class PoissonSplit():
    r"""
    Class for solving the Poisson equation
    $$ \Delta \Phi = \alpha \times \rho $$
    on unbounded domains via domain splitting + kelvin inversion.

    Attributes
    ----------
    alpha : float
        Physical parameter weighting the lhs of Poisson's equation
        (dimensionless).
    Rcut : float
        Radius of the interior domain (disk in 2d / sphere in 3d).
    solver : Solver instance
        The FEM solver instance to be run.

    """

    def __init__(self, alpha, densities_int, densities_ext, meshfiles,
                 coorsys='cartesian', **kwargs):
        """
        Construct a Poisson problem instance.

        Parameters
        ----------
        alpha : float
            Physical parameter weighting the rhs of Poisson's equation
            (dimensionless).
        densities_int : list
            List of density functions or constants for the interior domain.
            Thelength of this list must match the number of sub*domains in the
            mesh.
        densities_ext : list
            List of density functions or constants for the exterior domain.
            Thelength of this list must match the number of sub*domains in the
            mesh.
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
        relax_pp :  float
            Relaxation parameter for ping-pong iterations (between 0 and 1).
            The default is 0.7.
        entity_functions_int : list of 2-tuple
            List of tuples (dim, function) for manual entity selection in the
            interior domain. The default is [].
        entity_functions_ext : list of 2-tuple
            List of tuples (dim, function) for manual entity selection in the
            exterior domain. The default is [].
        verbose : bool
            Display user's information. The default is True.

        """

        if not isinstance(meshfiles, list): meshfiles = [meshfiles]
        if not isinstance(densities_int, list): densities_int = [densities_int]
        if not isinstance(densities_ext, list): densities_ext = [densities_ext]
        is_vacuum = (len(densities_ext)==1 and
                     ((densities_ext[0] is None) or (densities_ext[0]==0.0)))
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
        fem_order = kwargs.get('fem_order', 2)
        ent_func_int = kwargs.get('entity_functions_int', [])
        ent_func_ext = kwargs.get('entity_functions_ext', [])
        verbose = kwargs.get('verbose', True)
        rho_vac = densities_ext[-1]
        self.alpha = alpha
        self.Rcut = Rcut

        name_weak_int = "wfin"
        name_weak_out = "wfout"

        # Cartesian coordinates
        if coorsys == 'cartesian':

            if dim == 2:

                def matint(ts, coors, mode=None, **kwargs):
                    if mode != 'qp' : return
                    x = coors[:, 0]
                    val = abs(x).reshape(coors.shape[0],1,1)
                    return {'val' : val}

                def matrho(ts, coors, mode=None, rho=None, **kwargs):
                    if mode != 'qp' : return
                    if callable(rho): rho = rho(coors)
                    x = coors[:, 0]
                    val = (rho * abs(x)).reshape(coors.shape[0],1,1)
                    return {'val' : val}

                coeffsint = [1.0]
                termsint = [[matint, 'dw_laplace']]
                _complete_terms(termsint, densities_int, matrho)
                coeffsint += [alpha]*(len(densities_int)-densities_int.count(0))

                if is_vacuum:
                    def matext1(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                        if mode != 'qp' : return
                        xi, eta = coors[:, 0], coors[:, 1]
                        norm2 = xi**2 + eta**2
                        norm = sqrt(norm2)
                        val = norm2/Rcut**2 * abs(xi) * (5 - 4*norm/Rcut)
                        return {'val' : val.reshape(coors.shape[0],1,1)}
    
                    def matext2(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                        if mode != 'qp' : return
                        val = np.zeros((coors.shape[0], 2, 1))
                        xi, eta = coors[:, 0], coors[:, 1]
                        norm2 = xi**2 + eta**2
                        norm = sqrt(norm2)
                        val[:, 0, 0] = 20*abs(xi)/Rcut**2 * (1-norm/Rcut) * xi
                        val[:, 1, 0] = 20*abs(xi)/Rcut**2 * (1-norm/Rcut) * eta
                        return {'val' : val}
    
                    coeffsext = [1.0, 1.0]
                    termsext = [[matext1, 'dw_laplace'],
                                [matext2, 'dw_s_dot_mgrad_s']]
    
                else:
                    def matext1(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                        if mode != 'qp' : return
                        xi, eta = coors[:, 0], coors[:, 1]
                        norm2 = xi**2 + eta**2
                        norm = sqrt(norm2)
                        val = norm2**2/Rcut**4 * abs(xi) * (7 - 6*norm/Rcut)
                        return {'val' : val.reshape(coors.shape[0],1,1)}
    
                    def matext2(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                        if mode != 'qp' : return
                        val = np.zeros((coors.shape[0], 2, 1))
                        xi, eta = coors[:, 0], coors[:, 1]
                        norm2 = xi**2 + eta**2
                        norm = sqrt(norm2)
                        val[:, 0, 0] = 42*norm2*abs(xi)/Rcut**4 * (1-norm/Rcut) * xi
                        val[:, 1, 0] = 42*norm2*abs(xi)/Rcut**4 * (1-norm/Rcut) * eta
                        return {'val' : val}
                    
                    def matext3(ts, coors, mode=None, rho_vac=rho_vac, **kwargs):
                        if mode != 'qp' : return
                        xi, eta = coors[:, 0], coors[:, 1]
                        norm = sqrt(xi**2 + eta**2)
                        val = abs(xi) * (7 - 6*norm/Rcut) * rho_vac
                        return {'val' : val.reshape(coors.shape[0], 1, 1)}

    
                    coeffsext = [1.0, 1.0, alpha]
                    termsext = [[matext1, 'dw_laplace'],
                                [matext2, 'dw_s_dot_mgrad_s'],
                                [matext3, 'dw_volume_integrate']]

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
                    kwargsint['dirichlet_bc_facet'] = [0.0]

                wf1 = WeakForm(name_weak_int, meshfiles[0], termsint,
                               **kwargsint) # interior weak form

                vertex_bc = [0.0] # value of the field at infinity
                kwargsout = {'constcoeffs' : coeffsext,
                            'unknown_name' : 'u2',
                            'test_name' : 'v2',
                            'order' : fem_order,
                            'integral_name' : 'i2',
                            'domain_name' : 'exterior',
                            'dirichlet_bc_vertex' : vertex_bc,
                            'entity_functions' : ent_func_ext,
                            'verbose' : verbose}

                wf2 = WeakForm(name_weak_out, meshfiles[1], termsext,
                               **kwargsout) # exterior weak form

                # Specify the region of dimension D-1 shared by the two meshes
                cogammas = [wf1.facets[-1], wf2.facets[0]]

                solver_kwargs = {'cogammas' : cogammas,
                                 'conn' : conn,
                                 'is_bounded'  : False,
                                 'relax_pp' : kwargs.get('relax_pp', 0.7),
                                 'max_iter_pp' : kwargs.get('max_iter_pp', 7),
                                 'verbose' : verbose}
                merge_dicts(solver_kwargs, kwargs)
                wfs_dic = {'weakforms' : [wf1, wf2]}
                solver = Solver(wfs_dic, islinear=True, **solver_kwargs)

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
                    neumann_term = [neumann,
                                    'dw_surface_integrate',
                                    cogammas[1].name]
                    wf2.add_term(neumann_term, update=True, newcoeff=1.0)

                self.solver = solver

            # 3D FEM simulations only available in cartesian coordinates
            elif dim == 3:

                def matrho(ts, coors, mode=None, rho=None, Rcut=Rcut,
                           **kwargs):
                    if mode != 'qp' : return
                    if callable(rho):
                        val = rho(coors)
                    else:
                        val = rho * np.ones(coors.shape[0], dtype=np.float64)
                    return {'val' : val.reshape(coors.shape[0], 1, 1)}

                coeffsint = [1.0]
                termsint = [[None, 'dw_laplace']]
                _complete_terms(termsint, densities_int, matrho)
                coeffsint += [alpha]*(len(densities_int)-densities_int.count(0))

                if is_vacuum:
                    def matext1(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                        if mode != 'qp' : return
                        xi, eta, zeta = coors[:, 0], coors[:, 1], coors[:, 2]
                        norm2 = xi**2 + eta**2 + zeta**2
                        norm = sqrt(norm2)
                        val = norm2/Rcut**2 * (5 - 4*norm/Rcut)
                        return {'val' : val.reshape(coors.shape[0],1,1)}
        
                    def matext2(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                        if mode != 'qp' : return
                        val = np.zeros((coors.shape[0], 3, 1))
                        xi, eta, zeta = coors[:, 0], coors[:, 1], coors[:, 2]
                        norm = sqrt(xi**2 + eta**2 + zeta**2)
                        val[:, 0, 0] = 20/Rcut**2 * (1 - norm/Rcut) * xi
                        val[:, 1, 0] = 20/Rcut**2 * (1 - norm/Rcut) * eta
                        val[:, 2, 0] = 20/Rcut**2 * (1 - norm/Rcut) * zeta
                        return {'val' : val}

                    coeffsext = [1.0, 1.0]
                    termsext = [[matext1, 'dw_laplace'],
                                [matext2, 'dw_s_dot_mgrad_s']]
                else:
                    def matext1(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                        if mode != 'qp' : return
                        xi, eta, zeta = coors[:, 0], coors[:, 1], coors[:, 2]
                        norm2 = xi**2 + eta**2 + zeta**2
                        norm = sqrt(norm2)
                        val = norm2**2/Rcut**4 * (7 - 6*norm/Rcut)
                        return {'val' : val.reshape(coors.shape[0],1,1)}
        
                    def matext2(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                        if mode != 'qp' : return
                        val = np.zeros((coors.shape[0], 3, 1))
                        xi, eta, zeta = coors[:, 0], coors[:, 1], coors[:, 2]
                        norm2 = xi**2 + eta**2 + zeta**2
                        norm = sqrt(norm2)
                        val[:, 0, 0] = 42*norm2/Rcut**4 * (1-norm/Rcut) * xi
                        val[:, 1, 0] = 42*norm2/Rcut**4 * (1-norm/Rcut) * eta
                        val[:, 2, 0] = 42*norm2/Rcut**4 * (1-norm/Rcut) * zeta
                        return {'val' : val}
                    
                    def matext3(ts, coors, mode=None, rho_vac=rho_vac,
                                Rcut=Rcut, **kwargs):
                        if mode != 'qp' : return
                        xi, eta, zeta = coors[:, 0], coors[:, 1], coors[:, 2]
                        norm = sqrt(xi**2 + eta**2 + zeta**2)
                        val = (7 - 6*norm/Rcut) * rho_vac
                        return {'val' : val.reshape(coors.shape[0],1,1)}

                    coeffsext = [1.0, 1.0, alpha]
                    termsext = [[matext1, 'dw_laplace'],
                                [matext2, 'dw_s_dot_mgrad_s'],
                                [matext3, 'dw_volume_integrate']]
                
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
                    kwargsint['dirichlet_bc_facet'] = [0.0]

                wf1 = WeakForm(name_weak_int, meshfiles[0], termsint,
                               **kwargsint) # interior weak form

                vertex_bc = [0.0]
                kwargsout = {'constcoeffs' : coeffsext,
                            'unknown_name' : 'u2',
                            'test_name' : 'v2',
                            'order' : fem_order,
                            'integral_name' : 'i2',
                            'domain_name' : 'exterior',
                            'dirichlet_bc_vertex' : vertex_bc,
                            'entity_functions' : ent_func_ext,
                            'verbose' : verbose}
                wf2 = WeakForm(name_weak_out, meshfiles[1], termsext,
                               **kwargsout) # exterior weak form

                # Specify the region of dimension D-1 shared by the two meshes
                cogammas = [wf1.facets[-1], wf2.facets[0]]
                solver_kwargs = {'cogammas' : cogammas,
                                 'conn' : conn,
                                 'is_bounded' : False,
                                 'relax_pp' : kwargs.get('relax_pp', 0.7),
                                 'max_iter_pp' : kwargs.get('max_iter_pp', 7),
                                 'verbose' : verbose}
                merge_dicts(solver_kwargs, kwargs)
                wfs_dic = {'weakforms' : [wf1, wf2]}
                solver = Solver(wfs_dic, islinear=True, **solver_kwargs)

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
                    neumann_term = [neumann,
                                    'dw_surface_integrate',
                                    cogammas[1].name]
                    wf2.add_term(neumann_term, update=True, newcoeff=1.0)

                self.solver = solver

            else:
                raise Exception("Dimension %d is not valid" %dim)

        # Polar coordinates
        elif coorsys == 'polar' and dim == 1:

            def matint(ts, r, mode=None, **kwargs):
                if mode != 'qp' : return
                r = r.squeeze()
                return {'val' : (r**2).reshape(r.shape[0], 1, 1)}

            def matrho(ts, r, mode=None, rho=densities_int[0], **kwargs):
                if mode != 'qp' : return
                r = r.squeeze()
                if callable(rho): rho = rho(r)
                return {'val' : (r**2 * rho).reshape(r.shape[0], 1, 1)}

            coeffsint = [1.0, alpha]
            termsint = [[matint, 'dw_laplace'],
                       [matrho, 'dw_volume_integrate']]
            if is_vacuum:
                coeffsext = [Rcut**2]
                termsext = [[None, 'dw_laplace']]
            else:
                def matext1(ts, eta, mode=None, **kwargs):
                    if mode != 'qp' : return
                    eta = eta.squeeze()
                    val = eta**4/Rcut**2 * (5 - 4*eta/Rcut)
                    return {'val' : val.reshape(eta.shape[0], 1, 1)}
                
                def matext2(ts, eta, mode=None, Rcut=Rcut, **kwargs):
                    if mode != 'qp' : return
                    eta = eta.squeeze()
                    val = 20*eta**3/Rcut**2 * (1-eta/Rcut)
                    return {'val' : val.reshape(eta.shape[0], 1, 1)}
                
                def matext3(ts, eta, mode=None, rho_vac=rho_vac, Rcut=Rcut,
                            **kwargs):
                    if mode != 'qp' : return
                    eta = eta.squeeze()
                    val = Rcut**2 * (5 - 4*eta/Rcut) * rho_vac
                    return {'val' : val.reshape(eta.shape[0], 1, 1)}
                    
                coeffsext = [1.0, 1.0, alpha]
                termsext = [[matext1, 'dw_laplace'],
                            [matext2, 'dw_s_dot_mgrad_s'],
                            [matext3, 'dw_volume_integrate']]

            kwargsint = {'constcoeffs' : coeffsint,
                        'unknown_name' : 'u1',
                        'test_name' : 'v1',
                        'order' : fem_order,
                        'integral_name' : 'i1',
                        'domain_name' : 'interior',
                        'entity_functions' : ent_func_int,
                        'verbose' : verbose}

            # interior weak form
            wf1 = WeakForm(name_weak_int, meshfiles[0], termsint, **kwargsint)

            vertex_dbc2 = [0.0, None]
            kwargsext = {'constcoeffs' : coeffsext,
                        'unknown_name' : 'u2',
                        'test_name' : 'v2',
                        'order' : fem_order,
                        'integral_name' : 'i2',
                        'domain_name' : 'exterior',
                        'dirichlet_bc_vertex' : vertex_dbc2,
                        'entity_functions' : ent_func_ext,
                        'verbose' : verbose}

            # exterior weak form
            wf2 = WeakForm(name_weak_out, meshfiles[-1], termsext, **kwargsext)

            # Specify the region of dimension D-1 shared by the two meshes
            cogammas = [wf1.vertices[0], wf2.vertices[1]]
            solver_kwargs = {'cogammas' : cogammas,
                             'conn' : conn,
                             'is_bounded': False,
                             'relax_pp' : kwargs.get('relax_pp', 0.7),
                             'max_iter_pp' : kwargs.get('max_iter_pp', 7),
                             'verbose' : verbose}
            merge_dicts(solver_kwargs, kwargs)
            wfs_dic = {'weakforms' : [wf1, wf2]}
            solver = Solver(wfs_dic, islinear=True, **solver_kwargs)

            self.solver = solver

        # Polar coordinates
        elif coorsys == 'polar' and dim == 2:

            def matint1(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                if mode != 'qp' : return
                val = np.zeros((coors.shape[0], 2, 2))
                r = coors[:, 0]
                theta = coors[:, 1]
                val[:, 0, 0] = sin(theta)*r**2
                val[:, 1, 1] = sin(theta)
                return {'val' : val}

            def matrho(ts, coors, mode=None, Rcut=Rcut, rho=None,
                       **kwargs):
                if mode != 'qp' : return
                if callable(rho): rho = rho(coors)
                r = coors[:, 0]
                theta = coors[:, 1]
                val = rho*sin(theta)*r**2
                val.shape = (coors.shape[0], 1, 1)
                return {'val' : val}

            coeffsint = [1.0]
            termsint = [[matint1, 'dw_diffusion']]
            _complete_terms(termsint, densities_int, matrho)
            coeffsint += [alpha]*(len(densities_int)-densities_int.count(0))

            if is_vacuum:
                
                def matext1(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                    if mode != 'qp' : return
                    val = np.zeros((coors.shape[0], 2, 2))
                    eta = coors[:, 0]
                    theta = coors[:, 1]
                    val[:, 0, 0] = (3-2*eta/Rcut) * sin(theta) * eta**2
                    val[:, 1, 1] = (3-2*eta/Rcut) * sin(theta)
                    return {'val' : val}
    
                def matext2(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                    if mode != 'qp' : return
                    val = np.zeros((coors.shape[0], 2, 1))
                    eta = coors[:, 0]
                    theta = coors[:, 1]
                    val[:, 0, 0] = 6*eta*(1-eta/Rcut) * sin(theta)
                    return {'val' : val}
    
                coeffsext = [1.0, 1.0]
                termsext = [[matext1, 'dw_diffusion'],
                            [matext2, 'dw_s_dot_mgrad_s']]
                
            else:
                
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
    
                def matext3(ts, coors, mode=None, Rcut=Rcut, rho_vac=rho_vac,
                            **kwargs):
                    if mode != 'qp' : return
                    eta = coors[:, 0]
                    theta = coors[:, 1]
                    val = Rcut**2 * sin(theta) * (5 - 4*eta/Rcut) * rho_vac
                    return {'val' : val.reshape(eta.shape[0], 1, 1)}
    
    
                coeffsext = [1.0, 1.0, alpha]
                termsext = [[matext1, 'dw_diffusion'],
                            [matext2, 'dw_s_dot_mgrad_s'],
                            [matext3, 'dw_volume_integrate']]
    
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
                             'relax_pp' : kwargs.get('relax_pp', 0.7),
                             'max_iter_pp' : kwargs.get('max_iter_pp', 7),
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
            
        elif coorsys == 'polar_mu' and dim == 2:

            def matint1(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                if mode != 'qp' : return
                val = np.zeros((coors.shape[0], 2, 2))
                r, mu = coors[:, 0], coors[:, 1]
                val[:, 0, 0] = r**2
                val[:, 1, 1] = 1 - mu**2
                return {'val' : val}

            def matrho(ts, coors, mode=None, Rcut=Rcut, rho=None,
                       **kwargs):
                if mode != 'qp' : return
                if callable(rho): rho = rho(coors)
                r = coors[:, 0]
                val = rho*r**2
                val.shape = (coors.shape[0], 1, 1)
                return {'val' : val}

            coeffsint = [1.0]
            termsint = [[matint1, 'dw_diffusion']]
            _complete_terms(termsint, densities_int, matrho)
            coeffsint += [alpha]*(len(densities_int)-densities_int.count(0))

            if is_vacuum:
                
                def matext1(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                    if mode != 'qp' : return
                    val = np.zeros((coors.shape[0], 2, 2))
                    eta, mu = coors[:, 0], coors[:, 1]
                    val[:, 0, 0] = (3-2*eta/Rcut) * eta**2
                    val[:, 1, 1] = (3-2*eta/Rcut) * (1-mu**2)
                    return {'val' : val}
    
                def matext2(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                    if mode != 'qp' : return
                    val = np.zeros((coors.shape[0], 2, 1))
                    eta = coors[:, 0]
                    val[:, 0, 0] = 6*eta*(1-eta/Rcut)
                    return {'val' : val}
    
                coeffsext = [1.0, 1.0]
                termsext = [[matext1, 'dw_diffusion'],
                            [matext2, 'dw_s_dot_mgrad_s']]
                
            else:
                
                def matext1(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                    if mode != 'qp' : return
                    val = np.zeros((coors.shape[0], 2, 2))
                    eta, mu = coors[:, 0], coors[:, 1]
                    val[:, 0, 0] = eta**4/Rcut**2 * (5 - 4*eta/Rcut)
                    val[:, 1, 1] = (1-mu**2)*(eta/Rcut)**2 * (5 - 4*eta/Rcut)
                    return {'val' : val}
    
                def matext2(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                    if mode != 'qp' : return
                    val = np.zeros((coors.shape[0], 2, 1))
                    eta = coors[:, 0]
                    val[:, 0, 0] = 20*eta**3/Rcut**2 * (1-eta/Rcut)
                    return {'val' : val}
    
                def matext3(ts, coors, mode=None, Rcut=Rcut, rho_vac=rho_vac,
                            **kwargs):
                    if mode != 'qp' : return
                    eta = coors[:, 0]
                    val = Rcut**2 * (5 - 4*eta/Rcut) * rho_vac
                    return {'val' : val.reshape(eta.shape[0], 1, 1)}
    
    
                coeffsext = [1.0, 1.0, alpha]
                termsext = [[matext1, 'dw_diffusion'],
                            [matext2, 'dw_s_dot_mgrad_s'],
                            [matext3, 'dw_volume_integrate']]
    
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
                             'relax_pp' : kwargs.get('relax_pp', 0.7),
                             'max_iter_pp' : kwargs.get('max_iter_pp', 7),
                             'verbose' : verbose}
            merge_dicts(solver_kwargs, kwargs)
            wfs_dic = {'weakforms' : [wf1, wf2]}
            solver = Solver(wfs_dic, islinear=True, **solver_kwargs)

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
                    val = grad1[:, 0]
                    return {'val' : val.reshape(coors.shape[0], 1, 1)}
                neumann_term = [neumann, 'dw_surface_integrate',
                                cogammas[1].name]
                wf2.add_term(neumann_term, update=True, newcoeff=Rcut**2)

            self.solver = solver

        else:
            raise Exception("Not implemented coordinates system: %s" %coorsys)

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


class PoissonCompact():
    r"""
    Class for solving the Poisson equation
    $$ \Delta \Phi = \alpha \times \rho $$
    on the whole space via the compactification technique.

    Attributes
    ----------
    alpha : float
        Physical parameter weighting the lhs of Poisson's equation
        (dimensionless).
    solver : Solver instance
        The FEM solver instance to be run.

    """

    def __init__(self, alpha, densities, meshfile, coorsys='polar', **kwargs):
        """
        Construct a Poisson problem instance.

        Parameters
        ----------
        alpha : float
            Physical parameter weighting the rhs of Poisson's equation
            (dimensionless).
        densities : list
            List of density functions or constants. The length of this list
            must match the number of sub*domains in the mesh.
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
        entity_functions : list of 2-tuple
            List of tuples (dim, function) for manual entity selection.
            The default is [].
        verbose : bool
            Display user's information. The default is True.

        """

        if not isinstance(densities, list): densities = [densities]
        mesh_dir = kwargs.get('mesh_dir', None)
        if type(meshfile) == str:
            dim = get_meshdim(meshfile, mesh_dir=mesh_dir)
        else: # 1D meshes are not saved as VTK files
            dim = 1
        if 'Rcut' in kwargs.keys():
            Rcut = kwargs['Rcut']
        else:
            Rcut = get_rcut(meshfile, mesh_dir=mesh_dir, coorsys=coorsys)
        self.Rcut = Rcut
        fem_order = kwargs.get('fem_order', 2)
        ent_func = kwargs.get('entity_functions', [])
        verbose = kwargs.get('verbose', True)
        self.alpha = alpha

        name_weak = "wfCompact"

        if coorsys == 'cartesian' or dim != 2:
            raise NotImplementedError(
                "compactification only available in 2D polar")

        elif coorsys == 'polar' and dim == 2:

            def mat1(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                if mode != 'qp' : return
                val = np.zeros((coors.shape[0], 2, 2))
                eta = coors[:, 0]
                theta = coors[:, 1]
                val[:, 0, 0] = (eta * (Rcut-eta))**2 * sin(theta)/Rcut
                val[:, 1, 1] = sin(theta)*Rcut
                return {'val' : val}

            def mat2(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                if mode != 'qp' : return
                val = np.zeros((coors.shape[0], 2, 1))
                eta = coors[:, 0]
                theta = coors[:, 1]
                val[:, 0, 0] = eta**2 * (Rcut-eta) * sin(theta)/Rcut
                return {'val' : val}

            def matrho(ts, coors, mode=None, Rcut=Rcut, rho=None,
                       **kwargs):
                if mode != 'qp' : return
                if callable(rho): rho = rho(coors)
                eta, theta = coors[:, 0], coors[:, 1]
                val = np.zeros(coors.shape[0])
                idx = np.where(rho != 0)[0]
                val[idx] = Rcut*rho[idx]*sin(theta[idx]) \
                    * (eta[idx]/(Rcut-eta[idx]))**2
                return {'val' : val.reshape(coors.shape[0], 1, 1)}

            coeffs = [1.0, -2.0]
            terms = [[mat1, 'dw_diffusion'],
                     [mat2, 'dw_s_dot_mgrad_s']]
            _complete_terms(terms, densities, matrho)
            coeffs += [alpha]*(len(densities)-densities.count(0))
            
            kwargswf = {'constcoeffs' : coeffs,
                        'unknown_name' : 'u1',
                        'test_name' : 'v1',
                        'order' : fem_order,
                        'densities' : densities,
                        'dirichlet_bc_facet' : [0.0],
                        'integral_name' : 'i1',
                        'domain_name' : 'interior',
                        'entity_functions' : ent_func,
                        'verbose' : verbose}
            wf = WeakForm(name_weak, meshfile, terms, **kwargswf)
            wfs_dic = {'weakforms' : wf}
            solver = Solver(wfs_dic, islinear=True, is_bounded=False,
                            verbose=False)
            self.solver = solver
            
        elif coorsys == 'polar_mu' and dim == 2:

            def mat1(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                if mode != 'qp' : return
                val = np.zeros((coors.shape[0], 2, 2))
                eta, mu = coors[:, 0], coors[:, 1]
                val[:, 0, 0] = (eta * (Rcut-eta))**2 / Rcut
                val[:, 1, 1] = (1-mu**2)*Rcut
                return {'val' : val}

            def mat2(ts, coors, mode=None, Rcut=Rcut, **kwargs):
                if mode != 'qp' : return
                val = np.zeros((coors.shape[0], 2, 1))
                eta = coors[:, 0]
                val[:, 0, 0] = eta**2 * (Rcut-eta) / Rcut
                return {'val' : val}

            def matrho(ts, coors, mode=None, Rcut=Rcut, rho=None,
                       **kwargs):
                if mode != 'qp' : return
                if callable(rho): rho = rho(coors)
                eta = coors[:, 0]
                val = np.zeros(coors.shape[0])
                idx = np.where(rho != 0)[0]
                val[idx] = Rcut*rho[idx]*(eta[idx]/(Rcut-eta[idx]))**2
                return {'val' : val.reshape(coors.shape[0], 1, 1)}

            coeffs = [1.0, -2.0]
            terms = [[mat1, 'dw_diffusion'],
                     [mat2, 'dw_s_dot_mgrad_s']]
            _complete_terms(terms, densities, matrho)
            coeffs += [alpha]*(len(densities)-densities.count(0))
            
            kwargswf = {'constcoeffs' : coeffs,
                        'unknown_name' : 'u1',
                        'test_name' : 'v1',
                        'order' : fem_order,
                        'densities' : densities,
                        'dirichlet_bc_facet' : [0.0],
                        'integral_name' : 'i1',
                        'domain_name' : 'interior',
                        'entity_functions' : ent_func,
                        'verbose' : verbose}
            wf = WeakForm(name_weak, meshfile, terms, **kwargswf)
            wfs_dic = {'weakforms' : wf}
            solver = Solver(wfs_dic, islinear=True, is_bounded=False,
                            verbose=False)
            self.solver = solver
            
        else:
            raise Exception("Not implemented coordinates system: %s" %coorsys)


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


def _complete_terms(terms, densities, matrho):
    """Add the density integrals to the list of terms (one per subdomain)"""
    for k in range(len(densities)):
        if densities[k] != 0.0:
            terms.append([matrho, 'dw_volume_integrate', 'subomega30'+str(k)])