# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 12:36:29 2022

Tools for creating and manipulating weak form PDEs created out of Sfepy
objects.

@author: hlevy
"""

from femtoscope.inout.meshfactory import (get_meshsource,
                                          get_physical_group_ids)
import numbers
import numpy as np
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term, term_table
from sfepy.discrete import (FieldVariable, Integral, Equation, Material,
                            Function)
from sfepy.discrete.conditions import EssentialBC
from femtoscope.misc.util import get_default_args
from femtoscope.misc.util import numpyit


class WeakForm:
    r"""
    Class encapsulating all Sfepy objects for defining a weak form PDE.

    Attributes
    ----------
    name : str
        Name of the weak form.
    meshfile : str
        Name of the meshfile e.g. 'my_mesh.vtk' located in the specified mesh
        directory.
    terms : list
        List containing the terms of the weak form PDE. Each element of the
        list is a list containing:
            - a material function or None.
            - a Sfepy string of the term, e.g. 'dw_laplace', or
            'dw_volume'. https://sfepy.org/doc-devel/terms_overview.html.
            - a region's name e.g. 'facet200', 'subomega301', or None.
    verbose : bool
        Display user's information. The default is False.
    mesh_dir : str
        Absolute path to the directory containing meshfile. The default is
        None.
    ext : str
        Extension given to the mesh file. The default is '.vtk'.
    constcoeffs : list
        Coefficients weighting the terms such that the weak form equation
        reads
            $$ \sum_{i=1}^{N_{\mathrm{terms}}} \texttt{constcoeffs[i]} \times
            \texttt{terms[i]} = 0 $$
    unknown : `sfepy.discrete.variables.FieldVariable`
        Sfepy unknown variable.
    unknown_name : str
        Name of the unknown variable. The default is 'u'.
    test : `sfepy.discrete.variables.FieldVariable`
        Sfepy test variable.
    test_name : str
        Name of the test variable. The default is 'v'.
    order : int
        The FE approximation order. The default is 2.
    space : str
        The function space name. The default is 'H1'.
    poly_space_base : str
        The name of polynomial space base. The default is 'lagrange'.
    integral : `sfepy.discrete.integrals.Integral`
        Sfepy Integral instance.
    integral_name : str
        The integral name. The default is 'i'.
    domain : `sfepy.discrete.fem.domain.FEDomain`
        Sfepy Domain instance.
    domain_name : str
        The domain name. The default is 'domain'.
    omega : Region
        The whole domain over which the variable are defined.
    dimension : int
        Spatial dimension of the problem (3 is the maximum).
    physical_group_ids : list of int
        All distinct physical groups identifiers.
    region_dic : dict
        Dictionary of all regions.
    subomegas : list
        List of sub regions of dimension D.
    facets : list
        List of sub regions of dimension D-1.
    edges : list
        List of sub regions of dimension 1.
    vertices : list
        List of sub regions of dimension 0.
    epbcs : list
        List of periodic boundary conditions.
    lcbcs : list
        List of linear combination boundary conditions.
    ebcs : list
        List of essential boundary conditions.
    ndofs : int
        Number of degrees of freedom.
    entity_functions : list of 2-tuple
        List of tuples (dim, function) for manual entity selection.
        The default is [].

    """

    """
    Class variable `forbidden_mod_arg` is a list of material extra-arguments
    (strings) which are not to be interpreted as 'mod' terms. Any term having
    at least one extra-argument not contained in this list will be casted into
    the `self.eqn_mod` attribute.
    """
    forbidden_mod_arg = ['update_nl', 'rho', 'rho_vac', 'Rcut']

    def __init__(self, name, meshfile, terms, **kwargs):
        r"""
        Construct a WeakForm instance that is an object containing all the
        necessary information related to a given weak form PDE.

        Parameters
        ----------
        name : str
            Name of the weak form.
        meshfile : str
            Name of the meshfile e.g. 'my_mesh.vtk' located in the specified
            mesh directory.
        terms : list
            List containing the terms of the weak form PDE. Each element of the
            list is a list containing:
                - a material function or None.
                - a Sfepy string of the term, e.g. 'dw_laplace', or
                'dw_volume'. https://sfepy.org/doc-devel/terms_overview.html.
                - a region's name e.g. 'facet200', 'subomega301', or None.

        Other Parameters
        ----------------
        mesh_dir : str
            Directory where the mesh files are located. The default is None.
        ext : str
            Extension of the mesh file. The default is '.vtk'.
        constcoeffs : list
            Coefficients weighting the terms such that the weak form equation
            reads
                $$ \sum_{i=1}^{N_{\mathrm{terms}}} \texttt{constcoeffs[i]}
                \times \texttt{terms[i]} = 0 $$
        dirichlet_bc_facet : list
            List of dirichlet boundary conditions applied to facets. This list
            must be the same length as the number of facets defined in the
            domain. Elements of the list are such that:
                - if `dirichlet_bc_facet`[k] is None, then facets[k] is free of
                any essential boundary condition.
                - if `dirichlet_bc_facet`[k] is a number, then facets[k] is
                constrained by a constant essential boundary condition.
                - if `dirichlet_bc_facet`[k] is a function, then facets[k] is
                constrained by a dirichlet boundary condition calculated by the
                function.
            The default is None.
        dirichlet_bc_edge : list
            List of dirichlet boundary conditions applied to edges. This list
            must be the same length as the number of edges defined in the
            domain. The default is None.
        dirichlet_bc_vertex : list
            List of dirichlet boundary conditions applied to vertices. This
            list must be the same length as the number of edges defined in the
            domain. The default is None.
        verbose : bool
            Display user's information. The default is False.
        domain_name : str
            The domain name. The default is 'domain'.
        integral_name : str
            The integral name. The default is 'i'.
        space : str
            The function space name. The default is 'H1'.
        poly_space_base : str
            The name of polynomial space base. The default is 'lagrange'.
        test_name : str
            Name of the test variable. The default is 'v'.
        unknown_name : str
            Name of the unknown variable. The default is 'u'.
        densities : list
            List of density functions or constants. The length of this list
            must match the number of sub*domains in the mesh.
            The default is None.
        entity_functions : list of 2-tuple
            List of tuples (dim, function) for manual entity selection.
            The default is [].

        """

        # keyword arguments handling
        mesh_dir = kwargs.get('mesh_dir', None)
        ext = kwargs.get('ext', '.vtk')
        constcoeffs = kwargs.get('constcoeffs', [1]*len(terms))
        dirichlet_bc_vertex = kwargs.get('dirichlet_bc_vertex', None)
        dirichlet_bc_edge = kwargs.get('dirichlet_bc_edge', None)
        dirichlet_bc_facet = kwargs.get('dirichlet_bc_facet', None)
        dbc_list = [dirichlet_bc_vertex, dirichlet_bc_edge, dirichlet_bc_facet]

        self.verbose = kwargs.get('verbose', False)
        self.name = name
        self.constcoeffs = constcoeffs
        self.unknown_name = kwargs.get('unknown_name', 'u')
        self.test_name = kwargs.get('test_name', 'v')
        self.order = kwargs.get('order', 2)
        self.space = kwargs.get('space', 'H1')
        self.poly_space_base = kwargs.get('poly_space_base', 'lagrange')
        self.integral_name = kwargs.get('integral_name', 'i')
        self.domain_name = kwargs.get('domain_name', 'domain')
        self.nterms = len(terms)
        if type(meshfile) == str:
            self.meshfile = get_meshsource(meshfile, mesh_dir, ext)
            self.mesh = Mesh.from_file(self.meshfile)
            self.physical_group_ids = get_physical_group_ids(self.meshfile)
        else:
            self.mesh = meshfile
            self.physical_group_ids = []
        self.domain = FEDomain(self.domain_name, self.mesh)
        self.omega = self.domain.create_region('omega', 'all')
        self.dimension = self.omega.dim
        self.entity_functions = kwargs.get('entity_functions', [])

        assert len(terms)==len(constcoeffs)

        # Regions (cells, facets, edges, vertices):
        # - embedded in the mesh file
        self.region_dic = {'omega' : self.omega}
        subomegas, facets, edges, vertices = [], [], [], []
        # subool = len([x for x in self.physical_group_ids if x >= 300]) >= 2
        for group_id in self.physical_group_ids:
            if group_id >= 0 and group_id <= 99:
                vertices.append(self.domain.create_region(
                    'node%d' %group_id,
                    'vertices of group %d' %group_id, # selection of vertices
                    kind='vertex'))
                self.region_dic['node'+str(group_id)] = vertices[-1]
            elif group_id >= 100 and group_id <= 199:
                edges.append(self.domain.create_region(
                    'edge%d' %group_id,
                    'vertices of group %d' %group_id, # selection of vertices
                    kind='edge'))
                self.region_dic['edge'+str(group_id)] = edges[-1]
            elif group_id >= 200 and group_id <= 299:
                facets.append(self.domain.create_region(
                    'facet%d' %group_id,
                    'vertices of group %d' %group_id, # selection of vertices
                    kind='facet'))
                self.region_dic['facet'+str(group_id)] = facets[-1]
            elif group_id >= 300:
                subomegas.append(self.domain.create_region(
                    'subomega%d' %group_id,
                    'cells of group %d' %group_id, # selection of cells
                    kind='cell'))
                self.region_dic['subomega'+str(group_id)] = subomegas[-1]

        # - specified manually through a function of coordinates
        for dimFunc in self.entity_functions:
            dim, func = dimFunc[0], dimFunc[1]
            funcname = func.__name__
            if dim == 0:
                nb = len(vertices) + 10
                vertices.append(self.domain.create_region(
                    'node%d' %nb,
                    'vertices by %s' %funcname,
                    kind='vertex',
                    functions={funcname : func}))
                self.region_dic['node'+str(nb)] = vertices[-1]

            elif dim == 1 and self.dimension == 3:
                nb = 100 + len(edges) + 10
                edges.append(self.domain.create_region(
                    'edge%d' %nb,
                    'vertices by %s' %funcname,
                    kind='edge',
                    functions={funcname : func}))
                self.region_dic['edge'+str(nb)] = edges[-1]

            elif (dim == 2 and self.dimension == 3) or (
                    dim == 1 and self.dimension == 2):
                nb = 200 + len(facets) + 10
                facets.append(self.domain.create_region(
                    'facet%d' %nb,
                    'vertices by %s' %funcname,
                    kind='facet',
                    functions={funcname : func}))
                self.region_dic['facet'+str(nb)] = facets[-1]

            elif dim == 3:
                nb = 300 + len(subomegas) + 10
                subomegas.append(self.domain.create_region(
                    'subomega%d' %nb,
                    'cells by %s' %funcname,
                    kind='cell',
                    functions={funcname : func}))
                self.region_dic['subomega'+str(nb)] = subomegas[-1]

        self.subomegas = subomegas
        self.facets = facets
        self.edges = edges
        self.vertices = vertices

        # Density profiles
        densities = kwargs.get('densities', None)
        if densities is not None:
            if len(densities) > len(subomegas):
                raise Exception("More density profiles than sub-domains")
            self.densities = {}
            for k in range(len(densities)):
                self.densities[subomegas[k].name] = densities[k]

        # Field setup
        self.field = Field.from_args('fu', np.float64, 'scalar', self.omega,
                                     approx_order=self.order, space=self.space,
                                     poly_space_base=self.poly_space_base)
        self.ndofs = self.field.n_nod
        self.unknown = FieldVariable(self.unknown_name, 'unknown', self.field)
        self.test = FieldVariable(self.test_name, 'test', self.field,
                                  primary_var_name=self.unknown_name)
        self.integral = Integral(self.integral_name, order = 2*self.order+1)

        # Dirichlet boundary conditions on entities of dimension < D
        ent_list = [self.vertices, self.edges, self.facets]
        ebcs_list = []
        for k_dbcs, dbcs in enumerate(dbc_list):
            if dbcs is not None:
                assert len(dbcs) == len(ent_list[k_dbcs])
                for k_dbc, dbc in enumerate(dbcs):
                    if dbc is not None:
                        ebc_name = 'ebc_'+str(k_dbcs)+str(k_dbc)
                        if isinstance(dbc, numbers.Number):
                            ebc = EssentialBC(
                                ebc_name,
                                ent_list[k_dbcs][k_dbc],
                                {'%s.all' %self.unknown_name : dbc})
                        elif callable(dbc):
                            """ Function signature should be 
                            dbc(ts, coors, **kwargs) """
                            ebc_func_name = 'ebc_func_'+str(k_dbcs)+str(k_dbc)
                            ebc_func = Function(ebc_func_name, dbc)
                            ebc = EssentialBC(
                                ebc_name,
                                ent_list[k_dbcs][k_dbc],
                                {'%s.all' %self.unknown_name : ebc_func})
                        ebcs_list.append(ebc)
        self.ebcs = ebcs_list
        self.epbcs = []
        self.lcbcs = []

        # Terms definition
        self._terms = terms
        self.terms = []
        self.eqn = None
        for materm in terms:
            self.add_term(materm)

        # Equation
        self.eqn = WeakForm.make_equation(self.terms,
                                          self.constcoeffs,
                                          self.name)
        self.eqn_cst = None
        self.eqn_mod = None
        self.split_equation()

        # Display information for the user
        if self.verbose >= 2:
            print(self.__str__())

    def add_term(self, materm, update=False, newcoeff=1.0):
        """
        Add a term to the existing WeakForm object.

        Parameters
        ----------
        materm : list of length 2 or 3 (one element of terms).
            A list containing:
                - a material function or None.
                - a Sfepy string of the term, e.g. 'dw_laplace', or
                'dw_volume'. https://sfepy.org/doc-devel/terms_overview.html
                - a region's name e.g. 'facet200', 'subomega301', or None.
        update : bool, optional
            Update self.equation with new term. The default is False.
        newcoeff : float, optional
            If update, the new term multiplicative constant.
            The default is 1.0.

        Returns
        -------
        None.

        """
        material = materm[0]
        term_str = materm[1]
        ttkw = WeakForm.get_term_table_kw(term_table, term_str)
        
        # Determine the region over which integral is defined
        if len(materm)==3:
            if materm[2] is not None:
                name = materm[2] # the region's name was given by the user
        else:
            name = 'omega' # the region's name was not given, default is omega
            
        # assert term_str in term_table
        kwargs = {self.unknown_name : self.unknown,
                  self.test_name : self.test}

        if material is None:
            if 'state' not in ttkw and 'virtual/grad_state' not in ttkw:
                new_term = Term.new(
                    '%s(%s)' %(term_str, self.test_name),self.integral,
                    self.region_dic[name], **kwargs)
            else:
                new_term = Term.new(
                    '%s(%s, %s)' %(term_str,self.test_name,self.unknown_name),
                    self.integral, self.region_dic[name], **kwargs)

        elif callable(material):
            mat_func  = Function('mat_func', material)
            extra_args = get_default_args(material)
            matname = material.__name__
            # density profile linked to subregion
            if 'rho' in extra_args and extra_args['rho'] is None:
                extra_args['rho'] = self.densities[name]
                matname += '_' + name
            mat = Material(matname, kind='stationary', function=mat_func)
            mat.set_extra_args(**extra_args)
            kwargs[matname] = mat
            if 'state' not in ttkw and 'virtual/grad_state' not in ttkw:
                new_term = Term.new(
                    '%s(%s.val, %s)' %(term_str, matname, self.test_name),
                    self.integral, self.region_dic[name], **kwargs)
            else:
                new_term = Term.new(
                    '%s(%s.val, %s, %s)' %(term_str, matname, self.test_name,
                    self.unknown_name), self.integral,
                    self.region_dic[name], **kwargs)
        else:
            raise Exception("Material should be a function or None")

        self.terms.append(new_term)
        if update==True:
            self._terms.append(materm)
            self.constcoeffs.append(newcoeff)
            self.eqn = WeakForm.make_equation(self.terms,
                                              self.constcoeffs,
                                              self.name)
            self.split_equation()

    @staticmethod
    def make_equation(terms, constcoeffs, name):
        """Set the eqn attribute of the WeakForm object."""
        if terms:
            eqn = terms[0] * constcoeffs[0]
            for i in range(1, len(constcoeffs)):
                eqn = eqn.__add__(terms[i] * constcoeffs[i])
            eqn = Equation('eqn_%s' %name, eqn)
            return eqn

    def split_equation(self):
        """Split the equation into constant VS updatable equations."""
        terms_cst = []
        terms_mod = []
        constcoeffs_cst = []
        constcoeffs_mod = []
        mats = []
        terms = self._terms
        mats_aux = self.eqn.collect_materials()
        if len(mats_aux) != len(terms):
            i = 0
            for k in range(len(terms)):
                func = terms[k][0]
                if func is None:
                    mats.append(None)
                else:
                    mats.append(mats_aux[i])
                    i += 1
        else:
            mats = mats_aux

        for coeff, mat, term in zip(self.constcoeffs, mats, self.terms):
            isconstant = True
            if hasattr(mat, 'extra_args') and mat.extra_args != {}:
                extra_args = mat.extra_args # fetch material's extra-args dict
                for key in list(extra_args.keys()):
                    if key not in WeakForm.forbidden_mod_arg:
                        isconstant = False
                        break
            if isconstant:
                terms_cst.append(term)
                constcoeffs_cst.append(coeff)
            else:
                terms_mod.append(term)
                constcoeffs_mod.append(coeff)
        if terms_cst:
            self.eqn_cst = WeakForm.make_equation(terms_cst, constcoeffs_cst,
                                                  'eqn_cst')
        if terms_mod:
            self.eqn_mod = WeakForm.make_equation(terms_mod, constcoeffs_mod,
                                                  'eqn_mod')

    def __str__(self):
        string = f"""
        Weak Form:
            name: {self.name}
            """
        if hasattr(self, 'meshfile'):
            string += f"""
            mesh location: {self.meshfile}
            """
            string += f"""
            Ndofs: {self.ndofs}
            Approximation order: {self.order}
            Space: {self.space}
            Polynomial space base: {self.poly_space_base}
            Unknown variable: {self.unknown_name}
            Test variable: {self.test_name}
            Integral, Order: {self.integral.name}, {self.integral.order}
            Physical Groups: {self.physical_group_ids}
            Dirichlet BC: {self.ebcs}
            Periodic BC: {self.epbcs}
            Linear Combination BC: {self.lcbcs}
            Terms: {self.terms}
            Constant coefficients: {self.constcoeffs}
            Equation name: {self.eqn.name}
            """
        return string

    @staticmethod
    def get_term_table_kw(term_table, term_str):
        """Utility method for getting the syntax of a term given by its string
        identifier.  Might break with future Sfepy releases, see
        https://github.com/sfepy/sfepy/issues/773"""
        entry = np.unique(numpyit(term_table[term_str].arg_types).reshape(-1))
        return list(entry)