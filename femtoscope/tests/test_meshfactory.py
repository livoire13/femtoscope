# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:42:26 2024
Testing of the `meshfactory` module.
"""

import os
os.environ["FEMTOSCOPE_TEST_MODE"] = "1"

from pathlib import Path

import pytest

from femtoscope import MESH_DIR
from femtoscope.inout import meshfactory


@pytest.fixture
def mt2():
    return meshfactory.MeshingTools(2)


@pytest.fixture
def mt3():
    return meshfactory.MeshingTools(3)


def test_generate_mesh_from_gmsh_api(mt2):
    mt2.create_ellipse()
    mt2.create_subdomain()
    mt2.generate_mesh("test_mesh_from_gmsh_api.vtk", show_mesh=False)
    mesh_path_name = Path(MESH_DIR / "test_mesh_from_gmsh_api.vtk")
    assert mesh_path_name.is_file()
    mesh_path_name.unlink()


def test_split_spheres_from_mesh(mt3):
    mt3.Rcut = 1.0
    mt3.create_ellipsoid(rx=1,  ry=1, rz=1)
    mt3.create_subdomain()
    mt3.create_inversed_exterior_domain_3d()
    mt3.create_subdomain()
    mt3.generate_mesh("test_mesh_split_spheres.vtk", show_mesh=False)
    mesh_path_name = str(Path(MESH_DIR / "test_mesh_split_spheres.vtk"))
    meshfactory.split_spheres_from_mesh(mesh_path_name, 1.0)
    mesh_path_name_int = Path(MESH_DIR / "test_mesh_split_spheres_int.vtk")
    mesh_path_name_ext = Path(MESH_DIR / "test_mesh_split_spheres_ext.vtk")
    assert mesh_path_name_int.is_file() and mesh_path_name_ext.is_file()
    mesh_path_name_int.unlink()
    mesh_path_name_ext.unlink()


@pytest.mark.parametrize("dim", [2, 3])
def test_read_dimension_from_mesh_file(dim):
    mt = meshfactory.MeshingTools(dim)
    if dim == 2:
        mt.create_ellipse()
    if dim == 3:
        mt.create_ellipsoid()
    mt.create_subdomain()
    mesh_path_name = mt.generate_mesh("test_mesh_read_dimension.vtk")
    assert meshfactory.read_dimension_from_mesh_file(mesh_path_name) == dim
    Path(mesh_path_name).unlink()


@pytest.mark.parametrize("dim, coorsys",
                         [(2, 'cartesian'), (2, 'polar'), (3, 'cartesian')])
def test_read_Rcut_from_mesh_file(dim, coorsys):
    Rcut = 1.0
    mt = meshfactory.MeshingTools(dim)
    if dim == 2:
        mt.create_ellipse(rx=Rcut, ry=Rcut)
    if dim == 3:
        mt.create_ellipsoid(rx=Rcut, ry=Rcut, rz=Rcut)
    mt.create_subdomain()
    mesh_path_name = mt.generate_mesh("test_mesh_read_Rcut.vtk")
    r = meshfactory.read_Rcut_from_mesh_file(mesh_path_name, coorsys=coorsys)
    assert abs(Rcut - r) < 1e-8
    Path(mesh_path_name).unlink()


def test_generate_mesh_from_geo():
    param_dict = {'size': 0.2}
    mesh_name = 'rec_test.vtk'
    meshfactory.generate_mesh_from_geo(
        'rectangle_test.geo', param_dict=param_dict, mesh_name=mesh_name)
    mesh_path_name = Path(MESH_DIR / mesh_name)
    assert mesh_path_name.is_file()
    mesh_path_name.unlink()
