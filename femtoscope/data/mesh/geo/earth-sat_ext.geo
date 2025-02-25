// Gmsh project created on Thu Sep 07 11:24:05 2023

Rcut = 6;
maxSize = 0.2;
Ngamma = 120;
minSize = 0.01;

Point(1) = {0, 0, 0, 1e-2};
//+
Point(2) = {0, -Rcut, 0, 1.0};
//+
Point(3) = {0, Rcut, 0, 1.0};
//+
Circle(1) = {2, 1, 3};
//+
Line(2) = {2, 1};
//+
Line(3) = {1, 3};
//+
Curve Loop(1) = {1, -3, -2};
//+
Plane Surface(1) = {1};
//+
Physical Curve(200) = {1};
//+
Physical Surface(300) = {1};
//+
Physical Point(0) = {1};


// Mesh options

Transfinite Curve{1} = Ngamma;

Field[1] = Distance;
Field[1].PointsList = {1};

Field[2] = Threshold;
Field[2].DistMax = Rcut;
Field[2].DistMin = 1e-2;
Field[2].InField = 1;
Field[2].SizeMax = maxSize;
Field[2].SizeMin = minSize;

Background Field = 2;

Mesh.Algorithm = 6; // Frontal-Delaunay for 2D meshes

Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.Smoothing = 5;

