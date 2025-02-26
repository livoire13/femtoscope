// Gmsh project created on Thu Feb 09 11:11:05 2023

// Geometrical parameters
thetam = 0.05;
Rc = 7.0;

// Sizes
sizeMaxExt = 0.1;
sizeMinExt = 0.05;
sizeMinInt = 0.025;

Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0, thetam, 0, 1.0};
//+
Point(3) = {0, Pi, 0, 1.0};
//+
Point(4) = {Rc, Pi, 0, 1.0};
//+
Point(5) = {Rc, thetam, 0, 1.0};
//+
Point(6) = {Rc, 0.0, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 6};
//+
Line(6) = {6, 1};
//+
Line(7) = {2, 5};
//+
Curve Loop(1) = {2, 3, 4, -7};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {7, 5, 6, 1};
//+
Plane Surface(2) = {2};
//+
Physical Curve(201) = {4, 5};
//+
Physical Curve(203) = {2, 1};
//+
Physical Surface(300) = {1, 2};

Field[1] = MathEval;
Field[1].F = Sprintf("%g + %g*Atan(5*y)/(0.5*Pi) + %g", sizeMinInt, sizeMaxExt, thetam);

Field[2] = Distance;
Field[2].CurvesList = {1, 2};
Field[2].Sampling = 10000;
Field[3] = Threshold;
Field[3].DistMax = 0.5;
Field[3].DistMin = 0.0;
Field[3].InField = 2;
Field[3].Sigmoid = 1;
Field[3].SizeMax = sizeMaxExt;
Field[3].SizeMin = sizeMinExt;

Field[4] = Min;
Field[4].FieldsList = {1, 3};

Background Field = 4;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;

Mesh.Smoothing = 5;