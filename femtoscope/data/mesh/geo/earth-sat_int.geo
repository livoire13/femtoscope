// Gmsh project created on Thu Sep 07 08:53:24 2023

ysat = 1.1;
Lsat = 5e-07;
Lcur = 2e-06;
Rcut = 5;
raison = 2.5;
maxSize = 0.2;
minSize = Lsat/40;
Ngamma = 150;
circleSize = 0.008;

Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0, -Rcut, 0, 1.0};
//+
Point(3) = {0, -1, 0, 1.0};
//+
Point(4) = {0, 1, 0, 1.0};
//+
Point(5) = {0, Rcut, 0, 1.0};
//+
Point(6) = {0, ysat-Lsat/2, 0, 1.0};
//+
Point(7) = {Lsat/2, ysat-Lsat/2, 0, 1.0};
//+
Point(8) = {Lsat/2, ysat+Lsat/2, 0, 1.0};
//+
Point(9) = {0, ysat+Lsat/2, 0, 1.0};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 1};
//+
Line(5) = {1, 4};
//+
Line(6) = {6, 7};
//+
Line(7) = {7, 8};
//+
Line(8) = {8, 9};
//+
Line(9) = {9, 6};
//+
Circle(1) = {2, 1, 5};
//+
Circle(2) = {3, 1, 4};
//+
Curve Loop(1) = {2, -5, -4};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {6, 7, 8, 9};
//+
Plane Surface(2) = {2};
//+
Physical Surface(300) = {1};
//+
Physical Surface(301) = {2};
//+
Physical Surface(302) = {};
//+
Physical Curve(200) = {1};

For t In {0:11}
  Lcur *= raison;
  ind = 10 + 4*t;
  ind_prev = ind - 4;
  lnd = 10 + 5*t;
  lnd_prev = lnd - 5;
  jnd = 3 + t;
  jnd_prev = jnd - 1;
  Point(ind+0) = {0, ysat-Lcur/2, 0, 1.0};
  Point(ind+1) = {Lcur/2, ysat-Lcur/2, 0, 1.0};
  Point(ind+2) = {Lcur/2, ysat+Lcur/2, 0, 1.0};
  Point(ind+3) = {0, ysat+Lcur/2, 0, 1.0};
  Line(lnd+0) = {ind_prev+0, ind+0};
  Line(lnd+1) = {ind+0, ind+1};
  Line(lnd+2) = {ind+1, ind+2};
  Line(lnd+3) = {ind+2, ind+3};
  Line(lnd+4) = {ind+3, ind_prev+3};
  Curve Loop(jnd) = {lnd+0, lnd+1, lnd+2, lnd+3, lnd+4, -(lnd_prev+3), -(lnd_prev+2), -(lnd_prev+1)};
  Plane Surface(jnd) = {jnd};
  Physical Surface(302) += {jnd};
EndFor

Line(lnd+5) = {4, ind+0};
Line(lnd+6) = {ind+3, 5};
Curve Loop(jnd+1) = {3, 2, lnd+5, lnd+1, lnd+2, lnd+3, lnd+6, -1};
Plane Surface(jnd+1) = {jnd+1};
Physical Surface(302) += {jnd+1};


// Mesh options

Transfinite Curve{1} = Ngamma;
//Transfinite Curve{6} = Lsat/(2*minSize);
//Transfinite Curve{7} = 3*Lsat/minSize;
//Transfinite Curve{8} = Lsat/(2*minSize);
//Transfinite Curve{9} = 3*Lsat/minSize;
//Transfinite Surface {2};

Field[1] = Distance;
Field[1].CurvesList = {2};
Field[1].Sampling = 1000;

Field[2] = Threshold;
Field[2].DistMax = Rcut+2;
Field[2].DistMin = 0;
Field[2].InField = 1;
Field[2].SizeMax = maxSize;
Field[2].SizeMin = circleSize;

Field[3] = Box;
Field[3].VIn = minSize;
Field[3].VOut = maxSize;
Field[3].XMin = 0.0;
Field[3].XMax = Lsat/2;
Field[3].YMin = ysat - Lsat/2;
Field[3].YMax = ysat + Lsat/2;
Field[3].Thickness = 50*Lcur;

Field[4] = Min;
Field[4].FieldsList = {2, 3};

Background Field = 4;

Mesh.Algorithm = 6; // Frontal-Delaunay for 2D meshes

Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.Smoothing = 10;

