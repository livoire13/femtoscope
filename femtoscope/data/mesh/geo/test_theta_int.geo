// Gmsh project created on Sun Jan 15 10:08:13 2023

Rts = 0.0;
Rc = 5.0;
sa = 1.0;
size = 0.1;
Ngamma = 40;
size_min = 0.03;

//+
Point(1) = {Rts, 0, 0, 1.0};
//+
Point(2) = {Rts, Pi, 0, 1.0};
//+
Point(3) = {Rc, Pi, 0, 1.0};
//+
Point(4) = {Rc, 0, 0, 1.0};
//+
Point(5) = {sa, 0, 0, 1.0};
//+
Point(6) = {sa, Pi, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 6};
//+
Line(3) = {6, 3};
//+
Line(4) = {3, 4};
//+
Line(5) = {4, 5};
//+
Line(6) = {5, 1};
//+
Line(7) = {5, 6};
//+
Curve Loop(1) = {1, 2, -7, 6};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {7, 3, 4, 5};
//+
Plane Surface(2) = {2};
//+
Physical Curve(201) = {4};
//+
Physical Curve(203) = {1};
//+
Physical Surface(300) = {1};
//+
Physical Surface(301) = {2};
//+
Transfinite Curve {4} = Ngamma Using Progression 1;
//+
Field[1] = Constant;
//+
Field[1].IncludeBoundary = 1;
//+
Field[1].VIn = size;
//+
Field[1].VOut = size;
//+
Field[2] = Distance;
//+
Field[2].CurvesList = {7};
//+
Field[2].Sampling = 1000;
//+
Field[3] = Threshold;
//+
Field[3].DistMax = 0.7;
//+
Field[3].DistMin = 0;
//+
Field[3].InField = 2;
//+
Field[3].Sigmoid = 1;
//+
Field[3].SizeMax = size;
//+
Field[3].SizeMin = size_min;
//+
Field[4] = Min;
//+
Field[4].FieldsList = {1, 3};
//+
Background Field = 4;
