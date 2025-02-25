// Gmsh project created on Wed Feb 08 17:52:02 2023

// Geometrical parameters
thetam = 0.05;
hm = 0;
Re = 1.0;
Rc = 7;
Rts = 0.99;

// Lines
x1 = 1.2;
x2 = 2;
x3 = 3;
x4 = 4;
x5 = 6;

// Sizes
distMaxJump = 1.5;
distMinJump = 0.0;
sizeMinInt = 0.025;
sizeMaxInt = 0.1;
sizeJump = 0.01;


Point(1) = {Rts, 0, 0, 1.0};
//+
Point(2) = {Rts, thetam, 0, 1.0};
//+
Point(3) = {Rts, Pi, 0, 1.0};
//+
Point(4) = {Re, Pi, 0, 1.0};
//+
Point(5) = {x1, Pi, 0, 1.0};
//+
Point(6) = {x2, Pi, 0, 1.0};
//+
Point(7) = {x3, Pi, 0, 1.0};
//+
Point(8) = {x4, Pi, 0, 1.0};
//+
Point(9) = {x5, Pi, 0, 1.0};
//+
Point(10) = {Rc, Pi, 0, 1.0};
//+
Point(11) = {Rc, thetam, 0, 1.0};
//+
Point(12) = {Rc, 0, 0, 1.0};
//+
Point(13) = {x5, 0, 0, 1.0};
//+
Point(14) = {x5, thetam, 0, 1.0};
//+
Point(15) = {x4, 0.0, 0, 1.0};
//+
Point(16) = {x4, thetam, 0, 1.0};
//+
Point(17) = {x3, 0.0, 0, 1.0};
//+
Point(18) = {x3, thetam, 0, 1.0};
//+
Point(19) = {x2, 0.0, 0, 1.0};
//+
Point(20) = {x2, thetam, 0, 1.0};
//+
Point(21) = {x1, 0.0, 0, 1.0};
//+
Point(22) = {x1, thetam, 0, 1.0};
//+
Point(23) = {Re+hm, 0, 0, 1.0};
//+
Point(24) = {Re+hm, 0.3*thetam, 0, 1.0};
//+
Point(25) = {Re, 0.7*thetam, 0, 1.0};
//+
Point(26) = {Re, thetam, 0, 1.0};
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
Line(6) = {6, 7};
//+
Line(7) = {7, 8};
//+
Line(8) = {8, 9};
//+
Line(9) = {9, 10};
//+
Line(10) = {10, 11};
//+
Line(11) = {11, 12};
//+
Line(12) = {12, 13};
//+
Line(13) = {13, 15};
//+
Line(14) = {15, 17};
//+
Line(15) = {17, 19};
//+
Line(16) = {19, 21};
//+
Line(17) = {21, 23};
//+
Line(18) = {23, 1};
//+
Line(19) = {5, 22};
//+
Line(20) = {22, 21};
//+
Line(21) = {19, 20};
//+
Line(22) = {20, 6};
//+
Line(23) = {7, 18};
//+
Line(24) = {18, 17};
//+
Line(25) = {15, 16};
//+
Line(26) = {16, 8};
//+
Line(27) = {9, 14};
//+
Line(28) = {14, 13};
//+
Line(29) = {2, 26};
//+
Line(30) = {26, 22};
//+
Line(31) = {22, 20};
//+
Line(32) = {20, 18};
//+
Line(33) = {18, 16};
//+
Line(34) = {16, 14};
//+
Line(35) = {14, 11};
//+
Line(36) = {4, 26};
//+
BSpline(37) = {26, 26, 25, 24, 23, 23};
//+
Curve Loop(1) = {2, 3, 36, -29};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {29, 37, 18, 1};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {36, 30, -19, -4};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {30, 20, 17, -37};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {19, 31, 22, -5};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {16, -20, 31, -21};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {22, 6, 23, -32};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {32, 24, 15, 21};
//+
Plane Surface(8) = {8};
//+
Curve Loop(9) = {23, 33, 26, -7};
//+
Plane Surface(9) = {9};
//+
Curve Loop(10) = {33, -25, 14, -24};
//+
Plane Surface(10) = {10};
//+
Curve Loop(11) = {25, 34, 28, 13};
//+
Plane Surface(11) = {11};
//+
Curve Loop(12) = {27, -34, 26, 8};
//+
Plane Surface(12) = {12};
//+
Curve Loop(13) = {9, 10, -35, -27};
//+
Plane Surface(13) = {13};
//+
Curve Loop(14) = {35, 11, 12, -28};
//+
Plane Surface(14) = {14};
//+
Physical Curve(201) = {10, 11};
//+
Physical Curve(203) = {2, 1};
//+
Physical Surface(300) = {1, 2};
//+
Physical Surface(301) = {3, 4, 6, 5, 7, 8, 10, 9, 12, 11, 14, 13};

Field[1] = MathEval;
Field[1].F = Sprintf("%g + (1-Fabs(x-%g)/(%g-%g)) * Min( %g*(Atan(0.5*y)/(Pi/2)) , Ceil(y-%g) ) + (Fabs(x-%g)/(%g-%g)) * ( %g*Atan(5*y)/(0.5*Pi) + %g )", sizeMinInt, Re, Rc, Re, sizeMaxInt, thetam, Re, Rc, Re, sizeMaxInt, thetam);

Field[4] = Distance;
Field[4].CurvesList = {36, 37};
Field[4].Sampling = 10000;

Field[5] = Threshold;
Field[5].DistMax = distMaxJump;
Field[5].DistMin = distMinJump;
Field[5].InField = 4;
Field[5].Sigmoid = 1;
Field[5].SizeMax = sizeMaxInt;
Field[5].SizeMin = sizeJump;

Field[6] = Min;
Field[6].FieldsList = {1, 5};

Background Field = 6;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;

Mesh.Smoothing = 5;