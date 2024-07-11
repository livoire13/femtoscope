// Gmsh project created on Mon Jan 16 10:24:18 2023

Rc = 5.0;
size = 0.2;
Ngamma = 23;

//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0, Pi, 0, 1.0};
//+
Point(3) = {Rc, Pi, 0, 1.0};
//+
Point(4) = {Rc, 0, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Physical Curve(201) = {3};
//+
Physical Curve(203) = {1};
//+
Physical Surface(300) = {1};
//+
Transfinite Curve {3} = Ngamma Using Progression 1;
//+
Field[1] = Constant;
//+
Field[1].IncludeBoundary = 1;
//+
Field[1].VIn = size;
//+
Field[1].VOut = size;
//+
Background Field = 1;
