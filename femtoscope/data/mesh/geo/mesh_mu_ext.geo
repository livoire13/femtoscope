// Gmsh project created on Mon Jan 16 10:24:18 2023

Rc = 5.0;
size = 0.1;
Ngamma = 30;
better_gamma = 0;

//+
Point(1) = {0, -1, 0, 1.0};
//+
Point(2) = {0, 1, 0, 1.0};
//+
Point(3) = {Rc, 1, 0, 1.0};
//+
Point(4) = {Rc, -1, 0, 1.0};
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
If (better_gamma == 1)
  Field[2] = MathEval;
  Field[2].F = Sprintf("2*(1+%g-x)/%g", Rc, Ngamma);
  Field[3] = Min;
  Field[3].FieldsList = {1, 2};
  Background Field = 3;
Else
  Background Field = 1;
EndIf
