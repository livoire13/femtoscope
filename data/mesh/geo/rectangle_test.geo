// Gmsh project created on Wed Feb 28 09:05:13 2024
//+
size = 0.2;
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {2, 0, 0, 1.0};
//+
Point(3) = {2, 1, 0, 1.0};
//+
Point(4) = {0, 1, 0, 1.0};
//+
Point(5) = {1, 0, 0, 1.0};
//+
Point(6) = {1, 1, 0, 1.0};
//+
Line(1) = {1, 5};
//+
Line(2) = {5, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 6};
//+
Line(5) = {6, 5};
//+
Line(6) = {1, 4};
//+
Line(7) = {4, 6};
//+
Curve Loop(1) = {6, 7, 5, -1};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {4, 5, 2, 3};
//+
Plane Surface(2) = {2};
//+
Physical Curve(200) = {1, 2, 3, 4, 7, 6};
//+
Physical Surface(300) = {1};
//+
Physical Surface(301) = {2};
//+
Mesh.MeshSizeFactor = size;