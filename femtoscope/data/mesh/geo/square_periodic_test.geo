// Gmsh project created on Wed Sep 04 13:53:01 2024
SetFactory("OpenCASCADE");
//+
size = 1.0;
//+
Rectangle(1) = {0, 0, 0, 1, 1, 0};
//+
Periodic Curve {4} = {2};
//+
Periodic Curve {3} = {1};
//+
Physical Surface(300) = {1};
//+
Mesh.MeshSizeFactor = size;