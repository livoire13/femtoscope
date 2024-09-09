// Gmsh project created on Sat Jul 20 18:31:36 2024
SetFactory("OpenCASCADE");
//+
size = 0.2;
//+
Sphere(1) = {0, 0, 0, 0.25, -Pi/2, Pi/2, 2*Pi};
//+
Box(2) = {-0.5, -0.5, -0.5, 1, 1, 1};
//+
BooleanDifference(3) = { Volume{2}; Delete; }{ Volume{1}; };
//+
Periodic Surface {7} = {2} Translate {1, 0, 0};
//+
Periodic Surface {5} = {3} Translate {0, 1, 0};
//+
Periodic Surface {4} = {6} Translate {0, 0, 1};
//+
Physical Volume(300) = {1};
//+
Physical Volume(301) = {3};
//+
Mesh.MeshSizeFactor = size;