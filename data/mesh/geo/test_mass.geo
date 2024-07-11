// Gmsh project created on Fri Aug 06 10:19:21 2021
//+
Point(1) = {0, 0, -4, 1.0};
//+
Point(2) = {2, 0, -4, 1.0};
//+
Point(3) = {0, 2, -4, 1.0};
//+
Point(4) = {-2, 0, -4, 1.0};
//+
Point(5) = {0, -2, -4, 1.0};
//+
Rotate {{0, 0, 1}, {0, 0, 0}, Pi/10} {
  Duplicata { Point{2}; Point{3}; Point{4}; Point{5}; }
}
//+
Rotate {{0, 0, 1}, {0, 0, 0}, -Pi/10} {
  Duplicata { Point{2}; Point{3}; Point{4}; Point{5}; }
}
//+
Line(1) = {11, 7};
//+
Line(2) = {12, 8};
//+
Line(3) = {13, 9};
//+
Line(4) = {10, 6};
//+
Circle(5) = {7, 1, 12};
//+
Circle(6) = {8, 1, 13};
//+
Circle(7) = {9, 1, 10};
//+
Circle(8) = {6, 1, 11};
//+
Recursive Delete {
  Point{3}; Point{4}; Point{5}; Point{2}; 
}
//+
Point(14) = {1.5, 0, -4, 1.0};
//+
Point(15) = {0, 1.5, -4, 1.0};
//+
Point(16) = {-1.5, 0, -4, 1.0};
//+
Point(17) = {0, -1.5, -4, 1.0};
//+
Circle(9) = {14, 1, 15};
//+
Circle(10) = {15, 1, 16};
//+
Circle(11) = {16, 1, 17};
//+
Circle(12) = {17, 1, 14};
//+
Curve Loop(1) = {5, 2, 6, 3, 7, 4, 8, 1};
//+
Curve Loop(2) = {9, 10, 11, 12};
//+
Plane Surface(1) = {1, 2};
//+
Extrude {0, 0, 8} {
  Surface{1}; 
}
//+
Recursive Delete {
  Point{19}; 
}
