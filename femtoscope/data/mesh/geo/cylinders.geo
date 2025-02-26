// Gmsh project created on Fri Aug 06 09:40:19 2021
//+
Point(1) = {1, 0, -5, 1.0};
//+
Point(2) = {1, 0, -3, 1.0};
//+
Point(3) = {1, 0, 0, 1.0};
//+
Point(4) = {1, 0, 3, 1.0};
//+
Point(5) = {1, 0, 5, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/2} {
  Curve{1}; Curve{2}; Curve{3}; Curve{4}; 
}
//+
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/2} {
  Curve{5}; Curve{9}; Curve{13}; Curve{17}; 
}
//+
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/2} {
  Curve{21}; Curve{25}; Curve{29}; Curve{33}; 
}
//+
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/2} {
  Curve{37}; Curve{41}; Curve{45}; Curve{49}; 
}
//+
Physical Surface("Vp", 69) = {56, 40, 8, 24, 52, 20, 68, 36};
//+
Physical Surface("Y1Plus", 70) = {12};
//+
Physical Surface("Y2Plus", 71) = {16};
//+
Physical Surface("Z1Plus", 72) = {28};
//+
Physical Surface("Z2Plus", 73) = {32};
//+
Physical Surface("Y1Moins", 74) = {44};
//+
Physical Surface("Y2Moins", 75) = {48};
//+
Physical Surface("Z1Moins", 76) = {60};
//+
Physical Surface("Z2Moins", 77) = {64};
//+
Recursive Delete {
  Point{37}; 
}
//+
Recursive Delete {
  Point{37}; 
}
//+
Recursive Delete {
  Point{13}; 
}
//+
Point(54) = {3, 0, -5, 1.0};
//+
Point(55) = {3, 0, -3, 1.0};
//+
Point(56) = {3, 0, 0, 1.0};
//+
Point(57) = {3, 0, 3, 1.0};
//+
Point(58) = {3, 0, 5, 1.0};
//+
Line(68) = {54, 55};
//+
Line(69) = {55, 56};
//+
Line(70) = {56, 57};
//+
Line(71) = {57, 58};
//+
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/4} {
  Curve{68}; Curve{69}; Curve{70}; Curve{71}; 
}
//+
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/4} {
  Curve{78}; Curve{82}; Curve{86}; Curve{90}; 
}
//+
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/4} {
  Curve{94}; Curve{98}; Curve{102}; Curve{106}; 
}
//+
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/4} {
  Curve{110}; Curve{114}; Curve{118}; Curve{122}; 
}
//+
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/4} {
  Curve{126}; Curve{130}; Curve{134}; Curve{138}; 
}
//+
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/4} {
  Curve{142}; Curve{146}; Curve{150}; Curve{154}; 
}
//+
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/4} {
  Curve{158}; Curve{162}; Curve{166}; Curve{170}; 
}
//+
Extrude {{0, 0, 1}, {0, 0, 0}, Pi/4} {
  Curve{174}; Curve{178}; Curve{182}; Curve{186}; 
}
//+
Physical Surface("XMoins", 206) = {193, 81, 97, 113, 129, 145, 161, 177};
//+
Physical Surface("XPlus", 207) = {173, 189, 205, 93, 109, 125, 141, 157};
//+
Physical Surface("Phi1Plus", 208) = {149, 153};
//+
Physical Surface("Phi1Moins", 209) = {165, 169};
//+
Physical Surface("Phi2Plus", 210) = {181, 185};
//+
Physical Surface("Phi2Moins", 211) = {197, 201};
//+
Physical Surface("Phi3Plus", 212) = {85, 89};
//+
Physical Surface("Phi3Moins", 213) = {101, 105};
//+
Physical Surface("Phi4Plus", 214) = {117, 121};
//+
Physical Surface("Phi4Moins", 215) = {133, 137};
//+
Curve Loop(1) = {188, 204, 92, 108, 124, 140, 156, 172};
//+
Curve Loop(2) = {67, 19, 35, 51};
//+
Plane Surface(206) = {1, 2};
//+
Curve Loop(3) = {111, 127, 143, 159, 175, 191, 79, 95};
//+
Curve Loop(4) = {22, 38, 54, 6};
//+
Plane Surface(207) = {3, 4};
