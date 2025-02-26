// Gmsh project created on Fri Aug 06 09:40:19 2021

// PARAMETERS
H = 10.0; // height of the electrode holders
h = 8.0; // height of the test-mass
Ly = 5.0; // height of the Y-Z pairs
Lphi = 5.0; // height of the Phi pairs (must be equal to Ly)
Ry = 1.0; // radius of the inner electrode holder
Rphi = 3.0; // radius of the outer electrode holder
R1 = 1.5; // test-mass inner radius
R2 = 2.0; // test-mass outer radius

// Translation of the Test-Mass
Tbool = 0;
Tx = 0.0; // translation of the test-mass along x-axis
Ty = 0.0; // translation of the test-mass along y-axis
Tz = 0.0; // translation of the test-mass along z-axis

// Rotation of the Test-Mass
Rbool = 0;
// point
px = 0.0;
py = 0.0;
pz = 0.0;
// direction
ax = 0.0;
ay = 0.0;
az = 0.0;
// angle of rotation
rot_angle = 0.0; // [rad]

//+
Point(1) = {Ry, 0, -H/2, 1.0};
//+
Point(2) = {Ry, 0, -Ly/2, 1.0};
//+
Point(3) = {Ry, 0, 0, 1.0};
//+
Point(4) = {Ry, 0, Ly/2, 1.0};
//+
Point(5) = {Ry, 0, H/2, 1.0};
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
Rotate {{0, 0, 1}, {0, 0, 0}, -Pi/4} {
  Point{46}; Point{47}; Point{49}; Point{51}; Point{53}; Point{1}; Point{2}; Point{3}; Point{4}; Point{5}; Point{6}; Point{7}; Point{15}; Point{23}; Point{31}; Point{38}; Point{39}; Point{41}; Point{43}; Point{45}; 
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
Point(54) = {Rphi, 0, -H/2, 1.0};
//+
Point(55) = {Rphi, 0, -Lphi/2, 1.0};
//+
Point(56) = {Rphi, 0, 0, 1.0};
//+
Point(57) = {Rphi, 0, Lphi/2, 1.0};
//+
Point(58) = {Rphi, 0, H/2, 1.0};
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
//+
Point(115) = {0, 0, -h/2, 1.0};
//+
Point(116) = {R2, 0, -h/2, 1.0};
//+
Point(117) = {0, R2, -h/2, 1.0};
//+
Point(118) = {-R2, 0, -h/2, 1.0};
//+
Point(119) = {0, -R2, -h/2, 1.0};
//+
Rotate {{0, 0, 1}, {0, 0, 0}, Pi/12} {
  Duplicata { Point{116}; Point{117}; Point{118}; Point{119}; }
}
//+
Rotate {{0, 0, 1}, {0, 0, 0}, -Pi/12} {
  Duplicata { Point{116}; Point{117}; Point{118}; Point{119}; }
}
//+
Line(205) = {125, 121};
//+
Line(206) = {126, 122};
//+
Line(207) = {127, 123};
//+
Line(208) = {124, 120};
//+
Circle(209) = {121, 115, 126};
//+
Circle(210) = {122, 115, 127};
//+
Circle(211) = {123, 115, 124};
//+
Circle(212) = {120, 115, 125};
//+
Recursive Delete {
  Point{116}; Point{117}; Point{119}; Point{118}; 
}
//+
Point(128) = {R1, 0, -h/2, 1.0};
//+
Point(129) = {0, R1, -h/2, 1.0};
//+
Point(130) = {-R1, 0, -h/2, 1.0};
//+
Point(131) = {0, -R1, -h/2, 1.0};
//+
Circle(213) = {128, 115, 129};
//+
Circle(214) = {129, 115, 130};
//+
Circle(215) = {130, 115, 131};
//+
Circle(216) = {131, 115, 128};
//+
Curve Loop(5) = {209, 206, 210, 207, 211, 208, 212, 205};
//+
Curve Loop(6) = {214, 215, 216, 213};
//+
Extrude {0, 0, h} {
  Curve{216}; Curve{215}; Curve{214}; Curve{213}; Curve{207}; Curve{211}; Curve{208}; Curve{212}; Curve{205}; Curve{209}; Curve{206}; Curve{210}; 
}
//+
Curve Loop(7) = {261, 233, 237, 241, 245, 249, 253, 257};
//+
Curve Loop(8) = {221, 217, 229, 225};
//+
Plane Surface(265) = {7, 8};
//+
Plane Surface(266) = {5, 6};
//+
Physical Surface("Test_Mass", 267) = {265, 256, 266, 264, 220, 240, 244, 248, 232, 228, 252, 260, 224, 236};
//+
Surface Loop(1) = {207, 113, 97, 81, 193, 177, 161, 145, 129, 133, 117, 101, 85, 197, 181, 165, 149, 153, 137, 121, 105, 89, 201, 185, 169, 173, 157, 141, 125, 109, 93, 205, 189, 206, 52, 36, 20, 68, 64, 48, 32, 16, 12, 60, 44, 28, 24, 8, 56, 40};
//+
Surface Loop(2) = {260, 266, 256, 265, 264, 236, 240, 244, 248, 252, 228, 224, 220, 232};
//+
Volume(1) = {1, 2};
//+
Physical Volume("vacuum", 268) = {1};
//+
If (Tbool == 1)
  Translate {Tx, Ty, Tz} {
    Point{138}; Point{137}; Point{132}; Point{141}; Point{134}; Point{143}; Point{153}; Point{135}; Point{151}; Point{136}; Point{146}; Point{148}; Point{123}; Point{127}; Point{131}; Point{124}; Point{128}; Point{120}; Point{122}; Point{130}; Point{126}; Point{129}; Point{125}; Point{121}; Curve{233}; Curve{237}; Curve{217}; Curve{241}; Curve{261}; Curve{221}; Curve{257}; Curve{229}; Curve{225}; Curve{245}; Curve{249}; Curve{253}; Curve{235}; Curve{207}; Curve{234}; Curve{218}; Curve{211}; Curve{239}; Curve{216}; Curve{219}; Curve{208}; Curve{243}; Curve{210}; Curve{259}; Curve{215}; Curve{222}; Curve{255}; Curve{206}; Curve{226}; Curve{213}; Curve{214}; Curve{247}; Curve{212}; Curve{251}; Curve{205}; Curve{209}; Surface{265}; Surface{240}; Surface{236}; Surface{220}; Surface{244}; Surface{264}; Surface{224}; Surface{232}; Surface{248}; Surface{260}; Surface{228}; Surface{266}; Surface{252}; Surface{256}; 
  }
EndIf
//+
If (Rbool == 1)
  Rotate {{0, 1, 0}, {0, 0, 0}, Pi/50} {
    Point{138}; Point{137}; Point{141}; Point{132}; Point{134}; Point{143}; Point{153}; Point{135}; Point{136}; Point{151}; Point{146}; Point{148}; Point{123}; Point{127}; Point{124}; Point{131}; Point{128}; Point{120}; Point{122}; Point{130}; Point{129}; Point{126}; Point{125}; Point{121}; Curve{233}; Curve{237}; Curve{217}; Curve{241}; Curve{261}; Curve{221}; Curve{229}; Curve{257}; Curve{245}; Curve{225}; Curve{249}; Curve{253}; Curve{235}; Curve{207}; Curve{234}; Curve{211}; Curve{239}; Curve{218}; Curve{216}; Curve{219}; Curve{208}; Curve{243}; Curve{210}; Curve{259}; Curve{222}; Curve{215}; Curve{226}; Curve{213}; Curve{255}; Curve{206}; Curve{247}; Curve{212}; Curve{214}; Curve{251}; Curve{205}; Curve{209}; Surface{265}; Surface{240}; Surface{236}; Surface{220}; Surface{244}; Surface{264}; Surface{224}; Surface{232}; Surface{248}; Surface{260}; Surface{228}; Surface{266}; Surface{252}; Surface{256}; 
  }
EndIf
//+
Show "*";
//+
Hide {
  Point{1}; Point{5}; Point{6}; Point{11}; Point{13}; Point{21}; Point{29}; Point{31}; Point{37}; Point{38}; Point{45}; Point{46}; Point{53}; Point{115}; Point{133}; Curve{1}; Curve{4}; Curve{5}; Curve{6}; Curve{7}; Curve{17}; Curve{19}; Curve{21}; Curve{22}; Curve{33}; Curve{35}; Curve{37}; Curve{38}; Curve{49}; Curve{51}; Curve{54}; Curve{67}; Surface{8}; Surface{12}; Surface{20}; Surface{24}; Surface{36}; Surface{40}; Surface{52}; Surface{56}; Surface{68}; Surface{206}; Surface{207}; Volume{1}; 
}
