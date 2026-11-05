// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

// WATERBLOCK 3-D electron converter and actively cooled heat sink, in SI metres.
SetFactory("OpenCASCADE");
Geometry.OCCBooleanPreserveNumbering = 1;

xmin = -0.012;
xmax =  0.060;
ymin = -0.025;
ymax =  0.025;
zmin = -0.025;
zmax =  0.025;

// A tungsten target intercepts a normally incident beam ahead of the copper
// heat sink.
// Its swept, layered mesh resolves the 0.4 mm CSDA range without imposing
// the same 0.1 mm axial spacing on the transverse directions.  The first
// 0.4 mm of the thickness has four 0.1 mm layers; the rest is progressively
// coarsened.  The triangular source surface is swept into layered tetrahedra
// that meet the surrounding tetrahedra conformally.
Point(1001) = {-0.001, -0.007, -0.007, 0.002};
Point(1002) = {-0.001,  0.007, -0.007, 0.002};
Point(1003) = {-0.001,  0.007,  0.007, 0.002};
Point(1004) = {-0.001, -0.007,  0.007, 0.002};
Line(1001) = {1001, 1002};
Line(1002) = {1002, 1003};
Line(1003) = {1003, 1004};
Line(1004) = {1004, 1001};
Curve Loop(1001) = {1001, 1002, 1003, 1004};
Plane Surface(1001) = {1001};
SetMaxTag Volume(9);
target_extrusion[] = Extrude {0.002, 0, 0} {
  Surface{1001};
  Layers{{4, 3, 3}, {0.20, 0.50, 1.00}};
};
target_volume = target_extrusion[1];

// Copper heat sink with a U-shaped water passage.
Box(20) = {0.001, -0.018, -0.016, 0.027, 0.036, 0.032};
Cylinder(21) = {0.009, ymin, -0.008, 0, 0.040, 0, 0.0025};
Cylinder(22) = {0.021, ymin, -0.008, 0, 0.040, 0, 0.0025};
Cylinder(23) = {0.008,  0.014, -0.008, 0.014, 0, 0, 0.0025};
water[] = BooleanUnion{ Volume{21}; Delete; }
                       { Volume{22, 23}; Delete; };
copper[] = BooleanDifference{ Volume{20}; Delete; }{ Volume{water[]}; };
target[] = {target_volume};
copper[] = {20};

// Aluminum enclosure with a cylindrical beam aperture through the front wall.
Box(30) = {-0.004, -0.022, -0.020, 0.036, 0.044, 0.040};
Box(31) = {-0.002, -0.019, -0.017, 0.032, 0.038, 0.034};
housing0[] = BooleanDifference{ Volume{30}; Delete; }{ Volume{31}; Delete; };
Cylinder(32) = {-0.005, 0, 0, 0.010, 0, 0, 0.010};
housing1[] = BooleanDifference{ Volume{housing0[]}; Delete; }
                              { Volume{32}; Delete; };
housing[] = BooleanDifference{ Volume{housing1[]}; Delete; }
                             { Volume{water[]}; };

// A steel rear shield includes a line-of-sight photon aperture.
Box(40) = {0.040, ymin, zmin, 0.010, ymax-ymin, zmax-zmin};
Cylinder(41) = {0.039, 0, 0, 0.012, 0, 0, 0.006};
shield[] = BooleanDifference{ Volume{40}; Delete; }{ Volume{41}; Delete; };

// Make every material interface conformal before subtracting the assembly
// from the surrounding air.
assembly[] = BooleanFragments{ Volume{target[], copper[], water[]}; Delete; }
                              { Volume{housing[], shield[]}; Delete; };
target[] = {target_volume};
copper[] = {20};
water[] = {21};
housing[] = {30};
shield[] = {40};

// Air fills the chamber around the converter. A square beam column splits the
// upstream face into a finite source patch and its vacuum remainder.
Box(1) = {xmin, ymin, zmin, xmax-xmin, ymax-ymin, zmax-zmin};
domain_contact[] = BooleanFragments{ Volume{1}; Delete; }
                                    { Volume{target[], copper[], water[],
                                             housing[], shield[]}; Delete; };
air0[] = {41};
target[] = {target_volume};
copper[] = {20};
water[] = {21};
housing[] = {30};
shield[] = {40};
Box(2) = {xmin, -0.004, -0.004, 0.009, 0.008, 0.008};
air[] = BooleanFragments{ Volume{air0[]}; Delete; }{ Volume{2}; Delete; };

Physical Volume("tungsten", 101) = {target[]};
Physical Volume("copper", 102) = {copper[]};
Physical Volume("aluminum", 103) = {housing[]};
Physical Volume("air", 104) = {air[]};
Physical Volume("steel", 105) = {shield[]};
Physical Volume("water", 106) = {water[]};

eps = 1.e-7;
beam[] = Surface In BoundingBox{xmin-eps, -0.0041, -0.0041,
                                xmin+eps,  0.0041,  0.0041};
water_in[] = Surface In BoundingBox{0.0064, ymin-eps, -0.0106,
                                    0.0116, ymin+eps, -0.0054};
water_out[] = Surface In BoundingBox{0.0184, ymin-eps, -0.0106,
                                     0.0236, ymin+eps, -0.0054};
Physical Surface("beamInlet", 201) = {beam[]};
Physical Surface("waterInlet", 202) = {water_in[]};
Physical Surface("waterOutlet", 203) = {water_out[]};
outer_signed[] = CombinedBoundary{ Volume{target[], copper[], water[],
                                           housing[], air[], shield[]}; };
vacuum[] = {};
For i In {0:#outer_signed[]-1}
  vacuum[] += {Abs(outer_signed[i])};
EndFor
vacuum[] -= {beam[], water_in[], water_out[]};
Physical Surface("vacuum", 204) = {vacuum[]};

// Resolve the target and channel walls, then coarsen rapidly into the chamber.
Mesh.MeshSizeMin = 0.002;
Mesh.MeshSizeMax = 0.018;
Mesh.MeshSizeFromCurvature = 2;
Mesh.MinimumCirclePoints = 6;

Mesh.Algorithm = 6;
Mesh.Algorithm3D = 10;
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;
Mesh.MshFileVersion = 2.2;
