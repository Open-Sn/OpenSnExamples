# -*- coding: utf-8 -*-
"""
Reference points of Kobayashi Problem 3 and the steady-state case-ii fluxes:

    K. Kobayashi, N. Sugimura, and Y. Nagaya, "3D radiation transport benchmark problems
    and results for simple geometries with void region," Progress in Nuclear Energy 39(2),
    119-144 (2001); OECD/NEA NEA/NSC/DOC(2000)4.

Case ii (scattering ratio 0.5) with a 1 n/cm^3/s source in [0,10]^3. The time-dependent
benchmark keeps the same geometry and cross sections, so its time-integrated flux divided
by the 50 s source duration tends to these values.
"""

REFERENCE_POINTS = {
    "3A_1": (5.0, 5.0, 5.0),
    "3A_2": (5.0, 15.0, 5.0),
    "3A_3": (5.0, 25.0, 5.0),
    "3A_4": (5.0, 35.0, 5.0),
    "3A_5": (5.0, 45.0, 5.0),
    "3A_6": (5.0, 55.0, 5.0),
    "3A_7": (5.0, 65.0, 5.0),
    "3A_8": (5.0, 75.0, 5.0),
    "3A_9": (5.0, 85.0, 5.0),
    "3A_10": (5.0, 95.0, 5.0),
    "3B_2": (15.0, 55.0, 5.0),
    "3B_3": (25.0, 55.0, 5.0),
    "3B_4": (35.0, 55.0, 5.0),
    "3B_5": (45.0, 55.0, 5.0),
    "3B_6": (55.0, 55.0, 5.0),
    "3C_1": (5.0, 95.0, 35.0),
    "3C_2": (15.0, 95.0, 35.0),
    "3C_3": (25.0, 95.0, 35.0),
    "3C_4": (35.0, 95.0, 35.0),
    "3C_5": (45.0, 95.0, 35.0),
    "3C_6": (55.0, 95.0, 35.0),
}

# Point 3B_1 is the same location as 3A_6 and is not repeated.
STEADY_CASE_II_FLUX = {
    "3A_1": 8.61578e0,
    "3A_2": 2.16130e0,
    "3A_3": 8.93784e-1,
    "3A_4": 4.78052e-1,
    "3A_5": 2.89424e-1,
    "3A_6": 1.92698e-1,
    "3A_7": 1.04982e-1,
    "3A_8": 3.37544e-2,
    "3A_9": 1.08158e-2,
    "3A_10": 3.39632e-3,
    "3B_2": 6.72147e-2,
    "3B_3": 2.21799e-2,
    "3B_4": 9.90646e-3,
    "3B_5": 3.39066e-3,
    "3B_6": 1.05629e-3,
    "3C_1": 3.44804e-4,
    "3C_2": 2.91825e-4,
    "3C_3": 2.05793e-4,
    "3C_4": 2.62086e-4,
    "3C_5": 1.05367e-4,
    "3C_6": 4.44962e-5,
}
