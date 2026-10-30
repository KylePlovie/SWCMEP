G = 6.6743e-11
C_LIGHT = 299792458.0
K_B = 1.380649e-23
MU_0 = 1.25663706e-06
M_PROTON = 1.67262192e-27
M_ELECTRON = 9.1093837e-31
M_MEAN = M_PROTON / 2.0
M_SUN = 1.98892e+30
R_SUN = 695700000.0
L_SUN = 3.828e+26
T_SUN = 5778.0
OMEGA_SUN = 2.87e-06
AU = 149597870700.0
AU_KM = AU / 1000.0
R_INNER = 21.5 * R_SUN
R_OUTER = 1.1 * AU
V_SW_SLOW = 400000.0
V_SW_FAST = 750000.0
N_SW_1AU = 7000000.0
B_SW_1AU = 5e-09
T_SW_1AU = 100000.0
B_REF = 5e-09
R_REF = AU
I_RHO = 0
I_VR = 1
I_VPHI = 2
I_P = 3
I_BR = 4
I_BPHI = 5
I_PSI = 6
N_VARS_2D = 7
I_VTHETA = 7
I_BTHETA = 8
I_EINT = 9
N_VARS = 10

def ms_to_kms(v):
    return v / 1000.0

def kms_to_ms(v):
    return v * 1000.0

def m3_to_cm3(n):
    return n / 1000000.0

def T_to_nT(b):
    return b / 1e-09

def m_to_AU(r):
    return r / AU

def AU_to_m(r):
    return r * AU

def m_to_Rsun(r):
    return r / R_SUN
