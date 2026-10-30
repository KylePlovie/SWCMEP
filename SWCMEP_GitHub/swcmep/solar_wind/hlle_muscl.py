import torch
from swcmep.solar_wind.background_2d import I_BPHI, I_BR, I_PSI
from swcmep.solar_wind.mhd_solver import RHO, VPHI, VR, P
from swcmep.utils.constants import I_BTHETA, I_EINT, I_VTHETA
MU_0 = 1.2566370614e-06
BR = I_BR
BPHI = I_BPHI
PSI = I_PSI
VTHETA = I_VTHETA
BTHETA = I_BTHETA
EINT = I_EINT
_HLLE_INNER_ZONE = 4

def minmod(a, b):
    return torch.where(a * b > 0, torch.sign(a) * torch.minimum(torch.abs(a), torch.abs(b)), torch.zeros_like(a))

def _signed_clamp_nonzero(x, eps=1e-30):
    return torch.where(x >= 0, torch.clamp(x, min=eps), torch.clamp(x, max=-eps))

def prim_to_cons(U, gamma):
    W = torch.zeros_like(U)
    rho = torch.clamp(U[RHO], min=1e-30)
    v_r = U[VR]
    v_phi = U[VPHI]
    p = torch.clamp(U[P], min=1e-30)
    B_r = U[BR]
    B_phi = U[BPHI]
    B2 = B_r ** 2 + B_phi ** 2
    W[RHO] = rho
    W[VR] = rho * v_r
    W[VPHI] = rho * v_phi
    W[P] = p / (gamma - 1.0 + 1e-30) + 0.5 * rho * (v_r ** 2 + v_phi ** 2) + B2 / (2.0 * MU_0)
    W[BR] = B_r
    W[BPHI] = B_phi
    W[PSI] = U[PSI]
    return W

def cons_to_prim(W, gamma):
    U = torch.zeros_like(W)
    rho = torch.clamp(W[RHO], min=1e-30)
    U[RHO] = rho
    U[VR] = W[VR] / rho
    U[VPHI] = W[VPHI] / rho
    U[BR] = W[BR]
    U[BPHI] = W[BPHI]
    U[PSI] = W[PSI]
    B2 = W[BR] ** 2 + W[BPHI] ** 2
    E_mag = B2 / (2.0 * MU_0)
    E_kin = 0.5 * rho * (U[VR] ** 2 + U[VPHI] ** 2)
    U[P] = torch.clamp((gamma - 1.0) * (W[P] - E_kin - E_mag), min=1e-30)
    return U

def _mhd_flux_r(W, gamma, c_h=0.0, eps=1e-30):
    rho = torch.clamp(W[RHO], min=eps)
    vn = W[VR] / rho
    vt = W[VPHI] / rho
    bn = W[BR]
    bt = W[BPHI]
    psi = W[PSI]
    B2 = bn ** 2 + bt ** 2
    Ekin = 0.5 * rho * (vn ** 2 + vt ** 2)
    Etot = W[P]
    p = _pressure_dual_3d(W, rho, Ekin, B2, gamma, eps)
    ptot = p + B2 / (2.0 * MU_0)
    vdotb = vn * bn + vt * bt
    F = torch.zeros_like(W)
    F[RHO] = rho * vn
    F[VR] = rho * vn * vn + ptot - bn * bn / MU_0
    F[VPHI] = rho * vn * vt - bn * bt / MU_0
    F[P] = (Etot + ptot) * vn - bn * vdotb / MU_0 + psi * bn / MU_0
    F[BR] = psi
    F[BPHI] = bt * vn - vt * bn
    F[PSI] = c_h ** 2 * bn
    return F

def _mhd_flux_phi(W, gamma, c_h=0.0, eps=1e-30):
    rho = torch.clamp(W[RHO], min=eps)
    vn = W[VPHI] / rho
    vt = W[VR] / rho
    bn = W[BPHI]
    bt = W[BR]
    psi = W[PSI]
    B2 = bn ** 2 + bt ** 2
    Ekin = 0.5 * rho * (vt ** 2 + vn ** 2)
    Etot = W[P]
    p = _pressure_dual_3d(W, rho, Ekin, B2, gamma, eps)
    ptot = p + B2 / (2.0 * MU_0)
    vdotb = vn * bn + vt * bt
    G = torch.zeros_like(W)
    G[RHO] = rho * vn
    G[VR] = rho * vn * vt - bn * bt / MU_0
    G[VPHI] = rho * vn * vn + ptot - bn * bn / MU_0
    G[P] = (Etot + ptot) * vn - bn * vdotb / MU_0 + psi * bn / MU_0
    G[BR] = bt * vn - vt * bn
    G[BPHI] = psi
    G[PSI] = c_h ** 2 * bn
    return G

def _fast_speed(rho, p, bn, bt, gamma, eps=1e-30):
    rho = torch.clamp(rho, min=eps)
    p = torch.clamp(p, min=eps)
    a2 = gamma * p / rho
    b2 = (bn ** 2 + bt ** 2) / (MU_0 * rho)
    can2 = bn ** 2 / (MU_0 * rho)
    s = a2 + b2
    s_safe = torch.clamp(s, min=eps)
    q = torch.clamp(4.0 * (a2 / s_safe) * (can2 / s_safe), min=0.0, max=1.0)
    cf2 = 0.5 * s * (1.0 + torch.sqrt(torch.clamp(1.0 - q, min=0.0)))
    return torch.sqrt(torch.clamp(cf2, min=eps))

def hlle_flux_mhd(W_L, W_R, gamma, use_phi=False, c_h=0.0, eps=1e-30):
    rhoL = torch.clamp(W_L[RHO], min=eps)
    rhoR = torch.clamp(W_R[RHO], min=eps)
    if use_phi:
        vnL, vtL, bnL, btL = (W_L[VPHI] / rhoL, W_L[VR] / rhoL, W_L[BPHI], W_L[BR])
        vnR, vtR, bnR, btR = (W_R[VPHI] / rhoR, W_R[VR] / rhoR, W_R[BPHI], W_R[BR])
    else:
        vnL, vtL, bnL, btL = (W_L[VR] / rhoL, W_L[VPHI] / rhoL, W_L[BR], W_L[BPHI])
        vnR, vtR, bnR, btR = (W_R[VR] / rhoR, W_R[VPHI] / rhoR, W_R[BR], W_R[BPHI])
    B2L = bnL ** 2 + btL ** 2
    B2R = bnR ** 2 + btR ** 2
    EkL = 0.5 * rhoL * (vnL ** 2 + vtL ** 2)
    EkR = 0.5 * rhoR * (vnR ** 2 + vtR ** 2)
    pL = _pressure_dual_3d(W_L, rhoL, EkL, B2L, gamma, eps)
    pR = _pressure_dual_3d(W_R, rhoR, EkR, B2R, gamma, eps)
    bn_f = 0.5 * (bnL + bnR)
    cfL = _fast_speed(rhoL, pL, bn_f, btL, gamma, eps)
    cfR = _fast_speed(rhoR, pR, bn_f, btR, gamma, eps)
    SL = torch.minimum(vnL - cfL, vnR - cfR)
    SR = torch.maximum(vnL + cfL, vnR + cfR)
    if c_h > 0:
        SL = torch.minimum(SL, torch.full_like(SL, -c_h))
        SR = torch.maximum(SR, torch.full_like(SR, c_h))
    FL = _mhd_flux_phi(W_L, gamma, c_h, eps) if use_phi else _mhd_flux_r(W_L, gamma, c_h, eps)
    FR = _mhd_flux_phi(W_R, gamma, c_h, eps) if use_phi else _mhd_flux_r(W_R, gamma, c_h, eps)
    denom = torch.clamp(SR - SL, min=eps)
    F_hlle = (SR * FL - SL * FR + SL * SR * (W_R - W_L)) / denom
    mL = SL.unsqueeze(0).expand_as(FL) >= 0
    mR = SR.unsqueeze(0).expand_as(FR) <= 0
    return torch.where(mL, FL, torch.where(mR, FR, F_hlle))

def hlld_flux(W_L, W_R, gamma, c_h=0.0, eps=1e-30, bn_tol=1e-06):
    rhoL = torch.clamp(W_L[RHO], min=eps)
    rhoR = torch.clamp(W_R[RHO], min=eps)
    vnL = W_L[VR] / rhoL
    vtL = W_L[VPHI] / rhoL
    vnR = W_R[VR] / rhoR
    vtR = W_R[VPHI] / rhoR
    bnL = W_L[BR]
    btL = W_L[BPHI]
    bnR = W_R[BR]
    btR = W_R[BPHI]
    B2L = bnL ** 2 + btL ** 2
    B2R = bnR ** 2 + btR ** 2
    EkL = 0.5 * rhoL * (vnL ** 2 + vtL ** 2)
    EkR = 0.5 * rhoR * (vnR ** 2 + vtR ** 2)
    pL = _pressure_dual_3d(W_L, rhoL, EkL, B2L, gamma, eps)
    pR = _pressure_dual_3d(W_R, rhoR, EkR, B2R, gamma, eps)
    ptL = pL + B2L / (2.0 * MU_0)
    ptR = pR + B2R / (2.0 * MU_0)
    vdbL = vnL * bnL + vtL * btL
    vdbR = vnR * bnR + vtR * btR
    bn = 0.5 * (bnL + bnR)
    abs_bn = torch.abs(bn)
    sgn_bn = torch.where(abs_bn > eps, torch.sign(bn), torch.ones_like(bn))
    cfL = _fast_speed(rhoL, pL, bn, btL, gamma, eps)
    cfR = _fast_speed(rhoR, pR, bn, btR, gamma, eps)
    SL = torch.minimum(vnL - cfL, vnR - cfR)
    SR = torch.maximum(vnL + cfL, vnR + cfR)
    FL = _mhd_flux_r(W_L, gamma, c_h, eps)
    FR = _mhd_flux_r(W_R, gamma, c_h, eps)
    dL = rhoL * (SL - vnL)
    dR = rhoR * (SR - vnR)
    dSM = torch.where(torch.abs(dL - dR) > eps, dL - dR, torch.full_like(dL, eps))
    SM = (dL * vnL - dR * vnR - ptL + ptR) / dSM
    pT_star = ptL + rhoL * (SL - vnL) * (SM - vnL)
    den_sm_L = _signed_clamp_nonzero(SL - SM, eps)
    den_sm_R = _signed_clamp_nonzero(SR - SM, eps)
    ratio_L = torch.clamp((SL - vnL) / den_sm_L, min=0.1, max=10.0)
    ratio_R = torch.clamp((SR - vnR) / den_sm_R, min=0.1, max=10.0)
    rho_sL = rhoL * ratio_L
    rho_sR = rhoR * ratio_R
    denL = rhoL * (SL - vnL) * (SL - SM) - bn ** 2 / MU_0
    denR = rhoR * (SR - vnR) * (SR - SM) - bn ** 2 / MU_0
    safe_threshold_L = torch.clamp(0.0001 * torch.maximum(torch.abs(rhoL * (SL - vnL) * (SL - SM)), bn ** 2 / MU_0 + eps), min=eps)
    safe_threshold_R = torch.clamp(0.0001 * torch.maximum(torch.abs(rhoR * (SR - vnR) * (SR - SM)), bn ** 2 / MU_0 + eps), min=eps)
    safeL = torch.abs(denL) > safe_threshold_L
    safeR = torch.abs(denR) > safe_threshold_R
    vt_sL = torch.where(safeL, vtL - bn * btL * (SM - vnL) / (MU_0 * denL), vtL)
    vt_sR = torch.where(safeR, vtR - bn * btR * (SM - vnR) / (MU_0 * denR), vtR)
    bt_sL = torch.where(safeL, btL * (rhoL * (SL - vnL) ** 2 - bn ** 2 / MU_0) / denL, btL)
    bt_sR = torch.where(safeR, btR * (rhoR * (SR - vnR) ** 2 - bn ** 2 / MU_0) / denR, btR)
    bt_max = 2.0 * torch.maximum(torch.abs(btL), torch.abs(btR)) + eps
    bt_sL = torch.clamp(bt_sL, -bt_max, bt_max)
    bt_sR = torch.clamp(bt_sR, -bt_max, bt_max)
    vdb_sL = SM * bn + vt_sL * bt_sL
    vdb_sR = SM * bn + vt_sR * bt_sR
    E_sL = ((SL - vnL) * W_L[P] - ptL * vnL + pT_star * SM + bn * (vdbL - vdb_sL) / MU_0) / den_sm_L
    E_sR = ((SR - vnR) * W_R[P] - ptR * vnR + pT_star * SM + bn * (vdbR - vdb_sR) / MU_0) / den_sm_R
    U_sL = torch.zeros_like(W_L)
    U_sL[RHO] = rho_sL
    U_sL[VR] = rho_sL * SM
    U_sL[VPHI] = rho_sL * vt_sL
    U_sL[P] = E_sL
    U_sL[BR] = bn
    U_sL[BPHI] = bt_sL
    U_sL[PSI] = W_L[PSI]
    U_sR = torch.zeros_like(W_R)
    U_sR[RHO] = rho_sR
    U_sR[VR] = rho_sR * SM
    U_sR[VPHI] = rho_sR * vt_sR
    U_sR[P] = E_sR
    U_sR[BR] = bn
    U_sR[BPHI] = bt_sR
    U_sR[PSI] = W_R[PSI]
    F_sL = FL + SL.unsqueeze(0) * (U_sL - W_L)
    F_sR = FR + SR.unsqueeze(0) * (U_sR - W_R)
    sqL = torch.sqrt(torch.clamp(rho_sL, min=eps))
    sqR = torch.sqrt(torch.clamp(rho_sR, min=eps))
    SL_s = SM - abs_bn / (MU_0 ** 0.5 * sqL)
    SR_s = SM + abs_bn / (MU_0 ** 0.5 * sqR)
    dss = torch.clamp(sqL + sqR, min=eps)
    vt_ss = (sqL * vt_sL + sqR * vt_sR + (bt_sR - bt_sL) * sgn_bn) / dss
    bt_ss = (sqL * bt_sR + sqR * bt_sL + sqL * sqR * (vt_sR - vt_sL) * sgn_bn) / dss
    vdb_ss = SM * bn + vt_ss * bt_ss
    E_ssL = E_sL - sgn_bn * sqL * (vdb_sL - vdb_ss)
    E_ssR = E_sR - sgn_bn * sqR * (vdb_sR - vdb_ss)
    U_ssL = torch.zeros_like(W_L)
    U_ssL[RHO] = rho_sL
    U_ssL[VR] = rho_sL * SM
    U_ssL[VPHI] = rho_sL * vt_ss
    U_ssL[P] = E_ssL
    U_ssL[BR] = bn
    U_ssL[BPHI] = bt_ss
    U_ssL[PSI] = W_L[PSI]
    U_ssR = torch.zeros_like(W_R)
    U_ssR[RHO] = rho_sR
    U_ssR[VR] = rho_sR * SM
    U_ssR[VPHI] = rho_sR * vt_ss
    U_ssR[P] = E_ssR
    U_ssR[BR] = bn
    U_ssR[BPHI] = bt_ss
    U_ssR[PSI] = W_R[PSI]
    F_ssL = F_sL + SL_s.unsqueeze(0) * (U_ssL - U_sL)
    F_ssR = F_sR + SR_s.unsqueeze(0) * (U_ssR - U_sR)
    F = torch.zeros_like(W_L)
    m0 = SL >= 0
    m1 = (SL < 0) & (SL_s >= 0)
    m2 = (SL_s < 0) & (SM >= 0)
    m3 = (SM < 0) & (SR_s >= 0)
    m4 = (SR_s < 0) & (SR > 0)
    m5 = SR <= 0

    def em(m):
        return m.unsqueeze(0).expand_as(F)
    F = torch.where(em(m0), FL, F)
    F = torch.where(em(m1), F_sL, F)
    F = torch.where(em(m2), F_ssL, F)
    F = torch.where(em(m3), F_ssR, F)
    F = torch.where(em(m4), F_sR, F)
    F = torch.where(em(m5), FR, F)
    bad = (abs_bn < bn_tol) | ~safeL | ~safeR | ~torch.isfinite(SM) | ~torch.isfinite(pT_star) | ~torch.isfinite(E_sL) | ~torch.isfinite(E_sR) | ~torch.isfinite(E_ssL) | ~torch.isfinite(E_ssR) | (rho_sL <= 0) | (rho_sR <= 0)
    F_fb = hlle_flux_mhd(W_L, W_R, gamma, use_phi=False, c_h=c_h, eps=eps)
    F = torch.where(bad.unsqueeze(0).expand_as(F), F_fb, F)
    return F

def reconstruct_standard(W):
    dW = W[:, 1:, :] - W[:, :-1, :]
    slope = minmod(dW[:, :-1, :], dW[:, 1:, :])
    return (W[:, 1:-1, :] + 0.5 * slope, W[:, 2:, :] - 0.5 * slope)

def reconstruct_wb(W, W_eq):
    dWp = W - W_eq
    dW = dWp[:, 1:, :] - dWp[:, :-1, :]
    slope = minmod(dW[:, :-1, :], dW[:, 1:, :])
    W_eq_face = 0.5 * (W_eq[:, 1:-1, :] + W_eq[:, 2:, :])
    return (W_eq_face + (dWp[:, 1:-1, :] + 0.5 * slope), W_eq_face + (dWp[:, 2:, :] - 0.5 * slope))

def source_terms(W, r_gpu, grav_gpu, gamma, c_h=0.0, c_p_ratio=0.18, eps=1e-30):
    S = torch.zeros_like(W)
    rho = torch.clamp(W[RHO], min=eps)
    v_r = W[VR] / rho
    v_phi = W[VPHI] / rho
    B_r = W[BR]
    B_phi = W[BPHI]
    psi = W[PSI]
    B2 = B_r ** 2 + B_phi ** 2
    Ekin = 0.5 * rho * (v_r ** 2 + v_phi ** 2)
    p = torch.clamp((gamma - 1.0) * (W[P] - Ekin - B2 / (2.0 * MU_0)), min=eps)
    p_tot = p + B2 / (2.0 * MU_0)
    S[VR] = (rho * v_phi ** 2 + 2.0 * p_tot - B_phi ** 2 / MU_0) / r_gpu + rho * grav_gpu
    S[VPHI] = -(rho * v_r * v_phi - B_r * B_phi / MU_0) / r_gpu
    S[P] = rho * v_r * grav_gpu
    E_z = v_r * B_phi - v_phi * B_r
    S[BR] = torch.zeros_like(B_r)
    S[BPHI] = +E_z / r_gpu
    if c_h > 0:
        c_p2 = (c_p_ratio * c_h) ** 2
        S[PSI] = -c_h / c_p2 * psi
    return S

def compute_c_h(W, r_gpu, dr_gpu, dphi, gamma, cfl_factor=0.5, eps=1e-30):
    rho = torch.clamp(W[RHO], min=eps)
    vr = torch.abs(W[VR] / rho)
    vp = torch.abs(W[VPHI] / rho)
    B2 = W[BR] ** 2 + W[BPHI] ** 2
    Ekin = 0.5 * rho * ((W[VR] / rho) ** 2 + (W[VPHI] / rho) ** 2)
    p = torch.clamp((gamma - 1.0) * (W[P] - Ekin - B2 / (2.0 * MU_0)), min=eps)
    cf = _fast_speed(rho, p, W[BR], W[BPHI], gamma, eps)
    max_sr = (vr + cf).max()
    max_sp = (vp + cf).max()
    return float(torch.maximum(max_sr, max_sp))

def hlle_muscl_step(U, dr_gpu, r_gpu, dphi, grav_gpu, gamma, dt, G_MSun, U_eq=None, use_glm=True):
    nr = U.shape[1]
    W = prim_to_cons(U, gamma)
    W_eq = prim_to_cons(U_eq, gamma) if U_eq is not None else None
    c_h = compute_c_h(W, r_gpu, dr_gpu, dphi, gamma) if use_glm else 0.0
    S = source_terms(W, r_gpu, grav_gpu, gamma, c_h=c_h)
    dW_f = torch.roll(W, -1, dims=2) - W
    dW_b = W - torch.roll(W, 1, dims=2)
    slope_p = minmod(dW_b, dW_f)
    W_pL = W + 0.5 * slope_p
    W_pR = torch.roll(W - 0.5 * slope_p, -1, dims=2)
    F_phi = hlle_flux_mhd(W_pL, W_pR, gamma, use_phi=True, c_h=c_h)
    dFp = F_phi - torch.roll(F_phi, 1, dims=2)
    if W_eq is not None:
        W_L_r, W_R_r = reconstruct_wb(W, W_eq)
    else:
        W_L_r, W_R_r = reconstruct_standard(W)
    F_r_hi = hlld_flux(W_L_r, W_R_r, gamma, c_h=c_h)
    F_r_lo = hlld_flux(W[:, 1:-1, :], W[:, 2:, :], gamma, c_h=c_h)
    rhoL = torch.clamp(W_L_r[RHO], min=1e-30)
    rhoR = torch.clamp(W_R_r[RHO], min=1e-30)
    vL = W_L_r[VR] / rhoL
    vR = W_R_r[VR] / rhoR
    B2L = W_L_r[BR] ** 2 + W_L_r[BPHI] ** 2
    B2R = W_R_r[BR] ** 2 + W_R_r[BPHI] ** 2
    EkL = 0.5 * rhoL * (vL ** 2 + (W_L_r[VPHI] / rhoL) ** 2)
    EkR = 0.5 * rhoR * (vR ** 2 + (W_R_r[VPHI] / rhoR) ** 2)
    pL = torch.clamp((gamma - 1.0) * (W_L_r[P] - EkL - B2L / (2.0 * MU_0)), min=1e-30)
    pR = torch.clamp((gamma - 1.0) * (W_R_r[P] - EkR - B2R / (2.0 * MU_0)), min=1e-30)
    jump_rho = torch.abs(rhoR - rhoL) / torch.clamp(torch.minimum(rhoR, rhoL), min=1e-30)
    jump_vr = torch.abs(vR - vL)
    jump_p = torch.abs(pR - pL) / torch.clamp(torch.minimum(pR, pL), min=1e-30)
    BtL = torch.abs(W_L_r[BPHI]) + 1e-30
    BtR = torch.abs(W_R_r[BPHI]) + 1e-30
    jump_bt = torch.abs(W_R_r[BPHI] - W_L_r[BPHI]) / torch.minimum(BtL, BtR)
    troubled = (jump_rho > 5.0) | (jump_vr > 2000000.0) | (jump_p > 10.0) | (jump_bt > 3.0)
    F_r_m = torch.where(troubled.unsqueeze(0), F_r_lo, F_r_hi)
    n_blend = min(_HLLE_INNER_ZONE, F_r_m.shape[1])
    if n_blend > 0:
        F_r_hlle = hlle_flux_mhd(W_L_r[:, :n_blend, :], W_R_r[:, :n_blend, :], gamma, use_phi=False, c_h=c_h)
        alpha = torch.linspace(1.0, 0.0, n_blend, device=W.device, dtype=W.dtype)
        alpha = alpha.reshape(1, n_blend, 1)
        F_r_m[:, :n_blend, :] = alpha * F_r_hlle + (1.0 - alpha) * F_r_m[:, :n_blend, :]
    F_inner = hlle_flux_mhd(W[:, :1, :], W[:, 1:2, :], gamma, use_phi=False, c_h=c_h)
    F_outer = hlle_flux_mhd(W[:, -2:-1, :], W[:, -1:, :], gamma, use_phi=False, c_h=c_h)
    W_new = torch.zeros_like(W)
    W_new[:, 0, :] = W[:, 0, :]
    dr1 = dr_gpu[1:2, :]
    r1 = r_gpu[1:2, :]
    rfR1 = 0.5 * (r_gpu[1:2, :] + r_gpu[2:3, :])
    rfL1 = 0.5 * (r_gpu[0:1, :] + r_gpu[1:2, :])
    W_new[:, 1:2, :] = W[:, 1:2, :] - dt / dr1 * (rfR1 ** 2 * F_r_m[:, :1, :] - rfL1 ** 2 * F_inner) / r1 ** 2 - dt / (r1 * dphi) * dFp[:, 1:2, :] + dt * S[:, 1:2, :]
    if nr > 5:
        dr_i = dr_gpu[2:-2, :]
        r_i = r_gpu[2:-2, :]
        rfR = 0.5 * (r_gpu[2:-2, :] + r_gpu[3:-1, :])
        rfL = 0.5 * (r_gpu[1:-3, :] + r_gpu[2:-2, :])
        Fr = F_r_m[:, 1:nr - 3, :]
        Fl = F_r_m[:, 0:nr - 4, :]
        W_new[:, 2:-2, :] = W[:, 2:-2, :] - dt / dr_i * (rfR ** 2 * Fr - rfL ** 2 * Fl) / r_i ** 2 - dt / (r_i * dphi) * dFp[:, 2:-2, :] + dt * S[:, 2:-2, :]
    dr_e = dr_gpu[-2:-1, :]
    r_e = r_gpu[-2:-1, :]
    rfRe = 0.5 * (r_gpu[-2:-1, :] + r_gpu[-1:, :])
    rfLe = 0.5 * (r_gpu[-3:-2, :] + r_gpu[-2:-1, :])
    W_new[:, -2:-1, :] = W[:, -2:-1, :] - dt / dr_e * (rfRe ** 2 * F_outer - rfLe ** 2 * F_r_m[:, -1:, :]) / r_e ** 2 - dt / (r_e * dphi) * dFp[:, -2:-1, :] + dt * S[:, -2:-1, :]
    W_new[:, -1, :] = W[:, -1, :]
    U_out = cons_to_prim(W_new, gamma)
    U_out[P] = torch.clamp(U_out[P], min=1e-20)
    v_max = 5000000.0
    U_out[VR] = torch.clamp(U_out[VR], -v_max, v_max)
    U_out[VPHI] = torch.clamp(U_out[VPHI], -v_max, v_max)
    return U_out

def prim_to_cons_3d(U, gamma, eps=1e-30):
    W = torch.zeros_like(U)
    rho = torch.clamp(U[RHO], min=eps)
    v_r = U[VR]
    v_phi = U[VPHI]
    v_th = U[VTHETA]
    p = torch.clamp(U[P], min=eps)
    B_r = U[BR]
    B_phi = U[BPHI]
    B_th = U[BTHETA]
    B2 = B_r ** 2 + B_phi ** 2 + B_th ** 2
    v2 = v_r ** 2 + v_phi ** 2 + v_th ** 2
    W[RHO] = rho
    W[VR] = rho * v_r
    W[VPHI] = rho * v_phi
    W[VTHETA] = rho * v_th
    W[P] = p / (gamma - 1.0 + eps) + 0.5 * rho * v2 + B2 / (2.0 * MU_0)
    W[BR] = B_r
    W[BPHI] = B_phi
    W[BTHETA] = B_th
    W[PSI] = U[PSI]
    if U.shape[0] > EINT:
        W[EINT] = torch.maximum(torch.clamp(U[EINT], min=eps / (gamma - 1.0 + eps)), p / (gamma - 1.0 + eps))
    return W

def cons_to_prim_3d(W, gamma, eps=1e-30):
    U = torch.zeros_like(W)
    rho = torch.clamp(W[RHO], min=eps)
    U[RHO] = rho
    U[VR] = W[VR] / rho
    U[VPHI] = W[VPHI] / rho
    U[VTHETA] = W[VTHETA] / rho
    U[BR] = W[BR]
    U[BPHI] = W[BPHI]
    U[BTHETA] = W[BTHETA]
    U[PSI] = W[PSI]
    B2 = W[BR] ** 2 + W[BPHI] ** 2 + W[BTHETA] ** 2
    E_mag = B2 / (2.0 * MU_0)
    E_kin = 0.5 * rho * (U[VR] ** 2 + U[VPHI] ** 2 + U[VTHETA] ** 2)
    E_tot = W[P]
    thermal = E_tot - E_kin - E_mag
    p_cons = (gamma - 1.0) * thermal
    if W.shape[0] > I_EINT:
        e_aux = torch.clamp(W[I_EINT], min=eps / (gamma - 1.0))
        thermal_frac = thermal / torch.clamp(torch.abs(E_tot), min=eps)
        use_aux = ~torch.isfinite(p_cons) | (p_cons <= eps) | (thermal_frac < 0.001)
        e_thermal = torch.clamp(thermal, min=eps / (gamma - 1.0))
        e_out = torch.where(use_aux, e_aux, e_thermal)
        p = (gamma - 1.0) * e_out
    else:
        p = p_cons
    p = torch.clamp(p, min=eps)
    U[P] = p
    if U.shape[0] > I_EINT:
        U[I_EINT] = e_out
    return U

def _pressure_dual_3d(W, rho, Ekin, B2, gamma, eps=1e-30):
    thermal = W[P] - Ekin - B2 / (2.0 * MU_0)
    p_cons = (gamma - 1.0) * thermal
    if W.shape[0] > EINT:
        p_aux = (gamma - 1.0) * torch.clamp(W[EINT], min=eps / (gamma - 1.0 + eps))
        thermal_frac = thermal / torch.clamp(torch.abs(W[P]), min=eps)
        use_aux = ~torch.isfinite(p_cons) | (p_cons <= eps) | (thermal_frac < 0.001)
        p = torch.where(use_aux, p_aux, p_cons)
    else:
        p = p_cons
    return torch.clamp(p, min=eps)

def _mhd_flux_r_3d(W, gamma, c_h=0.0, eps=1e-30):
    rho = torch.clamp(W[RHO], min=eps)
    vn = W[VR] / rho
    vt1 = W[VPHI] / rho
    vt2 = W[VTHETA] / rho
    bn = W[BR]
    bt1 = W[BPHI]
    bt2 = W[BTHETA]
    psi = W[PSI]
    B2 = bn ** 2 + bt1 ** 2 + bt2 ** 2
    Ekin = 0.5 * rho * (vn ** 2 + vt1 ** 2 + vt2 ** 2)
    Etot = W[P]
    p = _pressure_dual_3d(W, rho, Ekin, B2, gamma, eps)
    ptot = p + B2 / (2.0 * MU_0)
    vdotb = vn * bn + vt1 * bt1 + vt2 * bt2
    F = torch.zeros_like(W)
    F[RHO] = rho * vn
    F[VR] = rho * vn * vn + ptot - bn * bn / MU_0
    F[VPHI] = rho * vn * vt1 - bn * bt1 / MU_0
    F[VTHETA] = rho * vn * vt2 - bn * bt2 / MU_0
    F[P] = (Etot + ptot) * vn - bn * vdotb / MU_0 + psi * bn / MU_0
    F[BR] = psi
    F[BPHI] = bt1 * vn - vt1 * bn
    F[BTHETA] = bt2 * vn - vt2 * bn
    F[PSI] = c_h ** 2 * bn
    if W.shape[0] > EINT:
        F[EINT] = W[EINT] * vn
    return F

def _mhd_flux_theta_3d(W, gamma, c_h=0.0, eps=1e-30):
    rho = torch.clamp(W[RHO], min=eps)
    vn = W[VTHETA] / rho
    vt1 = W[VR] / rho
    vt2 = W[VPHI] / rho
    bn = W[BTHETA]
    bt1 = W[BR]
    bt2 = W[BPHI]
    psi = W[PSI]
    B2 = bn ** 2 + bt1 ** 2 + bt2 ** 2
    Ekin = 0.5 * rho * (vn ** 2 + vt1 ** 2 + vt2 ** 2)
    Etot = W[P]
    p = _pressure_dual_3d(W, rho, Ekin, B2, gamma, eps)
    ptot = p + B2 / (2.0 * MU_0)
    vdotb = vn * bn + vt1 * bt1 + vt2 * bt2
    G = torch.zeros_like(W)
    G[RHO] = rho * vn
    G[VR] = rho * vn * vt1 - bn * bt1 / MU_0
    G[VPHI] = rho * vn * vt2 - bn * bt2 / MU_0
    G[VTHETA] = rho * vn * vn + ptot - bn * bn / MU_0
    G[P] = (Etot + ptot) * vn - bn * vdotb / MU_0 + psi * bn / MU_0
    G[BR] = bt1 * vn - vt1 * bn
    G[BPHI] = bt2 * vn - vt2 * bn
    G[BTHETA] = psi
    G[PSI] = c_h ** 2 * bn
    if W.shape[0] > EINT:
        G[EINT] = W[EINT] * vn
    return G

def _mhd_flux_phi_3d(W, gamma, c_h=0.0, eps=1e-30):
    rho = torch.clamp(W[RHO], min=eps)
    vn = W[VPHI] / rho
    vt1 = W[VR] / rho
    vt2 = W[VTHETA] / rho
    bn = W[BPHI]
    bt1 = W[BR]
    bt2 = W[BTHETA]
    psi = W[PSI]
    B2 = bn ** 2 + bt1 ** 2 + bt2 ** 2
    Ekin = 0.5 * rho * (vn ** 2 + vt1 ** 2 + vt2 ** 2)
    Etot = W[P]
    p = _pressure_dual_3d(W, rho, Ekin, B2, gamma, eps)
    ptot = p + B2 / (2.0 * MU_0)
    vdotb = vn * bn + vt1 * bt1 + vt2 * bt2
    G = torch.zeros_like(W)
    G[RHO] = rho * vn
    G[VR] = rho * vn * vt1 - bn * bt1 / MU_0
    G[VTHETA] = rho * vn * vt2 - bn * bt2 / MU_0
    G[VPHI] = rho * vn * vn + ptot - bn * bn / MU_0
    G[P] = (Etot + ptot) * vn - bn * vdotb / MU_0 + psi * bn / MU_0
    G[BR] = bt1 * vn - vt1 * bn
    G[BTHETA] = bt2 * vn - vt2 * bn
    G[BPHI] = psi
    G[PSI] = c_h ** 2 * bn
    if W.shape[0] > EINT:
        G[EINT] = W[EINT] * vn
    return G

def hlle_flux_3d(W_L, W_R, gamma, direction='r', c_h=0.0, eps=1e-30, return_speeds=False):
    rhoL = torch.clamp(W_L[RHO], min=eps)
    rhoR = torch.clamp(W_R[RHO], min=eps)
    if direction == 'r':
        vnL, bnL = (W_L[VR] / rhoL, W_L[BR])
        vnR, bnR = (W_R[VR] / rhoR, W_R[BR])
        B2L = W_L[BR] ** 2 + W_L[BPHI] ** 2 + W_L[BTHETA] ** 2
        B2R = W_R[BR] ** 2 + W_R[BPHI] ** 2 + W_R[BTHETA] ** 2
        bt_sq_L = W_L[BPHI] ** 2 + W_L[BTHETA] ** 2
        bt_sq_R = W_R[BPHI] ** 2 + W_R[BTHETA] ** 2
        flux_L = _mhd_flux_r_3d(W_L, gamma, c_h, eps)
        flux_R = _mhd_flux_r_3d(W_R, gamma, c_h, eps)
    elif direction == 'theta':
        vnL, bnL = (W_L[VTHETA] / rhoL, W_L[BTHETA])
        vnR, bnR = (W_R[VTHETA] / rhoR, W_R[BTHETA])
        B2L = W_L[BR] ** 2 + W_L[BPHI] ** 2 + W_L[BTHETA] ** 2
        B2R = W_R[BR] ** 2 + W_R[BPHI] ** 2 + W_R[BTHETA] ** 2
        bt_sq_L = W_L[BR] ** 2 + W_L[BPHI] ** 2
        bt_sq_R = W_R[BR] ** 2 + W_R[BPHI] ** 2
        flux_L = _mhd_flux_theta_3d(W_L, gamma, c_h, eps)
        flux_R = _mhd_flux_theta_3d(W_R, gamma, c_h, eps)
    else:
        vnL, bnL = (W_L[VPHI] / rhoL, W_L[BPHI])
        vnR, bnR = (W_R[VPHI] / rhoR, W_R[BPHI])
        B2L = W_L[BR] ** 2 + W_L[BPHI] ** 2 + W_L[BTHETA] ** 2
        B2R = W_R[BR] ** 2 + W_R[BPHI] ** 2 + W_R[BTHETA] ** 2
        bt_sq_L = W_L[BR] ** 2 + W_L[BTHETA] ** 2
        bt_sq_R = W_R[BR] ** 2 + W_R[BTHETA] ** 2
        flux_L = _mhd_flux_phi_3d(W_L, gamma, c_h, eps)
        flux_R = _mhd_flux_phi_3d(W_R, gamma, c_h, eps)
    Ekin_L = 0.5 * rhoL * ((W_L[VR] / rhoL) ** 2 + (W_L[VPHI] / rhoL) ** 2 + (W_L[VTHETA] / rhoL) ** 2)
    Ekin_R = 0.5 * rhoR * ((W_R[VR] / rhoR) ** 2 + (W_R[VPHI] / rhoR) ** 2 + (W_R[VTHETA] / rhoR) ** 2)
    pL = _pressure_dual_3d(W_L, rhoL, Ekin_L, B2L, gamma, eps)
    pR = _pressure_dual_3d(W_R, rhoR, Ekin_R, B2R, gamma, eps)
    bn_f = 0.5 * (bnL + bnR)
    bt_f_L = torch.sqrt(torch.clamp(bt_sq_L, min=0.0))
    bt_f_R = torch.sqrt(torch.clamp(bt_sq_R, min=0.0))
    cfL = _fast_speed(rhoL, pL, bn_f, bt_f_L, gamma, eps)
    cfR = _fast_speed(rhoR, pR, bn_f, bt_f_R, gamma, eps)
    SL = torch.minimum(vnL - cfL, vnR - cfR)
    SR = torch.maximum(vnL + cfL, vnR + cfR)
    if direction == 'phi':
        checks = {'rhoL': rhoL, 'rhoR': rhoR, 'vnL': vnL, 'vnR': vnR, 'bnL': bnL, 'bnR': bnR, 'B2L': B2L, 'B2R': B2R, 'pL': pL, 'pR': pR, 'bt_f_L': bt_f_L, 'bt_f_R': bt_f_R, 'cfL': cfL, 'cfR': cfR, 'SL': SL, 'SR': SR}
        for name, arr in checks.items():
            bad = ~torch.isfinite(arr)
            if bool(bad.any()):
                idx = bad.nonzero()[0]
                ri, ti, pi = (int(idx[0]), int(idx[1]), int(idx[2]))
                raise RuntimeError(f'Non-finite phi HLLE component {name} at r={ri}, th={ti}, ph={pi}: rhoL={rhoL[ri, ti, pi]}, rhoR={rhoR[ri, ti, pi]}, vnL={vnL[ri, ti, pi]}, vnR={vnR[ri, ti, pi]}, pL={pL[ri, ti, pi]}, pR={pR[ri, ti, pi]}, B2L={B2L[ri, ti, pi]}, B2R={B2R[ri, ti, pi]}, cfL={cfL[ri, ti, pi]}, cfR={cfR[ri, ti, pi]}')
    if c_h > 0:
        SL = torch.minimum(SL, torch.full_like(SL, -c_h))
        SR = torch.maximum(SR, torch.full_like(SR, c_h))
    denom = torch.clamp(SR - SL, min=eps)
    F_hlle = (SR * flux_L - SL * flux_R + SL * SR * (W_R - W_L)) / denom
    if direction == 'phi':
        bad_flux = ~torch.isfinite(F_hlle)
        if bool(bad_flux.any()):
            idx = bad_flux.nonzero()[0]
            vv, ri, ti, pi = (int(idx[0]), int(idx[1]), int(idx[2]), int(idx[3]))
            raise RuntimeError(f'Non-finite phi F_hlle var={vv} at r={ri}, th={ti}, ph={pi}: SL={SL[ri, ti, pi]}, SR={SR[ri, ti, pi]}, denom={denom[ri, ti, pi]}, flux_L={flux_L[vv, ri, ti, pi]}, flux_R={flux_R[vv, ri, ti, pi]}, W_L={W_L[vv, ri, ti, pi]}, W_R={W_R[vv, ri, ti, pi]}')
    mL = SL.unsqueeze(0).expand_as(flux_L) >= 0
    mR = SR.unsqueeze(0).expand_as(flux_R) <= 0
    flux_out = torch.where(mL, flux_L, torch.where(mR, flux_R, F_hlle))
    if return_speeds:
        a_max = torch.maximum(SR.abs(), SL.abs())
        return (flux_out, a_max)
    return flux_out

def source_terms_3d(W, r_gpu, cot_gpu, grav_gpu, gamma, c_h=0.0, c_p_ratio=0.18, eps=1e-30, dr_gpu=None, dtheta_gpu=None, dphi_gpu=None, ct_active=False, apply_powell=True):
    S = torch.zeros_like(W)
    rho = torch.clamp(W[RHO], min=eps)
    v_r = W[VR] / rho
    v_phi = W[VPHI] / rho
    v_th = W[VTHETA] / rho
    B_r = W[BR]
    B_phi = W[BPHI]
    B_th = W[BTHETA]
    psi = W[PSI]
    B2 = B_r ** 2 + B_phi ** 2 + B_th ** 2
    Ekin = 0.5 * rho * (v_r ** 2 + v_phi ** 2 + v_th ** 2)
    p = _pressure_dual_3d(W, rho, Ekin, B2, gamma, eps)
    p_tot = p + B2 / (2.0 * MU_0)
    S[VR] = (rho * v_th ** 2 + rho * v_phi ** 2 + 2.0 * p_tot - B_th ** 2 / MU_0 - B_phi ** 2 / MU_0) / r_gpu + rho * grav_gpu
    ntheta = W.shape[2]
    pole_mask = torch.ones(1, 1, ntheta, 1, device=W.device, dtype=W.dtype)
    pole_mask[:, :, 0, :] = 0.0
    pole_mask[:, :, -1, :] = 0.0
    S[VTHETA] = pole_mask * ((rho * v_phi ** 2 - B_phi ** 2 / MU_0) * cot_gpu + p_tot * cot_gpu - rho * v_r * v_th + B_r * B_th / MU_0) / r_gpu
    S[VPHI] = pole_mask * (-(rho * v_r * v_phi - B_r * B_phi / MU_0) - (rho * v_th * v_phi - B_th * B_phi / MU_0) * cot_gpu) / r_gpu
    S[P] = rho * v_r * grav_gpu
    if W.shape[0] > EINT and dtheta_gpu is not None and (dphi_gpu is not None):
        div_v = torch.zeros_like(v_r)
        r2vr = r_gpu ** 2 * v_r
        center_span_r = torch.clamp(r_gpu[2:] - r_gpu[:-2], min=eps)
        div_v[1:-1] += (r2vr[2:] - r2vr[:-2]) / (r_gpu[1:-1] ** 2 * center_span_r)
        theta_centers = (torch.arange(ntheta, device=W.device, dtype=W.dtype) + 0.5) * (torch.pi / ntheta)
        sin_theta = torch.sin(theta_centers).reshape(1, ntheta, 1)
        sin_vtheta = sin_theta * v_th
        div_v[:, 1:-1, :] += (sin_vtheta[:, 2:, :] - sin_vtheta[:, :-2, :]) / (2.0 * r_gpu * sin_theta[:, 1:-1, :] * dtheta_gpu)
        div_v += (torch.roll(v_phi, -1, dims=2) - torch.roll(v_phi, 1, dims=2)) / (2.0 * r_gpu * sin_theta * dphi_gpu)
        S[EINT] = -p * div_v
    glm_br_source = 2.0 * psi / r_gpu if c_h > 0 else torch.zeros_like(B_r)
    glm_btheta_source = cot_gpu * psi / r_gpu if c_h > 0 else torch.zeros_like(B_th)
    S[BR] = glm_br_source
    if not ct_active:
        E_phi = v_th * B_r - v_r * B_th
        S[BTHETA] = -E_phi / r_gpu + glm_btheta_source
        E_z = v_r * B_phi - v_phi * B_r
        E_r = v_th * B_phi - v_phi * B_th
        S[BPHI] = (+E_z - cot_gpu * E_r) / r_gpu
    if c_h > 0:
        if dr_gpu is not None and dtheta_gpu is not None and (dphi_gpu is not None):
            theta_centers_glm = (torch.arange(ntheta, device=W.device, dtype=W.dtype) + 0.5) * (torch.pi / ntheta)
            sin_theta_glm = torch.sin(theta_centers_glm).reshape(1, ntheta, 1)
            dx = torch.minimum(dr_gpu, r_gpu * dtheta_gpu)
            dx = torch.minimum(dx, r_gpu * sin_theta_glm * dphi_gpu)
            tau = torch.clamp(max(c_p_ratio, 0.001) * dx / c_h, min=eps)
            S[PSI] = -psi / tau
    if apply_powell and dr_gpu is not None and (dtheta_gpu is not None) and (dphi_gpu is not None):
        ntheta_ = W.shape[2]
        r2Br = r_gpu ** 2 * B_r
        divB_r = torch.zeros_like(B_r)
        center_span_r = torch.clamp(r_gpu[2:] - r_gpu[:-2], min=eps)
        divB_r[1:-1, :, :] = (r2Br[2:, :, :] - r2Br[:-2, :, :]) / (r_gpu[1:-1] ** 2 * center_span_r)
        sint_c = torch.sin(torch.linspace(0, torch.pi, ntheta_ + 1, device=W.device, dtype=W.dtype)[:-1] + 0.5 * (torch.pi / ntheta_)).reshape(1, ntheta_, 1)
        sinBth = sint_c * B_th
        divB_th = torch.zeros_like(B_th)
        divB_th[:, 1:-1, :] = (sinBth[:, 2:, :] - sinBth[:, :-2, :]) / (2.0 * sint_c[:, 1:-1, :] * r_gpu * dtheta_gpu)
        divB_phi_t = torch.zeros_like(B_phi)
        divB_phi_t[:, :, 1:-1] = (B_phi[:, :, 2:] - B_phi[:, :, :-2]) / (2.0 * r_gpu * sint_c * dphi_gpu)
        divB_phi_t[:, :, 0] = (B_phi[:, :, 1] - B_phi[:, :, -1]) / (2.0 * r_gpu[:, :, 0] * sint_c[0, :, 0] * dphi_gpu)
        divB_phi_t[:, :, -1] = (B_phi[:, :, 0] - B_phi[:, :, -2]) / (2.0 * r_gpu[:, :, 0] * sint_c[0, :, 0] * dphi_gpu)
        divB = divB_r + divB_th + divB_phi_t
        S[VR] -= divB * B_r / MU_0
        S[VPHI] -= divB * B_phi / MU_0
        S[VTHETA] -= divB * B_th / MU_0
        S[P] -= divB * (v_r * B_r + v_phi * B_phi + v_th * B_th) / MU_0
        S[BR] -= divB * v_r
        S[BPHI] -= divB * v_phi
        S[BTHETA] -= divB * v_th
    return S

def compute_c_h_3d(W, gamma, eps=1e-30):
    radial_margin = 3 if W.shape[1] > 7 else 1
    theta_margin = 3 if W.shape[2] > 7 else 1
    W_core = W[:, radial_margin:-radial_margin, theta_margin:-theta_margin, :]
    if W_core.numel() == 0:
        W_core = W
    rho = torch.clamp(W_core[RHO], min=eps)
    vr = W_core[VR]
    vphi = W_core[VPHI]
    vth = W_core[VTHETA]
    Br = W_core[BR]
    Bp = W_core[BPHI]
    Bt = W_core[BTHETA]
    B2 = Br ** 2 + Bp ** 2 + Bt ** 2
    Ekin = 0.5 * rho * ((vr / rho) ** 2 + (vphi / rho) ** 2 + (vth / rho) ** 2)
    p = _pressure_dual_3d(W_core, rho, Ekin, B2, gamma, eps)
    B_tot = torch.sqrt(torch.clamp(B2, min=0.0))
    cf = _fast_speed(rho, p, B_tot, torch.zeros_like(B_tot), gamma, eps)
    v_max = torch.maximum(torch.maximum(torch.abs(vr / rho), torch.abs(vphi / rho)), torch.abs(vth / rho))
    c_h = (v_max + cf).max()
    if not bool(torch.isfinite(c_h)):
        raise RuntimeError('non-finite GLM cleaning speed')
    return float(c_h)

def apply_pole_bcs_3d(U):
    U[RHO, :, 0, :] = U[RHO, :, 1, :]
    U[P, :, 0, :] = U[P, :, 1, :]
    U[VR, :, 0, :] = U[VR, :, 1, :]
    U[BR, :, 0, :] = U[BR, :, 1, :]
    U[BTHETA, :, 0, :] = U[BTHETA, :, 1, :]
    if U.shape[0] > EINT:
        U[EINT, :, 0, :] = U[EINT, :, 1, :]
    U[VPHI, :, 0, :] = 0.0
    U[VTHETA, :, 0, :] = 0.0
    U[BPHI, :, 0, :] = 0.0
    U[PSI, :, 0, :] = 0.0
    U[RHO, :, -1, :] = U[RHO, :, -2, :]
    U[P, :, -1, :] = U[P, :, -2, :]
    U[VR, :, -1, :] = U[VR, :, -2, :]
    U[BR, :, -1, :] = U[BR, :, -2, :]
    U[BTHETA, :, -1, :] = U[BTHETA, :, -2, :]
    if U.shape[0] > EINT:
        U[EINT, :, -1, :] = U[EINT, :, -2, :]
    U[VPHI, :, -1, :] = 0.0
    U[VTHETA, :, -1, :] = 0.0
    U[BPHI, :, -1, :] = 0.0
    U[PSI, :, -1, :] = 0.0
    return U

def hlle_muscl_step_3d(U, dr_gpu, r_gpu, dtheta, dphi, sin_th_edges_gpu, cot_gpu, grav_gpu, gamma, dt, c_h=None, c_p_ratio=0.18, U_eq=None, use_glm=True, eps=1e-30, Br_face=None, Bth_face=None, Bphi_face=None, r_edges_gpu=None, theta_edges_gpu=None, inner_guard_cells=3, outer_guard_cells=3, pole_guard_cells=9, spatial_order=2, shock_flattening=False, shock_threshold=2.0):
    nr, ntheta = (U.shape[1], U.shape[2])
    if spatial_order not in (1, 2):
        raise ValueError('spatial_order must be 1 or 2')
    if shock_threshold <= 1.0:
        raise ValueError('shock_threshold must be greater than 1')
    r_guard = min(max(int(inner_guard_cells), 1), max(nr - 4, 1))
    r_outer_guard = min(max(int(outer_guard_cells), 1), max(nr - r_guard - 2, 1))
    th_guard = min(max(int(pole_guard_cells), 1), max((ntheta - 2) // 2, 1))
    W = prim_to_cons_3d(U, gamma, eps)
    W_eq = prim_to_cons_3d(U_eq, gamma, eps) if U_eq is not None else None

    def _shift_with_edge(values, dim, offset, periodic):
        if periodic:
            return torch.roll(values, shifts=offset, dims=dim)
        if offset == -1:
            tail = values.select(dim, values.shape[dim] - 1).unsqueeze(dim)
            return torch.cat((values.narrow(dim, 1, values.shape[dim] - 1), tail), dim=dim)
        head = values.select(dim, 0).unsqueeze(dim)
        return torch.cat((head, values.narrow(dim, 0, values.shape[dim] - 1)), dim=dim)

    def _shock_mask(dim, periodic=False):
        log_rho = torch.log(torch.clamp(U[RHO], min=1e-25))
        log_p = torch.log(torch.clamp(U[P], min=1e-20))
        threshold = torch.log(torch.tensor(shock_threshold, dtype=U.dtype, device=U.device))
        sensor = torch.zeros_like(log_rho)
        for field in (log_rho, log_p):
            forward = _shift_with_edge(field, dim, -1, periodic)
            backward = _shift_with_edge(field, dim, 1, periodic)
            sensor = torch.maximum(sensor, torch.abs(forward - field))
            sensor = torch.maximum(sensor, torch.abs(field - backward))
        return sensor >= threshold
    if spatial_order == 2 and shock_flattening:
        flatten_r = _shock_mask(0)
        flatten_t = _shock_mask(1)
        flatten_p = _shock_mask(2, periodic=True)
    else:
        flatten_r = flatten_t = flatten_p = None

    def _sanitize_face_state(S):
        S[RHO] = torch.clamp(S[RHO], min=1e-25)
        if S.shape[0] > I_EINT:
            S[I_EINT] = torch.clamp(S[I_EINT], min=eps / (gamma - 1.0))
        mom_sq = S[VR] ** 2 + S[VPHI] ** 2 + S[VTHETA] ** 2
        e_kin = 0.5 * mom_sq / S[RHO]
        b_sq = S[BR] ** 2 + S[BPHI] ** 2 + S[BTHETA] ** 2
        e_mag = b_sq / (2.0 * MU_0)
        if S.shape[0] > I_EINT:
            e_thermal = S[I_EINT]
        else:
            e_thermal = torch.full_like(e_kin, eps / (gamma - 1.0))
        S[P] = torch.maximum(S[P], e_kin + e_mag + e_thermal)
        return S

    def _assert_finite_state(name, S):
        bad = ~torch.isfinite(S)
        if bool(bad.any()):
            idx = bad.nonzero()[0]
            vv, ri, ti, pi = (int(idx[0]), int(idx[1]), int(idx[2]), int(idx[3]))
            varnames = ['RHO', 'MOM_R/VR', 'MOM_PHI/VPHI', 'ETOT/P', 'BR', 'BPHI', 'PSI', 'MOM_THETA/VTHETA', 'BTHETA', 'EINT']
            vname = varnames[vv] if vv < len(varnames) else f'VAR{vv}'
            raise RuntimeError(f'Non-finite in {name}: var={vv} {vname} at r={ri}, th={ti}, ph={pi}')
    if use_glm:
        if c_h is None:
            c_h = compute_c_h_3d(W, gamma, eps)
    else:
        c_h = 0.0
    use_ct = use_glm and c_h > 0 and (Br_face is not None) and (Bth_face is not None) and (Bphi_face is not None) and (r_edges_gpu is not None) and (theta_edges_gpu is not None)
    if use_ct and W_eq is not None:
        raise ValueError('equilibrium-residual mode is not implemented for constrained transport; disable use_well_balanced when use_ct=True')
    dtheta_t = torch.tensor(dtheta, dtype=W.dtype, device=W.device)
    dphi_t = torch.tensor(dphi, dtype=W.dtype, device=W.device)
    if W_eq is not None:
        S_eq = source_terms_3d(W_eq, r_gpu, cot_gpu, grav_gpu, gamma, c_h=c_h, c_p_ratio=c_p_ratio, eps=eps, dr_gpu=dr_gpu, dtheta_gpu=dtheta_t, dphi_gpu=dphi_t, ct_active=use_ct, apply_powell=not use_ct)
        S = source_terms_3d(W, r_gpu, cot_gpu, grav_gpu, gamma, c_h=c_h, c_p_ratio=c_p_ratio, eps=eps, dr_gpu=dr_gpu, dtheta_gpu=dtheta_t, dphi_gpu=dphi_t, ct_active=use_ct, apply_powell=not use_ct)
    else:
        S_eq = None
        S = source_terms_3d(W, r_gpu, cot_gpu, grav_gpu, gamma, c_h=c_h, c_p_ratio=c_p_ratio, eps=eps, dr_gpu=dr_gpu, dtheta_gpu=dtheta_t, dphi_gpu=dphi_t, ct_active=use_ct, apply_powell=not use_ct)
    _assert_finite_state('source terms S', S)
    if S_eq is not None:
        _assert_finite_state('equilibrium source terms S_eq', S_eq)
    Wp = W - W_eq if W_eq is not None else W
    dW_f_p = torch.roll(Wp, -1, dims=3) - Wp
    dW_b_p = Wp - torch.roll(Wp, 1, dims=3)
    if spatial_order == 2:
        slope_p = minmod(dW_b_p, dW_f_p)
        if flatten_p is not None:
            slope_p = torch.where(flatten_p.unsqueeze(0), torch.zeros_like(slope_p), slope_p)
    else:
        slope_p = torch.zeros_like(Wp)
    W_pL = W + 0.5 * slope_p
    W_pR = torch.roll(W - 0.5 * slope_p, -1, dims=3)
    if use_ct and Bphi_face is not None and (Br_face is not None) and (Bth_face is not None):
        W_pL[BPHI] = Bphi_face
        W_pR[BPHI] = Bphi_face
        Br_c_phi = 0.5 * (Br_face[:-1] + Br_face[1:])
        W_pL[BR] = Br_c_phi
        W_pR[BR] = Br_c_phi
        Bth_c_phi = 0.5 * (Bth_face[:, :-1, :] + Bth_face[:, 1:, :])
        W_pL[BTHETA] = Bth_c_phi
        W_pR[BTHETA] = Bth_c_phi
    W_pL = _sanitize_face_state(W_pL)
    W_pR = _sanitize_face_state(W_pR)
    _assert_finite_state('W_pL before phi flux', W_pL)
    _assert_finite_state('W_pR before phi flux', W_pR)
    F_phi_3d = hlle_flux_3d(W_pL, W_pR, gamma, direction='phi', c_h=c_h, eps=eps)
    _assert_finite_state('F_phi_3d', F_phi_3d)
    dF_phi = F_phi_3d - torch.roll(F_phi_3d, 1, dims=3)
    dW_f_t = Wp[:, :, 1:, :] - Wp[:, :, :-1, :]
    dW_b_t = torch.cat([torch.zeros_like(dW_f_t[:, :, :1, :]), dW_f_t], dim=2)
    dW_f_t_pad = torch.cat([dW_f_t, torch.zeros_like(dW_f_t[:, :, -1:, :])], dim=2)
    if spatial_order == 2:
        slope_t = minmod(dW_b_t, dW_f_t_pad)
        if flatten_t is not None:
            slope_t = torch.where(flatten_t.unsqueeze(0), torch.zeros_like(slope_t), slope_t)
    else:
        slope_t = torch.zeros_like(Wp)
    W_tL_f = (W + 0.5 * slope_t)[:, :, :-1, :]
    W_tR_f = (W - 0.5 * slope_t)[:, :, 1:, :]
    if use_ct and Bth_face is not None and (Br_face is not None) and (Bphi_face is not None):
        Bth_at_thface = Bth_face[:, 1:-1, :]
        W_tL_f[BTHETA] = Bth_at_thface
        W_tR_f[BTHETA] = Bth_at_thface
        Br_c = 0.5 * (Br_face[:-1] + Br_face[1:])
        Br_at_thface = 0.5 * (Br_c[:, :-1, :] + Br_c[:, 1:, :])
        W_tL_f[BR] = Br_at_thface
        W_tR_f[BR] = Br_at_thface
        Bphi_at_thface = 0.5 * (Bphi_face[:, :-1, :] + Bphi_face[:, 1:, :])
        W_tL_f[BPHI] = Bphi_at_thface
        W_tR_f[BPHI] = Bphi_at_thface
    W_tL_f = _sanitize_face_state(W_tL_f)
    W_tR_f = _sanitize_face_state(W_tR_f)
    _assert_finite_state('W_tL_f before theta flux', W_tL_f)
    _assert_finite_state('W_tR_f before theta flux', W_tR_f)
    G_theta, a_th = hlle_flux_3d(W_tL_f, W_tR_f, gamma, direction='theta', c_h=c_h, eps=eps, return_speeds=True)
    _assert_finite_state('G_theta', G_theta)
    if not bool(torch.isfinite(a_th).all()):
        idx = (~torch.isfinite(a_th)).nonzero()[0]
        raise RuntimeError(f'Non-finite a_t at r={int(idx[0])}, th={int(idx[1])}, ph={int(idx[2])}')
    sin_faces = sin_th_edges_gpu[1:-1]
    sin_faces = sin_faces.reshape(1, 1, ntheta - 1, 1)
    G_sin = G_theta * sin_faces
    sin_centers = torch.sin(torch.linspace(0, torch.pi, ntheta + 1, device=W.device, dtype=W.dtype)[:-1] + 0.5 * (torch.pi / ntheta))
    sin_centers = sin_centers.reshape(1, 1, ntheta, 1)
    G_R = G_sin[:, :, 1:, :]
    G_L = G_sin[:, :, :-1, :]
    dG_interior = (G_R - G_L) / (sin_centers[:, :, 1:-1, :] * r_gpu * dtheta)
    dG = torch.zeros_like(W)
    dG[:, :, 1:-1, :] = dG_interior
    if W_eq is not None:
        Wp_r = W - W_eq
        dWp_f_r = Wp_r[:, 1:, :, :] - Wp_r[:, :-1, :, :]
        dWp_b_pad = torch.cat([torch.zeros_like(dWp_f_r[:, :1]), dWp_f_r], dim=1)
        dWp_f_pad = torch.cat([dWp_f_r, torch.zeros_like(dWp_f_r[:, -1:])], dim=1)
        if spatial_order == 2:
            slope_r = minmod(dWp_b_pad, dWp_f_pad)
            if flatten_r is not None:
                slope_r = torch.where(flatten_r.unsqueeze(0), torch.zeros_like(slope_r), slope_r)
        else:
            slope_r = torch.zeros_like(Wp_r)
        W_rL = W_eq + Wp_r + 0.5 * slope_r
        W_rR = W_eq + Wp_r - 0.5 * slope_r
    else:
        dW_f_r = W[:, 1:, :, :] - W[:, :-1, :, :]
        dW_b_r = W[:, 1:, :, :] - W[:, :-1, :, :]
        dW_b_pad = torch.cat([torch.zeros_like(dW_b_r[:, :1]), dW_b_r], dim=1)
        dW_f_pad = torch.cat([dW_f_r, torch.zeros_like(dW_f_r[:, -1:])], dim=1)
        if spatial_order == 2:
            slope_r = minmod(dW_b_pad, dW_f_pad)
            if flatten_r is not None:
                slope_r = torch.where(flatten_r.unsqueeze(0), torch.zeros_like(slope_r), slope_r)
        else:
            slope_r = torch.zeros_like(W)
        W_rL = W + 0.5 * slope_r
        W_rR = W - 0.5 * slope_r
    W_rL_f = W_rL[:, :-1, :, :]
    W_rR_f = W_rR[:, 1:, :, :]
    n_flat_inner = min(r_guard, nr - 1)
    n_flat_outer = min(r_outer_guard, nr - 1)
    W_rL_f[:, :n_flat_inner, :, :] = W[:, :n_flat_inner, :, :]
    W_rR_f[:, :n_flat_inner, :, :] = W[:, 1:n_flat_inner + 1, :, :]
    W_rL_f[:, -n_flat_outer:, :, :] = W[:, -n_flat_outer - 1:-1, :, :]
    W_rR_f[:, -n_flat_outer:, :, :] = W[:, -n_flat_outer:, :, :]
    if use_ct and Bth_face is not None and (Bphi_face is not None):
        Bth_c = 0.5 * (Bth_face[:, :-1, :] + Bth_face[:, 1:, :])
        Bth_at_rface = 0.5 * (Bth_c[:-1] + Bth_c[1:])
        W_rL_f[BTHETA] = Bth_at_rface
        W_rR_f[BTHETA] = Bth_at_rface
        Bphi_at_rface = 0.5 * (Bphi_face[:-1] + Bphi_face[1:])
        W_rL_f[BPHI] = Bphi_at_rface
        W_rR_f[BPHI] = Bphi_at_rface
    W_rL_f = _sanitize_face_state(W_rL_f)
    W_rR_f = _sanitize_face_state(W_rR_f)
    _assert_finite_state('W_rL_f before radial flux', W_rL_f)
    _assert_finite_state('W_rR_f before radial flux', W_rR_f)
    F_r_3d, a_r = hlle_flux_3d(W_rL_f, W_rR_f, gamma, direction='r', c_h=c_h, eps=eps, return_speeds=True)
    _assert_finite_state('F_r_3d', F_r_3d)
    if not bool(torch.isfinite(a_r).all()):
        idx = (~torch.isfinite(a_r)).nonzero()[0]
        raise RuntimeError(f'Non-finite a_r at r={int(idx[0])}, th={int(idx[1])}, ph={int(idx[2])}')
    if use_ct:
        from swcmep.solar_wind.ct_mhd_3d import ct_update_3d, face_to_center
        Br_new, Bth_new, Bphi_new = ct_update_3d(Br_face, Bth_face, Bphi_face, F_r_3d, G_theta, F_phi_3d, r_edges_gpu.reshape(-1), r_gpu.reshape(-1), dr_gpu.reshape(-1), theta_edges_gpu.reshape(-1), dtheta, dphi, dt, a_r=a_r, a_th=a_th)
        Br_c, Bth_c, Bphi_c = face_to_center(Br_new, Bth_new, Bphi_new)
    if r_edges_gpu is not None:
        radial_edges = r_edges_gpu.reshape(-1).to(dtype=W.dtype)
        r_edges_inner = radial_edges[1:-1].reshape(-1, 1, 1)
        radial_volume = ((radial_edges[1:] ** 3 - radial_edges[:-1] ** 3) / 3.0).reshape(-1, 1, 1)
    else:
        r_edges_inner = 0.5 * (r_gpu[:-1] + r_gpu[1:])
        radial_volume = r_gpu ** 2 * dr_gpu
    F_r_weighted = r_edges_inner ** 2 * F_r_3d
    dF_r = torch.zeros_like(W)
    dF_r[:, 1:-1, :, :] = (F_r_weighted[:, 1:, :, :] - F_r_weighted[:, :-1, :, :]) / radial_volume[1:-1]
    sin_safe = torch.clamp(sin_centers, min=1e-06)
    dF_phi_div = dF_phi / (r_gpu * sin_safe * dphi)
    if W_eq is not None:
        Wp_eq_L = _sanitize_face_state(W_eq.clone())
        Wp_eq_R = _sanitize_face_state(torch.roll(W_eq, -1, dims=3).clone())
        F_phi_eq = hlle_flux_3d(Wp_eq_L, Wp_eq_R, gamma, direction='phi', c_h=c_h, eps=eps)
        dF_phi_eq = F_phi_eq - torch.roll(F_phi_eq, 1, dims=3)
        dF_phi_eq_div = dF_phi_eq / (r_gpu * sin_safe * dphi)
        Wt_eq_L = _sanitize_face_state(W_eq[:, :, :-1, :].clone())
        Wt_eq_R = _sanitize_face_state(W_eq[:, :, 1:, :].clone())
        G_eq = hlle_flux_3d(Wt_eq_L, Wt_eq_R, gamma, direction='theta', c_h=c_h, eps=eps)
        G_eq_sin = G_eq * sin_faces
        dG_eq = torch.zeros_like(W_eq)
        dG_eq[:, :, 1:-1, :] = (G_eq_sin[:, :, 1:, :] - G_eq_sin[:, :, :-1, :]) / (sin_centers[:, :, 1:-1, :] * r_gpu * dtheta)
        Wr_eq_L = _sanitize_face_state(W_eq[:, :-1, :, :].clone())
        Wr_eq_R = _sanitize_face_state(W_eq[:, 1:, :, :].clone())
        F_r_eq = hlle_flux_3d(Wr_eq_L, Wr_eq_R, gamma, direction='r', c_h=c_h, eps=eps)
        F_r_eq_weighted = r_edges_inner ** 2 * F_r_eq
        dF_r_eq = torch.zeros_like(W_eq)
        dF_r_eq[:, 1:-1, :, :] = (F_r_eq_weighted[:, 1:, :, :] - F_r_eq_weighted[:, :-1, :, :]) / radial_volume[1:-1]
        _assert_finite_state('equilibrium radial divergence', dF_r_eq)
        _assert_finite_state('equilibrium theta divergence', dG_eq)
        _assert_finite_state('equilibrium phi divergence', dF_phi_eq_div)
        dF_r = dF_r - dF_r_eq
        dG = dG - dG_eq
        dF_phi_div = dF_phi_div - dF_phi_eq_div
        S = S - S_eq
    for A in (dF_r, dG, dF_phi_div):
        A[:, 0:r_guard, :, :] = 0.0
        A[:, -r_outer_guard:, :, :] = 0.0
        A[:, :, 0:th_guard, :] = 0.0
        A[:, :, -th_guard:, :] = 0.0
    S[:, 0:r_guard, :, :] = 0.0
    S[:, -r_outer_guard:, :, :] = 0.0
    S[:, :, 0:th_guard, :] = 0.0
    S[:, :, -th_guard:, :] = 0.0
    W_new = W - dt * dF_r - dt * dG - dt * dF_phi_div + dt * S
    if use_ct:
        W_new[BR] = Br_c
        W_new[BTHETA] = Bth_c
        W_new[BPHI] = Bphi_c
    W_new[:, 0:r_guard, :, :] = W[:, 0:r_guard, :, :]
    W_new[:, -r_outer_guard:, :, :] = W[:, -r_outer_guard:, :, :]
    W_new[:, :, 0:th_guard, :] = W[:, :, 0:th_guard, :]
    W_new[:, :, -th_guard:, :] = W[:, :, -th_guard:, :]
    _assert_finite_state('W_new before positivity repair', W_new)
    rho_floor = 1e-25
    p_floor = 1e-20
    rho_needs_floor = W_new[RHO] < rho_floor
    W_new[RHO] = torch.clamp(W_new[RHO], min=rho_floor)
    rho_safe = W_new[RHO]
    mom_sq = W_new[VR] ** 2 + W_new[VPHI] ** 2 + W_new[VTHETA] ** 2
    E_kin_c = 0.5 * mom_sq / rho_safe
    B2_c = W_new[BR] ** 2 + W_new[BPHI] ** 2 + W_new[BTHETA] ** 2
    E_mag_c = B2_c / (2.0 * MU_0)
    pressure_floor = torch.full_like(E_kin_c, p_floor)
    E_min = E_kin_c + E_mag_c + pressure_floor / (gamma - 1.0)
    p_th_prefl = (gamma - 1.0) * (W_new[P] - E_kin_c - E_mag_c)
    r_stop = -r_outer_guard if r_outer_guard > 0 else None
    th_stop = -th_guard if th_guard > 0 else None
    interior_p = p_th_prefl[r_guard:r_stop, th_guard:th_stop, :]
    E_min_active = E_min[r_guard:r_stop, th_guard:th_stop, :]
    E_active = W_new[P][r_guard:r_stop, th_guard:th_stop, :]
    rho_floor_active = rho_needs_floor[r_guard:r_stop, th_guard:th_stop, :]
    floor_active = int((E_active < E_min_active).sum().item())
    rho_floor_count = int(rho_floor_active.sum().item())
    active_cells = max(int(E_active.numel()), 1)
    repair_limit = max(32, int(0.01 * active_cells))
    if floor_active > repair_limit or rho_floor_count > repair_limit:
        pmin_val = float(interior_p.min())
        raise RuntimeError(f'Positivity repair runaway: energy_cells={floor_active}, rho_cells={rho_floor_count}, active_cells={active_cells}, pmin={pmin_val:.3e}')
    if floor_active or rho_floor_count:
        if not hasattr(hlle_muscl_step_3d, '_floor_log_counter'):
            hlle_muscl_step_3d._floor_log_counter = 0
        hlle_muscl_step_3d._floor_log_counter += 1
        if hlle_muscl_step_3d._floor_log_counter % 50 == 0 or floor_active > 100 or rho_floor_count > 100:
            pmin_idx = (interior_p == interior_p.min()).nonzero()[0]
            print(f'  [positivity] energy={floor_active} rho={rho_floor_count} active={active_cells} pmin={float(interior_p.min()):+.3e} at r={int(pmin_idx[0]) + r_guard}, th={int(pmin_idx[1]) + th_guard}, ph={int(pmin_idx[2])}')
    W_new[P] = torch.maximum(W_new[P], E_min)
    if W_new.shape[0] > I_EINT:
        eint_floor = pressure_floor / (gamma - 1.0)
        W_new[I_EINT] = torch.maximum(W_new[I_EINT], eint_floor)
    U_out = cons_to_prim_3d(W_new, gamma, eps)
    U_out[P] = torch.clamp(U_out[P], min=p_floor)
    U_out[RHO] = torch.clamp(U_out[RHO], min=rho_floor)
    _assert_finite_state('primitive state after update', U_out)
    U_out = apply_pole_bcs_3d(U_out)
    if use_ct:
        return (U_out, float(c_h), Br_new, Bth_new, Bphi_new)
    return (U_out, float(c_h))
