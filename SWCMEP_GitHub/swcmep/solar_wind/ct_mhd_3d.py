import numpy as np
import torch
from swcmep.solar_wind.magnetic_diagnostics import face_divergence_fields
RHO = 0
VR = 1
VPHI = 2
P = 3
BR = 4
BPHI = 5
PSI = 6
VTHETA = 7
BTHETA = 8

def init_face_B_from_centers(Br_c, Bth_c, Bphi_c):
    Br_face = torch.cat([Br_c[:1], 0.5 * (Br_c[:-1] + Br_c[1:]), Br_c[-1:]], dim=0)
    Bth_face = torch.cat([torch.zeros_like(Bth_c[:, :1, :]), 0.5 * (Bth_c[:, :-1, :] + Bth_c[:, 1:, :]), torch.zeros_like(Bth_c[:, -1:, :])], dim=1)
    Bphi_face = Bphi_c.clone()
    return (Br_face, Bth_face, Bphi_face)

def face_to_center(Br_face, Bth_face, Bphi_face):
    return (0.5 * (Br_face[:-1] + Br_face[1:]), 0.5 * (Bth_face[:, :-1, :] + Bth_face[:, 1:, :]), Bphi_face.clone())

def ct_update_3d(Br_face, Bth_face, Bphi_face, F_r, G_theta, F_phi, r_edges, r_centers, dr, theta_edges, dtheta, dphi, dt, a_r=None, a_th=None):
    device = Br_face.device
    dtype = Br_face.dtype
    nr = Bphi_face.shape[0]
    nth = Bphi_face.shape[1]
    nph = Bphi_face.shape[2]
    eps = 1e-30
    Fr_Bth = F_r[BTHETA]
    Fr_Bth_pad = torch.cat([Fr_Bth[:1], Fr_Bth, Fr_Bth[-1:]], dim=0)
    Ephi_r_c = 0.5 * (Fr_Bth_pad[:-1] + Fr_Bth_pad[1:])
    Ephi_r_te = 0.5 * (Ephi_r_c[:, :-1, :] + Ephi_r_c[:, 1:, :])
    Ephi_th = -G_theta[BR]
    if a_r is not None and a_th is not None:
        a_r_pad = torch.cat([a_r[:1], a_r, a_r[-1:]], dim=0)
        a_r_rc = 0.5 * (a_r_pad[:-1] + a_r_pad[1:])
        a_r_te = 0.5 * (a_r_rc[:, :-1, :] + a_r_rc[:, 1:, :])
        a_th_e = a_th
        denom_ph = torch.clamp(a_r_te + a_th_e, min=eps)
        E_phi = (a_th_e * Ephi_r_te + a_r_te * Ephi_th) / denom_ph
    else:
        E_phi = 0.5 * (Ephi_r_te + Ephi_th)
    Fr_Bphi_neg = -F_r[BPHI]
    Eth_r = torch.cat([Fr_Bphi_neg[:1], Fr_Bphi_neg, Fr_Bphi_neg[-1:]], dim=0)
    Fphi_BR = F_phi[BR]
    Fphi_BR_pad = torch.cat([Fphi_BR[:1], Fphi_BR, Fphi_BR[-1:]], dim=0)
    Eth_phi = 0.5 * (Fphi_BR_pad[:-1] + Fphi_BR_pad[1:])
    E_theta = 0.5 * (Eth_r + Eth_phi)
    Fphi_Bth_neg = -F_phi[BTHETA]
    Er_phi = 0.5 * (Fphi_Bth_neg[:, :-1, :] + Fphi_Bth_neg[:, 1:, :])
    E_r = 0.5 * (G_theta[BPHI] + Er_phi)

    def zeros_pole():
        return torch.zeros(nr, 1, nph, device=device, dtype=dtype)
    E_phi_full = torch.cat([zeros_pole(), E_phi, zeros_pole()], dim=1)
    E_r_full = torch.cat([zeros_pole(), E_r, zeros_pole()], dim=1)
    E_theta_full = E_theta
    re64 = r_edges.to(torch.float64)
    dr64 = dr.to(torch.float64)
    Ep64 = E_phi_full.to(torch.float64)
    Er64 = E_r_full.to(torch.float64)
    Eth64 = E_theta_full.to(torch.float64)
    te64 = theta_edges.to(torch.float64)
    cos_te64 = torch.cos(te64)
    sin_te64 = torch.sin(te64)
    cos_diff64 = (cos_te64[:-1] - cos_te64[1:]).clamp(min=1e-10)
    rsq_diff64 = 0.5 * (re64[1:] ** 2 - re64[:-1] ** 2)
    A_r64 = (re64 ** 2).reshape(nr + 1, 1, 1) * cos_diff64.reshape(1, nth, 1) * dphi
    A_th64 = rsq_diff64.reshape(nr, 1, 1) * sin_te64.clamp(min=1e-10).reshape(1, nth + 1, 1) * dphi
    A_phi64 = rsq_diff64.reshape(nr, 1, 1) * dtheta
    Ep64_rpad = torch.cat([Ep64[:1], Ep64, Ep64[-1:]], dim=0)
    Ep64_re = 0.5 * (Ep64_rpad[:-1] + Ep64_rpad[1:])
    sinEp64 = sin_te64.reshape(1, nth + 1, 1) * Ep64_re
    Eth_phi64 = torch.roll(Eth64, -1, dims=2) - Eth64
    dBr64 = torch.zeros(nr + 1, nth, nph, dtype=torch.float64, device=device)
    dBr64[1:-1] = -((sinEp64[1:-1, :-1, :] - sinEp64[1:-1, 1:, :]) * dphi - Eth_phi64[1:-1] * dtheta) / A_r64[1:-1].clamp(min=1e-60)
    rEp64 = dr64.reshape(nr, 1, 1) * Ep64[:, 1:-1, :]
    Er_phi64 = torch.roll(Er64[:, 1:-1, :], -1, dims=2) - Er64[:, 1:-1, :]
    dBth64 = torch.zeros(nr, nth + 1, nph, dtype=torch.float64, device=device)
    dBth64[:, 1:-1, :] = -(rEp64 * dphi - Er_phi64 * dr64.reshape(nr, 1, 1)) / A_th64[:, 1:-1, :].clamp(min=1e-60)
    rEth64 = re64[1:].reshape(nr, 1, 1) * Eth64[1:] - re64[:-1].reshape(nr, 1, 1) * Eth64[:-1]
    Er_th64 = Er64[:, 1:, :] - Er64[:, :-1, :]
    dBphi64 = -(rEth64 * dtheta - Er_th64 * dr64.reshape(nr, 1, 1)) / A_phi64.expand(nr, nth, nph).clamp(min=1e-60)
    dBr_dt = dBr64.to(dtype)
    dBth_dt = dBth64.to(dtype)
    dBphi_dt = dBphi64.to(dtype)
    N_GHOST = 13
    N_OUTER = 5
    N_POLE = 9
    dBr_dt[:N_GHOST + 1, :, :] = 0.0
    dBr_dt[-N_OUTER:, :, :] = 0.0
    dBr_dt[:, :N_POLE, :] = 0.0
    dBr_dt[:, -N_POLE:, :] = 0.0
    dBth_dt[:N_GHOST, :, :] = 0.0
    dBth_dt[-N_OUTER:, :, :] = 0.0
    dBth_dt[:, :N_POLE, :] = 0.0
    dBth_dt[:, -N_POLE:, :] = 0.0
    dBphi_dt[:N_GHOST, :, :] = 0.0
    dBphi_dt[-N_OUTER:, :, :] = 0.0
    dBphi_dt[:, :N_POLE, :] = 0.0
    dBphi_dt[:, -N_POLE:, :] = 0.0
    mx = max(dBr_dt.abs().max().item(), dBth_dt.abs().max().item(), dBphi_dt.abs().max().item())
    if mx > 1e-06:
        fi = dBth_dt.abs().argmax().item()
        ri = fi // ((nth + 1) * nph)
        ji = fi // nph % (nth + 1)
        ki = fi % nph
        print(f'  [CT] max_dB={mx:.2e} at r={ri},th={ji},phi={ki} Ephi={E_phi_full[min(ri, nr - 1), min(ji, nth), ki].item():.2e}')
    Br_new = Br_face + dt * dBr_dt
    Bth_new = Bth_face + dt * dBth_dt
    Bphi_new = Bphi_face + dt * dBphi_dt
    return (Br_new, Bth_new, Bphi_new)

def compute_divB_face(Br_face, Bth_face, Bphi_face, r_edges, theta_edges, dr, dtheta, dphi):
    fields = face_divergence_fields(Br_face, Bth_face, Bphi_face, r_edges, theta_edges, dphi)
    normalized = fields['normalized']
    return (float(normalized.max()), float(normalized.mean()))

def _smoke_test_zero_flux_update():
    print('CT zero-flux smoke check (not a divergence-preservation benchmark)')
    nr, nth, nph = (32, 20, 36)
    R_SUN, AU = (695700000.0, 149600000000.0)
    r_inner = 21.5 * R_SUN
    r_edges_np = np.geomspace(r_inner, 1.1 * AU, nr + 1)
    r_centers_np = 0.5 * (r_edges_np[:-1] + r_edges_np[1:])
    dr_np = np.diff(r_edges_np)
    th_edges_np = np.linspace(0, np.pi, nth + 1)
    dtheta = th_edges_np[1] - th_edges_np[0]
    dphi = 2 * np.pi / nph
    dt_val = torch.float64
    re = torch.tensor(r_edges_np, dtype=dt_val)
    rc = torch.tensor(r_centers_np, dtype=dt_val)
    dr = torch.tensor(dr_np, dtype=dt_val)
    te = torch.tensor(th_edges_np, dtype=dt_val)
    B0 = 5e-09 * (AU / r_inner) ** 2
    Br_face = torch.zeros(nr + 1, nth, nph, dtype=dt_val)
    for i in range(nr + 1):
        Br_face[i] = B0 * (r_inner / r_edges_np[i]) ** 2
    Bth_face = torch.zeros(nr, nth + 1, nph, dtype=dt_val)
    Bphi_face = torch.zeros(nr, nth, nph, dtype=dt_val)
    d0, _ = compute_divB_face(Br_face, Bth_face, Bphi_face, re, te, dr, dtheta, dphi)
    print(f'  Initial divB (normalized): {d0:.2e}')
    F_r = torch.zeros(9, nr - 1, nth, nph, dtype=dt_val)
    G_theta = torch.zeros(9, nr, nth - 1, nph, dtype=dt_val)
    F_phi = torch.zeros(9, nr, nth, nph, dtype=dt_val)
    for step in range(5):
        Br_face, Bth_face, Bphi_face = ct_update_3d(Br_face, Bth_face, Bphi_face, F_r, G_theta, F_phi, re, rc, dr, te, dtheta, dphi, dt=60.0)
        d, _ = compute_divB_face(Br_face, Bth_face, Bphi_face, re, te, dr, dtheta, dphi)
        print(f'  Step {step + 1}: divB = {d:.2e}')
    growth = abs(d - d0) / max(d0, 1e-30)
    print(f"  {('PASS' if growth < 1e-06 else 'FAIL')}: growth={growth:.2e}")
if __name__ == '__main__':
    _smoke_test_zero_flux_update()
