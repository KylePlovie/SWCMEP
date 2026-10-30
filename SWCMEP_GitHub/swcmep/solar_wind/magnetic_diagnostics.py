import numpy as np
import torch
from swcmep.utils.constants import AU

def _geometry(values, like):
    return torch.as_tensor(values, dtype=torch.float64, device=like.device).reshape(-1)

def _fields(radial, theta, phi, br, btheta, bphi, dx):
    divergence = radial + theta + phi
    magnitude = torch.sqrt(br ** 2 + btheta ** 2 + bphi ** 2)
    return {'radial': radial, 'theta': theta, 'phi': phi, 'divergence': divergence, 'normalized': divergence.abs() * dx / magnitude.clamp(min=1e-30), 'magnetic_magnitude': magnitude, 'length_scale': dx.expand_as(divergence)}

def face_divergence_fields(br, btheta, bphi, r_edges, theta_edges, dphi):
    br, btheta, bphi = (field.to(torch.float64) for field in (br, btheta, bphi))
    re = _geometry(r_edges, br)
    te = _geometry(theta_edges, br)
    nr, nt, nphi = bphi.shape
    if br.shape != (nr + 1, nt, nphi) or btheta.shape != (nr, nt + 1, nphi):
        raise ValueError('incompatible staggered magnetic-field shapes')
    if re.numel() != nr + 1 or te.numel() != nt + 1:
        raise ValueError('edge coordinates do not match magnetic-field shapes')
    dr = torch.diff(re).reshape(-1, 1, 1)
    dtheta = torch.diff(te).reshape(1, -1, 1)
    rc = (0.5 * (re[:-1] + re[1:])).reshape(-1, 1, 1)
    tc = (0.5 * (te[:-1] + te[1:])).reshape(1, -1, 1)
    cos_difference = (torch.cos(te[:-1]) - torch.cos(te[1:])).reshape(1, -1, 1)
    sin_edge = torch.sin(te)
    sin_edge = torch.where(sin_edge.abs() < 1e-15, 0.0, sin_edge).reshape(1, -1, 1)
    radial_area_factor = (0.5 * (re[1:] ** 2 - re[:-1] ** 2)).reshape(-1, 1, 1)
    volume = ((re[1:] ** 3 - re[:-1] ** 3) / 3.0).reshape(-1, 1, 1) * cos_difference * dphi
    flux_r = re.reshape(-1, 1, 1) ** 2 * cos_difference * dphi * br
    flux_theta = radial_area_factor * sin_edge * dphi * btheta
    flux_phi = radial_area_factor * dtheta * bphi
    radial = (flux_r[1:] - flux_r[:-1]) / volume
    theta = (flux_theta[:, 1:] - flux_theta[:, :-1]) / volume
    phi = (torch.roll(flux_phi, -1, dims=2) - flux_phi) / volume
    dx = torch.minimum(torch.minimum(dr, rc * dtheta), rc * torch.sin(tc) * dphi)
    return _fields(radial, theta, phi, 0.5 * (br[:-1] + br[1:]), 0.5 * (btheta[:, :-1] + btheta[:, 1:]), 0.5 * (bphi + torch.roll(bphi, -1, dims=2)), dx)

def centered_divergence_fields(br, btheta, bphi, r_centers, theta_centers, dr, dtheta, dphi):
    br, btheta, bphi = (field.to(torch.float64) for field in (br, btheta, bphi))
    r = _geometry(r_centers, br).reshape(-1, 1, 1)
    theta = _geometry(theta_centers, br).reshape(1, -1, 1)
    radial_width = _geometry(dr, br).reshape(-1, 1, 1)
    sint = torch.sin(theta)
    r2br = r ** 2 * br
    radial = ((r2br[2:] - r2br[:-2]) / (r[1:-1] ** 2 * (r[2:] - r[:-2])))[:, 1:-1]
    sin_btheta = sint * btheta
    polar = ((sin_btheta[:, 2:] - sin_btheta[:, :-2]) / (r * sint[:, 1:-1] * (theta[:, 2:] - theta[:, :-2])))[1:-1]
    azimuthal = ((torch.roll(bphi, -1, dims=2) - torch.roll(bphi, 1, dims=2)) / (2.0 * r * sint * dphi))[1:-1, 1:-1]
    dx = torch.minimum(torch.minimum(radial_width, r * dtheta), r * sint * dphi)[1:-1, 1:-1]
    return _fields(radial, polar, azimuthal, br[1:-1, 1:-1], btheta[1:-1, 1:-1], bphi[1:-1, 1:-1], dx)

def summarize_divergence(fields, grid, *, region=None, offset=(0, 0, 0)):
    region = region or (slice(None), slice(None), slice(None))
    normalized = fields['normalized'][region]
    local_index = np.unravel_index(int(normalized.argmax()), tuple(normalized.shape))
    field_index = tuple((selection.indices(size)[0] + int(index) * selection.indices(size)[2] for selection, size, index in zip(region, fields['normalized'].shape, local_index)))
    i, j, k = tuple((index + shift for index, shift in zip(field_index, offset)))
    return {'max': float(normalized.max()), 'mean': float(normalized.mean()), 'rms': float(torch.sqrt(torch.mean(normalized ** 2))), 'cell_count': normalized.numel(), 'worst_cell': {'index': [i, j, k], 'r_au': float(grid.r_centers[i] / AU), 'latitude_deg': float(90.0 - np.degrees(grid.theta_centers[j])), 'longitude_deg': float(np.degrees(grid.phi_centers[k])), 'divergence_T_per_m': float(fields['divergence'][field_index]), 'radial_T_per_m': float(fields['radial'][field_index]), 'theta_T_per_m': float(fields['theta'][field_index]), 'phi_T_per_m': float(fields['phi'][field_index]), 'magnetic_magnitude_T': float(fields['magnetic_magnitude'][field_index]), 'length_scale_m': float(fields['length_scale'][field_index])}}
