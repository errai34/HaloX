import astropy.units as u
import gala.dynamics as gd
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import (
    CartesianDifferential,
    CartesianRepresentation,
    Galactocentric,
    SkyCoord,
)
from astropy.io import fits
from astropy.table import Table
from gala.action_angle import actionAngleStaeckel  # New import for computing actions
from gala.potential import MilkyWayPotential

# -------------------------------
# 1. Load the DESI VAC and merge extensions
# -------------------------------
filename = "/Users/iciuca/Desktop/Research/ResearchData/DESI/mwsall-pix-fuji.fits"

with fits.open(filename) as hdul:
    for idx, ext in enumerate(hdul):
        print(f"Extension {idx}: {ext.name}")

rvt = Table.read(filename, hdu="RVTAB")
spt = Table.read(filename, hdu="SPTAB")
gaia = Table.read(filename, hdu="GAIA")

combined = rvt.copy()
for col in gaia.colnames + spt.colnames:
    if col not in combined.colnames:
        # Use the column from gaia if it exists; otherwise from spt
        combined[col] = gaia[col] if col in gaia.colnames else spt[col]

# -------------------------------
# 2. Select the "clean-star" sample
# -------------------------------
sel_clean = (
    (combined["RVS_WARN"] == 0)
    & (combined["RR_SPECTYPE"] == "STAR")
    & (np.abs(combined["VRAD"]) < 600)
    & (combined["VRAD_ERR"] < 20)
    & (combined["VSINI"] < 50)
    & (combined["FEH"] > -3.9)
    & (combined["FEH"] < 0.5)
    & (combined["FEH_ERR"] < 0.2)
)
selected_stars = combined[sel_clean]

# -------------------------------
# 3. Compute distances and filter
# -------------------------------
# Convert parallax (in mas) to arcsec then compute distance in kpc.
parallax_arcsec = selected_stars["PARALLAX"].astype(float) * 1e-3
valid_parallax = parallax_arcsec > 0
selected_stars = selected_stars[valid_parallax]
# Distance in kpc: 1/parallax (arcsec) gives distance in parsecs, so divide by 1000.
dist_kpc = 1.0 / (parallax_arcsec[valid_parallax] * 1000.0)
selected_stars["DIST_KPC"] = dist_kpc

# -------------------------------
# 4. Build a SkyCoord object
# -------------------------------
mask_finite = (
    np.isfinite(selected_stars["TARGET_RA"])
    & np.isfinite(selected_stars["TARGET_DEC"])
    & np.isfinite(selected_stars["PMRA"])
    & np.isfinite(selected_stars["PMDEC"])
    & np.isfinite(selected_stars["DIST_KPC"])
    & np.isfinite(selected_stars["VRAD"])
)
selected_stars = selected_stars[mask_finite]
selected_stars = selected_stars[selected_stars["DIST_KPC"] <= 100]

# --- Decouple the radial_velocity unit if it already exists ---
if hasattr(selected_stars["VRAD"], "quantity"):
    vrad_val = selected_stars["VRAD"].quantity.value.astype(float)
else:
    vrad_val = selected_stars["VRAD"].data.astype(float)

coords = SkyCoord(
    ra=selected_stars["TARGET_RA"] * u.deg,
    dec=selected_stars["TARGET_DEC"] * u.deg,
    distance=selected_stars["DIST_KPC"] * u.kpc,
    pm_ra_cosdec=selected_stars["PMRA"] * u.mas / u.yr,
    pm_dec=selected_stars["PMDEC"] * u.mas / u.yr,
    radial_velocity=vrad_val * u.km / u.s,
    frame="icrs",
)

galcen_frame = Galactocentric(
    galcen_distance=8.178 * u.kpc,
    z_sun=20.8 * u.pc,
    roll=0 * u.deg,
    galcen_v_sun=[11.1, 250.24, 7.25] * u.km / u.s,
)
galcen = coords.transform_to(galcen_frame)

X = galcen.x.to(u.kpc).value
Y = galcen.y.to(u.kpc).value
Z = galcen.z.to(u.kpc).value
vX = galcen.v_x.to(u.km / u.s).value
vY = galcen.v_y.to(u.km / u.s).value
vZ = galcen.v_z.to(u.km / u.s).value

R = np.sqrt(X**2 + Y**2)
phi = np.arctan2(Y, X)
vR = (X * vX + Y * vY) / R
vT = (-Y * vX + X * vY) / R

# -------------------------------
# 6. Compute Orbital Properties using gala
# -------------------------------
mw_pot = MilkyWayPotential()

# Create a PhaseSpacePosition object
pos = CartesianRepresentation(x=X * u.kpc, y=Y * u.kpc, z=Z * u.kpc)
vel = CartesianDifferential(
    d_x=vX * u.km / u.s, d_y=vY * u.km / u.s, d_z=vZ * u.km / u.s
)
pos = pos.with_differentials(vel)
w0 = gd.PhaseSpacePosition(pos)

# Compute actions using the actionAngleStaeckel method.
aAS = actionAngleStaeckel(pot=mw_pot, delta=0.45 * u.kpc)
actions = aAS(w0)
Jr = actions["Jr"].value
# Note: actionAngleStaeckel returns 'Lz' instead of 'Jphi'; they are equivalent.
Jphi = actions["Lz"].value
Jz = actions["Jz"].value

# Compute energy directly from the potential
E = mw_pot.energy(w0).value

# -------------------------------
# 7. Apply Halo Selection
# -------------------------------
V_c = mw_pot.circular_velocity(R * u.kpc).to(u.km / u.s).value
epsilon = Jphi / (R * V_c)
halo_mask = (np.abs(epsilon) <= 0.75) & (Jr < 1e4)
halo_stars = selected_stars[halo_mask]
print("Number of halo stars:", len(halo_stars))

R_halo = R[halo_mask]
vR_halo = vR[halo_mask]
vT_halo = vT[halo_mask]
epsilon_halo = epsilon[halo_mask]
Jphi_halo = Jphi[halo_mask]
E_halo = E[halo_mask]
Jr_halo = Jr[halo_mask]

# -------------------------------
# 8. Visualization of Orbital Properties
# -------------------------------
plt.figure(figsize=(8, 5))
plt.hist(epsilon_halo, bins=30, alpha=0.7, color="C0")
plt.xlabel("Orbital Circularity (ε)")
plt.ylabel("Number of Stars")
plt.title("Distribution of Orbital Circularity in Halo Subset")
plt.show()

# -------------------------------
# 9. (Optional) Compute and visualize orbital eccentricity for a sample
# -------------------------------
n_sample = min(1000, len(halo_stars))
sample_indices = np.random.choice(
    np.arange(len(halo_stars)), size=n_sample, replace=False
)

pos_sample = (
    np.vstack(
        (
            X[halo_mask][sample_indices],
            Y[halo_mask][sample_indices],
            Z[halo_mask][sample_indices],
        )
    ).T
    * u.kpc
)
vel_sample = np.vstack(
    (
        vX[halo_mask][sample_indices],
        vY[halo_mask][sample_indices],
        vZ[halo_mask][sample_indices],
    )
).T * (u.km / u.s)
w0_sample = gd.PhaseSpacePosition(pos=pos_sample, vel=vel_sample)

orbit_sample = mw_pot.integrate_orbit(
    w0_sample, dt=1 * u.Myr, t1=0 * u.Myr, t2=10 * u.Gyr
)

r_peri = np.zeros(n_sample)
r_apo = np.zeros(n_sample)
ecc = np.zeros(n_sample)
for i, orbit in enumerate(orbit_sample):
    R_orbit = np.sqrt(orbit.pos.x**2 + orbit.pos.y**2).to(u.kpc).value
    r_peri[i] = np.min(R_orbit)
    r_apo[i] = np.max(R_orbit)
    ecc[i] = (r_apo[i] - r_peri[i]) / (r_apo[i] + r_peri[i])

plt.figure(figsize=(8, 5))
plt.hist(ecc, bins=30, alpha=0.7, color="C3")
plt.xlabel("Eccentricity")
plt.ylabel("Number of Stars")
plt.title("Orbital Eccentricity Distribution for a Sample of Halo Stars")
plt.show()
