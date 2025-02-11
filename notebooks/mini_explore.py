import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import (
    Galactocentric,
    SkyCoord,
)
from astropy.io import fits
from astropy.table import Table

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

# Take first 100 stars but sort by total velocity to get more bound ones
total_pm = np.sqrt(selected_stars["PMRA"] ** 2 + selected_stars["PMDEC"] ** 2)
total_vel = np.sqrt(total_pm**2 + selected_stars["VRAD"] ** 2)
sorted_idx = np.argsort(total_vel)
selected_stars = selected_stars[sorted_idx[:100]]

# Print velocity statistics
print("\nVelocity statistics for selected stars:")
print(
    f"Radial velocity range: {np.min(selected_stars['VRAD']):.1f} to {np.max(selected_stars['VRAD']):.1f} km/s"
)
print(
    f"Proper motion RA range: {np.min(selected_stars['PMRA']):.1f} to {np.max(selected_stars['PMRA']):.1f} mas/yr"
)
print(
    f"Proper motion Dec range: {np.min(selected_stars['PMDEC']):.1f} to {np.max(selected_stars['PMDEC']):.1f} mas/yr"
)

# Take just first 10 stars for testing
selected_stars = selected_stars

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


# print the units of X, Y, Z, vX, vY, vZ, R, phi, vR, vT
print(
    f"X range: {np.min(X):.1f} to {np.max(X):.1f} kpc, "
    f"mean: {np.mean(X):.1f}, std: {np.std(X):.1f} kpc"
    f"Y range: {np.min(Y):.1f} to {np.max(Y):.1f} kpc, "
    f"mean: {np.mean(Y):.1f}, std: {np.std(Y):.1f} kpc"
    f"Z range: {np.min(Z):.1f} to {np.max(Z):.1f} kpc, "
    f"mean: {np.mean(Z):.1f}, std: {np.std(Z):.1f} kpc"
    f"vX range: {np.min(vX):.1f} to {np.max(vX):.1f} km/s, "
    f"mean: {np.mean(vX):.1f}, std: {np.std(vX):.1f} km/s"
    f"vY range: {np.min(vY):.1f} to {np.max(vY):.1f} km/s, "
    f"mean: {np.mean(vY):.1f}, std: {np.std(vY):.1f} km/s"
    f"vZ range: {np.min(vZ):.1f} to {np.max(vZ):.1f} km/s, "
    f"mean: {np.mean(vZ):.1f}, std: {np.std(vZ):.1f} km/s"
    f"R range: {np.min(R):.1f} to {np.max(R):.1f} kpc, "
    f"mean: {np.mean(R):.1f}, std: {np.std(R):.1f} kpc"
    f"phi range: {np.min(phi):.1f} to {np.max(phi):.1f} rad, "
    f"mean: {np.mean(phi):.1f}, std: {np.std(phi):.1f} rad"
    f"vR range: {np.min(vR):.1f} to {np.max(vR):.1f} km/s, "
    f"mean: {np.mean(vR):.1f}, std: {np.std(vR):.1f} km/s"
    f"vT range: {np.min(vT):.1f} to {np.max(vT):.1f} km/s, "
    f"mean: {np.mean(vT):.1f}, std: {np.std(vT):.1f} km/s"
)

# plot a histrogram of X, Y, Z, vX, vY, vZ, R, phi, vR, vT

plt.hist2d(X, Y, bins=50, cmap="viridis")
plt.show()
plt.hist2d(R, Z, bins=50, cmap="viridis")
plt.show()
plt.hist2d(vX, vY, bins=50, cmap="viridis")
plt.show()
plt.hist2d(R, phi, bins=50, cmap="viridis")
plt.show()
plt.hist2d(vR, vT, bins=50, cmap="viridis")

plt.show()
