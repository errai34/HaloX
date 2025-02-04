import astropy.units as u
import numpy as np
from astropy.coordinates import Galactocentric, SkyCoord
from astropy.io import fits
from astropy.table import Table

# -------------------------------
# 1. Load the DESI VAC and merge extensions
# -------------------------------
filename = "/Users/iciuca/Desktop/Research/ResearchData/DESI/mwsall-pix-fuji.fits"

# List available extensions
with fits.open(filename) as hdul:
    for idx, ext in enumerate(hdul):
        print(f"Extension {idx}: {ext.name}")

# Read the three extensions
rvt = Table.read(filename, hdu="RVTAB")
spt = Table.read(filename, hdu="SPTAB")
gaia = Table.read(filename, hdu="GAIA")

print("\nRVTAB columns:")
print(rvt.colnames)
print("\nSPTAB columns:")
print(spt.colnames)
print("\nGAIA columns:")
print(gaia.colnames)

print("\nNumber of rows:")
print("RVTAB:", len(rvt))
print("SPTAB:", len(spt))
print("GAIA:", len(gaia))

# Merge the data by simply adding columns from GAIA and SPTAB that are not already in RVTAB.
combined = rvt.copy()
for col in gaia.colnames:
    if col not in combined.colnames:
        combined[col] = gaia[col]
for col in spt.colnames:
    if col not in combined.colnames:
        combined[col] = spt[col]

print("\nFinal combined table has", len(combined.colnames), "columns:")
print(combined.colnames)

# -------------------------------
# 2. Select the “clean‐star” sample.
# (These cuts follow the criteria in the paper.)
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
print("\nNumber of clean stars:", len(selected_stars))

# Verify that required columns exist
required_cols = ["TARGET_RA", "TARGET_DEC", "PMRA", "PMDEC", "PARALLAX", "VRAD"]
for col in required_cols:
    if col in selected_stars.colnames:
        print(f"Column '{col}' found.")
    else:
        print(f"Warning: Column '{col}' NOT found!")

# -------------------------------
# 3. Compute distances from parallax and filter out non-positive parallaxes.
# (Assuming PARALLAX is in mas.)
# -------------------------------
parallax_arcsec = selected_stars["PARALLAX"].astype(float) * 1e-3  # mas -> arcsec
valid_parallax = parallax_arcsec > 0
print(f"\nStars with valid parallax: {np.sum(valid_parallax)}")
print(
    f"Stars with non-positive parallax: {len(selected_stars) - np.sum(valid_parallax)}"
)
selected_stars = selected_stars[valid_parallax]

# Compute distance in kpc: distance (pc) = 1/parallax(arcsec) => in kpc: 1/(parallax_arcsec*1000)
dist_kpc = 1.0 / (parallax_arcsec[valid_parallax] * 1000.0)
selected_stars["DIST_KPC"] = dist_kpc

# # Keep only stars within 5 kpc (local halo sample)
# nearby = dist_kpc < 5.0
# print("\nDistance filtering:")
# print(f"Total stars with valid parallax: {len(dist_kpc)}")
# print(f"Stars within 5 kpc: {np.sum(nearby)}")
# selected_stars = selected_stars[nearby]
# selected_stars["DIST_KPC"] = dist_kpc[nearby]

# -------------------------------
# 4. Build a SkyCoord object (filtering out any rows with NaN in key columns)
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
print("After filtering for finite coordinates, number of stars:", len(selected_stars))

# Print some basic VRAD info
print("\nVRAD column information:")
print(f"VRAD dtype: {selected_stars['VRAD'].dtype}")
print(f"VRAD sample values: {selected_stars['VRAD'][:5]}")
print(f"VRAD min: {np.min(selected_stars['VRAD'])}")
print(f"VRAD max: {np.max(selected_stars['VRAD'])}")
print(f"VRAD mean: {np.mean(selected_stars['VRAD'])}")
print(f"VRAD std: {np.std(selected_stars['VRAD'])}")

print("\nInput data statistics:")
print(
    f"RA range: {np.min(selected_stars['TARGET_RA']):.2f} to {np.max(selected_stars['TARGET_RA']):.2f} deg"
)
print(
    f"DEC range: {np.min(selected_stars['TARGET_DEC']):.2f} to {np.max(selected_stars['TARGET_DEC']):.2f} deg"
)
print(
    f"Distance range: {np.min(selected_stars['DIST_KPC']):.2f} to {np.max(selected_stars['DIST_KPC']):.2f} kpc"
)
print(
    f"PMRA range: {np.min(selected_stars['PMRA']):.2f} to {np.max(selected_stars['PMRA']):.2f} mas/yr"
)
print(
    f"PMDEC range: {np.min(selected_stars['PMDEC']):.2f} to {np.max(selected_stars['PMDEC']):.2f} mas/yr"
)

# Filter out stars with unreasonable distances (> 100 kpc)
dist_mask = selected_stars["DIST_KPC"] <= 100
selected_stars = selected_stars[dist_mask]
print(
    f"\nAfter filtering for reasonable distances (<=100 kpc), number of stars: {len(selected_stars)}"
)
print(
    f"Updated distance range: {np.min(selected_stars['DIST_KPC']):.2f} to {np.max(selected_stars['DIST_KPC']):.2f} kpc"
)

# -------------------------------
# 5. Create coordinates in the Galactocentric frame.
# -------------------------------
# First create ICRS coordinates without velocities
icrs_coords = SkyCoord(
    ra=selected_stars["TARGET_RA"] * u.deg,
    dec=selected_stars["TARGET_DEC"] * u.deg,
    distance=selected_stars["DIST_KPC"] * u.kpc,
    frame="icrs",
)

# Convert proper motions to velocities
# The conversion factor is: pm [mas/yr] * distance [kpc] * 4.74 = v [km/s]
k = 4.74  # Conversion factor from mas/yr*kpc to km/s
dist_kpc = selected_stars["DIST_KPC"]

# Calculate tangential velocities
vl = selected_stars["PMRA"] * k * dist_kpc  # km/s
vb = selected_stars["PMDEC"] * k * dist_kpc  # km/s
vr = selected_stars["VRAD"].data.astype(float)  # km/s

# Create full space motion coordinates
coords = SkyCoord(
    ra=selected_stars["TARGET_RA"] * u.deg,
    dec=selected_stars["TARGET_DEC"] * u.deg,
    distance=dist_kpc * u.kpc,
    pm_ra_cosdec=selected_stars["PMRA"] * u.mas / u.yr,
    pm_dec=selected_stars["PMDEC"] * u.mas / u.yr,
    radial_velocity=vr * u.km / u.s,
    frame="icrs",
)

print("\nFirst few stars in ICRS frame:")
for i in range(5):
    print(
        f"Star {i}: RA, DEC = {coords[i].ra:.2f}, {coords[i].dec:.2f}; Distance = {coords[i].distance:.2f}; PM = ({coords[i].pm_ra_cosdec:.2f}, {coords[i].pm_dec:.2f}); RV = {coords[i].radial_velocity:.2f}"
    )

# Transform to Galactocentric frame using modern values
# Values from Gravity Collaboration 2019 and Reid & Brunthaler 2020
galcen_frame = Galactocentric(
    galcen_distance=8.178 * u.kpc,  # Distance to Galactic Center
    z_sun=20.8 * u.pc,  # Height of Sun above Galactic plane
    roll=0 * u.deg,  # No roll in transformation
    galcen_v_sun=[11.1, 250.24, 7.25]
    * u.km
    / u.s,  # Solar motion relative to Galactic center
)
galcen = coords.transform_to(galcen_frame)

# Extract positions and velocities
X = galcen.x.to(u.kpc).value
Y = galcen.y.to(u.kpc).value
Z = galcen.z.to(u.kpc).value
vX = galcen.v_x.to(u.km / u.s).value
vY = galcen.v_y.to(u.km / u.s).value
vZ = galcen.v_z.to(u.km / u.s).value

# Convert to cylindrical coordinates
R = np.sqrt(X**2 + Y**2)
phi = np.arctan2(Y, X)
vR = (X * vX + Y * vY) / R  # Radial velocity
vT = (-Y * vX + X * vY) / R  # Tangential (azimuthal) velocity

print("\nCylindrical coordinate statistics:")
print(f"R range: {np.nanmin(R):.2f} to {np.nanmax(R):.2f} kpc")
print(f"phi range: {np.nanmin(phi):.2f} to {np.nanmax(phi):.2f} rad")
print(f"vR range: {np.nanmin(vR):.2f} to {np.nanmax(vR):.2f} km/s")
print(f"vT range: {np.nanmin(vT):.2f} to {np.nanmax(vT):.2f} km/s")
print(f"Z range: {np.nanmin(Z):.2f} to {np.nanmax(Z):.2f} kpc")
print(f"vZ range: {np.nanmin(vZ):.2f} to {np.nanmax(vZ):.2f} km/s")
