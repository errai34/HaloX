import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import Galactocentric, SkyCoord
from astropy.io import fits
from astropy.table import Table
from galpy.actionAngle import actionAngleStaeckel
from galpy.orbit import Orbit
from galpy.potential import HernquistPotential, MiyamotoNagaiPotential, NFWPotential, evaluatePotentials

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

# Keep only stars within 5 kpc (local halo sample)
nearby = dist_kpc < 5.0
print("\nDistance filtering:")
print(f"Total stars with valid parallax: {len(dist_kpc)}")
print(f"Stars within 5 kpc: {np.sum(nearby)}")
selected_stars = selected_stars[nearby]
selected_stars["DIST_KPC"] = dist_kpc[nearby]

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
    frame="icrs"
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
    pm_ra_cosdec=selected_stars["PMRA"] * u.mas/u.yr,
    pm_dec=selected_stars["PMDEC"] * u.mas/u.yr,
    radial_velocity=vr * u.km/u.s,
    frame="icrs"
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
    galcen_v_sun=[11.1, 250.24, 7.25] * u.km/u.s  # Solar motion relative to Galactic center
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

# -------------------------------
# 6. Create an Orbit object (using galpy) from the cylindrical coordinates.
# -------------------------------
# Note: Orbit expects [R, vR, vT, z, vz, phi]
orbits = Orbit(vxvv=[R, vR, vT, Z, vZ, phi], ro=8.122, vo=229.0)

# -------------------------------
# 7. Halo selection using orbital circularity and a radial action cut.
# -------------------------------
# Compute orbital circularity: epsilon = |vphi|/v_circ (v_circ = 229 km/s assumed)
vphi = orbits.vphi()  # azimuthal velocity from galpy
epsilon = np.abs(vphi) / 229.0
halo_mask = epsilon <= 0.75

# Use the computed JR from the later step (if available) for the radial action cut.
# Here we postpone the final halo selection until after the action calculation.
initial_halo_sel = halo_mask  # temporary selection based on orbital circularity

# -------------------------------
# 8. Compute actions with the Stäckel approximation.
# -------------------------------
# Set up the Galactic potential (using updated parameters)
# Potential parameters based on more recent estimates
ro = 8.178  # kpc
vo = 229.0  # km/s

disk = MiyamotoNagaiPotential(amp=6.8e10, a=3.0, b=0.28)
bulge = HernquistPotential(amp=5.0e9, a=0.5)
halo = NFWPotential(amp=8.0e11, a=16.0)
total_potential = [disk, bulge, halo]

# Scale the potential to match our ro and vo
for pot in total_potential:
    pot.turn_physical_off()

# Use Staeckel approximation for action calculation in normalized units
aAS = actionAngleStaeckel(pot=total_potential, delta=0.45)

# Use the orbit arrays from the Orbit object.
R_arr = orbits.R()
vR_arr = orbits.vR()
vT_arr = orbits.vT()
z_arr = orbits.z()
vz_arr = orbits.vz()
phi_arr = orbits.phi()

valid_mask = (
    (~np.isnan(R_arr))
    & (~np.isnan(vR_arr))
    & (~np.isnan(vT_arr))
    & (~np.isnan(z_arr))
    & (~np.isnan(vz_arr))
    & (~np.isnan(phi_arr))
)
print(f"\nNumber of stars with valid orbit data: {np.sum(valid_mask)}")

# Convert to galpy natural units
ro = 8.0  # kpc
vo = 220.0  # km/s

# Print some statistics before normalization
print("\nBefore normalization:")
print(f"R range: {np.min(R_arr[valid_mask]):.2f} to {np.max(R_arr[valid_mask]):.2f} kpc")
print(f"vR range: {np.min(vR_arr[valid_mask]):.2f} to {np.max(vR_arr[valid_mask]):.2f} km/s")
print(f"vT range: {np.min(vT_arr[valid_mask]):.2f} to {np.max(vT_arr[valid_mask]):.2f} km/s")

# Convert to natural units
R_nat = R_arr[valid_mask] / ro
vR_nat = vR_arr[valid_mask] / vo
vT_nat = vT_arr[valid_mask] / vo
z_nat = z_arr[valid_mask] / ro
vz_nat = vz_arr[valid_mask] / vo

# Print statistics after normalization
print("\nAfter normalization (in natural units):")
print(f"R range: {np.min(R_nat):.2f} to {np.max(R_nat):.2f}")
print(f"vR range: {np.min(vR_nat):.2f} to {np.max(vR_nat):.2f}")
print(f"vT range: {np.min(vT_nat):.2f} to {np.max(vT_nat):.2f}")

# Compute actions
jr_vals, lz_vals, jz_vals = aAS.actionsFreqs(
    R_nat,
    vR_nat,
    vT_nat,
    z_nat,
    vz_nat,
    phi_arr[valid_mask],
)[:3]

# Print actions in natural units before scaling back
print("\nActions in natural units:")
print(f"JR range: {np.min(jr_vals):.2f} to {np.max(jr_vals):.2f}")
print(f"Lz range: {np.min(lz_vals):.2f} to {np.max(lz_vals):.2f}")
print(f"Jz range: {np.min(jz_vals):.2f} to {np.max(jz_vals):.2f}")

# Convert back to physical units: (kpc * km/s)
print(f"\nScaling factor (ro * vo): {ro * vo:.2f} kpc·km/s")
jr_vals = jr_vals * ro * vo
lz_vals = lz_vals * ro * vo
jz_vals = jz_vals * ro * vo

# Print final actions in physical units
print("\nActions in physical units (kpc·km/s):")
print(f"JR range: {np.min(jr_vals):.2f} to {np.max(jr_vals):.2f}")
print(f"Lz range: {np.min(lz_vals):.2f} to {np.max(lz_vals):.2f}")
print(f"Jz range: {np.min(jz_vals):.2f} to {np.max(jz_vals):.2f}")

# Create full-length arrays and insert computed actions
all_jr = np.full(len(R_arr), np.nan)
all_lz = np.full(len(R_arr), np.nan)
all_jz = np.full(len(R_arr), np.nan)
all_jr[valid_mask] = jr_vals
all_lz[valid_mask] = lz_vals
all_jz[valid_mask] = jz_vals

# Add computed actions to the selected_stars table
selected_stars["JR"] = all_jr
selected_stars["LZ"] = all_lz
selected_stars["JZ"] = all_jz

print("\nAction statistics:")
print(f"JR range: {np.nanmin(all_jr):.2f} to {np.nanmax(all_jr):.2f} kpc·km/s")
print(f"LZ range: {np.nanmin(all_lz):.2f} to {np.nanmax(all_lz):.2f} kpc·km/s")
print(f"JZ range: {np.nanmin(all_jz):.2f} to {np.nanmax(all_jz):.2f} kpc·km/s")

# Now apply the radial action cut.
# Use JR < 150 kpc·km/s as a reasonable cut for halo stars
JR_cut = 150.0  # kpc·km/s
# Calculate total energy in physical units
v_total_sq = vX**2 + vY**2 + vZ**2

# Convert positions to natural units for potential calculation
R_nat = np.sqrt(X**2 + Y**2) / ro
z_nat = Z / ro

# Evaluate potential and convert to physical units
phi_vals = evaluatePotentials(MWPotential2014, R_nat, z_nat) * vo**2

# Total energy = KE + PE
E = 0.5 * v_total_sq + phi_vals

# Create diagnostic plots
plt.figure(figsize=(15, 10))

# Plot 1: PMRA distribution
plt.subplot(221)
plt.hist(selected_stars['PMRA'], bins=50, alpha=0.6)
plt.xlabel('PMRA (mas/yr)')
plt.ylabel('Count')
plt.title('PMRA Distribution')

# Plot 2: JZ distribution (normalized)
plt.subplot(222)
plt.hist(all_jz / 1e3, bins=50, alpha=0.6)
plt.xlabel('JZ (10³ kpc·km/s)')
plt.ylabel('Count')
plt.title('JZ Distribution')

# Plot 3: LZ distribution (normalized)
plt.subplot(223)
plt.hist(all_lz / 1e3, bins=50, alpha=0.6)
plt.xlabel('LZ (10³ kpc·km/s)')
plt.ylabel('Count')
plt.title('LZ Distribution')

# Plot 4: Energy distribution (normalized)
plt.subplot(224)
plt.hist(E / 1e5, bins=50, alpha=0.6)
plt.xlabel('E (10⁵ km²/s²)')
plt.ylabel('Count')
plt.title('Energy Distribution')

plt.tight_layout()
plt.savefig('diagnostic_plots.png')
plt.close()

print("\nDistribution Statistics:")
print(f"PMRA: mean = {np.mean(selected_stars['PMRA']):.2f}, std = {np.std(selected_stars['PMRA']):.2f} mas/yr")
print(f"JZ: mean = {np.mean(all_jz/1e3):.2f}, std = {np.std(all_jz/1e3):.2f} (10³ kpc·km/s)")
print(f"LZ: mean = {np.mean(all_lz/1e3):.2f}, std = {np.std(all_lz/1e3):.2f} (10³ kpc·km/s)")
print(f"E: mean = {np.mean(E/1e5):.2f}, std = {np.std(E/1e5):.2f} (10⁵ km²/s²)")

print(f"\nUsing JR cut value: {JR_cut:.2f} kpc·km/s")
final_halo_sel = initial_halo_sel & (all_jr < JR_cut)
halo_subset = selected_stars[final_halo_sel]
print(
    f"\nAfter applying the JR cut, number of stars in the halo subset: {len(halo_subset)}"
)

# -------------------------------
# 9. Recreate Fig. 6 (Left Panel): 2D Histogram of Orbital Energy vs Lz, color-coded by median [FEH]
# -------------------------------
# For this, we need to compute the orbital energy (E) for each star in the halo subset.
# Here we use a simple estimator of energy: E = 0.5*v^2 + Phi(R, z), where Phi is the potential.
# We'll compute v^2 from the Galactocentric velocities.
# Note: A more precise energy computation would involve integrating orbits in the full potential.

# Calculate total energy in physical units
v_total_sq = vX**2 + vY**2 + vZ**2

# Convert positions to natural units for potential calculation
R_nat = np.sqrt(X**2 + Y**2) / ro
z_nat = Z / ro

# Evaluate potential and convert to physical units
phi_vals = evaluatePotentials(total_potential, R_nat, z_nat) * vo**2

# Total energy = KE + PE
E = 0.5 * v_total_sq + phi_vals

# Compute Phi(R,z) in the total potential. We use the galpy potential objects.
# galpy potentials have a method 'Phi' that expects inputs in natural units.
# Convert positions to natural units.
from galpy.potential import evaluatePotentials

x_nat = X / ro
y_nat = Y / ro
z_nat_full = Z / ro

# Compute cylindrical radius for the potential evaluation.
R_cyl = np.sqrt(x_nat**2 + y_nat**2)

# Evaluate potential (in vo^2 units) and convert back.
phi_vals = evaluatePotentials(total_potential, R_cyl, z_nat_full) * vo**2

# Compute total energy in physical units (km^2/s^2)
E = 0.5 * v_total_sq + phi_vals

# For the halo subset, extract E and Lz (from our computed actions)
# We assume that the LZ we computed is equivalent to the z-component of angular momentum.
mask_halo = final_halo_sel  # boolean mask for halo_subset in selected_stars order
E_halo = E[mask_halo]
Lz_halo = selected_stars["LZ"][mask_halo].data  # in kpc km/s
feh_halo = halo_subset["FEH"].data

# Create 2D bins for E and Lz.
nbins = 50
E_bins = np.linspace(np.nanmin(E_halo), np.nanmax(E_halo), nbins)
Lz_bins = np.linspace(np.nanmin(Lz_halo), np.nanmax(Lz_halo), nbins)

# Create a 2D grid to hold median [FEH] per bin.
med_feh = np.full((nbins - 1, nbins - 1), np.nan)

# Loop over bins and compute the median [FEH] for stars in each bin.
for i in range(nbins - 1):
    for j in range(nbins - 1):
        in_bin = (
            (E_halo >= E_bins[i])
            & (E_halo < E_bins[i + 1])
            & (Lz_halo >= Lz_bins[j])
            & (Lz_halo < Lz_bins[j + 1])
        )
        if np.sum(in_bin) > 0:
            med_feh[i, j] = np.median(feh_halo[in_bin])

# Plot the 2D map
plt.figure(figsize=(10, 8))
# Use extent to map bins onto the plot
extent = [Lz_bins[0], Lz_bins[-1], E_bins[0], E_bins[-1]]
im = plt.imshow(med_feh, origin="lower", extent=extent, aspect="auto", cmap="viridis")
plt.xlabel("Lz (kpc km/s)")
plt.ylabel("Orbital Energy E (km²/s²)")
plt.title("Fig. 6 (Left): Median [FEH] in E vs. Lz bins for Halo Stars")
cbar = plt.colorbar(im)
cbar.set_label("Median [FEH]")
plt.tight_layout()
plt.show()

# -------------------------------
# 10. (Optional) Print the fraction of metal‐poor stars ([FEH] < -1.0) in the halo subset.
# -------------------------------
mp_all = selected_stars["FEH"] < -1.0
mp_halo = halo_subset["FEH"] < -1.0
if np.sum(mp_all) > 0:
    frac_mp = np.sum(mp_halo) / np.sum(mp_all) * 100.0
    print(
        "\nFraction of metal-poor ([FEH] < -1.0) stars in the halo subset: %.1f%%"
        % frac_mp
    )
else:
    print("\nNo metal-poor stars found in the sample; cannot compute fraction.")
