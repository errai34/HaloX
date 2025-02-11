import astropy.units as u
import gala.dynamics as gd
import gala.potential as gp
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import Galactocentric, SkyCoord
from astropy.io import fits
from astropy.table import Table
from gala.dynamics.actionangle import get_staeckel_fudge_delta
from galpy.actionAngle import actionAngleStaeckel

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
# 3. Compute distances and apply heliocentric distance cut (< 5 kpc)
# -------------------------------
parallax_arcsec = selected_stars["PARALLAX"].astype(float) * 1e-3
valid_parallax = parallax_arcsec > 0
selected_stars = selected_stars[valid_parallax]
dist_kpc = 1.0 / (parallax_arcsec[valid_parallax] * 1000.0)
selected_stars["DIST_KPC"] = dist_kpc

selected_stars = selected_stars[selected_stars["DIST_KPC"] <= 5]

print("\nCatalog mean values (full sample after distance cut):")
print("Mean Distance [kpc]:", np.mean(selected_stars["DIST_KPC"]))
print("Mean VRAD [km/s]:", np.mean(selected_stars["VRAD"]))
print("Mean PMRA [mas/yr]:", np.mean(selected_stars["PMRA"]))
print("Mean PMDEC [mas/yr]:", np.mean(selected_stars["PMDEC"]))

mask_pm = (np.abs(selected_stars["PMRA"]) < 200) & (
    np.abs(selected_stars["PMDEC"]) < 200
)
selected_stars = selected_stars[mask_pm]
print("\nAfter applying proper-motion cut (|PMRA|<200, |PMDEC|<200):")
print("Mean PMRA [mas/yr]:", np.mean(selected_stars["PMRA"]))
print("Mean PMDEC [mas/yr]:", np.mean(selected_stars["PMDEC"]))

N_sample = len(selected_stars)
if len(selected_stars) > N_sample:
    np.random.seed(42)
    indices = np.random.choice(
        np.arange(len(selected_stars)), size=N_sample, replace=False
    )
    selected_stars = selected_stars[indices]
print(
    "\nNumber of stars after distance & PM cuts and random sampling:",
    len(selected_stars),
)

total_pm = np.sqrt(selected_stars["PMRA"] ** 2 + selected_stars["PMDEC"] ** 2)
total_vel = np.sqrt(total_pm**2 + selected_stars["VRAD"] ** 2)
print("\nRandom sample velocity statistics (mean values):")
print(f"Mean Radial velocity: {np.mean(selected_stars['VRAD']):.1f} km/s")
print(f"Mean Proper motion RA: {np.mean(selected_stars['PMRA']):.1f} mas/yr")
print(f"Mean Proper motion Dec: {np.mean(selected_stars['PMDEC']):.1f} mas/yr")

print("\nUsing all {} stars for orbit integration.".format(len(selected_stars)))

# -------------------------------
# 4. Build a SkyCoord object
# -------------------------------
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
print("\nSkyCoord object created:")
print("Number of stars:", len(coords))
print("Sample coordinate (star 0):", coords[0])

# -------------------------------
# 5. Transform to a Galactocentric frame and compute orbital circularity
# -------------------------------
galcen_frame = Galactocentric(
    galcen_distance=8.178 * u.kpc,
    z_sun=20.8 * u.pc,
    roll=0 * u.deg,
    galcen_v_sun=[11.1, 250.24, 7.25] * u.km / u.s,
)
galcen = coords.transform_to(galcen_frame)
print("\nTransformed to Galactocentric frame.")
print("Sample Galactocentric coordinate (star 0):", galcen[0])

X = galcen.x.to(u.kpc).value
Y = galcen.y.to(u.kpc).value
Z = galcen.z.to(u.kpc).value
vX = galcen.v_x.to(u.km / u.s).value
vY = galcen.v_y.to(u.km / u.s).value
vZ = galcen.v_z.to(u.km / u.s).value

print("\nSample positions (kpc):")
print("X[0] =", X[0], "Y[0] =", Y[0], "Z[0] =", Z[0])
print("Sample velocities (km/s):")
print("vX[0] =", vX[0], "vY[0] =", vY[0], "vZ[0] =", vZ[0])

R = np.sqrt(X**2 + Y**2)
phi = np.arctan2(Y, X)
vT = (-Y * vX + X * vY) / R
v_phi = -vT  # so that prograde orbits have v_phi > 0.
Lz = R * v_phi

potential = gp.MilkyWayPotential()
pos_for_vcirc = np.vstack([R, np.zeros_like(R), np.zeros_like(R)]) * u.kpc
v_circ = potential.circular_velocity(pos_for_vcirc).to(u.km / u.s).value
Lz_circ = R * v_circ
epsilon = Lz / Lz_circ

print("\nCylindrical coordinate mean values:")
print("Mean R [kpc]:", np.mean(R))
print("Mean v_phi [km/s]:", np.mean(v_phi))
print("Mean epsilon (circularity):", np.mean(epsilon))
print("Note: For halo stars we expect epsilon <= 0.75.")

mask_halo = epsilon <= 0.75
print("Stars before halo cut:", len(R), "after halo cut:", np.sum(mask_halo))
print("First 10 epsilon values:", epsilon[:10])

if np.sum(mask_halo) == 0:
    raise ValueError(
        "No halo stars remain after applying the circularity cut. Relax threshold for testing."
    )

X, Y, Z = X[mask_halo], Y[mask_halo], Z[mask_halo]
vX, vY, vZ = vX[mask_halo], vY[mask_halo], vZ[mask_halo]
print("\nAfter halo cut, number of stars:", len(X))

# Attach units to positions and velocities for the PhaseSpacePosition
X = np.atleast_1d(X)
Y = np.atleast_1d(Y)
Z = np.atleast_1d(Z)
vX = np.atleast_1d(vX)
vY = np.atleast_1d(vY)
vZ = np.atleast_1d(vZ)

pos = np.vstack([X, Y, Z]) * u.kpc
vel = np.vstack([vX, vY, vZ]) * (u.km / u.s)
w0 = gd.PhaseSpacePosition(pos=pos, vel=vel)
print("Phase space position shape:", w0.pos.shape)
if w0.pos.ndim == 1:
    print("First star's position (with units):", w0.pos)
else:
    print("First star's position (with units):", w0.pos[:, 0])

# -------------------------------
# 6. Integrate orbits using the potential
# -------------------------------
H = gp.Hamiltonian(potential)
orbit = H.integrate_orbit(w0, dt=0.5 * u.Myr, n_steps=2000)
print("Orbit shape:", orbit.shape)
print("Sample orbit for first star (first 5 time steps):")
for t in range(5):
    print(
        f"t = {t}: x = {orbit[t, 0].x:.2f}, y = {orbit[t, 0].y:.2f}, z = {orbit[t, 0].z:.2f}"
    )

# -------------------------------
# 7. Compute orbital energy and actions using Galpy's Stäckel Fudge
# -------------------------------
energies = H.energy(w0)
energies_kms2 = energies.to(u.km**2 / u.s**2)
print("\nOrbital energies (km²/s²):")
for i, E in enumerate(energies_kms2):
    print(f"Star {i}: E = {E:.2f}")

# Instead of using the O2GF method, we use Galpy's Stäckel Fudge.
# Import the necessary functions from gala and galpy:

# Convert our potential to a galpy potential:
galpy_potential = potential.as_interop("galpy")

Nstars = orbit.shape[1]
J = np.zeros((3, Nstars))
Omega = np.zeros((3, Nstars))
for n in range(Nstars):
    # Convert the integrated orbit for star n to a galpy orbit:
    o = orbit[:, n].to_galpy_orbit()
    # Compute the appropriate focal length delta:
    delta = get_staeckel_fudge_delta(potential, orbit[:, n])
    # Create the Galpy Stäckel Fudge action-angle object:
    staeckel = actionAngleStaeckel(pot=galpy_potential, delta=delta)
    # Compute actions, frequencies, and angles along the orbit:
    af = staeckel.actionsFreqs(o)
    # The returned af is a tuple of arrays (actions, angles, freqs); we take the mean over time:
    af_mean = np.array([np.mean(arr) for arr in af])
    J[:, n] = af_mean[:3]
    Omega[:, n] = af_mean[3:]

print("\nActions computed using Galpy's Stäckel Fudge:")
print("Shape of actions array:", J.shape)
for n in range(Nstars):
    print(f"Star {n}: J = {J[:, n]}, Omega = {Omega[:, n]}")

# -------------------------------
# 8. Plot the integrated orbits (Galactocentric x-y plane)
# -------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
for i in range(orbit.shape[1]):
    x = orbit[:, i].x.to(u.kpc).value
    y = orbit[:, i].y.to(u.kpc).value
    if i < 5:
        ax.plot(x, y, lw=1.5, alpha=0.8, label=f"Star {i}")
    else:
        ax.plot(x, y, lw=1.5, alpha=0.8)
ax.set_xlabel("x [kpc]")
ax.set_ylabel("y [kpc]")
ax.set_title("Integrated Orbits of Selected Halo Stars")
ax.legend()
plt.tight_layout()
plt.show()
