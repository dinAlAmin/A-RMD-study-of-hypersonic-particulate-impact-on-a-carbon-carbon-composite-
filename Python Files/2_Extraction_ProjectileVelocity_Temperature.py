# -*- coding: utf-8 -*-
'''

Extraction of projectile velocity and temperature from a LAMMPS dump file at a specific frame.
--------------------------------------------------Din----------------------------------------

'''

from pathlib import Path
import math
import numpy as np

DUMP_FILE = Path("projectile_energy.dump")
FRAME_NUMBER = 10         # 1-based
TIMESTEP_FS = 0.1         # from LAMMPS input
PRINT_ATOMS = False

R_KCAL_PER_MOL_K = 0.00198720425864083
REAL_KE_FACTOR = 0.5 * 1.0e-3 * (1.0e5 ** 2) / 4184.0

def read_requested_frame(filename, frame_number):
    if frame_number < 1:
        raise ValueError("FRAME_NUMBER must be >= 1.")
    current = 0
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        while True:
            line = f.readline()
            if not line:
                raise IndexError(f"Frame {frame_number} not found; only {current} frame(s) read.")
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            current += 1
            timestep = int(f.readline().strip())

            if not f.readline().startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError("Unexpected dump format.")
            n_atoms = int(f.readline().strip())

            box_header = f.readline().strip()
            if not box_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Unexpected dump format: BOX BOUNDS missing.")
            f.readline(); f.readline(); f.readline()

            atom_header = f.readline().strip()
            if not atom_header.startswith("ITEM: ATOMS"):
                raise ValueError("Unexpected dump format: ATOMS header missing.")
            cols = atom_header.split()[2:]

            rows = []
            for _ in range(n_atoms):
                vals = f.readline().split()
                rows.append(vals)

            if current == frame_number:
                return timestep, cols, rows

def idx(cols, *names):
    for n in names:
        if n in cols:
            return cols.index(n)
    raise KeyError(f"None of {names} found. Available columns: {cols}")

def main():
    timestep, cols, rows = read_requested_frame(DUMP_FILE, FRAME_NUMBER)

    i_id = idx(cols, "id")
    i_mass = idx(cols, "mass")
    i_vx = idx(cols, "vx")
    i_vy = idx(cols, "vy")
    i_vz = idx(cols, "vz")

    ke_col = None
    for name in ("c_KEATOM", "c_keatom", "ke"):
        if name in cols:
            ke_col = cols.index(name)
            break

    ids = np.array([int(r[i_id]) for r in rows])
    m = np.array([float(r[i_mass]) for r in rows])
    vx = np.array([float(r[i_vx]) for r in rows])
    vy = np.array([float(r[i_vy]) for r in rows])
    vz = np.array([float(r[i_vz]) for r in rows])

    M = m.sum()
    vcm_x = np.sum(m*vx)/M
    vcm_y = np.sum(m*vy)/M
    vcm_z = np.sum(m*vz)/M
    vcm_mag = math.sqrt(vcm_x**2 + vcm_y**2 + vcm_z**2)

    if ke_col is not None:
        ke_total = sum(float(r[ke_col]) for r in rows)
        ke_source = cols[ke_col]
    else:
        v2 = vx**2 + vy**2 + vz**2
        ke_total = REAL_KE_FACTOR * np.sum(m*v2)
        ke_source = "mass/velocity calculation"

    ke_com = REAL_KE_FACTOR * M * (vcm_mag**2)
    ke_internal = ke_total - ke_com
    if ke_internal < 0 and abs(ke_internal) < 1e-8*max(abs(ke_total),1.0):
        ke_internal = 0.0

    N = len(rows)
    dof = 3*N - 3
    temperature_K = 2.0*ke_internal/(dof*R_KCAL_PER_MOL_K)

    time_ps = timestep*TIMESTEP_FS/1000.0

    print("="*68)
    print("PROJECTILE STATE AT REQUESTED FRAME")
    print("="*68)
    print(f"Frame number (1-based) : {FRAME_NUMBER}")
    print(f"LAMMPS timestep        : {timestep}")
    print(f"Physical time          : {time_ps:.6f} ps")
    print(f"Projectile atoms       : {N}")
    print()
    print("Center-of-mass velocity")
    print(f"  Vx   = {vcm_x:.8e} A/fs = {vcm_x*100:.6f} km/s")
    print(f"  Vy   = {vcm_y:.8e} A/fs = {vcm_y*100:.6f} km/s")
    print(f"  Vz   = {vcm_z:.8e} A/fs = {vcm_z*100:.6f} km/s")
    print(f"  |V|  = {vcm_mag:.8e} A/fs = {vcm_mag*100:.6f} km/s")
    print()
    print("Heating / internal motion")
    print(f"  Total KE             = {ke_total:.6f} kcal/mol ({ke_source})")
    print(f"  COM translational KE = {ke_com:.6f} kcal/mol")
    print(f"  Internal KE          = {ke_internal:.6f} kcal/mol")
    print(f"  Temperature          = {temperature_K:.3f} K")
    print("="*68)

    if PRINT_ATOMS:
        for atom_id, mass, ax, ay, az in zip(ids, m, vx, vy, vz):
            print(atom_id, mass, ax, ay, az)

if __name__ == "__main__":
    main()
