# -*- coding: utf-8 -*-
"""

Analysis of energy partitioning during hypervelocity impact simulations. 
--------------------------------------Din & Codex----------------------

Expected files from simulation
------------------------------
1. energy_partition.dat
2. projectile_energy.dump
3. species_evolution.txt

Outputs
-------
- energy_analysis_processed.csv
- projectile_com_energy.csv
- species_processed.csv
- 01_projectile_kinetic_energy_loss.png
- 02_target_energy_changes.png
- 03_reaxff_energy_components.png
- 04_species_production.png
- 05_energy_changes_normalized_by_projectile_KE.png

Interpretation
--------------
This script deliberately does NOT label one ReaxFF energy term as a unique
"chemical dissipation energy." ReaxFF bond/angle/conjugation energies respond
to both mechanical distortion and reaction. Instead, it combines:

  mechanical/thermal indicators:
    projectile translational KE loss
    target KE change
    target PE change
    thermostat energy exchange

  chemical/reactive indicators:
    ReaxFF bond-energy change
    ReaxFF conjugation-energy change
    ReaxFF Coulomb-energy change
    species production (e.g. CO and CO2)

Changes are referenced to REFERENCE_STEP (or to the first available sample).
"""

from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

ENERGY_FILE = Path("energy_partition.dat")
PROJECTILE_DUMP = Path("projectile_energy.dump")
SPECIES_FILE = Path("species_evolution.txt")
OUT_DIR = Path("energy_analysis_results")
OUT_DIR.mkdir(exist_ok=True)

# LAMMPS timestep from the input file, in fs.
TIMESTEP_FS = 0.1

# If None, the first sample is used as the reference.
# For IMPACT-ONLY analysis, set this to the timestep immediately before
# first projectile-target contact.
#REFERENCE_STEP = None
REFERENCE_STEP = 127000

# Species to show if present in the ReaxFF species output.
SPECIES_TO_PLOT = ["CO", "CO2", "O2"]

# Output/plot energy unit.
# Supported: "kcal/mol", "eV", "keV"
ENERGY_UNIT = "eV"

# LAMMPS "real" velocity is Angstrom/fs and mass is g/mol.
# KE[kcal/mol] = REAL_KE_FACTOR * sum(m_i * v_i^2)
# where REAL_KE_FACTOR includes the 1/2 factor.
REAL_KE_FACTOR = 0.5 * 1.0e-3 * (1.0e5 ** 2) / 4184.0

KCAL_MOL_TO_EV = 0.0433641153087705


# ============================================================
# PLOTTING DEFAULTS
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 300


# ============================================================
# UNIT HELPERS
# ============================================================

def energy_factor():
    u = ENERGY_UNIT.lower()
    if u == "kcal/mol":
        return 1.0
    if u == "ev":
        return KCAL_MOL_TO_EV
    if u == "kev":
        return KCAL_MOL_TO_EV / 1000.0
    raise ValueError("ENERGY_UNIT must be 'kcal/mol', 'eV', or 'keV'.")


def eunit_label():
    if ENERGY_UNIT.lower() == "kcal/mol":
        return "kcal/mol"
    if ENERGY_UNIT.lower() == "ev":
        return "eV"
    return "keV"


EFAC = energy_factor()


# ============================================================
# ENERGY HISTORY
# ============================================================

ENERGY_COLUMNS = [
    "Step",
    "Time_fs",
    "KE_SiO",
    "KE_Target",
    "PE_SiO",
    "PE_Target",
    "Ebond",
    "Eatom",
    "Elp",
    "Emol",
    "Eval",
    "Epen",
    "Ecoa",
    "Ehbond",
    "Etorsion",
    "Econj",
    "Evdw",
    "Ecoul",
    "Efield",
    "Eqeq",
    "ThermostatExchange",
    "PE_All",
    "KE_All",
    "Etot_All",
]


def read_energy_history(path):
    rows = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            parts = s.split()

            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue

            if len(vals) != len(ENERGY_COLUMNS):
                warnings.warn(
                    f"Skipping energy row with {len(vals)} columns; "
                    f"expected {len(ENERGY_COLUMNS)}."
                )
                continue

            rows.append(vals)

    if not rows:
        raise RuntimeError(
            f"No valid numerical data found in {path}. "
            "Check that the modified LAMMPS input completed far enough "
            "to write energy_partition.dat."
        )

    df = pd.DataFrame(rows, columns=ENERGY_COLUMNS)
    df["Step"] = df["Step"].astype(int)
    df["Time_ps"] = df["Time_fs"] / 1000.0
    return df


# ============================================================
# PROJECTILE DUMP PARSER
# ============================================================

def iter_lammps_custom_dump(path):
    """
    Minimal parser for a standard LAMMPS dump custom trajectory.
    Yields (timestep, dataframe).
    """
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        while True:
            line = fh.readline()
            if not line:
                break

            if not line.startswith("ITEM: TIMESTEP"):
                continue

            step = int(fh.readline().strip())

            line = fh.readline()
            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise RuntimeError("Unexpected dump format after TIMESTEP.")
            natoms = int(fh.readline().strip())

            line = fh.readline()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError("Unexpected dump format before BOX BOUNDS.")

            # Three box-bound lines.
            fh.readline()
            fh.readline()
            fh.readline()

            header = fh.readline().strip()
            if not header.startswith("ITEM: ATOMS"):
                raise RuntimeError("Unexpected dump format before ATOMS data.")

            cols = header.split()[2:]

            data = []
            for _ in range(natoms):
                vals = fh.readline().split()
                data.append(vals)

            frame = pd.DataFrame(data, columns=cols)

            for col in cols:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

            yield step, frame


def projectile_energy_history(path):
    records = []

    for step, frame in iter_lammps_custom_dump(path):
        needed = {"mass", "vx", "vy", "vz"}
        missing = needed - set(frame.columns)
        if missing:
            raise RuntimeError(
                f"Projectile dump is missing columns: {sorted(missing)}"
            )

        m = frame["mass"].to_numpy(float)
        vx = frame["vx"].to_numpy(float)
        vy = frame["vy"].to_numpy(float)
        vz = frame["vz"].to_numpy(float)

        mtot = np.sum(m)

        vcmx = np.sum(m * vx) / mtot
        vcmy = np.sum(m * vy) / mtot
        vcmz = np.sum(m * vz) / mtot
        vcm2 = vcmx**2 + vcmy**2 + vcmz**2

        # Translational kinetic energy of the projectile COM.
        ke_trans = REAL_KE_FACTOR * mtot * vcm2

        # If c_KEATOM exists, sum it for total projectile kinetic energy.
        if "c_KEATOM" in frame.columns:
            ke_total = frame["c_KEATOM"].sum()
            ke_internal = ke_total - ke_trans
        else:
            ke_total = np.nan
            ke_internal = np.nan

        # COM position is useful for identifying contact.
        xcm = np.sum(m * frame["x"].to_numpy(float)) / mtot
        ycm = np.sum(m * frame["y"].to_numpy(float)) / mtot
        zcm = np.sum(m * frame["z"].to_numpy(float)) / mtot

        records.append(
            {
                "Step": int(step),
                "Time_ps": step * TIMESTEP_FS / 1000.0,
                "Mass_total_g_per_mol": mtot,
                "Xcm_A": xcm,
                "Ycm_A": ycm,
                "Zcm_A": zcm,
                "Vcm_x_A_per_fs": vcmx,
                "Vcm_y_A_per_fs": vcmy,
                "Vcm_z_A_per_fs": vcmz,
                "Vcm_A_per_fs": np.sqrt(vcm2),
                "KE_trans_kcal_per_mol": ke_trans,
                "KE_total_kcal_per_mol": ke_total,
                "KE_internal_kcal_per_mol": ke_internal,
            }
        )

    if not records:
        raise RuntimeError(f"No projectile frames found in {path}.")

    return pd.DataFrame(records)


# ============================================================
# REAXFF SPECIES PARSER
# ============================================================

def read_reaxff_species(path):
    records = []
    current_species = None

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue

            if s.startswith("#"):
                toks = s.lstrip("#").split()

                # Find a header containing Timestep and species names.
                lower = [t.lower() for t in toks]
                if "timestep" in lower and len(toks) >= 3:
                    # Standard first 3 fields are Timestep No_Moles No_Specs.
                    current_species = toks[3:]
                continue

            if current_species is None:
                continue

            parts = s.split()
            try:
                numeric = [float(x) for x in parts]
            except ValueError:
                continue

            if len(numeric) < 3:
                continue

            step = int(numeric[0])
            no_moles = numeric[1]
            no_specs = numeric[2]
            counts = numeric[3:]

            rec = {
                "Step": step,
                "Time_ps": step * TIMESTEP_FS / 1000.0,
                "No_Moles": no_moles,
                "No_Specs": no_specs,
            }

            for formula, count in zip(current_species, counts):
                rec[formula] = count

            records.append(rec)

    if not records:
        warnings.warn(
            f"No species records could be parsed from {path}. "
            "The energy plots can still be generated."
        )
        return pd.DataFrame()

    # Species headers can change with time. concat/normalization gives missing
    # species a count of zero.
    df = pd.DataFrame(records).fillna(0.0)
    df = df.sort_values("Step").reset_index(drop=True)

    # Multiple blocks at same timestep are not expected, but combine if present.
    numeric_cols = [c for c in df.columns if c not in ["Step", "Time_ps"]]
    if df["Step"].duplicated().any():
        df = df.groupby("Step", as_index=False)[numeric_cols].sum()
        df["Time_ps"] = df["Step"] * TIMESTEP_FS / 1000.0

    return df


# ============================================================
# REFERENCE / CHANGES
# ============================================================

def choose_reference_step(energy_df):
    if REFERENCE_STEP is None:
        return int(energy_df["Step"].iloc[0])

    available = energy_df["Step"].to_numpy()
    idx = int(np.argmin(np.abs(available - REFERENCE_STEP)))
    chosen = int(available[idx])

    if chosen != REFERENCE_STEP:
        warnings.warn(
            f"REFERENCE_STEP={REFERENCE_STEP} not present; "
            f"using nearest energy sample {chosen}."
        )

    return chosen


def add_energy_changes(df, reference_step):
    out = df.copy()
    ref = out.loc[out["Step"] == reference_step].iloc[0]

    energy_cols = [
        "KE_SiO", "KE_Target", "PE_SiO", "PE_Target",
        "Ebond", "Eatom", "Elp", "Emol", "Eval", "Epen", "Ecoa",
        "Ehbond", "Etorsion", "Econj", "Evdw", "Ecoul", "Efield", "Eqeq",
        "ThermostatExchange", "PE_All", "KE_All", "Etot_All",
    ]

    for col in energy_cols:
        out["d_" + col] = out[col] - ref[col]

    return out


# ============================================================
# PLOTS
# ============================================================

def save_line_plot(x, series, xlabel, ylabel, filename, zero_line=False):
    """
    series: list of (label, y-array)
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for label, y in series:
        ax.plot(x, y, linewidth=1.6, label=label)

    if zero_line:
        ax.axhline(0.0, linewidth=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    #ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=600, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    for required in [ENERGY_FILE, PROJECTILE_DUMP]:
        if not required.exists():
            raise FileNotFoundError(
                f"{required} was not found. Run the modified LAMMPS input first."
            )

    energy = read_energy_history(ENERGY_FILE)
    projectile = projectile_energy_history(PROJECTILE_DUMP)

    species = (
        read_reaxff_species(SPECIES_FILE)
        if SPECIES_FILE.exists()
        else pd.DataFrame()
    )

    ref_step = choose_reference_step(energy)
    energy = add_energy_changes(energy, ref_step)

    # Reference projectile sample nearest to reference step.
    pidx = np.argmin(np.abs(projectile["Step"].to_numpy() - ref_step))
    p_ref = projectile.iloc[pidx]
    projectile_ref_step = int(p_ref["Step"])
    Eproj0 = float(p_ref["KE_trans_kcal_per_mol"])

    projectile["Projectile_KE_loss_kcal_per_mol"] = (
        Eproj0 - projectile["KE_trans_kcal_per_mol"]
    )

    # Converted columns for easy plotting.
    projectile["KE_trans_plot"] = projectile["KE_trans_kcal_per_mol"] * EFAC
    projectile["KE_loss_plot"] = (
        projectile["Projectile_KE_loss_kcal_per_mol"] * EFAC
    )
    projectile["KE_internal_plot"] = (
        projectile["KE_internal_kcal_per_mol"] * EFAC
    )

    # Save processed tables.
    energy.to_csv(OUT_DIR / "energy_analysis_processed.csv", index=False)
    projectile.to_csv(OUT_DIR / "projectile_com_energy.csv", index=False)

    if not species.empty:
        species.to_csv(OUT_DIR / "species_processed.csv", index=False)

    # ---------------- Figure 1: projectile translational KE ----------------
    save_line_plot(
        projectile["Time_ps"],
        [
            ("Projectile translational KE", projectile["KE_trans_plot"]),
            ("Cumulative translational KE loss", projectile["KE_loss_plot"]),
        ],
        "Time (ps)",
        f"Energy ({eunit_label()})",
        "01_projectile_kinetic_energy_loss.png",
    )

    # ---------------- Figure 2: target KE and PE changes ----------------
    save_line_plot(
        energy["Time_ps"],
        [
            ("Target ΔKE", energy["d_KE_Target"] * EFAC),
            ("Target ΔPE", energy["d_PE_Target"] * EFAC),
            (
                "Thermostat cumulative exchange",
                energy["d_ThermostatExchange"] * EFAC,
            ),
        ],
        "Time (ps)",
        f"Energy change relative to step {ref_step} ({eunit_label()})",
        "02_target_energy_changes.png",
        zero_line=True,
    )

    # ---------------- Figure 3: ReaxFF components ----------------
    save_line_plot(
        energy["Time_ps"],
        [
            ("Δ bond energy", energy["d_Ebond"] * EFAC),
            ("Δ conjugation energy", energy["d_Econj"] * EFAC),
            ("Δ Coulomb energy", energy["d_Ecoul"] * EFAC),
            ("Δ valence-angle energy", energy["d_Eval"] * EFAC),
        ],
        "Time (ps)",
        f"ReaxFF energy change ({eunit_label()})",
        "03_reaxff_energy_components.png",
        zero_line=True,
    )

    # ---------------- Figure 4: species production ----------------
    if not species.empty:
        available = [s for s in SPECIES_TO_PLOT if s in species.columns]

        if not available:
            metadata = {"Step", "Time_ps", "No_Moles", "No_Specs"}
            candidates = [c for c in species.columns if c not in metadata]

            # Choose species with the largest count variation.
            variation = {
                c: float(species[c].max() - species[c].min())
                for c in candidates
            }
            available = [
                k for k, _ in sorted(
                    variation.items(),
                    key=lambda kv: kv[1],
                    reverse=True
                )[:5]
                if variation[k] > 0
            ]

            warnings.warn(
                "None of SPECIES_TO_PLOT were found. "
                f"Plotting the most variable species instead: {available}"
            )

        if available:
            series = []

            # Plot production relative to each species count at/nearest reference.
            sidx = np.argmin(
                np.abs(species["Step"].to_numpy() - ref_step)
            )

            for sp in available:
                baseline = species.iloc[sidx][sp]
                series.append(
                    (f"Δ {sp}", species[sp] - baseline)
                )

            save_line_plot(
                species["Time_ps"],
                series,
                "Time (ps)",
                "Change in number of molecules",
                "04_species_production.png",
                zero_line=True,
            )

    # ---------------- Figure 5: normalized comparison ----------------
    # This is a comparison of indicators, NOT an exact additive partition.
    if abs(Eproj0) > 1e-20:
        save_line_plot(
            energy["Time_ps"],
            [
                (
                    "Target ΔKE / KEp,0",
                    energy["d_KE_Target"] / Eproj0,
                ),
                (
                    "Target ΔPE / KEp,0",
                    energy["d_PE_Target"] / Eproj0,
                ),
                (
                    "Δ Reax bond / KEp,0",
                    energy["d_Ebond"] / Eproj0,
                ),
                (
                    "Δ Reax conjugation / KEp,0",
                    energy["d_Econj"] / Eproj0,
                ),
                (
                    "Δ Reax Coulomb / KEp,0",
                    energy["d_Ecoul"] / Eproj0,
                ),
                (
                    "Thermostat exchange / KEp,0",
                    energy["d_ThermostatExchange"] / Eproj0,
                ),
            ],
            "Time (ps)",
            "Energy change / reference projectile translational KE",
            "05_energy_changes_normalized_by_projectile_KE.png",
            zero_line=True,
        )

    print("=" * 72)
    print("Energy/species post-processing complete")
    print("=" * 72)
    print(f"Reference energy step      : {ref_step}")
    print(f"Nearest projectile step    : {projectile_ref_step}")
    print(
        "Reference projectile translational KE: "
        f"{Eproj0 * EFAC:.6g} {eunit_label()}"
    )
    print()
    print("Generated directory:", OUT_DIR.resolve())
    print("  energy_analysis_processed.csv")
    print("  projectile_com_energy.csv")
    if not species.empty:
        print("  species_processed.csv")
    print("  01_projectile_kinetic_energy_loss.png")
    print("  02_target_energy_changes.png")
    print("  03_reaxff_energy_components.png")
    if not species.empty:
        print("  04_species_production.png")
    print("  05_energy_changes_normalized_by_projectile_KE.png")
    print()
    print(
        "Important: Figure 5 is a normalized comparison of coupled energy "
        "indicators, not a mathematically unique mechanical-vs-chemical "
        "energy partition."
    )


if __name__ == "__main__":
    main()
