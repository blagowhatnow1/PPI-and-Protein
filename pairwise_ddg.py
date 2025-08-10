# heuristic_ddg_pipeline.py. Under review. Not meant to be a rigorous calculation replacing FEP. 
import os
import json
import hashlib
import logging
from datetime import datetime

import numpy as np
import openmm
from openmm import app, unit

# Constants (precise)
R_KJ_PER_MOL_K = 0.00831446261815324  # kJ / (mol K)
# Note: Boltzmann constant per molecule is 1.380649e-23 J/K; we work in per-mol units.

# ===== Utility Functions =====

# Configure module-level logger (safe if module is imported)
logger = logging.getLogger("heuristic_ddg")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(ch)


def setup_file_logger(logfile="results/logs/run.log"):
    """Attach a file logger in addition to the console logger."""
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    fh = logging.FileHandler(logfile, mode="a")
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(fh)
    return logger


def _short_hash(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def calculate_entropy_from_fluctuations(energy_values, temperature, tref=298.15):
    """
    Estimate entropy change relative to tref using crude fluctuation formula.

    Parameters
    ----------
    energy_values : sequence of openmm.unit.Quantity or floats
        Energies expected in kJ/mol unit; if Quantity, convert inside.
    temperature : openmm.unit.Quantity (Kelvin)
    tref : float, Kelvin
        Reference temperature for relative entropy: S(T) - S(Tref) = C_v * ln(T/Tref).

    Returns
    -------
    S_rel : openmm.unit.Quantity (kJ/mol/K)
        Estimated entropy difference S(T) - S(Tref). Warning: crude.
    C_v : float (kJ/mol/K)
        Estimated heat capacity used.
    """
    T = float(temperature.value_in_unit(unit.kelvin))
    # Convert energies to floats in kJ/mol
    E_kj = np.array(
        [float(e.value_in_unit(unit.kilojoule_per_mole)) if hasattr(e, "value_in_unit") else float(e)
         for e in energy_values],
        dtype=float,
    )

    if E_kj.size < 3:
        raise ValueError("Need at least 3 energy samples to estimate fluctuations.")

    # Use unbiased variance (ddof=1)
    var_E = np.var(E_kj, ddof=1)  # (kJ/mol)^2

    # Use R (kJ/mol/K) because energies are per-mol
    C_v = var_E / (R_KJ_PER_MOL_K * T**2)  # kJ/mol/K

    # Return entropy difference relative to tref (assume approx constant C_v)
    S_rel = C_v * np.log(T / tref)  # kJ/mol/K (dimensionless ln times C_v)

    # Attach units only at the end to keep numeric math simple
    return S_rel * unit.kilojoule_per_mole / unit.kelvin, C_v


def run_md(
    modeller,
    system,
    temperature,
    equil_steps,
    md_steps,
    interval,
    save_dcd=False,
    dcd_filename=None,
    seed=None,
    platform_name=None,
):
    """
    Run MD and return time series of potential, kinetic, and total energies (as quantities).
    - temperature: openmm.unit.Quantity (Kelvin)
    - interval: sample every `interval` steps
    - seed: optional integer for reproducibility (passed to setVelocitiesToTemperature)
    """
    # Integrator and simulation
    integrator = openmm.LangevinIntegrator(temperature, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)
    if platform_name:
        platform = openmm.Platform.getPlatformByName(platform_name)
        simulation = app.Simulation(modeller.topology, system, integrator, platform)
    else:
        simulation = app.Simulation(modeller.topology, system, integrator)

    simulation.context.setPositions(modeller.positions)
    # deterministically set velocities if seed provided
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)
    simulation.context.setVelocitiesToTemperature(temperature, int(seed))

    # Equilibration
    logger.info("Starting equilibration...")
    simulation.step(int(equil_steps))
    logger.info("Equilibration complete. Starting production MD...")

    # Reporters: if saving DCD, add reporter
    if save_dcd:
        if not dcd_filename:
            raise ValueError("dcd_filename must be provided when save_dcd=True")
        simulation.reporters.append(app.DCDReporter(dcd_filename, int(interval)))
    # Also add a StateDataReporter to console/file if desired (optional)
    # simulation.reporters.append(app.StateDataReporter(stream, interval, step=True, potentialEnergy=True, temperature=True))

    potential_ts = []
    kinetic_ts = []
    total_ts = []

    # Sample energies every `interval` steps in a loop to allow safe getState calls
    n_steps = int(md_steps)
    step = 0
    while step < n_steps:
        step_block = min(int(interval), n_steps - step)
        simulation.step(step_block)
        step += step_block

        state = simulation.context.getState(getEnergy=True, getKineticEnergy=True)
        epot = state.getPotentialEnergy()       # Quantity (kJ/mol)
        ekin = state.getKineticEnergy()         # Quantity (kJ/mol)
        etot = epot + ekin

        potential_ts.append(epot)
        kinetic_ts.append(ekin)
        total_ts.append(etot)

    return {
        "potential": potential_ts,
        "kinetic": kinetic_ts,
        "total": total_ts,
        "n_samples": len(total_ts),
        "seed": int(seed),
    }


def approximate_relative_free_energy(energy_timeseries, temperature, tref=298.15):
    """
    Compute F ≈ <U> - T * S_rel where S_rel = S(T) - S(tref) is estimated via fluctuations.
    Inputs:
        energy_timeseries: dict from run_md (with 'total' list)
        temperature: openmm.unit.Quantity (Kelvin)
    Returns:
        F : openmm.unit.Quantity (kJ/mol)
        stderr_F : float (kJ/mol) approximate standard error of mean of F (from energy sem)
        metadata dict (C_v, n_samples, seed)
    Notes:
        This is a heuristic approximation — see code comments for caveats.
    """
    total = energy_timeseries["total"]
    n = len(total)
    if n < 3:
        raise ValueError("Need at least 3 total-energy samples to estimate free energy.")

    # Convert to floats in kJ/mol
    total_kj = np.array([float(e.value_in_unit(unit.kilojoule_per_mole)) for e in total], dtype=float)
    avg_U = float(np.mean(total_kj))  # kJ/mol
    sem_U = float(np.std(total_kj, ddof=1) / np.sqrt(n))  # approximate standard error (no decorrelation correction)

    # Estimate entropy difference and heat capacity from fluctuations
    S_rel_q, C_v = calculate_entropy_from_fluctuations(total, temperature, tref=tref)
    # S_rel_q is in kJ/mol/K as a Quantity
    S_rel = float(S_rel_q.value_in_unit(unit.kilojoule_per_mole / unit.kelvin))  # numeric

    T = float(temperature.value_in_unit(unit.kelvin))
    F_val = avg_U - T * S_rel  # kJ/mol (float)

    # propagated standard error: mainly from avg_U; uncertainty from entropy is nontrivial and neglected here
    stderr_F = sem_U

    return F_val * unit.kilojoule_per_mole, stderr_F, {"C_v": C_v, "n_samples": n, "seed": energy_timeseries.get("seed", None)}


def calculate_ddg(free_energy_dict, sequences):
    """
    Compute pairwise ΔΔG. free_energy_dict maps full sequence -> (F_value_float_kJ, stderr)
    Returns dict mapping "seqA -> seqB" -> {"ddg": float, "stderr": float}
    """
    ddgs = {}
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            a, b = sequences[i], sequences[j]
            Fa, se_a = free_energy_dict[a]["F"], free_energy_dict[a]["stderr"]
            Fb, se_b = free_energy_dict[b]["F"], free_energy_dict[b]["stderr"]
            ddg = Fb - Fa
            ddg_stderr = np.sqrt(se_a**2 + se_b**2)
            ddgs[f"{a}->{b}"] = {"ddg": ddg, "stderr": ddg_stderr}
    return ddgs


def process_sequences(sequences, pdb_paths, config):
    """
    Top-level runner. Returns ddg dict.
    """
    if len(sequences) != len(pdb_paths):
        raise ValueError("sequences and pdb_paths must be the same length.")

    T = config["temperature"] * unit.kelvin
    results_dir = config.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    free_energies = {}
    for i, seq in enumerate(sequences):
        short = seq[:10]
        logger.info(f"\nProcessing sequence {short} (index {i})")

        if not os.path.exists(pdb_paths[i]):
            logger.error("PDB not found: %s", pdb_paths[i])
            raise FileNotFoundError(pdb_paths[i])

        modeller, system = load_structure(pdb_paths[i], padding=config.get("padding_nm", 1.0))
        dcd_out = os.path.join(results_dir, f"{short}_{_short_hash(seq)}.dcd") if config.get("save_dcd", False) else None

        energies = run_md(
            modeller=modeller,
            system=system,
            temperature=T,
            equil_steps=config["equil_steps"],
            md_steps=config["md_steps"],
            interval=config["sample_interval"],
            save_dcd=config.get("save_dcd", False),
            dcd_filename=dcd_out,
            seed=config.get("random_seed", None),
            platform_name=config.get("platform", None),
        )

        F_q, stderr_F, meta = approximate_relative_free_energy(energies, T)
        # note: above uses the new signature -- we pass the whole energies dict
        free_energies[seq] = {"F": float(F_q.value_in_unit(unit.kilojoule_per_mole)), "stderr": stderr_F, "meta": meta}

        logger.info(f"Estimated F for {short}: {free_energies[seq]['F']:.2f} ± {free_energies[seq]['stderr']:.2f} kJ/mol")
    ddgs = calculate_ddg(free_energies, sequences)

    # write results
    outpath = os.path.join(results_dir, "ddg_results.json")
    with open(outpath, "w") as fh:
        json.dump(ddgs, fh, indent=2)

    logger.info("Finished. Results written to %s", outpath)
    return ddgs


# Keep your load_structure implementation (slightly modified to accept padding)
def load_structure(pdb_file, padding=1.0):
    """Loads and solvates a PDB structure and returns modeller and system objects."""
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB not found: {pdb_file}")

    pdb = app.PDBFile(pdb_file)
    forcefield = app.ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(forcefield, model='tip3p', padding=padding * unit.nanometer)
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds
    )
    return modeller, system

# ===== Main Entrypoint =====

# remove old `log` wrapper (delete the following)
# def log(msg):
#     global logger
#     logger(msg)

if __name__ == "__main__":
    # attach file logger in addition to console logger
    setup_file_logger()  # adds file handler to module-level `logger`

    # Example input
    sequences = [
        "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK",
        "PIAQIHIGRGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK",
        "PIAAHHIGRGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK"
    ]

    pdb_files = [
        "p_prepared1.pdb",
        "p_prepared2.pdb",
        "p_prepared3.pdb"
    ]

    config = {
        "temperature": 310,         # in K
        "equil_steps": 5000,
        "md_steps": 50000,
        "sample_interval": 500,
        "save_dcd": True,
        "results_dir": "results",
        "random_seed": 42,
        # "platform": "CUDA"  # set if you want specific platform
    }

    ddg_results = process_sequences(sequences, pdb_files, config)
    logger.info("ΔΔG run complete. Results: %s", ddg_results)

