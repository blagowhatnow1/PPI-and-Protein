#This code is experimental
#Might need further review 
#Hybrid Potts-Like Model
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from Bio.Align import PairwiseAligner
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import substitution_matrices
import openmm
from openmm import app, unit
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#==== STRUCTURE LOADING ====

def load_pdb_structure(pdb_file):
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    pdb = app.PDBFile(pdb_file)
    forcefield = app.ForceField('amber99sb.xml', 'amber99_obc.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff)
    return modeller, system


# ==== ENERGY CALCULATION ====

def calculate_distance(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)


def extract_nonbonded_force(system):
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            return force
    raise ValueError("NonbondedForce not found in the system.")


def calculate_atom_interaction(index1, index2, positions, nonbonded_force, cutoff=0.1):
    q1, sigma1, eps1 = [x.value_in_unit(y) for x, y in zip(nonbonded_force.getParticleParameters(index1), 
                                                            [unit.elementary_charge, unit.nanometer, unit.kilojoule_per_mole])]
    q2, sigma2, eps2 = [x.value_in_unit(y) for x, y in zip(nonbonded_force.getParticleParameters(index2), 
                                                            [unit.elementary_charge, unit.nanometer, unit.kilojoule_per_mole])]

    pos1, pos2 = positions[index1], positions[index2]
    r = calculate_distance(pos1, pos2)
    if r < cutoff:
        return 0, 0

    # Lennard-Jones
    epsilon = np.sqrt(eps1 * eps2)
    sigma = 0.5 * (sigma1 + sigma2)
    lj = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

    # Coulombic energy
    k_e = 8.9875517923e9
    q1_coulomb = q1 * 1.602e-19
    q2_coulomb = q2 * 1.602e-19
    coulomb_joule = k_e * q1_coulomb * q2_coulomb / (r * 1e-9)
    coulomb_kjmol = (coulomb_joule / 1000) * 6.022e23

    return lj, coulomb_kjmol

def calculate_residue_interaction(modeller, system):
    integrator = openmm.VerletIntegrator(0.002 * unit.picoseconds)
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()

    nb_force = extract_nonbonded_force(system)
    residues = list(modeller.topology.residues())
    positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
    n = len(residues)
    interaction_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            energy = 0
            for atom1 in residues[i].atoms():
                for atom2 in residues[j].atoms():
                    lj, coul = calculate_atom_interaction(atom1.index, atom2.index, positions, nb_force)
                    energy += lj + coul
            interaction_matrix[i, j] = interaction_matrix[j, i] = energy

    return interaction_matrix


# ==== ALIGNMENT + SCORING ====

def msa_to_sequence_index_map(aligned_seq):
    """
    For a gapped aligned sequence, returns a list where:
    - msa_index_map[i] = original residue index
    - Or None if that position is a gap
    """
    seq_index = 0
    msa_index_map = []
    for aa in aligned_seq:
        if aa == '-':
            msa_index_map.append(None)
        else:
            msa_index_map.append(seq_index)
            seq_index += 1
    return msa_index_map

def align_sequences(sequences):
    aligner = PairwiseAligner()
    aligner.mode = 'global'

    # Start with the first sequence as the initial alignment base
    msa = [sequences[0]]

    for i in range(1, len(sequences)):
        aligned = aligner.align(msa[0], sequences[i])[0]

        aligned_seq1 = str(aligned[0])
        aligned_seq2 = str(aligned[1])

        # Update the reference alignment with gaps if needed
        updated_msa = []
        ref_index = 0
        new_index = 0

        while ref_index < len(aligned_seq1) and new_index < len(aligned_seq2):
            a1 = aligned_seq1[ref_index]
            a2 = aligned_seq2[new_index]

            for j in range(len(msa)):
                if len(updated_msa) <= j:
                    updated_msa.append("")

                if a1 == '-':
                    updated_msa[j] += '-'
                else:
                    updated_msa[j] += msa[j][0]
                    msa[j] = msa[j][1:]

            ref_index += 1
            new_index += 1

        msa = updated_msa + [aligned_seq2]

    # Convert to SeqRecord for compatibility
    msa_records = [SeqRecord(Seq(seq), id=f"seq_{i}") for i, seq in enumerate(msa)]

    # Alignment score matrix (optional here)
    matrix = np.zeros((len(msa_records), len(msa_records)))

    return msa_records, matrix

def build_substitution_matrix():
    raw = substitution_matrices.load('BLOSUM62')
    matrix = {}
    for (a1, a2), score in raw.items():
        matrix[(a1, a2)] = matrix[(a2, a1)] = score
    return matrix

def calculate_hybrid_energy(seq1, seq2, mat1, mat2, substitution_matrix=None, alpha=1.0, beta=1.0):
    seq1, seq2 = str(seq1), str(seq2)
    assert len(seq1) == len(seq2), "Aligned sequences must be equal length."
    L = len(seq1)

    map1 = msa_to_sequence_index_map(seq1)
    map2 = msa_to_sequence_index_map(seq2)

    evo_energy_sum = 0.0
    phys_energy_sum = 0.0
    valid_pairs = 0

    for i in range(L):
        for j in range(i + 1, L):
            if '-' in (seq1[i], seq2[i], seq1[j], seq2[j]):
                continue

            # Mapping from MSA to original residue indices
            idx1_i = map1[i]
            idx1_j = map1[j]
            idx2_i = map2[i]
            idx2_j = map2[j]

            if None in (idx1_i, idx1_j, idx2_i, idx2_j):
                continue

            valid_pairs += 1

            # Evolutionary score
            if substitution_matrix:
                evo_score_1 = substitution_matrix.get((seq1[i], seq2[i]), 0)
                evo_score_2 = substitution_matrix.get((seq1[j], seq2[j]), 0)
                evo_energy_sum += 0.5 * (evo_score_1 + evo_score_2)

            # Physical energy
            try:
                phys_energy_sum += mat1[idx1_i, idx1_j] + mat2[idx2_i, idx2_j]
            except IndexError:
                continue  # Skip if out-of-bounds (likely due to bad map)

    if valid_pairs == 0:
        raise ValueError("No valid residue pairs found.")

    evo_energy_avg = evo_energy_sum / valid_pairs
    phys_energy_avg = phys_energy_sum / valid_pairs

    phys_energy_avg /= 10.0  # optional scaling

    total_energy = alpha * (-evo_energy_avg) + beta * phys_energy_avg
    return total_energy

# ==== MAIN WORKFLOW ====

def calculate_msa_fitness(msa, pdb_files, alignment_matrix, alpha=1.0, beta=1.0, max_workers=4):
    fitness = {}
    interaction_matrices = {}
    for pdb in pdb_files:
        if pdb not in interaction_matrices:
            modeller, system = load_pdb_structure(pdb)
            interaction_matrices[pdb] = calculate_residue_interaction(modeller, system)

    substitution_matrix = build_substitution_matrix()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i in range(len(msa)):
            for j in range(i + 1, len(msa)):
                seq1, seq2 = msa[i].seq, msa[j].seq
                m1, m2 = interaction_matrices[pdb_files[i]], interaction_matrices[pdb_files[j]]
                futures[(i, j)] = executor.submit(
                    calculate_hybrid_energy, seq1, seq2, m1, m2, substitution_matrix, alpha, beta
                )

        for (i, j), fut in futures.items():
            try:
                fitness[(i, j)] = fut.result(timeout=600)
            except Exception as e:
                print(f"Error for pair ({i}, {j}): {e}")

    return fitness

def plot_relative_hybrid_energies(fitness, ref_index=0, n_structures=4):
    relative_energies = []
    labels = []
    for i in range(n_structures):
        if i == ref_index:
            continue
        key = (min(ref_index, i), max(ref_index, i))
        energy = fitness.get(key)
        if energy is not None:
            relative_energies.append(energy)
            labels.append(f"{ref_index} vs {i}")

    # Use a colormap to assign unique colors
    cmap = cm.get_cmap('tab10', len(relative_energies))
    colors = [cmap(i) for i in range(len(relative_energies))]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, relative_energies, color=colors, width=0.5)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.ylabel("Hybrid Energy (lower = more stable/compatible)")
    plt.title(f"Relative Hybrid Energy vs. Reference Structure {ref_index}")
    plt.grid(axis='y', linestyle='--', linewidth=0.5)

    # Add value labels
    for bar, energy in zip(bars, relative_energies):
        plt.text(bar.get_x() + bar.get_width() / 2, energy + 0.05,
                 f"{energy:.3f}", ha='center', va='bottom', fontsize=9)

    # Adjust y-axis to provide visual separation between bars
    y_margin = 0.3
    min_y = min(relative_energies)
    max_y = max(relative_energies)
    plt.ylim(min_y - y_margin, max_y + y_margin)

    plt.tight_layout()
    plt.show()

def process_sequences_and_pdbs(sequences, pdb_files, alpha=1.0, beta=1.0):
    msa, align_scores = align_sequences(sequences)
    fitness = calculate_msa_fitness(msa, pdb_files, align_scores, alpha, beta)
    
    print("Hybrid Fitness Energies:")
    for (i, j), energy in fitness.items():
        print(f"Pair ({i}, {j}) → Energy: {energy:.3f}")
    
    # Plot hybrid energy relative to reference (structure 0)
    plot_relative_hybrid_energies(fitness, ref_index=0, n_structures=len(sequences))

# ==== USAGE EXAMPLE ====

if __name__ == "__main__":
    sequences = [
        "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK",
        "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKIGKKGGGELASK",
        "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFIGGIGLASK", 
        "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFLMQAAASKA"
        ]
    pdb_files = ["m1.pdb", "m2.pdb", "m3.pdb","m4.pdb"]
    process_sequences_and_pdbs(sequences, pdb_files)
