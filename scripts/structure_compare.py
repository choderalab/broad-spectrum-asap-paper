import MDAnalysis as mda
import numpy as np
from scipy.spatial.distance import cdist
from Bio import pairwise2
from scripts.lddt import lddt
from pathlib import Path
from asapdiscovery.spectrum.calculate_rmsd import rmsd_alignment
from asapdiscovery.spectrum.blast import pdb_to_seq


def convert_chain_id(chain):
    """Convert a chain identifier between letter and number representations."""
    if chain.isalpha():
        return ord(chain.lower()) - 96
    elif chain.isdigit():
        return chr(int(chain) + 96)
    return chain  # Return as is if neither


def get_residue_mapping(seq_ref, seq_mob):
    """Aligns two sequences and returns the correct start and end residue indices,
    ignoring gaps."""
    alignments = pairwise2.align.globalxx(seq_ref, seq_mob)
    aligned_ref, aligned_mob = alignments[0][:2]  # Extract first alignment

    ref_idx, mob_idx = [], []
    r, m = 0, 0  # Residue counters (no gaps)

    for a, b in zip(aligned_ref, aligned_mob):
        ref_idx.append(r if a != "-" else None)
        mob_idx.append(m if b != "-" else None)
        r += a != "-"
        m += b != "-"

    # Find first and last matched residues
    start_idx = next(
        i
        for i in range(len(ref_idx))
        if ref_idx[i] is not None and mob_idx[i] is not None
    )
    end_idx = next(
        i
        for i in range(len(ref_idx) - 1, -1, -1)
        if ref_idx[i] is not None and mob_idx[i] is not None
    )

    return start_idx + 1, end_idx + 1  # 1-based indexing


def find_bsite_resids(
    pdb,
    pdb_ref,
    aligned_temp,
    ligres="UNK",
    chain_m="A",
    chain_r="A",
    bsite_dist=4.5,
    mode="ligand",
    res_threshold=5,
):
    """Find binding site residues in a protein-ligand complex based on ligand proximity."""
    rmsd, pdb_aln = rmsd_alignment(pdb, pdb_ref, aligned_temp, chain_m, chain_r)
    u = mda.Universe(pdb_aln)
    u_ref = mda.Universe(pdb_ref)

    # Initial atom selections
    bs_atoms = u_ref.select_atoms(
        f"protein and chainid {chain_r} and around {bsite_dist} resname {ligres}"
    )
    lig_atoms = u_ref.select_atoms(f"chainid {chain_r} and resname {ligres}")
    ca_mob = u.select_atoms(f"protein and chainid {chain_m} and name CA").atoms

    # Handle incorrect chain selections
    if len(ca_mob) == 0:
        print("The mobile chain is incorrect, attempting to fix")
        chain_m = convert_chain_id(chain_m)
        ca_mob = u.select_atoms(f"protein and chainid {chain_m} and name CA").atoms

    if len(bs_atoms) == 0:
        print("The reference chain is incorrect, attempting to fix")
        chain_r = convert_chain_id(chain_r)
        bs_atoms = u_ref.select_atoms(
            f"protein and chainid {chain_r} and around {bsite_dist} resname {ligres}"
        )
        lig_atoms = u_ref.select_atoms(f"chainid {chain_r} and resname {ligres}")

    # Define reference positions based on mode
    if mode == "bsite":
        ref_pos = bs_atoms.select_atoms("name CA").positions
    elif mode == "ligand":
        ref_pos = lig_atoms.positions
    else:
        raise NotImplementedError("Only 'bsite' and 'ligand' modes are allowed.")

    bs_ref = np.unique(bs_atoms.resids)
    bs_ref = bs_ref[
        bs_ref >= res_threshold
    ]  # In case terminal residue is categorized as binding site
    n_res = len(bs_ref)

    if len(bs_ref) == 0:
        raise ValueError(f"No binding site residues have an idx above {res_threshold}")

    distances = cdist(ca_mob.positions, ref_pos, metric="euclidean")

    sorted_flat = np.argsort(distances.ravel())
    rows, _ = np.unravel_index(sorted_flat, distances.shape)

    res_seen = set()
    bs_mob = [
        ca_mob.resids[r]
        for r in rows
        if (ca_mob.resids[r] not in res_seen and not res_seen.add(ca_mob.resids[r]))
    ]

    return np.sort(bs_mob[:n_res]), bs_ref


def get_binding_site_rmsd(
    file_mob,
    file_ref,
    bsite_dist=4.5,
    rmsd_mode="CA",
    chain_mob="1",
    chain_ref="1",
    ligres="UNK",
    chain_ref2="A",
    lig_ref_pdb=None,
    aligned_temp=None,
):
    """Calculate RMSD of the binding site between two PDB structures."""
    u = mda.Universe(file_mob).select_atoms(
        f"protein and segid {chain_mob} and not resname ACE and not resname NME"
    )
    u_ref = mda.Universe(file_ref).select_atoms(
        f"protein and segid {chain_ref} and not resname ACE and not resname NME"
    )

    u_ref_l = mda.Universe(file_ref).select_atoms(
        f"(protein and segid {chain_ref} and not resname ACE and not resname NME) or resname {ligres}"
    )
    u_lig = u_ref_l.select_atoms(f"resname {ligres}")
    bs_atoms = u_ref_l.select_atoms(f"protein and around {bsite_dist} resname {ligres}")

    # Handle reference with no ligand
    if len(u_lig) == 0:
        if lig_ref_pdb and aligned_temp:
            bs_ids, __ = find_bsite_resids(
                file_mob,
                lig_ref_pdb,
                aligned_temp,
                ligres,
                chain_mob,
                chain_ref2,
                bsite_dist,
                mode="ligand",
                res_threshold=5,
            )
            bs_atoms = u_ref_l.select_atoms(" or ".join([f"resid {r}" for r in bs_ids]))
        else:
            raise ValueError(
                f"No ligand found in ref with resname {ligres}. Provide a correct ligand name or a second reference PDB."
            )

    # Align sequences to ensure residue numbering is consistent
    seq_mob = pdb_to_seq(file_mob, chain=str(chain_mob)).seq.replace("X", "")
    seq_ref = pdb_to_seq(file_ref, chain=str(chain_ref)).seq.replace("X", "")

    start_resid, end_resid = get_residue_mapping(seq_ref, seq_mob)
    u_ref = u_ref.select_atoms(f"resid {start_resid}:{end_resid}")
    u_ref_l = u_ref_l.select_atoms(
        f"resid {start_resid}:{end_resid} or resname {ligres}"
    )
    u = u.select_atoms(f"resid {start_resid}:{end_resid}")

    # Select binding site residues
    binding_site = [
        r.resid - start_resid + 1 for r in bs_atoms.residues if "CA" in r.atoms.names
    ]
    binding_site_n = [
        str(r.resname) for r in bs_atoms.residues if "CA" in r.atoms.names
    ]

    binding_site_m = []
    binding_site_r = []
    for i, r in enumerate(binding_site):
        res = u.residues[r - 1]
        if binding_site_n[i] == res.resname:
            binding_site_m.append(res.resid)
            binding_site_r.append(r)
        else:
            print(f"{binding_site_n[i]} != {res.resname}")

    sel_bs = " or ".join(f"resid {r}" for r in binding_site_r)
    sel_bs_m = " or ".join(f"resid {r}" for r in binding_site_m)

    if len(sel_bs_m) > 0:
        u_bs_m = u.select_atoms(sel_bs_m)
    else:
        alignment = pairwise2.align.globalms(seq_mob, seq_ref, 2, -1, -0.8, -0.5)[0]
        print(
            "The sequences may be different! \n", pairwise2.format_alignment(*alignment)
        )
        return -1

    u_bs_ref = u_ref.select_atoms(sel_bs)

    rmsd_sel = "name CA" if rmsd_mode == "CA" else "not name H*"
    m_pos, ref_pos = u_bs_m.select_atoms(rmsd_sel), u_bs_ref.select_atoms(rmsd_sel)

    # Match common atoms per residue
    mpos_list, refpos_list = [], []
    for mob_res, ref_res in zip(m_pos.residues, ref_pos.residues):
        common_atoms = set(mob_res.atoms.names) & set(ref_res.atoms.names)
        mob_common, ref_common = mob_res.atoms.select_atoms(
            f"name {' or name '.join(common_atoms)}"
        ), ref_res.atoms.select_atoms(f"name {' or name '.join(common_atoms)}")
        if len(mob_common) == len(ref_common):
            mpos_list.append(mob_common.positions)
            refpos_list.append(ref_common.positions)

    try:
        rmsd = np.sqrt(
            ((np.vstack(mpos_list) - np.vstack(refpos_list)) ** 2).sum(-1).mean()
        )
    except ValueError:
        print(f"Error: Mismatched lengths ({len(m_pos)} vs {len(ref_pos)})")
        rmsd = -1

    return rmsd


def compare_pairwise_alignment(align, matches, original):
    result = []
    index = -1
    for char, match in zip(align, matches):
        if char == "-":
            index += 0
        else:
            index += 1
        if match == "|":
            assert original[index] == char
            result.append(index)

    return result


def find_matching_residues(seq_pred, seq_ref, print_alignment=True):
    """Find the residues matching from the two seqs, excluding gaps"""
    alignments = pairwise2.align.globalxx(seq_pred, seq_ref)
    format_alignment = pairwise2.format_alignment(*alignments[0])
    if print_alignment:
        print(format_alignment)
    align_pred, matches, align_ref, score = format_alignment.splitlines()
    match_idx_pred = compare_pairwise_alignment(align_pred, matches, seq_pred)
    match_idx_ref = compare_pairwise_alignment(align_ref, matches, seq_ref)

    return match_idx_pred, match_idx_ref


def calculate_pddt(pred_protein, ref_protein, print_alignment=False):
    seq_pred = pdb_to_seq(pdb_input=pred_protein, chain="A").seq.replace("X", "")
    seq_ref = pdb_to_seq(pdb_input=ref_protein, chain="A").seq.replace("X", "")
    pred_mask, ref_mask = find_matching_residues(
        seq_pred, seq_ref, print_alignment=print_alignment
    )

    ref_atoms = mda.Universe(ref_protein).select_atoms(
        "protein and chainID A and not resname NME and not resname ACE"
    )
    ref_coords = np.array(
        [res.atoms.select_atoms("name CA").positions[0] for res in ref_atoms.residues]
    )
    ref_coords = ref_coords[ref_mask]
    ref_coords = ref_coords[None, :, :]

    pred_atoms = mda.Universe(pred_protein).select_atoms(
        "protein and not resname NME and not resname ACE"
    )
    pred_coords = np.array(
        [res.atoms.select_atoms("name CA").positions[0] for res in pred_atoms.residues]
    )

    pred_coords = pred_coords[pred_mask]
    pred_coords = pred_coords[None, :, :]

    true_points = np.ones((1, pred_coords.shape[1], 1))
    p_res_lddt = lddt(
        pred_coords, ref_coords, true_points_mask=true_points, per_residue=True
    )
    return p_res_lddt[0], ref_mask

def get_bsite(pdb_label, pdb_ref, pdb_dir, chain_mob, chain_ref, lig_mob, lig_ref):
    from MDAnalysis.lib.util import convert_aa_code
    pdb_dir = Path(pdb_dir)
    pdb = list(pdb_dir.glob(f"{pdb_label}*.pdb"))
    if len(pdb) == 0:
        print(f"PDB wasn't found for label {pdb_label}")
        return None, None
    u_ref = mda.Universe(pdb_ref)
    u = mda.Universe(pdb[0])
    res_ref = u_ref.select_atoms(f"protein and chainid {chain_ref} and not name H* and around 4.5 resname {lig_ref} and not resname ACE and not resname NME and not resname NMA").residues
    res_mob = u.select_atoms(f"protein and chainid {chain_mob} and not name H* and around 4.5 resname {lig_mob} and not resname ACE and not resname NME and not resname NMA").residues
    bs_ref = []
    bs_mob = []
    ref_len , mob_len = len(res_ref), len(res_mob)
    seq_len = max(ref_len, mob_len)
    for i in range(seq_len):
        ref = convert_aa_code(res_ref[i].resname) if i < ref_len else '-'
        mob = convert_aa_code(res_mob[i].resname) if i < mob_len else '-'
        bs_ref.append(ref)
        bs_mob.append(mob)
    return bs_ref, bs_mob

def calculate_bsite_score(aligned_sequences, 
                          pdb_ref, 
                          pdb_dir,  
                          chain_mob='A', 
                          chain_ref='A', 
                          lig_mob='LIG', 
                          lig_ref='LIG', 
                          ref_idx=0,):
    from Bio import pairwise2
    num_alignments = len(aligned_sequences)
    scores = []
    ids = []
    for s in range(num_alignments):
        id_comp = aligned_sequences[s].id.split("|")[1].split(".")
        label = f"{id_comp[0]}_{id_comp[1]}"
        bs_ref, bs_mob = get_bsite(label, pdb_ref, pdb_dir, chain_mob, chain_ref, lig_mob, lig_ref)
        if bs_ref is None:
            continue
        seq_ref = "".join(bs_ref)
        seq_mob = "".join(bs_mob)
        align_score = pairwise2.align.globalms(seq_ref, seq_mob, 2, -1, -1, -.5, score_only=True,)   
        if s == ref_idx:
            max_score = align_score
        id_score = (align_score / max_score) * 100 
        scores.append(id_score)
        ids.append(label)

    return ids, scores