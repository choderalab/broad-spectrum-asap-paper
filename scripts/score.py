from asapdiscovery.data.schema.complex import Complex, PreppedComplex
from asapdiscovery.docking.docking import DockingInputPair
from asapdiscovery.docking.scorer import MetaScorer

from asapdiscovery.docking.openeye import POSITDocker

from asapdiscovery.data.backend.openeye import (
    load_openeye_pdb,
    save_openeye_sdf,
)
from asapdiscovery.modeling.modeling import split_openeye_mol
from asapdiscovery.data.backend.openeye import oechem
from asapdiscovery.spectrum.calculate_rmsd import rmsd_alignment
from asapdiscovery.simulation.simulate import VanillaMDSimulator

import os
from rdkit import Chem
from typing import Union
import pandas as pd
from pathlib import Path
import subprocess


def dock_and_score(
    pdb_complex,
    comp_name,
    target_name,
    scorers,
    label,
    pdb_ref=None,
    aligned_folder=None,
    allow_clashes=True,
    align_chain="A",
    align_chain_ref="A",
):
    """Re-dock ligand in a complex and return pose scores

    Parameters
    ----------
    pdb_complex : Union[Path, str]
        PDB path for protein-ligand complex to score
    comp_name : str
        Name of to give to complex. Can be arbitrary.
    target_name : str
        Name of reference target (see asapdiscovery documentation).
    scorers : List
        List with scorer objects. For ChemGauss use ChemGauss4Scorer().
    pdb_ref : Union[Path, str], optional
        PDB of reference structure that will be used to align pdb_complex, by default None
    aligned_folder : Path, optional
        Folder where aligned complex PDB will be save, only if pdb_ref is provided, by default None
    allow_clashes : bool, optional
       Dock allowing clashes on final pose, by default True
    align_chain : str, optional
        Chain by which , by default "A"
    align_chain : str, optional
        Chain in target to align with ref, by default "A"
    align_chain_ref : str, optional
        Chain in ref to align with target, by default "A"

    Returns
    -------
    tuple : (Pandas DataFrame, PreppedComplex, Ligand, Path)
        DataFrame with scores, Prepared complex, Ligand pose, Path to aligned PDB
    """
    if pdb_ref:
        pdb_complex, aligned = rmsd_alignment(
            pdb_complex,
            pdb_ref,
            aligned_folder / f"{label}_a.pdb",
            align_chain,
            align_chain_ref,
        )
    else:
        aligned = pdb_complex
        print(aligned)
    cmp = Complex.from_pdb(
        aligned,
        ligand_kwargs={"compound_name": comp_name},
        target_kwargs={"target_name": target_name},
    )
    try:
        prepped_cmp = PreppedComplex.from_complex(cmp)
        docker = POSITDocker(allow_final_clash=allow_clashes)
        input_pair = DockingInputPair(ligand=cmp.ligand, complex=prepped_cmp)
        results = docker.dock(inputs=[input_pair])
        ligand_pose = results[0].posed_ligand

        metascorer = MetaScorer(scorers=scorers)
        scores_df = metascorer.score(results, return_df=True)
    except Exception as e:
        scores_df = pd.DataFrame([None], columns=["docking-score-POSIT"])
        prepped_cmp = cmp
        ligand_pose = cmp.ligand
        print("!!!! Prep and dock unsuccessful", e)

    return scores_df, prepped_cmp, ligand_pose, aligned


def ligand_rmsd_oechem(
    refmol: oechem.OEGraphMol, fitmol: oechem.OEGraphMol, overlay=False
):
    """Helper function to calculate ligand RMSD with OEChem"""
    nConfs = 1
    _ = oechem.OEDoubleArray(nConfs)
    automorf = True
    heavyOnly = True
    rotmat = oechem.OEDoubleArray(9 * nConfs)
    transvec = oechem.OEDoubleArray(3 * nConfs)

    success = oechem.OERMSD(
        refmol, fitmol, automorf, heavyOnly, overlay, rotmat, transvec
    )
    if not success:
        print("RMSD calculation failed")
    return success


def ligand_rmsd_rdkit(target_sdf, ref_sdf):
    """Helper function to calculate ligand RMSD with RDKit"""
    target_sdf = str(target_sdf)
    ref_sdf = str(ref_sdf)
    with Chem.SDMolSupplier(target_sdf) as supp:
        mol_target = supp[0]
    with Chem.SDMolSupplier(ref_sdf) as supp:
        mol_ref = supp[0]
    try:
        rmsd = Chem.rdMolAlign.CalcRMS(mol_target, mol_ref)
    except Exception:
        try:
            rmsd = Chem.AllChem.AlignMol(mol_target, mol_ref)
        except Exception:
            rmsd = -1
    return rmsd


def get_ligand_rmsd(
    target_pdb: str,
    ref_pdb: str,
    addHs=True,
    pathT="",
    pathR="",
    rmsd_mode="oechem",
    overlay=False,
) -> float:
    """Calculate RMSD of a molecule against a reference

    Parameters
    ----------
    target_pdb : str
        Path to PDB of protein with ligand to align.
    ref_pdb : str
        Path to PDB to align target against.
    addHs : bool, optional
        Add explicit Hs with OEChem tools, by default True
    pathT : str, optional
        Temporary path to save the protein target pdb, as needed for rdkit rmsd mode, by default ""
    pathR : str, optional
        Temporary path to save the ligand sdf, as needed for rdkit rmsd mode, by default ""
    rmsd_mode : str, optional
        Tool to use for RMSD calculation between ["oechem", "rdkit"], by default "oechem"
    overlay : bool, optional
        Whether to overlay pose for RMSD, by default False

    Returns
    -------
    float
        RMSD after alignment

    Raises
    ------
    ValueError
        When pathT and pathR aren't provided in rdkit mode.
    NotImplementedError
        When incorrect rmsd_mode is provided.
    """
    target_complex = load_openeye_pdb(target_pdb)
    ref_complex = load_openeye_pdb(ref_pdb)

    target_dict = split_openeye_mol(target_complex)
    ref_dict = split_openeye_mol(ref_complex)

    # Add Hs
    target_lig = target_dict["lig"]
    ref_lig = ref_dict["lig"]
    if addHs:
        oechem.OEAddExplicitHydrogens(target_lig)
        oechem.OEAddExplicitHydrogens(ref_lig)
    else:
        oechem.OESuppressHydrogens(target_lig)
        oechem.OESuppressHydrogens(ref_lig)

    path_target = path_ref = ""
    if pathT and pathR:
        path_target = save_openeye_sdf(target_lig, pathT)
        path_ref = save_openeye_sdf(ref_lig, pathR)

    if ref_lig.NumAtoms() != target_lig.NumAtoms():
        print(f"Ref {ref_lig.NumAtoms()} and  target {target_lig.NumAtoms()}")

    if not (pathT and pathR) and (rmsd_mode == "rdkit" or rmsd_mode == "both"):
        raise ValueError(
            "for rdkit mode. a path to save/load sdf mols must be provided"
        )

    rmsd_oechem = ligand_rmsd_oechem(ref_lig, target_lig, overlay)
    if rmsd_mode == "oechem":
        return rmsd_oechem
    elif rmsd_mode == "rdkit":
        rmsd_rdkit = ligand_rmsd_rdkit(path_target, path_ref)
        return rmsd_rdkit
    elif rmsd_mode == "both":
        rmsd_rdkit = ligand_rmsd_rdkit(path_target, path_ref)
        return [rmsd_oechem, rmsd_rdkit]
    else:
        raise NotImplementedError("Provide a valid value for rmsd_mode")


def score_autodock_vina(
    receptor_pdb=Path,
    ligand_pdb=Path,
    box_center=None,
    box_size=[20, 20, 20],
    dock=False,
    path_to_prepare_file="./",
):
    """Score ligand pose with AutoDock Vina

    Parameters
    ----------
    receptor_pdb : Path
        Path to pdb of target (no ligand).
    ligand_pdb : Path
        Path to sdf of ligand.
    box_center : List, optional
        Center of ligand box as [x, y, z], by default None and the box will be calculated.
    box_size : list, optional
        Size of docking box, by default [20, 20, 20]
    dock : bool, optional
        Whether to redock ligand with AutoDock Vina, by default False
    path_to_prepare_file : str, optional
        Path to Python file which prepares ligand box if not provided (copied from AutoDock Vina repo), by default "./"

    Returns
    -------
    tuple: (pd.DataFrame, str)
        (DataFrame with scores, path to docked pose)

    Raises
    ------
    ValueError
        Path to target file is neither of pdb or pdbqt allowed formats
    """
    from vina import Vina

    df_scores = pd.DataFrame(index=[0])

    if Path(receptor_pdb).suffix == ".pdb":
        # Prepare receptor
        subprocess.run(
            f"prepare_receptor -r {receptor_pdb} -o {receptor_pdb}qt", shell=True
        )
    elif Path(receptor_pdb).suffix == ".pdbqt":
        receptor_pdb = Path(str(receptor_pdb)[:-2])
    else:
        raise ValueError("Only allowed formats are .pdb and .pdbqt")
    # Prepare ligand
    subprocess.run(
        f"mk_prepare_ligand.py -i {ligand_pdb} -o {str(ligand_pdb)[:-3]}pdbqt",
        shell=True,
    )
    v = Vina(sf_name="vina")

    # First check if prep was successful
    if (
        not Path(f"{receptor_pdb}qt").is_file()
        or not Path(f"{str(ligand_pdb)[:-3]}pdbqt").is_file()
    ):
        df_scores["Vina-score-premin"] = None
        df_scores["Vina-score-min"] = None
        if dock:
            df_scores["Vina-dock-score"] = None
        return df_scores, None

    # get coordinates of box
    if box_center is None:
        parent_dir = ligand_pdb.resolve().parents[0]
        p = subprocess.Popen(
            f"pythonsh {path_to_prepare_file}/prepare_gpf.py -l {ligand_pdb.stem}.pdbqt -r {receptor_pdb.stem}.pdbqt -y",
            cwd=parent_dir,
            shell=True,
            stdout=subprocess.PIPE,
        )
        (output, err) = p.communicate()
        # The grid needs some time to compute
        p.wait()
        with open(f"{parent_dir/receptor_pdb.stem}.gpf", "r") as f:
            for line in f:
                if line.startswith("gridcenter"):
                    # Split the line into columns
                    comps = line.split()
                    x = float(comps[1])
                    y = float(comps[2])
                    z = float(comps[3])
                    break
        box_center = [x, y, z]
    v.set_receptor(f"{receptor_pdb}qt")

    v.set_ligand_from_file(f"{str(ligand_pdb)[:-3]}pdbqt")
    v.compute_vina_maps(center=box_center, box_size=box_size)

    # Score the current pose
    energy = v.score()
    print("Score before minimization: %.3f (kcal/mol)" % energy[0])
    df_scores["Vina-score-premin"] = energy[0]

    # Minimized locally the current pose
    energy_minimized = v.optimize()
    print("Score after minimization : %.3f (kcal/mol)" % energy_minimized[0])
    df_scores["Vina-score-min"] = energy_minimized[0]
    v.write_pose(f"{str(receptor_pdb)[:-4]}_minimized.pdbqt", overwrite=True)
    out_pose = None

    if dock:
        # Dock the ligand
        v.dock(exhaustiveness=32, n_poses=20)
        v.write_poses(
            f"{str(receptor_pdb)[:-4]}_vina_out.pdbqt", n_poses=1, overwrite=True
        )
        df_scores["Vina-dock-score"] = v.score()[0]
        # Convert pose in pdbqt to calculate rmsd
        out_pose = f"{str(receptor_pdb)[:-4]}_vina_out.pdb"
        subprocess.run(
            f"babel -ipdbqt '{str(receptor_pdb)[:-4]}_vina_out.pdbqt' -opdb '{out_pose}'",
            shell=True,
        )
    return df_scores, out_pose


def score_gnina(pdb_target, sdf_ligand, pdb_dir, home_dir, gnina_script):
    logfile = f"out_{pdb_target[:-4]}.log"
    env = os.environ.copy()
    env["SDF"] = sdf_ligand
    env["PDB"] = pdb_target
    env["PDB_DIR"] = pdb_dir
    env["home_data"] = home_dir
    env["LOGFILE"] = logfile

    process = subprocess.Popen(
        ["bash", gnina_script],
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )
    # keep only the last line
    last_line = None
    for line in process.stdout:
        last_line = line.strip()
    process.wait()
    if last_line:
        data = [last_line.split(",")]
        df = pd.DataFrame(
            data,
            columns=[
                "gnina-RMSD",
                "gnina-Affinity",
                "gnina-Affinity-var",
                "CNNscore",
                "CNNaffinity",
                "CNNvariance",
            ],
        )
    else:
        df = pd.DataFrame(
            columns=[
                "gnina-RMSD",
                "gnina-Affinity",
                "gnina-Affinity-var",
                "CNNscore",
                "CNNaffinity",
                "CNNvariance",
            ]
        )
    return df


def minimize_structure(
    pdb_complex: Union[Path, str],
    min_out: Union[Path, str],
    out_dir: Union[Path, str],
    md_platform: str,
    comp_name: str,
    target_name: str,
) -> str:
    """MD energy minimization a protein ligand complex

    Parameters
    ----------
    pdb_complex : Union[Path, str]
        Path to protein ligand complex pdb
    min_out : Union[Path, str]
        Output file with minimized pdb
    out_dir : Union[Path, str]
        Directory to save output minimized complex
    md_platform : str
        MD OpenMM platform [CPU, CUDA, OpenCL, Reference, Fastest]
    comp_name : str
        Name of to give to complex. Can be arbitrary.
    target_name : str
        Name of reference target (see asapdiscovery documentation).

    Returns
    -------
    str
       Path to minimized file
    """

    if Path(min_out).is_file():
        print(f"the file {min_out} already exists")
        return min_out
    cmp = Complex.from_pdb(
        pdb_complex,
        ligand_kwargs={"compound_name": comp_name},
        target_kwargs={"target_name": target_name},
    )
    prepped_cmp = PreppedComplex.from_complex(cmp)
    prepped_cmp.target.to_pdb_file(out_dir / "target.pdb")
    cmp.ligand.to_sdf(out_dir / "ligand.sdf")

    md_simulator = VanillaMDSimulator(
        output_dir=out_dir,
        openmm_platform=md_platform,
        minimize_only=True,
        reporting_interval=1250,
        equilibration_steps=5000,
        num_steps=1,
    )
    simulation_results = md_simulator.simulate(
        [(out_dir / "target.pdb", out_dir / "ligand.sdf")],
        outpaths=[out_dir],
        failure_mode="skip",
    )
    min_path = simulation_results[0].minimized_pdb_path
    subprocess.run(f"mv {min_path} {min_out}", shell=True)
    subprocess.run(f"rm -r {out_dir}/target_ligand", shell=True)
    subprocess.run(f"rm {out_dir}/target.pdb", shell=True)
    subprocess.run(f"rm {out_dir}/ligand.sdf", shell=True)
    return min_out
