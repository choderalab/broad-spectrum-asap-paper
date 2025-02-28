from asapdiscovery.docking.scorer import ChemGauss4Scorer, MLModelScorer
from asapdiscovery.ml.models import ASAPMLModelRegistry
from asapdiscovery.data.schema.complex import Complex
from score import (
    dock_and_score,
    get_ligand_rmsd,
    score_autodock_vina,
    minimize_structure,
    score_gnina,
)
from structure_compare import get_binding_site_rmsd
import pandas as pd
from pathlib import Path
import re
import click

import logging


@click.command()
@click.option(
    "-d",
    "--docking-dir",
    type=str,
    default="./",
    help="Path to directory where docked structures are stored.",
)
@click.option(
    "-f",
    "--pdb_ref",
    type=str,
    default="./",
    help="Path to directory/file where crystal structures are stored.",
)
@click.option(
    "-o",
    "--out-csv",
    type=str,
    default="scores.csv",
    help="Path to file where scoring results will be stored.",
)
@click.option(
    "--docking-csv",
    type=str,
    default="",
    help="Path to csv files with docking results.",
)
@click.option(
    "--comp-name",
    type=str,
    default="test",
    help="Name of complex.",
)
@click.option(
    "--target-name",
    type=str,
    default="",
    help="Name of target.",
)
@click.option(
    "--vina-score",
    is_flag=True,
    default=False,
    help="Whether to run vina scoring.",
)
@click.option(
    "--vina-box-x",
    type=float,
    help="coordinate x of vina box.",
)
@click.option(
    "--vina-box-y",
    type=float,
    help="coordinate y of vina box.",
)
@click.option(
    "--vina-box-z",
    type=float,
    help="coordinate z of vina box.",
)
@click.option(
    "--docking-vina",
    is_flag=True,
    default=False,
    help="Whether to run docking on vina.",
)
@click.option(
    "--path-to-grid-prep",
    type=str,
    default="./",
    help="Path to file that calculated grid",
)
@click.option(
    "--ligand-regex",
    type=str,
    default="ASAP-[0-9]+",
    help="Pattern for extracting ligand ID from string.",
)
@click.option(
    "--protein-regex",
    type=str,
    default="YP_[0-9]+_[0-9]+|NP_[0-9]+_[0-9]+",
    help="Pattern for extracting ligand ID from string.",
)
@click.option(
    "--minimize",
    is_flag=True,
    default=False,
    help="Whether to minimize the pdb structures before running scoring.",
)
@click.option(
    "--ml-score",
    is_flag=True,
    default=False,
    help="Whether to employ asap-implemented ML models to score poses.",
)
@click.option(
    "--bsite-rmsd",
    is_flag=True,
    default=False,
    help="Whether to calculate binding site RMSD (only relevant when the ref pdb is the same target as the docked complex).",
)
@click.option(
    "--chain-dock",
    type=str,
    default="1",
    help="Chain ID of main chain in docked complex pdb(s).",
)
@click.option(
    "--chain-ref",
    type=str,
    default="A",
    help="Chain ID of main chain in reference pdb(s).",
)
@click.option(
    "--lig-resname",
    type=str,
    default="LIG",
    help="Residue name of ligand in reference pdb(s).",
)
@click.option(
    "--gnina-score",
    is_flag=True,
    default=False,
    help="Whether to run gnina scoring.",
)
@click.option(
    "--gnina-script",
    type=str,
    default="gnina_script.sh",
    help="Path to bash script that runs Gnina CLI.",
)
@click.option(
    "--home-dir",
    type=str,
    default="./",
    help="Path to directory to process gnina files. Gnina has problems with remote directories so location in $HOME is recommended.",
)
@click.option(
    "--log-level",
    type=str,
    default="INFO",
    help="Logger print mode. [info|debug|warning|error|critical]",
)
def score_complexes(
    docking_dir: str,
    pdb_ref: str,
    docking_csv: str,
    out_csv: str,
    comp_name: str,
    target_name: str,
    ligand_regex: str,
    protein_regex: str,
    chain_dock: str,
    chain_ref: str,
    lig_resname: str,
    vina_score=False,
    vina_box_x=None,
    vina_box_y=None,
    vina_box_z=None,
    docking_vina=False,
    path_to_grid_prep="./",
    minimize=False,
    ml_score=False,
    bsite_rmsd=False,
    gnina_score=False,
    gnina_script=None,
    home_dir=None,
    log_level="info",
):
    # Log level
    if log_level.lower() == "info":
        level = logging.INFO
    elif log_level.lower() == "debug":
        level = logging.DEBUG
    elif log_level.lower() == "warning":
        level = logging.WARNING
    elif log_level.lower() == "error":
        level = logging.ERROR
    else:
        level = logging.CRITICAL
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    path_ref = Path(pdb_ref)
    docking_dir = Path(docking_dir)

    all_scores = []

    if len(docking_csv):
        # Match the protein and ligand regex on docking output file
        logging.info("Reading docking CSV file: %s", docking_csv)
        docking_csv = Path(docking_csv)

        df_dock = pd.read_csv(docking_csv)
        df_dock = pd.read_csv(docking_csv)
        df_dock["lig-ID"] = df_dock["input"].apply(
            lambda s: (
                re.search(ligand_regex, s).group(0)
                if re.search(ligand_regex, s)
                else None
            )
        )
        df_dock["prot-ID"] = df_dock["input"].apply(
            lambda s: (
                re.search(protein_regex, s).group(0)
                if re.search(protein_regex, s)
                else None
            )
        )
        df_dock = df_dock[["lig-ID", "prot-ID", "docking-score-POSIT"]]

    logging.debug("Docking CSV file processed successfully.")
    logging.info("Starting scoring for docking directory %s", docking_dir.name)
    for file_min in docking_dir.glob("*.pdb"):
        # Extracting protein and ligand IDs and pre-calculated scores
        prot_id = re.search(protein_regex, file_min.stem)
        prot_id = prot_id.group(0) if prot_id else None

        ligand = re.search(ligand_regex, file_min.stem)
        ligand = ligand.group(0) if ligand else None

        # Default pre-min score if not found
        pre_min_score = 0

        if prot_id is not None:
            try:
                # 1 ligand, multiple targets
                pre_min_score = df_dock.loc[
                    df_dock["prot-ID"] == prot_id, "docking-score-POSIT"
                ].iloc[0]
            except (IndexError, KeyError):
                logging.warning("No pre-docking score found for protein %s", prot_id)
            tag = prot_id
        else:
            tag = ""

        if ligand is not None:
            try:
                # 1 target, multiple ligands
                pre_min_score = df_dock.loc[
                    df_dock["lig-ID"] == ligand, "docking-score-POSIT"
                ].iloc[0]
            except (IndexError, KeyError):
                logging.warning("No pre-docking score found for ligand %s", ligand)

            if prot_id is not None:
                # multiple targets, multiple ligands
                try:
                    pre_min_score = df_dock.loc[
                        (df_dock["lig-ID"] == ligand) & (df_dock["prot-ID"] == prot_id),
                        "docking-score-POSIT",
                    ].iloc[0]
                except (IndexError, KeyError):
                    logging.warning(
                        "No pre-docking score found for ligand %s and protein %s",
                        ligand,
                        prot_id,
                    )
                tag += "_"

            tag += f"{ligand}"

        logging.info(
            "Scoring protein %s and ligand %s from file %s", prot_id, ligand, file_min
        )
        # Reference structure
        if path_ref.is_dir():
            files_in_dir = list(path_ref.glob(f"*{ligand}*.pdb"))
            if len(files_in_dir) > 0:
                file_ref = files_in_dir[0]  # return first find
                logging.info("The ref %s was found for %s", file_ref, tag)
            else:
                logging.error("A reference was not found for %s", tag)
                continue
        else:
            file_ref = path_ref

        # Run minimization if requested
        min_out = file_min
        if minimize:
            min_folder = docking_dir / "minimized"
            min_folder.mkdir(parents=True, exist_ok=True)
            new_docking_dir = min_folder
            md_openmm_platform = "CUDA"
            try:
                min_out = f"{min_folder}/{tag}_min.pdb"
                logging.info("Running MD minimization of %s", tag)
                minimize_structure(
                    file_min,
                    min_out,
                    min_folder,
                    md_openmm_platform,
                    target_name,
                    comp_name,
                )
                chain_dock = "1"  # Standard in OpenMM output file
            except Exception as error:
                logging.error("Couldn't minimize %s: %s", file_min, error)
                continue
        else:
            new_docking_dir = docking_dir
        # Directory to save aligned complexes
        docked_aligned = new_docking_dir / "aligned"
        docked_aligned.mkdir(parents=True, exist_ok=True)

        scorers = [ChemGauss4Scorer()]
        # load addtional ml scorers
        if ml_score:
            logging.info("Loading additional ML scorers")
            # check which endpoints are availabe for the target
            models = ASAPMLModelRegistry.reccomend_models_for_target(target_name)
            ml_scorers = MLModelScorer.load_model_specs(models=models)
            scorers.extend(ml_scorers)
        # Prepare complex, re-dock and score
        logging.info("Running protein prep, docking and scoring of %s", min_out)
        scores_df, prepped_cmp, ligand_pose, aligned = dock_and_score(
            min_out,
            comp_name,
            target_name,
            scorers,
            label=tag,
            allow_clashes=True,
            pdb_ref=file_ref,
            aligned_folder=docked_aligned,
            align_chain=chain_dock,
            align_chain_ref=chain_ref,
        )
        logging.debug(
            "Columns of scoring dataset from asapdiscovery: %s", scores_df.columns
        )
        scores_df["premin-score-POSIT"] = pre_min_score
        df_save = scores_df[["premin-score-POSIT", "docking-score-POSIT"]]
        if ml_score:  # Add ML scores
            if scores_df["docking-score-POSIT"].values:
                df_save = scores_df[
                    [
                        "premin-score-POSIT",
                        "docking-score-POSIT",
                        "computed-SchNet-pIC50",
                        "computed-E3NN-pIC50",
                        "computed-GAT-pIC50",
                    ]
                ]
            else:
                df_save = pd.DataFrame(
                    columns=[
                        "premin-score-POSIT",
                        "docking-score-POSIT",
                        "computed-SchNet-pIC50",
                        "computed-E3NN-pIC50",
                        "computed-GAT-pIC50",
                    ]
                )

        if ligand is not None:
            df_save.insert(loc=0, column="lig-ID", value=ligand)
        if prot_id is not None:
            df_save.insert(loc=0, column="prot-ID", value=prot_id)
        logging.debug("The DataFrame after docking is %s", df_save)

        # Now save files for later scoring steps
        docked_prepped = new_docking_dir / "prepped"
        docked_prepped.mkdir(parents=True, exist_ok=True)
        pdb_prep = docked_prepped / (aligned.stem + "_target.pdb")
        sdf_ligand = docked_prepped / (aligned.stem + "_ligand.sdf")
        # Really annoying, but Target and PreppedTarget have different functions for gen the PDB
        if type(prepped_cmp) is Complex:
            prepped_cmp.target.to_pdb(pdb_prep)
        else:
            prepped_cmp.target.to_pdb_file(pdb_prep)
        ligand_pose.to_sdf(sdf_ligand)
        logging.info(
            "Saved prepped target as %s and ligand as %s",
            pdb_prep.stem,
            sdf_ligand.stem,
        )

        # RMSD score
        logging.debug("The aligned file was saved in %s", aligned)
        logging.info("Calculating RMSD of the ligand")
        lig_rmsd = get_ligand_rmsd(
            aligned, file_ref, True, rmsd_mode="oechem", overlay=False
        )
        df_save.insert(loc=len(df_save.columns), column="Lig-RMSD", value=lig_rmsd)
        pout = f"Calculated RMSD of POSIT ligand pose = {lig_rmsd} with ref {file_ref.stem}"

        if lig_rmsd < 0:
            # Retry ligand rmsd
            logging.info("Trying Ligand RMSD on a different method")
            sdf_ref = docked_prepped / (file_ref.stem + "_ligand.sdf")
            lig_rmsd = get_ligand_rmsd(
                aligned,
                file_ref,
                True,
                rmsd_mode="rdkit",
                overlay=False,
                pathT=str(sdf_ligand),
                pathR=str(sdf_ref),
            )

            if lig_rmsd < 0:
                logging.warning("Retry ligand RMSD calculation failed.")

        if bsite_rmsd:
            logging.info("Calculating RMSD of the binding site")
            try:
                bsite_rmsd = get_binding_site_rmsd(
                    aligned,
                    file_ref,
                    bsite_dist=5.0,
                    ligres=lig_resname,
                    chain_mob=chain_dock,
                    chain_ref=chain_ref,
                    rmsd_mode="heavy",
                    aligned_temp=aligned,
                )
            except Exception as e:
                bsite_rmsd = -1
                logging.warning("Binding site RMSD couldn't be calculated: %s", e)
            df_save.insert(
                loc=len(df_save.columns), column="Bsite-RMSD", value=bsite_rmsd
            )
            pout += f" and {bsite_rmsd} for binding site"
        logging.info(pout)

        if vina_score:
            logging.info("Calculating the affinity score with AutoDock Vina")
            if vina_box_x is None:
                vina_box = None
                logging.info("The grid box will be calculated for the complex")
            else:
                vina_box = [vina_box_x, vina_box_y, vina_box_z]

            df_vina, out_pose = score_autodock_vina(
                pdb_prep,
                sdf_ligand,
                vina_box,
                box_size=[20, 20, 20],
                dock=docking_vina,
                path_to_prepare_file=path_to_grid_prep,
            )
            if out_pose is not None:
                logging.info("Vina docking pose was successfully generated")
                try:
                    lig_rmsd = get_ligand_rmsd(
                        out_pose,
                        file_ref,
                        True,
                        rmsd_mode="oechem",
                        overlay=False,
                    )
                    logging.info("The RMSD of the vina pose was: %s", lig_rmsd)
                except Exception as e:
                    lig_rmsd = -1
                    logging.warning("The vina RMSD couldn't be calculated: %s", e)
                df_vina["Vina-pose-RMSD"] = lig_rmsd
            df_save = pd.concat([df_save, df_vina], axis=1, join="inner")

        logging.debug("%s", gnina_score)
        if gnina_score:
            if Path(gnina_script).exists():
                logging.info("Calculating the affnity with Gnina")
                try:
                    df_gnina = score_gnina(
                        f"{pdb_prep.stem}.pdb",
                        f"{sdf_ligand.stem}.sdf",
                        docked_prepped,
                        home_dir,
                        gnina_script,
                    )
                    logging.debug("Gnina df output is: %s", df_gnina)
                    df_save = pd.concat([df_save, df_gnina], axis=1, join="inner")
                except Exception as e:
                    logging.error("The Gnina calculation failed: %s", e)
            else:
                logging.error(
                    "A gnina bash script must be provided to calculate gnina scores. Won't calculate."
                )

        all_scores.append(df_save)
    all_scores = pd.concat(all_scores)
    all_scores.to_csv(out_csv, index=False)
    return


if __name__ == "__main__":
    score_complexes()
