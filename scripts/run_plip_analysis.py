"""
The goal of this script is to convert PLIP interactions into a framework that can be used to calculate a "score".

Example Usage:
python run_plip_analysis.py output_directory/crystal_8dz0_min_interactions.csv output_directory

Although the above will also compare the crystal structure against itself, it's nice to make sure things are working.
"""
import pandas as pd
import scripts.plip_analysis_schema as pa
from pathlib import Path
import click

LABELS = {
    'Variant': 'Viral Variant',
    'tversky_index': 'PLIF Recall',
    'ByTotalInteractions': 'Total Number of Interactions',
    'ByEverything': 'Atomic Level',
    'ByInteractionType': 'Interaction Type',
    'ByInteractionTypeAndAtomTypes': 'Interaction Type and Atom Type',
    'ByInteractionTypeAndResidueTypeAndBBorSC': 'Interaction Type and Residue Type and Backbone vs Sidechain',
    'ByInteractionTypeAndResidueTypeAndNumber': 'Interaction Type and Residue Type and Number'
}
def get_plip_analysis(crystal_csv: Path, docked_path: Path, output_dir: Path) -> pd.DataFrame:
    """
    Analyze PLIP interaction data from crystal and docked structures and generate a DataFrame with scores.

    Parameters
    ----------
    crystal_csv : Path
        Path to the CSV file containing PLIP interaction data for the crystal structure.
    docked_path : Path
        Path to the directory containing CSV files for docked structures.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the PLIP analysis results.
    """
    # Setup output directory
    if output_dir is None:
        output_dir = Path("./analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    docked_reports = {
        path.stem[7:]: pa.PLIntReport.from_csv(path)
        for path in docked_path.glob("*.csv")
    }
    if not docked_reports:
        raise FileNotFoundError(f"No csv files were found in {docked_path}. ")

    crystal_report =pa.PLIntReport.from_csv(crystal_csv)

    # Calculate scores
    score_list = []
    for level in pa.FingerprintLevel:
        for name, docked_report in docked_reports.items():
            score = pa.InteractionScore.from_fingerprints(
                crystal_report, docked_report, level
            )
            score_list.append({'Structure': name, **score.dict()})

    # Create dataframe and calculate ratios
    df = pd.DataFrame.from_records(score_list)
    df["ratio_of_intersection"] = (
            df["number_of_interactions_in_intersection"] /
            df["number_of_interactions_in_reference"]
    )
    df["ratio_of_query"] = (
            df["number_of_interactions_in_query"] /
            df["number_of_interactions_in_reference"]
    )

    # Process data for plotting
    df['Type of Interactions'] = df.provenance.apply(lambda x: LABELS.get(x))
    df['Fingerprint Specificity'] = df.provenance.apply(lambda x: LABELS.get(x))

    # Save result
    df.to_csv(output_dir / "results.csv")

    return df

@click.command()
@click.argument('crystal-csv', type=click.Path(exists=True, path_type=Path))
@click.argument('docked-path', type=click.Path(exists=True, path_type=Path))
@click.option('--output-dir', type=click.Path(path_type=Path), default=None, help="Output directory for analysis files. Defaults to ./analysis")
def main(crystal_csv: Path, docked_path: Path, output_dir: Path):
    """Analyze PLIP interaction data and generate comparison plots.
    CRYSTAL_CSV: Path to the CSV file containing PLIP interaction data for the crystal structure.
    DOCKED_PATH: Path to the directory containing CSV files for docked structures.
    """
    df = get_plip_analysis(crystal_csv, docked_path, output_dir)


if __name__ == '__main__':
    main()

