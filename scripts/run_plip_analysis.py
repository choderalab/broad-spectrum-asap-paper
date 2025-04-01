from plip_analysis_schema import PLIntReport
from pathlib import Path
import click
import yaml
from pebble import ProcessPool
from functools import partial
from warnings import warn


def analyze_structure(structure: Path, name: str, output_dir: Path, create_pymol_session: bool):
    outpath = output_dir / f"{name}_{structure.stem}_interactions.csv"
    pymol_path = outpath.with_suffix(".pse") if create_pymol_session else None
    interactions = PLIntReport.from_complex_path(
        complex_path=structure,
        create_pymol_session=create_pymol_session,
        pymol_session_path=pymol_path
    )
    interactions.to_csv(outpath)
    click.echo(f"Saved interactions to {outpath}")
    return outpath


@click.command()
@click.option(
    "--yaml-input",
    type=click.Path(exists=True, path_type=Path),
    help="Path to input yaml file containing name: path pairs",
    required=True
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Path to output directory",
    required=True
)
@click.option(
    "--create-pymol-session",
    is_flag=True,
    help="Create a PyMOL session for each structure"
)
@click.option(
    "--ncpus",
    type=int,
    default=1,
    help="Number of cpus to use for parallel processing"
)
def main(yaml_input: Path, output_dir: Path, create_pymol_session: bool, ncpus: int):
    """Get PLIP interactions"""
    output_dir.mkdir(exist_ok=True)

    if create_pymol_session:
        click.echo("Creating PyMOL sessions will be enabled for each structure.")
        warn("The PyMol code, while providing a useful output, is very buggy and hard to debug. "
             "Use it at your own risk, and consider disabling it if you encounter issues.")

    with open(yaml_input, "r") as f:
        input_dict = yaml.safe_load(f)

    for name, structure_dir in input_dict.items():
        structure_dir = Path(structure_dir)
        if not structure_dir.exists():
            raise FileNotFoundError(f"{structure_dir} does not exist")

        click.echo(f"Loading all pdb structures in {structure_dir}")
        structures = [structure for structure in structure_dir.glob("*.pdb")]

        click.echo(f"Analyzing {len(structures)} structures")

        analyze_structure_partial = partial(
            analyze_structure,
            name=name,
            output_dir=output_dir,
            create_pymol_session=create_pymol_session
        )

        # parallelize with pebble
        with ProcessPool(max_workers=ncpus) as pool:
            result = pool.map(analyze_structure_partial, structures)
        click.echo(result.result())


if __name__ == "__main__":
    main()