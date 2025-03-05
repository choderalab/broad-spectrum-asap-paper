# broad-spectrum-asap-paper
Structure-based prediction of affinity across viral families.

## Intro
The rapid emergence of viruses with pandemic and epidemic potential presents a continuous threat for public health worldwide. With the typical drug discovery pipeline taking an average of 6 years to     reach the pre-clinical stage, there is an urgent need for new strategies to design broad-spectrum antivirals that can target multiple viral family members and variants of concern. We present a structure-based computational pipeline designed to identify and evaluate broad-spectrum inhibitors across viral family members for a given target.

Publication: _In progress_ 

## Contributors
- Maria A. Castellanos
- Alexander M. Payne
- Hugo MacDermott-Opeskin

## Contents
Scripts and input files for the paper on structure-based prediction of affinity across the coronavirus family.

- Run sequence search and alignment with `asap-spectrum` from [`asapdiscovery`](https://github.com/asapdiscovery/asapdiscovery)
- Run protein folding with ColabFold and structure alignment with `asap-spectrum`
- Ligand transfer, docking and refinement with [`asap-discovery`](https://github.com/asapdiscovery/asapdiscovery)
- Scoring of poses with the `score_complexes` cli

## Installation
To install this repository, follow these steps:
 
1. Clone the repository, then enter the source tree:
  
 ```
git clone https://github.com/choderalab/broad-spectrum-asap-paper.git
cd broad-spectrum-asap-paper
```
 
2. Install the dependencies into a new conda environment, and activate it:
 
```
mamba env create -f conda-envs/test_env.yml
conda activate broad-spectrum
```

3. Install [ColabFold](https://github.com/sokrypton/ColabFold). This can be done locally using [localfold](https://github.com/YoshitakaMo/localcolabfold), or via Docker, following the instructions on the ColabFold repo. The example will assume the program is installed on a module `colabfold/v1.5.2`.
 
4. Install [gnina](https://github.com/gnina/gnina). Also not on available in `conda-forge` but can be installed via a Docker image.
`gnina` is used for scoring docked poses and it not strictly required (the alternative is AutoDock Vina), but it is more accurate.

## License
* This software is licensed under the [MIT license](https://opensource.org/licenses/MIT) - a copy of this license is provided as `SOFTWARE_LICENSE`
* The data in this repository is made available under the Creative Commons [CC0 (“No Rights Reserved”) License](https://creativecommons.org/share-your-work/public-domain/cc0/) - a copy of this license is provided as `DATA_LICENSE`

### Copyright
 
Copyright (c) 2025, Maria A. Castellanos
