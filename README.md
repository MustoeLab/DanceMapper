DANCE-MaPper
=======================================================================================
ML clustering code and associated analysis scripts for deconvoluting RNA ensembles
from single-molecule DMS-MaP datasets

```text
Copywrite 2023 Anthony Mustoe
This project is licensed under the terms of the MIT license
Developed by:
Anthony Mustoe Lab, Baylor College of Medicine
Kevin Weeks Lab, University of North Carolina
Contact: anthony.mustoe@bcm.edu
```


Dependencies
---------------------------------------------------------------------------------------

- python 2.7
- numpy
- cython

**_For `--rings` and `--pairmap`_**
- Ringmapper/Pairmapper package (v1.2): https://github.com/Weeks-UNC/RingMapper

**_For `plotClusters.py` and `foldClusters.py`_**
- StructureAnalysisTools: https://github.com/MustoeLab/StructureAnalysisTools
- RNAstructure package: https://rna.urmc.rochester.edu/RNAstructure.html
  - *Fold* for default usage
  - *ShapeKnots* for `--pk`
  - *partition* and *ProbabilityPlot* for `--prob`


Installation
---------------------------------------------------------------------------------------

- Compile accessoryFunctions.pyx cython routines by running: `python setup.py build_ext --inplace`


ShapeMapper preprocessing
---------------------------------------------------------------------------------------

- DanceMapper requires initial preprocessing of sequencing reads by ShapeMapper2 (v2.2 is preferred). 
- ShapeMapper2 should be run with the `--output-parsed-mutations` option.
- ShapeMapper2 can be obtained at https://github.com/Weeks-UNC/shapemapper2


DanceMapper.py
---------------------------------------------------------------------------------------

Run `DanceMapper.py --help` for complete usage information.

> **_Read depths:_** We recommend at least 250k mapped reads (ideally >500k) for
  reliable DANCE deconvolution. Accurate deconvolution may be possible with lower read
  depths, but we do not presently know the lower bound. For PAIR and RING analysis of
  deconvoluted reads, we recommend >1M reads, and ideally >1M reads per state.

> **_Memory requirements_** DanceMapper is very memory intensive. As a rough guideline,
  you will need 50 x N x R bytes, where N is the RNA length and R is the # of reads. So
  for a 400 nt long RNA with 1M reads, this would be 20 GB. We plan to release a memory
  calculator tool with future releases.

The current script is serial (single cpu). Run times vary based on RNA size, number of
reads, and number of final clusters. When performing primary clustering (`--fit`),
anticipate between 1-24 hours. When running PAIR or RING analysis (`--pairmap` and/or
`--ring`) anticipate 12-48 hours each. 

Input:

    parsed.mut
        file output by ShapeMapper

    profile.txt
        file output by ShapeMapper

Output when using `--fit`:

        .bm file 
            save file of the Bernoulli mixture model

        -reactivities.txt file
            normalized reactivities for each structure

Output when using `--ring`:

        [i]-rings.txt file
            RINGs for state i (window=1)

Output when using `--pairmap`:

    [i]-pairmap.txt file
        PAIRs for state i

    [i]-pairmap.bp file
        PAIR energy restraints for state i

    [i]-allcorrs.txt file
        RINGs (window=3) for state i

> **_Note_**: PAIR/RING calculations have modestly changed from v1.0.
> To run using original parameters, use the following flags:
> `--oldDMSnorm --pm_secondary_reactivity 0.5 --mincount 50`


foldClusters.py
---------------------------------------------------------------------------------------

Script for performing RNAstructure modeling based on clustered reactivities and
plotting results using arcPlot. Takes -reactivities.txt file as input. Can also accept
-pairmap.bp restraints.

Run `foldClusters.py --help` for options and usage information

foldClusters.py will generate sequence ([out].seq) and normalized dms files
([out]-[i].dms) for performing RNAstructure modeling using the -dmsnt option. (Note
that no math is being done, it simply disaggregated the -reactivites.txt file).

Folding is then done using Fold, and structure models will be written as [out]-[i].ct. 
PK folding is also available using the --pk option. Note that PK folding will use the
hierarchical foldPK script distributed as part of RNAtools, which wraps around
ShapeKnots allowing discovery of multiple PKs (see RNATools README for more
information). Structure models are written as CT format files (see RNAstructure
documentation for details). For PK folding, multiple CT files may be generated as
part of the hierarchical folding process. These are denoted as [out]-[i].1.ct, .2.ct,
etc. The final solution will be named [out]-[i].f.ct

Finally, arcPlots are generated using arcPlot and saved as [out]-[i].pdf. PDFs show
the MFE structure, the DMS reactivity profile, and PAIR data (if the --bp flag is used).

Note that the pairing probability option is currently not supported in standard
distributions of RNAstructure. We are working on making this option available. Please
contact us for more information in the meantime.


plotClusters.py
---------------------------------------------------------------------------------------

Script for visualizing and comparing reactivities of DanceMaP identified clusters.
(Makes step plots, also known as skyline plots).

Run `plotClusters.py --help` for usage information.


Examples
---------------------------------------------------------------------------------------

Some example data and commands are provided in the `./example/` directory. 

Some generic example commands are below:

**_Preprocess data_**

```bash
shapemapper --target add.fa --name example --amplicon --output-parsed --dms \
    --modified --R1 example-mod_R1.fastq.gz --R2 example-mod_R2.fastq.gz \
    --untreated --R1 example-neg_R1.fastq.gz --R2 example-neg_R2.fastq.gz
```

**_Run DanceMapper with PAIR and RING analysis_**

```bash
python DanceMapper.py --mod example_Modified_add_parsed.mut \
    --unt example_Untreated_add_parsed.mut --prof example_add_profile.txt \
    --out example --fit --pair --ring
```

**_Fold each ensemble state (MFE) using PAIR restraints and get arcPlot visualization, including of PAIRs_**

```bash
python foldClusters.py --bp example example-reactivities.txt example
```
