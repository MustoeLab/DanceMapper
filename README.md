# DANCE-MaPper
ML clustering code and associated analysis scripts for deconvoluting RNA ensembles
from single-molecule DMS-MaP datasets

-----------------------------------------
Copywrite 2021 Anthony Mustoe

This project is licensed under the terms of the MIT license

Developed by:
Anthony Mustoe Lab, Baylor College of Medicine
Kevin Weeks Lab, University of North Carolina

Contact: anthony.mustoe@bcm.edu



General Description
------------
Forthcoming


Dependencies
------------
- python 2.7 + numpy

- cython 

- Ringmapper/Pairmapper package (v1.1)
    available at https://github.com/Weeks-UNC/RingMapper

- RNATools2 (needed for plotClusters and foldClusters)
    available at https://github.com/Weeks-UNC/RNATools

- arcPlot (needed for foldClusters)
    available at https://github.com/Weeks-UNC/arcPlot
    
- RNAstructure (needed for foldClusters)
    available at https://rna.urmc.rochester.edu/RNAstructure.html



Installation
------------
- Open externalpaths.py in a text editor and insert correct paths to dependencies

- Compile accessoryFunctions.pyx cython routines by running:
    python setup.py build_ext --inplace



ShapeMapper preprocessing
-----------------------
DanceMapper requires initial preprocessing of sequencing reads by ShapeMapper2 (v2.1.5 is preferred). 
ShapeMapper2 should be run with the *--output-parsed-mutations* option.
ShapeMapper2 can be obtained at https://github.com/Weeks-UNC/shapemapper2


DANCE-MaP
---------------------------
### DanceMapper.py
Run DanceMapper.py --help for complete usage information

Note that we recommend having at least 250,000 mapped reads for reliable DANCE deconvolution, and ideally
>500,000 reads. Deconvolution may be possible with lower read depths, but we do not presently know the 
lower bound. 

For PAIR and RING analysis of deconvoluted reads, we recommend having at least 1,000,000 reads, and ideally
>1,000,000 reads per state.

The current script is serial (single cpu). Run times vary based RNA size, number of reads, and number
of final clusters. When performing primary clustering (*--fit*), anticipate 4-24 hours. When running
PAIR or RING analysis (*--pairmap* or *--ring*) anticipate 12-48 hours each. 


Input:
    
    parsed.mut file output by ShapeMapper
    
    profile.txt file output by ShapeMapper
    

Output:
    .bm file 
        save file of the Bernoulli mixture model. Only generated when using the --fit option

    -reactivities.txt file
        normalized reactivities for each structure. Only generated when using the --fit option

    [i]-rings.txt file
        RINGs for state i (window=1). Only generated when using the --ring option

    [i]-pairmap.txt file
        PAIRs for state i. Only generated when using the --pairmap option

    [i]-pairmap.bp file
        PAIR energy restraints for state i. Only generated when using the --pairmap option

    [i]-allcorrs.txt file
        RINGs (window=3) for state i. Only generated when using the --pairmap option



### foldClusters.py
Script for performing RNAstructure modeling based on clustered reactivities and plotting results 
using arcPlot. Takes -reactivities.txt file as input. Can also accept -pairmap.bp restraints. 

foldClusters.py will generate sequence ([out].seq) and normalized dms files ([out]-[i].dms) for
performing RNAstructure modeling using the -dmsnt option. (Note that no math is being done, 
it simply disaggregated the -reactivites.txt file).

Folding is then done using Fold, and structure models will be written as [out]-[i].ct. 
PK folding is also available using the --pk option. Note that PK folding will use the hierarchical 
foldPK script distributed as part of RNAtools, which wraps around ShapeKnots allowing discovery of 
multiple PKs (see RNATools README for more information). Structure models are written as CT format files 
(see RNAstructure documentation for details). For PK folding, multiple CT files may be generated as 
part of the hierarchical folding process. These are denoted as [out]-[i].1.ct, .2.ct, etc. 
The final solution will be named [out]-[i].f.ct

Finally, arcPlots are generated using arcPlot and saved as [out]-[i].pdf. PDFs show
the MFE structure, the DMS reactivity profile, and PAIR data (if the --bp flag is used).

Run foldClusters.py --help for additional options and usage information




### plotClusters.py 
Script for visualizing and comparing reactivities of DanceMaP identified clusters.
(Makes step plots, also known as skyline plots).
Run plotClusters.py --help for usage information




Example
------------

*Preprocess data*

    shapemapper --target add.fa --name example --amplicon --output-parsed \
    --modified --R1 example-mod_R1.fastq.gz --R2 example-mod_R2.fastq.gz \
    --untreated --R1 example-neg_R1.fastq.gz --R2 example-neg_R2.fastq.gz

*Run DanceMapper with PAIR and RING analysis*

    DanceMapper.py --mod example_Modified_add_parsed.mut --unt example_Untreated_add_parsed.mut --prof example_addl_profile.txt --out example --fit --pair --ring


*Fold and plot structure states (using PAIR restraints and computing pairing probabilities)*
    
    foldClusters.py --bp example --prob example-reactivities.txt example



Complete class description 
--------------------------
Forthcoming




