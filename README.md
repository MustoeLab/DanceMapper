# DANCE-MaPper
ML clustering code and associated analysis scripts for deconvoluting RNA ensembles
from single-molecule DMS-MaP datasets

-----------------------------------------
Copywrite 2023 Anthony Mustoe

This project is licensed under the terms of the MIT license

Developed by:

Anthony Mustoe Lab, Baylor College of Medicine

Kevin Weeks Lab, University of North Carolina

Contact: anthony.mustoe@bcm.edu



Dependencies
------------
- python 2.7 + numpy

- cython 

- Ringmapper/Pairmapper package (v1.2)
    available at https://github.com/Weeks-UNC/RingMapper

- StructureAnalysisTools (needed for plotClusters and foldClusters)
    available at https://github.com/MustoeLab/StructureAnalysisTools
    
- RNAstructure (needed for foldClusters)
    available at https://rna.urmc.rochester.edu/RNAstructure.html



Installation
------------
- Open externalpaths.py in a text editor and insert correct paths to dependencies

- Compile accessoryFunctions.pyx cython routines by running:
    python setup.py build_ext --inplace



ShapeMapper preprocessing
-----------------------
DanceMapper requires initial preprocessing of sequencing reads by ShapeMapper2 (v2.2 is preferred). 
ShapeMapper2 should be run with the *--output-parsed-mutations* option.
ShapeMapper2 can be obtained at https://github.com/Weeks-UNC/shapemapper2


DanceMapper.py
--------------
Run DanceMapper.py --help for complete usage information

Note that we recommend having at least 250,000 mapped reads for reliable DANCE deconvolution, and ideally
>500,000 reads. Deconvolution may be possible with lower read depths, but we do not presently know the 
lower bound. 

For PAIR and RING analysis of deconvoluted reads, we recommend having at least 1,000,000 reads, and ideally
>1,000,000 reads per state.

The current script is serial (single cpu). Run times vary based RNA size, number of reads, and number
of final clusters. When performing primary clustering (*--fit*), anticipate between 1-24 hours. When running
PAIR or RING analysis (*--pairmap* or *--ring*) anticipate 12-48 hours each. 

Note that DanceMapper is very memory intensive. As a rough guideline, you will need 50 x N x R bytes, where
N is the RNA length and R is the # of reads. So for a 400 nt long RNA with 1M reads, this would be 20 GB. 
We plan to release a memory calculator tool with future releases.


Input:
    
    parsed.mut 
        file output by ShapeMapper
    
    profile.txt 
        file output by ShapeMapper
    

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



*Note* PAIR/RING calculations have modestly changed from v1.0. To run using original parameters, 
use the following flags:
--oldDMSnorm --pm_secondary_reactivity 0.5 --mincount 50


DanceMapper.py usage related to N7
--------------
To use DanceMapper.py to deconvolve N7-G reactivities in addition to traditional N1/3 reactivities
ShapeMapper 2.3+ must be run with --dms and --N7 flags in addition to the --output-parsed-mutations
option to produce N7-G related files.

Input:
    
    parsed.mut 
        file output by ShapeMapper
    
    profile.txt 
        file output by ShapeMapper
    
    --concat
        flag directing DanceMapper to concatenate the parsed.mut and parsed.mutga files to deconvolve
        N7-G reactivities in addition to N1/3 reactivities

*Note* The parsed.mut file must have a corresponding parsed.mutga file located in the same directory.
Additionally, the profile.txt file must have a corresponding profile.txtga file located in the same
directory as well. When ShapeMapper 2.3+ is run with the --N7 flag the parsed.mut and profile.txt as
well as the corresponding parsed.mutga and profile.txtga files are all produced in the same directory
by default.

Output:

    concat.bm file 
        save file of the Bernoulli mixture model encompassing both N1/3 and N7-G nucleotides. Only 
        generated when using the --fit option

    N13.bm file 
        save file of the Bernoulli mixture model encompassing the N1/3 nucleotides. Only generated 
        when using the --fit option

    N7.bm file 
        save file of the Bernoulli mixture model encompassing the N7 nucleotides. Only generated 
        when using the --fit option

    -concat-reactivities.txt file
        normalized N1/3 and N7-G reactivities for each structure. Only generated when using the 
        --fit option

    -N13-reactivities.txt file
        normalized N1/3 reactivities for each structure. Only generated when using the --fit
        option

    -N7-reactivities.txt file
        normalized N7 reactivities for each structure. Only generated when using the --fit
        option

    [i]-N1rings.txt file
        N1/3-N1/3 RINGs for state i (window=1). Only generated when using the --ring option

    [i]-N7rings.txt file
        N7-N7 RINGs for state i (window=1). Only generated when using the --ring option
        
    [i]-N1N7rings.txt file
        N1/3-N7 RINGs for state i (window=1). Only generated when using the --ring option
       

*Additional Notes*
--pairmap flag not currently compatible with --concat flag

In order to visualize deconvolved N1/3 and N7-G data it is currently recommended that the 
dance-reactivites_profile_converter.py script be used. This script will split the
-concat-reactivities.txt file into profile.txt and profile.txtga files corresponding
to the states of the deconvolved model. eg:

"python dance-reactivites_profile_converter.py -concat-reactivities.txt --output output_prefix"

These profile files can then be visualized via arcplot.



foldClusters.py
----------------
Script for performing RNAstructure modeling based on clustered reactivities and plotting results 
using ArcPlot. Takes -reactivities.txt file as input. Can also accept -pairmap.bp restraints. 

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

Finally, ArcPlots are generated using ArcPlot and saved as [out]-[i].pdf. PDFs show
the MFE structure, the DMS reactivity profile, and PAIR data (if the --bp flag is used).

Run foldClusters.py --help for additional options and usage information

Note that the pairing probability option is currently not supported in standard distributions of RNAstructure.
We are working on making this option available. Please contact us for more information in the meantime.




plotClusters.py
----------------
Script for visualizing and comparing reactivities of DanceMaP identified clusters.
(Makes step plots, also known as skyline plots).

Run plotClusters.py --help for usage information



Example
========

Some example data and commands are provided in the *example* directory. 


Some generic example commands are below:


*Preprocess data*

    shapemapper --target add.fa --name example --amplicon --output-parsed --dms \
    --modified --R1 example-mod_R1.fastq.gz --R2 example-mod_R2.fastq.gz \
    --untreated --R1 example-neg_R1.fastq.gz --R2 example-neg_R2.fastq.gz

*Run DanceMapper with PAIR and RING analysis*

    python DanceMapper.py --mod example_Modified_add_parsed.mut --unt example_Untreated_add_parsed.mut --prof example_add_profile.txt --out example --fit --pair --ring


*Fold each ensemble state (MFE) using PAIR restraints and get arcPlot visualization, including of PAIRs*
    
    python foldClusters.py --bp example example-reactivities.txt example





