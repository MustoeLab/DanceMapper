# Ensemble-Mapper
EM clustering code and associated analysis scripts for DMS-MaP datasets

-----------------------------------------
Copywrite 2020 Anthony Mustoe

This project is licensed under the terms of the MIT license

Developed jointly by:
Anthony Mustoe Lab
Baylor College of Medicine

Kevin Weeks Lab
University of North Carolina

Contact: anthony.mustoe@bcm.edu



General Description
------------
Forthcoming


Dependencies
------------
- python 2.7 + numpy
- cython 
- Ringmapper/Pairmapper package
- (Needed for plotClusters/foldClusters): RNATools, arcPlot, RNAstructure



Installation
------------
- Open externalpaths.py in a text editor and insert correct paths to dependencies

- Compile accessoryFunctions.pyx cython routines by running:
    python setup.py build_ext --inplace



User-facing Programs
---------------------
The following programs have command line interfaces:

### EnsembleMap.py
Main clustering script. Run EnsembleMap.py --help for usage information

### plotClusters.py 
Script for visualizing and comparing the reactivites of EnsembleMaP identified clusters.
(Makes step plots, also known as skyline plots).
Run plotClusters.py --help for usage information

### foldClusters.py
Script for performing RNAstructure modeling based on clustered reactivities and plotting results 
using arcPlot. Takes -reactivities.txt file as input. Can also accept -pairmap.bp restraints. 

foldClusters.py will generate sequence ([out].seq) and normalized dms files ([out]-[clust#].dms) for
performing RNAstructure modeling using the -dmsnt option. (Note that no math is being done, 
it simply disaggregated the -reactivites.txt file).

Folding is then done using Fold, and structure models will be written as [out]-[clust#].ct. 
PK folding is also available using the --pk option. Note that PK folding will use the hierarchical 
foldPK script distributed as part of RNAtools, which wraps around ShapeKnots allowing discovery of 
multiple PKs (see RNATools README for more information). Structure models are written as CT format files 
(see RNAstructure documentation for details). For PK folding, multiple CT files may be generated as 
part of the hierarchical folding process. These are denoted as [out]-[clust#].1.ct, .2.ct, etc. 
The final solution will be named [out]-[clust#].f.ct

Finally, arcPlots are generated using arcPlot and saved as [out]-[clust#].pdf. PDFs show
the MFE structure, the DMS reactivity profile, and clustered pairmap data (if the --bp flag is used).

Run foldClusters.py --help for additional options and usage information


Complete class description and accessory codes
----------------------------------------------
Forthcoming


Example
-------
Forthcoming




