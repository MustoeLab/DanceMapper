
Example explanation
-------------------

You will first want to unzip the 'data' files (note that unzipped, these files collectively take ~1 GB of space).

>tar -xf data.tgz

This tarball contains preprocessed mutation string files (.mut) and profile.txt from ShapeMapper. These are the raw inputs into DanceMapper. 
Output files are also provided for reference (labeled with output prefix).


Once data are unzipped and DanceMapper and other dependencies are installed (see primary README) you are ready to run. 
Several example commands are provided below:


(1)  Perform a complete run, fitting both for consituent structures and then solving for PAIRs/RINGs. Because we know this RNA will only deconvolute into 2 structures,
here we terminate the search at 2 rather than trying to find 3 or more structures (--maxc 2). This speeds up runtime substantially. Performing the fit should <10 min, and the
ring and pair fits will each take ~30 minutes, resulting in a total run time of ~1 hr. If you only want to check performance of the fit option (<10 min) run the following:

>python ../DanceMapper.py --mod wt-0_Modified_addWTfull_parsed.mut --unt wt-0_Untreated_addWTfull_parsed.mut --prof wt-0_addWTfull_profile.txt --pair --ring --fit --maxc 2 --out test


(2) Just perform primary fit (ie bm file). This usually takes <15 minutes.

>python ../DanceMapper.py --mod wt-0_Modified_addWTfull_parsed.mut --unt wt-0_Untreated_addWTfull_parsed.mut --prof wt-0_addWTfull_profile.txt --fit --maxc 2 --out test


(3) Measure RINGs based on prior fit. This will take ~30 minutes

>python ../DanceMapper.py --mod wt-0_Modified_addWTfull_parsed.mut --unt wt-0_Untreated_addWTfull_parsed.mut --prof wt-0_addWTfull_profile.txt --readfrom output-wt-0.bm --ring --out test


Note that above commands will write to files beginning with 'test'. You can change the output prefix using the --out flag.


---


To compare reactivity profile / popolation fits (bm files) you can use the plotClusters.py tool:

>python ../plotClusters.py --bm1 test.bm --bm2 output-wt-0.bm

Note that DANCE is a stochastic algorithm and will naturally give rise to slight variations in final fitted parameters. However, the profile fits should be highly
correlated (R>0.99) and populations should be +/- 0.02


To compare RINGs or PAIRs, you can visualize the data using arcPlot or manually compare the files. As noted above, some stochastic variation is expected, 
but answers should closely correspond. Note that if you used option (3) above -- i.e. using the prior fit -- the RINGs should be exactly 
reproducible with the following exception: 1 or 2 RINGs may newly appear or dropout of the refitted dataset. This dropout/appearance effect is because of 
stochasticity in the synthetic control simulations used to screen out potential read-assignment artifacts.



Data source
-----------
Data in this file correspond to WT adenine riboswitch, 0 ligand, rep 1 (GEO = GSM5531248)
Raw data were processed using ShapeMapper v2.1.5

