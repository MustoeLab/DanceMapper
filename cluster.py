# helper function for using emtools

from emtool3 import BernoulliMixture
import numpy as np
import argparse


def parseArguments():

    parser = argparse.ArgumentParser(description = "Cluster data")
    parser.add_argument('inputFile', help='Path to mutation string file')
    parser.add_argument('seqFile', help="Fasta Molecule")
    parser.add_argument('output', default='', help="Results output. Default is print to stdout")

    
    parser.add_argument('ignore', default='', help="Comma-seperated list of nts to ignore")
    parser.add_argument('maxbg', type=float, default=0.01,"Maximum BG reactivity allowed (if BG file provided)")
   





# reads to be clustered, in binary format
readfile = 'bic_100.b' 

bgarray = None 

# Read in background reactivity rates for BG quality filtering
# NOTE -- this is optional. You can pass None value to BernoulliMixture
# If assigned, bgarray should be 1D array

# options for reading include:
bgarray = np.loadtxt('minus_react.np.txt')

# alternatively, you could read it using plotTools:
# from plotTools import ReactivityProfile
# bg = ReactivityProfile('minus_react.csv').
# bgarray = bg.rawprofile


# specify read positions you don't want to cluster. For example,
# structure cassette regions...
# NOTE -- these are 0-indexed
ignore = range(14)+range(134,176)

# Define cutoffs to use for BG quality filtering
maxbg  = 0.01 # exclude nts with (-) reactivity above this 
minsig = 0.002 # exclude nts with (+) reactivty below this
minsig_bg = 0.002 # exclude nts with (+)-(-) reactivity below this

m = BernoulliMixture(readfile,  ignoreCols=ignore, 
                     bgarray=bgarray, maxbg=maxbg, 
                     minsig=minsig, minsig_bg=minsig_bg)



#fit, s = m.bestFitEM( 2, trials=2, 
#                     verbal=True, assign=True, 
#                     convergeThresh=5e-5)

m.fitModel(maxcomponents = 4, verbal=True, convergeThresh=5e-5, trials=5)
m.printParams()





