#!/usr/bin/env python

import sys, argparse, itertools, math
import numpy as np

import optbuildComutationMatrix as buildMatrix



def parseArguments():

    parser = argparse.ArgumentParser(description = "Compute correlations from mutation string file")
    parser.add_argument('inputFile', help='Path to mutation string file')
    parser.add_argument('outputFile', help='Path for correlation output file')
    parser.add_argument('molsize', type=int, help="""Size of the molecule""") 

    parser.add_argument('--phred_cut', type=int, default=30, help="Set phred cutoff (default = 30)")
    
    parser.add_argument('--eventsep', type=int, default=5, help="""Number of non-mutations required between
            mutation events (default = 5)""")
    
    parser.add_argument('--maxdelsize', type=int, default=10000, help="""Maximum deletion/insert allowed in read 
    (default = 10000). Note the default of 10000 disables the features""")
    
    parser.add_argument('--events', default="AGCT-", help='String of valid mutation events (default = AGCT-)')   
    
    return parser.parse_args()   
        



if __name__ == '__main__':


    args = parseArguments()
    
    # fill matrices
    buildMatrix.filter_write(args.inputFile, args.outputFile, args.molsize,
                             args.phred_cut, args.events, args.eventsep, args.maxdelsize)













# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
