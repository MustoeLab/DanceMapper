# helper function for using emtools

import sys
sys.path.append('/nas02/home/a/m/amustoe/BernoulliModeling')

from emtool3 import BernoulliMixture
import numpy as np
import argparse
import time

def readCSV(fname):
    """Hacky -- need to replace with better version/workflow in future"""
    
    seq = ''
    rate = []

    with open(fname) as inp:
        
        # pop off the headers
        for i in xrange(3): inp.readline()
        
        for line in inp:
            spl = line.split(',')
            if len(spl[0]) == 0:
                continue

            seq += spl[1]
            rate.append(float(spl[18]))
        
    return seq, np.array(rate)




def printBMfit(output, bm, error):
    """Print out the fitted parameters. 
    """

    output.write('{0} component model\n'.format(bm.components))
    
    # write the first header
    for c in xrange(bm.components):
        output.write("Pop{0}\tError{0}\t".format(c))
    output.write("\n")

    # write out the populations
    for c in xrange(bm.components):
        output.write("{0:.3f}\t{1:.3f}\t".format(bm.p[c], error[0][c] )) 
    output.write("\n")

    # write out the second header
    output.write("Nt\t")
    for c in xrange(bm.components): 
        output.write("Mu{0}\tError{0}\t".format(c))
    output.write("Status\n")


    # compute the total number of nts (both valid and invalid)
    totnts = len(bm.invalidindices)+len(bm.columnindices)

    for i in xrange(totnts):
        output.write("{0}\t".format(i+1))
        
        try:
            bmidx = int(np.argwhere(i==bm.columnindices))
            #print i, bmidx
            for c in xrange(bm.components):
                output.write("{0:.6f}\t{1:.3f}\t".format(bm.mu[c, bmidx], error[1][c, bmidx]))
            
            if bm.inactive_mask[bmidx]:
                output.write("inactive")
            output.write("\n")

        except TypeError:
            for c in xrange(bm.components):
                output.write("nan\tnan\t")

            output.write("invalid\n")
 

def printBMfit_bgsub(output, bm, error, sequence, bgarray):
    """Print out the fitted parameters. 
    """

    output.write('{0} component model\n'.format(bm.components))
    
    # write the first header
    for c in xrange(bm.components):
        output.write("Pop{0}\tError{0}\t".format(c))
    output.write("\n")

    # write out the populations
    for c in xrange(bm.components):
        output.write("{0:.3f}\t{1:.3f}\t".format(bm.p[c], error[0][c] )) 
    output.write("\n")

    # write out the second header
    output.write("Nt\tSeq\t")
    for c in xrange(bm.components): 
        output.write("Mu{0}\tMu_bg{0}\tError{0}\t".format(c))
    output.write("Status\n")


    # compute the total number of nts (both valid and invalid)
    totnts = len(bm.invalidindices)+len(bm.columnindices)

    for i in xrange(totnts):
        output.write("{0}\t{1}\t".format(i+1, sequence[i]))
        
        try:
            bmidx = int(np.argwhere(i==bm.columnindices))

            for c in xrange(bm.components):
                output.write("{0:.3f}\t{1:.3f}\t{2:.3f}\t".format(bm.mu[c, bmidx], 
                                                                  bm.mu[c, bmidx]-bgarray[i],
                                                                  error[1][c, bmidx]))
            
            if bm.inactive_mask[bmidx]:
                output.write("inactive")
            output.write("\n")

        except TypeError:
            for c in xrange(bm.components):
                output.write("nan\tnan\tnan\t")

            output.write("invalid\n")
 


def printBMprops(bm):
    """Print out the nt and read properties of data to be clustered"""

    print 'Ignored nan nts :: {0}'.format(bm.invalidindices)
    print 'Inactive nts :: {0}'.format(bm.columnindices[bm.inactive_mask]+1)
    print '-'*10
    # print number of active, inactive nts
    print "{0}\ttotal nts".format( len(bm.invalidindices)+ len(bm.columnindices) )
    print "{0}\tnan nts ignored".format( len(bm.invalidindices) )
    print "{0}\tinactive nts".format( np.sum(bm.inactive_mask) )
    print "{0}\tactive nts".format( np.sum(bm.active_mask) )
    
    print "\n{0} reads".format( bm.numreads )
        
    print '-'*10


def parseArguments():
    
    parser = argparse.ArgumentParser(description = "Cluster data")

    # eventually want to make binary processing a single step
    parser.add_argument('inputFile', type=str, help='Path to binary file')
    #parser.add_argument('seqFile', help="Fasta Molecule")

    parser.add_argument('--output', type=str, default='stdout', 
                        help="Results output. Default is print to stdout")
    parser.add_argument('--bgfile', type=str, default='', 
                        help="CSV file containing BG mutation rates")
    parser.add_argument('--maxcomponents', type=int, default=4, 
                        help="Maximum number of model components to search")
    parser.add_argument('--convergethresh', type=float, default=1e-5, 
                        help="EM convergence threshold")
    parser.add_argument('--emtrials', type=int, default=5, 
                        help="Number of fit trials for each model size")
    
    parser.add_argument('--inactivents', type=str, default='',
                        help="Comma-seperated list of nts to treat inactive during clustering")
    parser.add_argument('--invalidnts', type=str, default='', 
                        help="Comma-seperated list of invalid nts to ignore (set to nan)")
    parser.add_argument('--maxbg', type=float, default=0.01, 
                        help="Maximum BG mutation rate (if BG file provided)")
    parser.add_argument('--minrx', type=float, default=0.002, 
                        help="Minimum RX mutation rate to be actively clustered")
    parser.add_argument('--minrxbg', type=float, default=0.002, 
                        help="Minimum RX-BG rate to be actively clustered")
    

    args = parser.parse_args()

    # parse the ignorents arguments
    if args.invalidnts:
        args.invalidnts = set([ int(x)-1 for x in args.invalidnts.split(',') ])
    else:
        args.invalidnts = set([])
    
    if args.inactivents:
        args.inactivents = set([ int(x)-1 for x in args.inactivents.split(',') ])
    else:
        args.inactivents = set([])
    


    return args





if __name__ == '__main__':

    args = parseArguments()


    bgreactivity = None


    if args.bgfile:
        
        # read bg file
        sequence, bgreactivity = readCSV(args.bgfile)
        
        # get high bg posititions and add them to set of "high bg ignore nts"
        args.invalidnts.update( np.where(bgreactivity > args.maxbg)[0])

    
    # initialize the model
    BM = BernoulliMixture(args.inputFile, invalidcols=args.invalidnts, 
                          inactivecols=args.inactivents,
                          bgarray=bgreactivity, maxbg=args.maxbg, 
                          minrx=args.minrx, minrxbg=args.minrxbg, verbal=False)
    
    # print stats to stdout
    printBMprops(BM)
    
    t = time.time()

    BM.fitModel(maxcomponents = args.maxcomponents, 
               convergeThresh = args.convergethresh,
               trials = args.emtrials,
               verbal = True)
    
    print "Fit time = {0:.1f}".format(time.time()-t)
    
    t= time.time()
    # compute the error from the information matrix
    #imerror = BM.compute_IM_Error()
    
    #if imerror[0] is None:
    #    imerror = [np.ones(BM.p.shape)*100, np.ones(BM.mu.shape)*100]

    imerror = [np.zeros(BM.p.shape), np.zeros(BM.mu.shape)]

    print "Error time = {0:.1f}".format(time.time()-t)


    if args.output == 'stdout':
        output = sys.stdout
        output.write('*'*20+'\n')
    else:
        output = open(args.output, 'w')
    
    if args.bgfile:
        printBMfit_bgsub(output, BM, imerror, sequence, bgreactivity)
    else:
        printBMfit(output, BM, imerror)
    
    if args.output != 'stdout':
        output.close()
    






# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
