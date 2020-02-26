#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=2

from libc.math cimport log, exp
from libc.stdio cimport FILE, fopen, fclose, getline, printf, fflush, stdout

import time
import numpy as np
cimport numpy as np

from readMutStrings cimport READ, parseLine, fillReadMut, incrementArrays
from dSFMT cimport dsfmt_t, dsfmt_init_gen_rand, dsfmt_genrand_close_open


###########################################################################
#
# Fill mutation matrices
#
###########################################################################


def fillReadMatrices(str inputFile, int seqlen, int mincoverage, int undersample=-1):
    """ inputFile = path to the classified mutations files """
    
    # mem alloc for reading the file using c
    cdef FILE* cfile
    cdef size_t lsize = 0
    cdef ssize_t endfile
    cdef char* line = NULL
    
    # mem alloc for read parsing and for loops   
    cdef READ r
    cdef int i
    
    # open the file
    cfile = fopen(inputFile, "r")
    if cfile == NULL:
        raise IOError(2, "No such file or directory: '{0}'".format(inputFile))


    allreadstr = []
    allmutstr = []


    cdef int linenum = -1
    cdef int skipped_reads = 0
    cdef int skipped_coverage = 0

    # iterate through lines
    while True and linenum<1e9:
        
        linenum += 1 
        
        # get the line of text using c-function
        endfile = getline(&line, &lsize, cfile)
        if endfile == -1:
            break
        

        try:
            r = parseLine(line, 3)
            if r.read == NULL:
                raise IndexError()
            elif r.stop >= seqlen:
                print "Skipping line {0} with out-of-array-bounds = ({1}, {2})".format(linenum, r.start, r.stop)
                continue

            mutstr= np.zeros(seqlen, dtype=np.int8) # will contain 1 if mutated
            readstr = np.zeros(seqlen, dtype=np.int8) # will contain 1 if read 
            coverage = fillReadMutArrays(readstr, mutstr, r)
            
            if coverage >= mincoverage:
                allreadstr.append(readstr)
                allmutstr.append(mutstr)
            else:
                skipped_coverage+=1
                            
        except:
            skipped_reads +=1
            pass
    

    fclose(cfile)

    if skipped_reads>0:
        print("skipped {} non-aligned reads ".format(skipped_reads))
 
    if skipped_coverage>0:
        print("skipped {0} reads not passing {1} coverage threshold".format(skipped_coverage, mincoverage))
    
    
    if undersample > 0:
        linenum = len(allreadstr)
        
        if linenum < undersample:
            print("Dataset could not be undersampled at {0} :: using all {1} available reads".format(undersample, linenum))

        else:
            print("Undersampling {0} reads from {1} total passing quality thresholds".format(undersample, linenum))

            # this is not optimized, but should work for now
            idx = np.random.choice(linenum, undersample, replace=False)
            allreadstr = [allreadstr[i] for i in idx]
            allmutstr = [allmutstr[i] for i in idx]

    return np.array(allreadstr, dtype=np.int8), np.array(allmutstr, dtype=np.int8)




cdef int fillReadMutArrays(np.int8_t[:] readstr, np.int8_t[:] mutstr, READ r):
    """fill readstr and mutstr from Read r"""
        
    cdef int i, idx
    cdef int coverage = 0
    
    for i in xrange(r.stop - r.start + 1):
        idx = i+r.start       
        readstr[idx] = r.read[i]-r.subcode
        mutstr[idx] = r.muts[i]-r.subcode
        coverage += r.read[i]-r.subcode
            
    return coverage




##################################################################################

def compute1Dprofile(str inputFile, int seqlen, int mincoverage):
    """Compute the 1D read depth and mutation rate from the provided mutation string file"""
    
    # mem alloc for reading the file using c
    cdef FILE* cfile
    cdef size_t lsize = 0
    cdef ssize_t endfile
    cdef char* line = NULL
    
    # mem alloc for read parsing and for loops   
    cdef READ r
    cdef int i, idx, coverage
    
    cdef double[:] mutationrate = np.zeros(seqlen, dtype=np.float64) 
    cdef int[:] readdepth = np.zeros(seqlen, dtype=np.int32) 
 

    # open the file
    cfile = fopen(inputFile, "r")
    if cfile == NULL:
        raise IOError(2, "No such file or directory: '{0}'".format(inputFile))

    cdef int linenum = -1
    
    # iterate through lines
    while True and linenum<1e9:
        
        linenum += 1 
        
        # get the line of text using c-function
        endfile = getline(&line, &lsize, cfile)
        if endfile == -1:
            break
        
        r = parseLine(line, 3)
        if r.read == NULL:
            continue
        elif r.stop >= seqlen:
            print("Skipping line {0} with out-of-array-bounds = ({1}, {2})".format(linenum, r.start, r.stop))
            continue
            
        coverage = 0
        for i in xrange(r.stop - r.start + 1):
            coverage += r.read[i]-r.subcode
            
        if coverage >= mincoverage:
            for i in xrange(r.stop - r.start + 1):
                idx = i+r.start       
                readdepth[idx] += r.read[i]-r.subcode
                mutationrate[idx] += r.muts[i]-r.subcode

    fclose(cfile)

    
    for i in xrange(seqlen):
        if readdepth[i] > 0:
            mutationrate[i] /= readdepth[i]

    return np.array(mutationrate), np.array(readdepth)




##################################################################################


def loglikelihoodmatrix(double[:,::1] loglike, char[:,::1] reads, char[:,::1] mutations, int[:] activecols, 
                        double[:,::1] mu, double[:] p):
    """Compute the (natural) loglikelihoodmatrix for each read and model component"""   

    cdef int modeldim = p.shape[0]
    cdef int numreads = reads.shape[0]
    cdef int actlen = activecols.shape[0]

    cdef int d,i,j,col

    cdef double[:,::1] logMu = np.empty((modeldim, actlen), dtype=np.float64)    
    cdef double[:,::1] clogMu = np.empty((modeldim, actlen), dtype=np.float64)
    

    for d in xrange(modeldim):
        for j in xrange(actlen):
            col = activecols[j]
            logMu[d,j] = log( mu[d,col] )
            clogMu[d,j] = log( 1-mu[d,col] )

    cdef double[:] logp = np.empty(modeldim, dtype=np.float64)
    for d in xrange(modeldim):
        logp[d] = log( p[d] )
    

    for d in xrange(modeldim):
        for i in xrange(numreads):
            
            loglike[d,i] = logp[d]

            for j in xrange(actlen):
                col = activecols[j]
                if mutations[i, col]:
                    loglike[d,i] += logMu[d,j]
                elif reads[i, col]:
                    loglike[d,i] += clogMu[d,j]
    



##################################################################################

def maximizeP(double[:] p, double[:,::1] readWeights):
    """Update p parameters based on readWeights
    The 'maximization' step of the EM algorithm"""
 
    cdef int d, i
    cdef double modelweight
    cdef int numreads = readWeights.shape[1]
    
    for d in xrange(p.shape[0]):

        modelweight = 0.0
        for i in xrange(numreads):
            modelweight += readWeights[d,i]
        
        #p[d] = (modelweight + pPrior[d] -1) / (numreads + sumpPrior - modeldim)
        p[d] = modelweight/numreads




def maximizeMu(double[:,::1] mu, double[:,::1] readWeights, 
               char[:,::1] reads, char[:,::1] mutations, int[:] activecols,
               double[:,::1] priorA, double[:,::1] priorB):
    """Update mu parameters based on readWeights
    The 'maximization' step of the EM algorithm"""
    
    cdef int modeldim = mu.shape[0]
    cdef int numreads = reads.shape[0]
    cdef int actlen = activecols.shape[0]
    cdef int d, i, j, col
    
    #reset mu and add prior to the numerator
    for d in xrange(modeldim):
        for j in xrange(actlen):
            col = activecols[j]
            mu[d,col] = priorA[d,col]

    # initialize denominator with priors
    cdef double[:,::1] positionweight = np.zeros((modeldim, actlen))
    
    for d in xrange(modeldim):
        for j in xrange(actlen):
            col = activecols[j]
            positionweight[d,j] = priorA[d,col]+priorB[d,col]
    
    # accumulate numerator and denominator
    for d in xrange(modeldim):

        for i in xrange(numreads):
            for j in xrange(actlen):
                
                col = activecols[j]

                if mutations[i,col]:
                    mu[d,col] += readWeights[d,i]  
                
                if reads[i,col]:
                    positionweight[d,j] += readWeights[d,i]
    

    # do final division
    for d in xrange(modeldim):
        for j in xrange(actlen):
            col = activecols[j]
            mu[d,col] /= positionweight[d,j]



##################################################################################


def fillRINGMatrix(char[:,::1] reads, char[:,::1] mutations, char[:] activestatus,
                   double[:,::1] mu, double[:] p, int window, double assignprob): 
    """active status is array containing 0/1 whether or not column is to be included 
    posterior prob calculations"""
    

    # initialize RING matrices
    cdef int[:,:,::1] read_arr = np.zeros((p.shape[0], mu.shape[1], mu.shape[1]), dtype=np.int32)
    cdef int[:,:,::1] comut_arr = np.zeros((p.shape[0], mu.shape[1], mu.shape[1]), dtype=np.int32)
    cdef int[:,:,::1] inotj_arr = np.zeros((p.shape[0], mu.shape[1], mu.shape[1]), dtype=np.int32)


    # declare counters
    cdef int n,i,j,m

    cdef int pdim = p.shape[0]

    # setup the random number generator
    cdef dsfmt_t dsfmt
    dsfmt_init_gen_rand(&dsfmt, np.uint32(time.time()))


    # compute logp
    cdef double[:] logp = np.log(p)
    
    # compute logmu and clogmu
    cdef double[:,::1] logmu = np.zeros((mu.shape[0], mu.shape[1]))
    cdef double[:,::1] clogmu = np.zeros((mu.shape[0], mu.shape[1]))
    for i in xrange(pdim):
        for j in xrange(mu.shape[1]):
            if mu[i,j] > 0:
                logmu[i,j] = log( mu[i,j] )
                clogmu[i,j] = log( 1-mu[i,j] )
    

    # declare other needed containers
    cdef double[:] loglike = np.empty(pdim) # container for read loglike of each model
    cdef double[:] ll_i = np.empty(pdim) # container for loglike subtracting i
    cdef double[:] ll_ij = np.empty(pdim) # container for loglike subtracting i & j
    cdef double[:] weights = np.empty(pdim) # container for normalized probabilties
    
    # codes for contigency table
    cdef int icode    
    cdef int jcode 
    

    # traverse over reads
    for n in xrange(reads.shape[0]):
        
        if n%1000==0:
            printf("\r%d",n)
            fflush(stdout)

        # compute overall loglike of the read
        readloglike(loglike, activestatus, reads[n,:], mutations[n,:], logp, logmu, clogmu)
        
        
        # now iterate through all i/j pairs
        for i in xrange(read_arr.shape[1]-window+1):
            
            # compute mut code, and skip if not read at all
            icode = _computeMutCode(reads[n,:], mutations[n,:], i, window)
            if icode < 0: continue
            
            # reset ll_i
            for m in xrange(pdim):
                ll_i[m] = loglike[m]
            
            # subtract window i
            _subtractloglike(ll_i, i, window, reads[n,:], mutations[n,:], activestatus, logmu, clogmu)
            
            
            # compute weight of read ignoring i
            _loglike2prob(ll_i, weights)
            
            # increment the diagonal for keeping track of overall mutation rate
            for m in xrange(pdim):
                if (assignprob>=0 and weights[m] >= assignprob) or \
                        (assignprob<0 and weights[m] >= dsfmt_genrand_close_open(&dsfmt)):
                    read_arr[m,i,i] += 1
                    if icode==1:
                        comut_arr[m,i,i] += 1


            for j in xrange(i+1, read_arr.shape[1]-window+1):

                jcode = _computeMutCode(reads[n,:], mutations[n,:], j, window)
                if jcode < 0: continue
                
                # reset ll_ij
                for m in xrange(pdim):
                    ll_ij[m] = ll_i[m]
                
                # subtract j
                _subtractloglike(ll_ij, j, window, reads[n,:], mutations[n,:], activestatus, logmu, clogmu)
                
                # compute weight of read ignoring i & j
                _loglike2prob(ll_ij, weights) 
                

                # now iterate through models and increment RING matrices
                for m in xrange(pdim):
                    
                    # add the read
                    if (assignprob>=0 and weights[m] >= assignprob) or \
                            (assignprob<0 and weights[m] >= dsfmt_genrand_close_open(&dsfmt)):
                    
                        read_arr[m,i,j] += 1
            
                        if icode == 1 and jcode == 1:
                            comut_arr[m,i,j] += 1
                        elif icode == 1 and jcode == 0:
                            inotj_arr[m,i,j] += 1
                        elif icode == 0 and jcode == 1:
                            inotj_arr[m,j,i] += 1
    

    # reset cursor to new line
    printf("\n\n")
    fflush(stdout)

    return read_arr, comut_arr, inotj_arr                                    




cdef void _subtractloglike(double[:] loglike, int i_index, int window, 
                           char[:] read, char[:] mutation, char[:] activestatus,
                           double[:,::1] logmu, double[:,::1] clogmu):
    """Subtract window i from loglike"""
    
    cdef int p,w,index
    
    # subtract off i_index
    for p in xrange(loglike.shape[0]):
        for w in xrange(window):
            index = i_index+w
            if activestatus[index]:
                if mutation[index]:
                    loglike[p] -= logmu[p,index]
                elif read[index]:
                    loglike[p] -= clogmu[p,index]
            


cdef void _loglike2prob(double[:] loglike, double[:] prob):

    # convert loglike to weights 
    cdef double total = 0.0
    cdef int p

    for p in xrange(loglike.shape[0]):
        prob[p] = exp( loglike[p] )
        total += prob[p]

    for p in xrange(loglike.shape[0]):
        prob[p] = prob[p]/total



cdef int _computeMutCode(char[:] read, char[:] mutation, int index, int window):
    """return -1,0,1 if no-data, read but no mutation, or mutation, 
    in the window, respectively"""
    
    cdef int i
    cdef int rcounter = 0
    cdef int mcounter = 0

    for i in xrange(window):
        if read[index+i]:
            rcounter+=1
        if mutation[index+i]:
            mcounter+=1
    
    if rcounter>0 and mcounter>0:
        return 1
    elif rcounter>0:
        return 0
    else:
        return -1




cdef void readloglike(double[:] loglike, char[:] activestatus, char[:] read, char[:] mutation, 
                      double[:] logp, double[:,::1] logmu, double[:,::1] clogmu):
    """Compute the loglike of a read"""
    
    cdef int p,i
    
    for p in xrange(logp.shape[0]):
        
        loglike[p] = logp[p]

        for i in xrange(logmu.shape[1]):

            if activestatus[i]:
                if mutation[i]:
                    loglike[p] += logmu[p,i]
                elif read[i]:
                    loglike[p] += clogmu[p,i]

             


##################################################################################

def computeInformationMatrix(double[:] p, double[:,::1] mu, double[:,::1] readWeights, 
                             char[:,::1] reads, char[:,::1] mutations, int[:] activecols,
                             double[:,::1] priorA, double[:,::1] priorB):
    """Return information matrix computed based on the data
    Imatrix is symmetric square np.array of dimension (p - 1) + (p x activecols)

    Matrix is computed from the complete data likelihood.
    References:
    T. A. Louis, J. R. Statist. Soc. B (1982)
    M. J. Walsh, NUWC-NPT Technical Report 11768 (2006)
    McLachlan and Peel, Finite Mixture Models (2000)
    """
    
    # compute index values used recurrently in loops
    cdef int ppar = p.shape[0]
    cdef int ppar1 = ppar-1  
    cdef int mupar = activecols.shape[0]
    cdef int imatsize = ppar1 + p.shape[0]*mupar
    

    # initialize output information matrix
    cdef double[:,::1] Imat = np.zeros((imatsize, imatsize), dtype=np.float64)
    
    # initialize internal variables
    cdef double[:] svect = np.empty(imatsize, dtype=np.float64)
    cdef int i,j, d, d1, d2
    cdef int col, idx, idx1, idx2
    cdef double value
    cdef double curweight
    
    # iterate over all reads
    for i in xrange(reads.shape[0]):
        
        # complete p portion of svect
        for d in xrange(ppar1):
            svect[d] = readWeights[d,i] / p[d] - readWeights[ppar1,i] / p[ppar1]
        
        # compute mu portion of svect
        idx = ppar1
        for d in xrange(ppar):
            curweight = readWeights[d,i]

            for j in xrange(mupar):
                col = activecols[j]

                if mutations[i,col]:
                    svect[idx] = curweight/mu[d,col]
                elif reads[i,col]:
                    svect[idx] = -curweight/(1-mu[d,col])
                else:
                    svect[idx] = 0

                idx += 1
        

        for idx1 in xrange(imatsize):
            
            d1 = (idx1 - ppar1) / mupar

            curweight = 1-1/readWeights[d1,i]

            for idx2 in xrange(idx1, imatsize):
                
                d2 = (idx2 - ppar1) / mupar
                
                # Handle E[SiSj] terms for mu elements of imat 
                # Only applies for mu elements (d1>=0); d1==d2; and not diagonol (idx1 != idx2)
                # all other E[SiSj] terms cancel (==0)
                if idx1>=ppar1 and d1==d2 and idx1 != idx2:
                    Imat[idx1,idx2] += svect[idx1]*svect[idx2]*curweight
                # otherwise just have E[Si]E[Sj] terms
                else:
                    Imat[idx1,idx2] += svect[idx1]*svect[idx2]
                

    
    # make transpose
    for idx1 in xrange(imatsize-1):
        for idx2 in xrange(idx1+1, imatsize):
            Imat[idx2, idx1] = Imat[idx1, idx2]

    
    # Compute and add Iprior to Imat (only mu terms have prior)
    idx = ppar1
    for d in xrange(ppar):
        for j in xrange(mupar):
            col = activecols[j]
            Imat[idx,idx] += (priorA[d,col])/mu[d,col]**2 + (priorB[d,col])/(1-mu[d,col])**2
            idx+=1
    
    return np.asarray(Imat)
    




# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
