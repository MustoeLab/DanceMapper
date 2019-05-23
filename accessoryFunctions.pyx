#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True


from libc.math cimport log
from libc.stdio cimport FILE, fopen, fclose, getline

import numpy as np
cimport numpy as np

from readMutStrings cimport READ, parseLine, fillReadMut, incrementArrays


###########################################################################
#
# Fill mutation matrices
#
###########################################################################


def fillReadMatrices(str inputFile, int seqlen, int mincoverage):
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
    while True and linenum<1e12:
        
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



def maximization(double[:] p, double[:,::1] mu, double[:,::1] readWeights, 
        char[:,::1] reads, char[:,::1] mutations, int[:] activecols,
        double[:] priorA, double[:] priorB):
    """Update p and mu parameters based on readWeights
    The 'maximization' step of the EM algorithm"""
    
    cdef int modeldim = p.shape[0]
    cdef int numreads = reads.shape[0]
    cdef int actlen = activecols.shape[0]
    cdef int d, i, j, col
    
    # update p params
    cdef double modelweight
    
    for d in xrange(modeldim):

        modelweight = 0
        for i in xrange(numreads):
            modelweight += readWeights[d,i]
        
        p[d] = modelweight/numreads
    

    # update mu params 
    
    #reset mu and add prior to the numerator
    for d in xrange(modeldim):
        for j in xrange(actlen):
            col = activecols[j]
            mu[d,col] = priorA[col]-1

    # initialize denominator with priors
    cdef double[:,::1] positionweight = np.zeros((modeldim, actlen))
    
    for d in xrange(modeldim):
        for j in xrange(actlen):
            col = activecols[j]
            positionweight[d,j] = priorA[col]+priorB[col]-2
    
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


def fillRINGMatrix(int[:,::1] read_arr, int[:,::1] comut_arr, int[:,::1] inotj_arr,
                   char[:,::1] reads, char[:,::1] mutations, double[:] weights, 
                   int maxreads, int window):
    

    # handle 'no limit' on maxreads
    if maxreads < 0:
        maxreads = reads.shape[0]


    cdef int i, j, n, idx

    # initialize read/mut arrays that are filled by fillReadMut
    cdef int[:] f_read = np.zeros( reads.shape[1]+1, dtype=np.intc)
    cdef int[:] f_mut = np.zeros( reads.shape[1]+1, dtype=np.intc)

    # initialize the READ object which we'll send to fillReadMut
    cdef READ R
    R.start = 0
    R.stop = reads.shape[1] - 1
    R.subcode = 0
    

    # quick but memory inefficent way to generate random numbers
    cdef int[:] readindices = np.where(np.random.random(weights.shape[0])<=weights)[0].astype(np.intc)
    

    # subsample if exceeds maxreads
    if maxreads>0 and len(readindices) > maxreads:
        readindices = np.random.choice(readindices, maxreads, replace=False)
    

    for n in xrange(len(readindices)):
        
        idx = readindices[n]

        # update the read/muts struct to point to current read
        R.read = &reads[idx,0]
        R.muts = &mutations[idx,0]
        
        # fill f_read and f_mut, with mincoverage=0 since reads have already been filtered
        fillReadMut(f_read, f_mut, R, window, 0)
        
        # increment read_arr, comut_arr, inotj_arr
        incrementArrays(read_arr, comut_arr, inotj_arr, f_read, f_mut)


    return len(readindices)




 

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
