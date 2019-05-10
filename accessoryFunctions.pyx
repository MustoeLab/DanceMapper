# cython: boundscheck=False, wraparound=False, nonecheck=False


from libc.math cimport log
from libc.stdio cimport FILE, fopen, fclose, getline
from libc.stdlib cimport atoi
from libc.string cimport strsep, strcmp

import numpy as np
cimport numpy as np



###########################################################################
# Define READ object

cdef struct READ:
    int start
    int stop
    char* read
    char* muts
    char* qual


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
            r = parseLine(line)
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
    """fill readstr (invert so that 1 = no data) and mutstr from Read r"""
        
    cdef int i, idx
    cdef coverage = 0
    
    for i in xrange(r.stop - r.start + 1):
        idx = i+r.start       
        readstr[idx] = r.read[i]-48
        mutstr[idx] = r.muts[i]-48
        coverage += r.read[i]-48
            
    return coverage
    


##################################################################################

cdef READ parseLine(char* line):
    # accessory function for fillMatrices
    # uses c functions to read through line; avoids having to use python split
    # which outputs python array and slows things down

    cdef READ r
    cdef int i

    # init token values
    cdef char* running = line
    cdef char* token = strsep(&running, " \t")
    cdef int tokindex = 1
    cdef int valueset = 0
    
    cdef fileformat = 3

    # pop off leader information so that token = r.start
    for i in xrange(2):
        token = strsep(&running, " \t")
    
   
    while token:
        
        if tokindex == 1:
            r.start = atoi(token)

        elif tokindex == 2:
            r.stop = atoi(token)
        
        elif fileformat==3:
            if tokindex==3 and strcmp(token, "INCLUDED") != 0:
                break
            elif tokindex == 6:
                r.read = token
            elif tokindex == 7:
                r.muts = token
                valueset = 1
                break
        else:
            if tokindex == 3:
                r.read = token
            elif tokindex == 4:
                r.muts = token
                valueset = 1
                break
        
        token = strsep(&running, " \t")
        tokindex+=1
    

    if valueset != 1 or r.start >= r.stop:
        r.start = 0
        r.stop = 0
        r.read = NULL
    

    return r



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


 


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
