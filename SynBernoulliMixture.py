
import numpy as np
import itertools, sys

import ringmapperpath
sys.path.append(ringmapperpath.path())

from EnsembleMap import EnsembleMap
from ReactivityProfile import ReactivityProfile



class SynBernoulliMixture():
    """Class for synthetic Bernoulli Mixture dataset
    Inherits generic BernoulliMixture
    """

    def __init__(self, p=None, mu=None):
        """p is 1D array with population of each model
        mu is MxN 2D array with Bernoulli probs of each state
        """
            
        defpar = int(p is None) + int(mu is None)
            
        if defpar == 1:
            raise ValueError("Illegal to define p/mu without the other")

        elif defpar == 0:
            self.p = p
            self.mu = mu
            self.compileModel()

        else:
            self.p = []
            self.mu = []
    
    
    def compileModel(self):
        """Convert arrays to numpy arrays and check for errors"""
        
        # check to see if it has already been converted to right type
        if not isinstance(self.mu, np.ndarray) or self.mu.dtype is np.dtype(object):
            self.mu = np.vstack(self.mu)
            self.p = np.array(self.p)
        
        if self.p.sum() != 1:
            raise AttributeError('Model populations don\'t sum to 1!')
        if self.p.size != self.mu.shape[0]:
            raise AttributeError('P and mu have inconsistent dimensions: p={0}, mu={1}'.format(self.p.size, self.mu.shape))



    def addModel(self, mu, p):
        """Add model to the mixture.
        Mu can be either array/list or file
        p is its population"""

        if isinstance(mu, basestring):
            self.mu.append( self.readParFile(mu) )
        
        else:
            self.mu.append( np.array(mu) )

        self.p.append(p)


    def readParFile(self, inpfile):
        """Read model parameters from file"""

        data = []
        with open(inpfile) as inp:
            for line in inp:
                spl = line.split()
                data.append(float(spl[-1]))

        return np.array(data)

    

    def generateReads(self, num_reads, nodata_rate = 0.05):
        
        # finalize the model if not done so...
        self.compileModel()       

        num_models, seqlen = self.mu.shape
        
        # array for holding model assignments
        assignments = np.random.choice(num_models, num_reads, p=self.p)
    

        # generate the mutation matrix
        # by default, things are not mutated (=0)
        muts = np.zeros((num_reads, seqlen), dtype=np.int8)

        for i in xrange(num_models):
            
            # create mask of items to select
            mask = np.ones(muts.shape, dtype=bool)

            # only select rows with desired assignment
            mask[(assignments != i),] = False

            # generate random matrix; deselect positions no meeting cutoff
            mask[np.random.random(muts.shape) > self.mu[i,:]] = False
            
            # positions that remain True in mask are mutated
            muts[mask] = 1
        

        # generate the reads matrix; default is nts are read
        reads = np.ones((num_reads, seqlen), dtype=np.int8)
        
        # zero out data
        reads[np.random.random(reads.shape) < nodata_rate] = 0

        
        return reads, mutations

        


    def generateEMobject(self, num_reads, bgrates, nodata_rate=0.05, **kwargs):


        EM = EnsembleMap(seqlen=self.mu.shape[1])     
        
        EM.numreads = num_reads
        EM.reads, EM.mutations = self.generateReads(num_reads, nodata_rate=nodata_rate)

        EM.profile = ReactivityProfile()
        
        mutrate = np.sum(EM.mutations, axis=0, dtype=float)
        mutrate /= np.sum(EM.reads, axis=0, dtype=float)

        EM.profile.rawprofile = mutrate
        
        EM.profile.backprofile = bgrates

        EM.profile.backgroundSubtract(normalize=True, DMS=True)

        EM.initializeActiveCols(**kwargs)

        return EM

   

    def simulateCorrelations(self, modelnum, corrlist):
        """Add in correlations to synthetic reads originating from model=modelnum
        corrlist is list of (i,j,alpha) tuples, where i and j are nts (1-based numbering)
        and alpha is strength of correlation:
            pij = alpha*pi*pj
        """

        asgn_mask = (self.read_assignments == modelnum)

        for n1,n2,alpha in corrlist:
            
            n1 -= 1
            n2 -= 1

            # compute the joint probability (independent muts)
            prob = (alpha-1)*self.mu[modelnum,n1]
            
            # select positions that should be mutated
            rmask = (np.random.random(len(asgn_mask)) < prob)
            # build total mask
            totmask = asgn_mask & rmask & self.reads[:,n2]
            
            # update the reads
            self.reads[totmask, n1] = 1




