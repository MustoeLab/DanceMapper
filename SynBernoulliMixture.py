
import numpy as np
import itertools, sys

import ringmapperpath
sys.path.append(ringmapperpath.path())

from EnsembleMap import EnsembleMap
from BernoulliMixture import BernoulliMixture
from ReactivityProfile import ReactivityProfile



class SynBernoulliMixture():

    def __init__(self, p=None, mu=None, bgrate=None,
                 active_columns=None, inactive_columns=None, fname=None):
        """p is 1D array with population of each model
        mu is MxN 2D array with Bernoulli probs of each state
        """
        
        defpar = int(p is None) + int(mu is None)
            
        if defpar == 1:
            raise ValueError("Illegal to define p/mu without the other")

        elif defpar == 0:
            self.p = p
            self.mu = mu
            self.correlations = [ [] for x in self.p ]
            self.compileModel()

        else:
            self.p = []
            self.mu = []
            self.correlations = []
        
        self.bgrate = bgrate
        self.active_columns = active_columns
        self.inactive_columns = inactive_columns
    

        if fname is not None:
            self.readModelfromFile(fname)

    
    def readModelfromFile(self, fname):

        BM = BernoulliMixture()
        BM.readModelFromFile(fname)

        self.p = BM.p
        self.mu = BM.mu

        self.correlations = [ [] for x in self.p ] 

        self.active_columns = BM.active_columns
        self.inactive_columns = BM.inactive_columns


            
    def compileModel(self):
        """Convert arrays to numpy arrays and check for errors"""
        
        # check to see if it has already been converted to right type
        if not isinstance(self.mu, np.ndarray) or self.mu.dtype is np.dtype(object):
            self.mu = np.vstack(self.mu)
            self.p = np.array(self.p)
        

        # make sure all values are defined
        self.mu[np.isnan(self.mu)] = -1


        if np.abs(1-self.p.sum()) > 1e-8:
            raise AttributeError('Model populations don\'t sum to 1!')
        if self.p.size != self.mu.shape[0]:
            raise AttributeError('P and mu have inconsistent dimensions: p={0}, mu={1}'.\
                    format(self.p.size, self.mu.shape))
        

        if len(self.correlations) != len(self.p):
            raise AttributeError('Correlation array doesn\'t match model dimension')



    def addModel(self, mu, p):
        """Add model to the mixture.
        Mu can be either array/list or file
        p is its population"""

        if isinstance(mu, basestring):
            self.mu.append( self.readParFile(mu) )
        
        else:
            self.mu.append( np.array(mu) )

        self.p.append(p)

        self.correlations.append( [] )



    def readParFile(self, inpfile):
        """Read model parameters from file"""

        data = []
        with open(inpfile) as inp:
            for line in inp:
                spl = line.split()
                data.append(float(spl[-1]))

        return np.array(data)

    
    
    def addCorrelation(self, i, j, modelnum, coupling):
    
        i_marg = self.mu[modelnum, i]
        j_marg = self.mu[modelnum, j]

        
        joint = min(coupling*i_marg*j_marg, 0.5*i_marg, 0.5*j_marg)

        probarray = np.array([1 - i_marg - j_marg + joint,
                              i_marg - joint,
                              j_marg - joint, 
                              joint])
        

        if min(probarray) < 0 or not np.isclose(probarray.sum(), 1):
            print probarray
            raise ValueError('Invalid correlation parameters')


        self.correlations[modelnum].append( (i,j,probarray) ) 




    def generateReads(self, num_reads, nodata_rate = 0.0, savedata=False):
        
        # finalize the model if not done so...
        self.compileModel()       

        num_models, seqlen = self.mu.shape
        
        # array for holding model assignments
        assignments = np.random.choice(num_models, num_reads, p=self.p)
    

        # generate the mutation matrix
        # by default, things are not mutated (=0)
        muts = np.zeros((num_reads, seqlen), dtype=np.int8)

        for m in xrange(num_models):
            
            # create mask of items to select
            mask = np.zeros(muts.shape, dtype=bool)

            # set mask based on reactivity, for now treating all 
            # reads as belonging to model m
            mask[np.random.random(muts.shape) <= self.mu[m,:]] = True
            
            # deselect rows that aren't from model m
            mask[(assignments != m),] = False
            
            # set muts
            muts[mask] = 1
        

            for corr in self.correlations[m]:

                selector = np.random.choice(4, num_reads, p=corr[2])

                mask = (assignments == m) & (selector == 0)
                muts[mask, corr[0]] = 0
                muts[mask, corr[1]] = 0

                mask = (assignments == m) & (selector == 1)
                muts[mask, corr[0]] = 1
                muts[mask, corr[1]] = 0

                mask = (assignments == m) & (selector == 2)
                muts[mask, corr[0]] = 0
                muts[mask, corr[1]] = 1

                mask = (assignments == m) & (selector == 3)
                muts[mask, corr[0]] = 1
                muts[mask, corr[1]] = 1

        # generate the reads matrix; default is nts are read
        reads = np.ones((num_reads, seqlen), dtype=np.int8)
        
        # zero out data
        mask = np.random.random(reads.shape) <= nodata_rate
        reads[mask] = 0
        muts[mask] = 0
        
        if savedata:
            self.readassignments = assignments
        
        return reads, muts
    

    def filterShortRange(self, reads, muts):

        for i in range(reads.shape[0]):
            
            lastmut = reads.shape[1]+100
            for j in range(reads.shape[1]-1, -1, -1):
                if lastmut-j<5:
                    if muts[i,j]:
                        lastmut = j
                    muts[i,j] = 0
                    reads[i,j] = 0
                
                elif muts[i,j]:
                    lastmut = j
    

    
    def getEMobject(self, num_reads, nodata_rate=0.0, savedata=False, **kwargs):
        
        reads, muts = self.generateReads(num_reads, nodata_rate=nodata_rate, savedata=savedata)
        
        EM = self.constructEM(reads, muts, **kwargs)
        
        if savedata:
            self.EM = EM

        return EM



    def constructEM(self, reads, muts, **kwargs):

        EM = EnsembleMap(seqlen=self.mu.shape[1])     
        
        EM.numreads = reads.shape[0]

        EM.reads = reads
        EM.mutations = muts 
        EM.checkDataIntegrity()

        EM.sequence = 'A'*self.mu.shape[1]
        EM.profile = ReactivityProfile()
        
        mutrate = np.sum(EM.mutations, axis=0, dtype=float)
        mutrate /= np.sum(EM.reads, axis=0, dtype=float)
        
        EM.profile.rawprofile = mutrate
        
        if self.bgrate is not None:
            EM.profile.backprofile = self.bgrate
        else:
            EM.profile.backprofile = np.zeros(self.mu.shape[1])+0.0001

        # normalize with DMS false because we don't have sequence info (it doesn't matter anyways)
        EM.profile.backgroundSubtract(normalize=True, DMS=False) 


        if self.active_columns is None or self.inactive_columns is None:
            EM.initializeActiveCols(**kwargs)
        else:
            EM.setColumns(activecols=self.active_columns, inactivecols=self.inactive_columns)

        return EM

    


    def getComponentEMobjects(self):
        """Return EM objects for reads/mutations from initial assignments"""

        em_list = []       
        
        for p in range(len(self.p)):
            mask = (self.readassignments == p)
            EM = self.constructEM(self.EM.reads[mask, :], self.EM.mutations[mask, :])
            em_list.append(EM)

        return em_list



    def writeParams(self, output):
        
        sortidx = range(len(self.p))
        sortidx.sort(key=lambda x: self.p[x], reverse=True)


        with open(output, 'w') as OUT:
            
            OUT.write('# P\n')
            np.savetxt(OUT, self.p[sortidx], fmt='%.4f', newline=' ')
            OUT.write('\n# P_err\n')
            OUT.write('-1 '*len(self.p))

            OUT.write('\n\n# Mu ; bg\n')

            for i in range(self.mu.shape[1]):
                OUT.write('{} '.format(i+1))
                np.savetxt(OUT, self.mu[sortidx,i], fmt='%.4f', newline=' ')
                if self.bgrate is not None:
                    OUT.write('; {0:.4f}'.format(self.bgrate[i]))
                OUT.write('\n')
            
            
            for m in range(len(self.p)):
                OUT.write('\n# Correlations {}\n'.format(m))
                
                corrs = self.correlations[m]
                corrs.sort(key=lambda x:x[1])
                corrs.sort(key=lambda x:x[0])
                for c in corrs:
                    OUT.write('{0} {1} {2:.4f} {3:.4f} {4:.4f}\n'.format(c[0]+1, c[1]+1, c[2][3], self.mu[m,c[0]], self.mu[m,c[1]]))

    
    def returnBM(self):
        """return model as a BernoulliMixture object"""
        
        model = BernoulliMixture(pdim=self.p.shape[0], mudim=self.mu.shape[1])
        
        model.p = self.p
        model.mu = self.mu

        if self.active_columns is None:
            model.active_columns = np.arange(self.mu.shape[1])
            model.inactive_columns = np.array([])
        else:
            model.active_columns = self.active_columns
            model.inactive_columns = self.inactive_columns


        return model
        
        




