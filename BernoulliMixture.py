

import numpy as np
import itertools, sys, copy, time
from collections import deque
import accessoryFunctions



class ConvergenceError(Exception):
    """Exception class for convergence errors encountered during EM fitting"""
    
    def __init__(self, msg, step, badcolumns = []):
        """msg = identifying message
        step = abortion step
        p = population parameters at time of abortion
        mu = mu params at time of abortion
        """
        self.msg = msg
        self.step = step
        self.badcolumns = badcolumns


    def __str__(self, idxmap = None):
        
        msg = self.msg
        if len(self.badcolumns) == 0:
            msg += " :: aborted at step {0}".format(self.step)
        elif idxmap is None:
            msg += " at col {0} :: aborted at step {1}".format(self.badcolumns, self.step)
        else:
            ntnum = idxmap[self.badcolumns]
            msg += " at nt {0} :: aborted at step {1}".format(ntnum, self.step)
           
        return msg




class ConvergenceMonitor(object):
    """Object to monitor convergence of EM algorithm"""

    def __init__(self, activecols, convergeThresh=1e-4, maxsteps=1000, initsteps=21):

        self.step = 0
        self.lastp = None
        self.lastmu = None
        self.rms = 0
        self.rmshistory = deque()
        self.initsteps = initsteps
        self.convergeThresh = convergeThresh
        self.maxsteps = maxsteps
        self.converged = False
        self.iterate = True
        self.error = None

        self.active_columns = activecols


    def update(self, p, mu):
        
        self.step += 1

        if self.step < self.initsteps:
            self.checkConvergence(p,mu)

        elif self.step > self.maxsteps:
            self.error = ConvergenceError('Maximum iterations exceeded', self.step)
            self.iterate = False
        
        else:
            try:
                self.checkParamBounds(p, mu)
                self.checkDegenerate(p,mu)
                self.checkConvergence(p, mu)

            except ConvergenceError as e:
                self.error = e
                self.iterate = False

        
    def checkConvergence(self, p, mu):
        """Compare new params to prior params.
        If within tolerance, set converged to True
        """
        
        activemu = mu[:,self.active_columns]

        if self.lastp is None:
            self.lastp = np.copy(p)
            self.lastmu = np.copy(activemu)
            return
        

        pdiff = np.abs(p - self.lastp) 
        mdiff = np.abs(activemu - self.lastmu) 
       
        if max(np.max(pdiff), np.max(mdiff)) < self.convergeThresh:
            self.postConvergeCheck(p, mu)
            self.converged = True
            self.iterate = False
        
        self.lastp = np.copy(p)
        self.lastmu = np.copy(activemu)



    def checkParamBounds(self, p, mu):
        """Make sure p and mu params are within allowed bounds"""
        

        minp = np.min(p)
        if minp < 0.001:
            raise ConvergenceError('Low Population = {0}'.format(minp), self.step)
        
        # go through active columns and make sure params are legit
        toohi = []
        toolo = []
            
        for i in self.active_columns:
            if np.any(mu[:,i] < 5e-5):
                toolo.append(i)
            if np.any(mu[:,i] > 0.5):
                toohi.append(i)
        
        if len(toohi)>0:
            vals = [list(mu[:,i]) for i in toohi]
            raise ConvergenceError('Columns with high Mu', self.step, toohi)
        
        if len(toolo)>0:
            vals = [list(mu[:,i]) for i in toolo]
            raise ConvergenceError('Columns with low Mu', self.step, toolo)
 

    def checkDegenerate(self, p, mu):
        """Check to see if converging to degenerate parameters
        Uses trend of rmsdiff to prematurely terminate 
        """
        
        activemu = mu[:, self.active_columns]

        excludepercent = 98

        rmsdiff = 10
        for i,j in itertools.combinations(range(mu.shape[0]), 2):
            
            d = np.square(activemu[i,:] - activemu[j,:])
            excludevalue = np.percentile(d, excludepercent)
            diff = np.sqrt( np.mean( d[d<=excludevalue] ) )
            
            if diff < rmsdiff:
                rmsdiff = diff
        
        change = rmsdiff - self.rms

        self.rms = rmsdiff
        self.rmshistory.append(change)
        
        # look at trend over last 50 steps
        if len(self.rmshistory) > 50:
            self.rmshistory.popleft()

            if rmsdiff < 0.005 and np.sum(self.rmshistory)<0:
                raise ConvergenceError('Degenerate Mu: RMS diff={0:.4f}'.format(rmsdiff), self.step)



    def postConvergeCheck(self, p, mu):
        
        activemu = mu[:, self.active_columns]
        
        toolo = []
        toohi = []

        for i in self.active_columns:
            if np.any(mu[:,i] < 1e-4):
                toolo.append(i)
            if np.any(mu[:,i] > 0.5):
                toohi.append(i)
        
        if len(toolo) > 0:
            raise ConvergenceError('Low Converged Mu', self.step, toolo)
        
        if len(toohi) > 0:
            raise ConvergenceError('High Converged Mu',self.step, toohi)
        




class BernoulliMixture(object):
    """This class contains parameters and fitting methods for a Bernoulli Mixture
    The number of model components is fixed
    """

    def __init__(self, pdim = None, mudim = None, p_initial=None, mu_initial=None,
                 active_columns = None, inactive_columns = None, idxmap = None,
                 priorA=2, priorB=2, **kwargs):
        """Flexibly initialize BM object
        pdim             = dimension of the p vector -- i.e. number of model components
        mudim            = dimension of the mu vector -- i.e. number of data columns
        p_initial        = initial p parameters  (pdim array)
        mu_initial       = initial mu parameters (mudim x pdim array)
        active_columns   = list of columns to cluster
        inactive_columns = list of inactive_columns to impute
        idxmap           = mudim array of nt indices

        priorA     = int or arraylike of A parameters for the beta prior
        priorB     = int or arraylike of B parameters for the beta prior

        Note that if p_initial and/or mu_initial are provided, their dimension
                will overide components, ncols parameters
        """
        
        
        self.pdim = pdim
        self.mudim = mudim
        self.p_initial  = p_initial
        self.mu_initial = mu_initial
        
        if self.p_initial is not None:
            self.pdim = self.p_initial.size
        
        if self.mu_initial is not None:
            self.mudim = self.mu_initial.size
        
        if self.pdim is not None and self.p_initial is None:
            self.initP()
        
        if self.mudim is not None and self.mu_initial is None:
            self.initMu(**kwargs)
        
        if self.pdim is not None and self.mudim is not None:
            self.setConstantPriors(priorA, priorB)
        
        self.dynamicprior = False

        self.p = None
        self.mu = None
        self.p_err = None
        self.mu_err = None
        self.converged = False
        
        self.cError = None
        self.loglike = None
        self.BIC = None
        
        self.idxmap = idxmap

        self.active_columns = None
        self.inactive_columns = None

        if inactive_columns is not None:
            self.inactive_columns = np.array(inactive_columns, dtype=np.int32)
        
        if active_columns is not None or self.mudim is not None:
            self.set_active_columns(active_columns)

    

    def copy(self):
        """return deep copy of BM"""
        return copy.deepcopy(self)



    def set_active_columns(self, cols=None):
        """cols should be list of columns to perform clustering on
        if None, all columns are set to active
        """
        
        # reset these values
        self.converged = False
        self.loglike = None
        self.BIC = None
        
        try:
            if cols is None:
                if self.inactive_columns is not None:
                    cols = np.arange(self.mudim)
                    mask = np.isin(cols, self.inactive_columns, invert=True)
                    self.active_columns = np.array(cols[mask], dtype=np.int32)

                else:
                    self.active_columns = np.arange(self.mudim, dtype=np.int32)

            else:
                self.active_columns = np.array(cols, dtype=np.int32)    
        

        except TypeError:
            
            if cols is None and self.mudim is None:
                raise TypeError("mudim is not defined")
            else:
                raise TypeError("cols={} is not a valid argument".format(cols))

    


    def initP(self):
        """Initialize p params to equiprobable"""
        if self.pdim is None:
            raise AttributeError("pdim is not defined")

        self.p_initial = np.ones(self.pdim) / self.pdim
        self.converged = False
    

    def initMu(self, mu_lowb = 0.005, mu_upb=0.15):
        """Compute random initial starting conditions for Mu, bounded by lowb and upb"""
        
        if self.pdim is None:
            raise AttributeError("pdim is not defined")
        if self.mudim is None:
            raise AttributeError("mudim is not defined")

        #mu = np.random.random((self.pdim, self.mudim))
        #self.mu_initial = mu*(mu_upb-mu_lowb) + mu_lowb
        
        self.mu_initial = np.random.beta(1,40, (self.pdim, self.mudim))+0.001
        self.converged = False

    

    def setConstantPriors(self, priorA, priorB):
        """set the beta priors
        priorA and priorB can be int or arraylike. If arraylike, must be 1D mudim array
        """
        
        if not isinstance(priorA, (float, int)):
            priorA = np.asarray(priorA)
            if priorA.shape[-1] != self.mudim:
                raise IndexError("priorA size = {0} is not equal to mudim = {1}".format(priorA.size, self.mudim))
            if len(priorA.shape) == 1:
                priorA = priorA*np.ones((self.pdim, self.mudim))

        else:
            priorA = priorA*np.ones((self.pdim, self.mudim))

        if not isinstance(priorB, (float, int)):
            priorB = np.asarray(priorB)
            if priorB.shape[-1] != self.mudim:
                raise IndexError("priorB size = {0} is not equal to mudim = {1}".format(priorB.size, self.mudim))
            if len(priorB.shape) == 1:
                priorB = priorB*np.ones((self.pdim, self.mudim))
        else:
            priorB = priorB*np.ones((self.pdim, self.mudim))


        self.priorA = priorA
        self.priorB = priorB
        self.converged = False

    

    def setDynamicPriors(self, weight, baserate):

        if not 0<weight<=1:
            raise ValueError('DynamicPrior weight = {} is invalid, must be 0< <=1'.format(weight))

        self.dynamic_weight = weight
        self.dynamic_baserate = baserate

        self.dynamicprior = True
        



    def computePosteriorProb(self, reads, mutations):
        
        # init the weight matrix
        W = np.zeros((self.pdim, reads.shape[0]), dtype=np.float64)

        # fill the weight matrix with loglikelihood of each component
        accessoryFunctions.loglikelihoodmatrix(W, reads, mutations, self.active_columns, self.mu, self.p)
        
        # covert to probability space
        W = np.exp(W)
        
        # convert to posterior prob
        W /= W.sum(axis=0)

        return W
 


    def maximization(self, reads, mutations, W, verbal=False):
        
        accessoryFunctions.maximizeP(self.p, W)
        
        if self.dynamicprior:
            self.updateDynamicPrior( reads.shape[0], verbal=verbal )

        accessoryFunctions.maximizeMu(self.mu, W, reads, mutations, 
                                      self.active_columns, self.priorA, self.priorB)
        
    
    
    def updateDynamicPrior(self, numreads, verbal=False):
    
        totalweight = self.dynamic_weight*numreads*self.p.reshape((-1,1))
        self.priorA = totalweight*self.dynamic_baserate
        self.priorB = totalweight*(1-self.dynamic_baserate)
        
        

    def fitEM(self, reads, mutations, maxiterations = 1000, convergeThresh=1e-4, verbal=False, **kwargs):
        """Fit model to data using EM 
        
        maxiterations = maximum allowed iterations
        convergeThresh = terminate once maximum abs. change in params between iterations
                         falls below this threshold
        """
        

        # make sure parameters are initialized, etc.
        if self.p_initial is None:
            self.initP()
        
        if self.mudim is None:
            self.mudim = reads.shape[1]
        elif reads.shape[1] != self.mudim:
            raise ValueError("Reads does not have the same shape as mudim")
        
        if self.mu_initial is None:
            self.initMu()
        
        if self.active_columns is None:
            self.set_active_columns()


        numreads = reads.shape[0]
        
        # init the parameters
        self.p = np.copy( self.p_initial )
        self.mu = np.copy( self.mu_initial )


        CM = ConvergenceMonitor(self.active_columns, maxsteps=maxiterations, convergeThresh=convergeThresh)
        
        timestart = time.time()

        while CM.iterate: 
            
            # expectation step
            W = self.computePosteriorProb(reads, mutations)
            
            self.maximization(reads, mutations, W)

            # this will throw ConvergenceErrors if bad soln
            CM.update(self.p, self.mu)
            
        
        self.converged = CM.converged
        self.cError = CM.error
        
        t1 = time.time()
        # make sure information matrix is defined
        if self.converged:
            try:
                self.computeUncertainty(reads, mutations, W)
            except ConvergenceError as e:
                self.converged = False
                self.cError = e

        print '***',time.time()-t1


        # compute loglike and BIC        
        if self.converged:
            self.computeModelLikelihood(reads, mutations)

        # print outcome
        if verbal:
            if self.converged:
                print('EM converged in {0} steps ({1:.0f} seconds); BIC={2:.1f}'.format(CM.step, time.time()-timestart, self.BIC))
            else:
                print(self.cError.__str__(self.idxmap))
            
            
    
    
    def computeModelLikelihood(self, reads, mutations):
        """Compute the (natural) log-likelihood of the data given the BM model
             --> assigns self.loglike 
        Compute the BIC of the model
             --> assigns self.BIC

        returns loglike, BIC
        """

        llmat = np.zeros((self.pdim, reads.shape[0]), dtype=np.float64)
        
        accessoryFunctions.loglikelihoodmatrix(llmat, reads, mutations, self.active_columns, self.mu, self.p)
        
        # determine the likelihood of each read by summing over components
        readl = np.sum(np.exp(llmat), axis=0) 
    
        # total log-likelihood --> the product of individual read likelihoods
        self.loglike = np.sum( np.log( readl ) )

        # number of parameters
        npar = len(self.active_columns)*self.pdim + self.pdim-1
        
        # BIC = -2*ln(LL) + npar*ln(n)
        self.BIC = -2*self.loglike + npar*np.log(reads.shape[0])

        return self.loglike, self.BIC

   


    def computeUncertainty(self, reads, mutations, readWeights=None):
        """Compute the uncertainty of the model parameters from the information matrix
        
        NOTE: will raise ConvergenceError exception if information matrix is poorly defined
        """
       
        if readWeights is None:
            readWeights = self.computePosteriorProb(reads, mutations)    


        Imat = accessoryFunctions.computeInformationMatrix(self.p, self.mu, readWeights, reads, 
                                                           mutations, self.active_columns, 
                                                           self.priorA, self.priorB)
        
        np.savetxt('imat.txt', Imat)


        # compute the inverse of the information matrix
        try:
            Imat = np.linalg.inv(Imat)
        except np.linalg.linalg.LinAlgError as e:
            raise ConvergenceError('Information matrix invalid: '+str(e), 'END')

        # check to make sure matrix doesn't have negative values
        if np.min(np.diag(Imat)) < 0:
            raise ConvergenceError('Information matrix invalid: inverted matrix has negative values', 'END')

        
        # compute p errors
        p_err = np.zeros(self.p.shape)
        # convert imat to stderrs
        p1 = self.pdim-1
        p_err[:-1] = np.sqrt(np.diag(Imat[:p1,:p1]))

        # compute error for p[-1] via error propagation (p[-1] = 1 - p[0] - p[1] ...)
        a = -1* np.ones((1,p1))
        p_err[-1] = np.sqrt(np.dot( np.dot(a, Imat[:p1, :p1]), a.transpose()))
        
        # compute mu errors
        mu_err = -1*np.ones(self.mu.shape) # initialize to -1, which will be value inactive/invalid
        
        for d in range(self.pdim):
            for i, col in enumerate(self.active_columns):
                idx = p1 + d*len(self.active_columns) + i
                mu_err[d, col] = np.sqrt(Imat[idx,idx])


        self.p_err = p_err
        self.mu_err = mu_err



    def alignModel(self, BM2):
        """Align some BM2 of same dimension to current BM
        Alignment is done to minimize RMS difference between Mus
        
        returns idx, rmsdiff
        """
        
        if not np.array_equal(self.active_columns, BM2.active_columns) and \
                len(self.active_columns) < len(BM2.active_columns):
            actlist = self.active_columns
        else:
            actlist = BM2.active_columns
            #raise ValueError("active_columns of two BernoulliMixture objects are not the same")


        mindiff = 1000
        for idx in itertools.permutations(range(self.pdim)):
            
            d = self.mu - BM2.mu[idx,]

            rmsdiff = np.square(d[:, actlist])
            rmsdiff = np.sqrt( np.mean(rmsdiff) )

            if rmsdiff < mindiff:
                minidx = idx
                mindiff = rmsdiff

        return minidx, mindiff



    def modelDifference(self, BM2, func=np.max):
        """compute the difference between two BM models. 
        The difference is evaluated using func
        """
        
        if not np.array_equal(self.active_columns, BM2.active_columns) and \
                len(self.active_columns) < len(BM2.active_columns):
            actlist = self.active_columns
        else:
            actlist = BM2.active_columns
            #raise ValueError("active_columns of two BernoulliMixture objects are not the same")
        
        

        idx, rmsdiff = self.alignModel(BM2)
        
        d = np.abs(self.mu - BM2.mu[idx,])
        mudiff = func(d[:, actlist])
        
        pdiff = func(np.abs(self.p-BM2.p[idx,]))
        

        return pdiff, mudiff

    


    def _writeModelParams(self, OUT):
        """Write out the params of the model to object OUT in a semi-human readable form"""
            
        # sort model components by population
        sortidx = range(self.pdim)
        sortidx.sort(key=lambda x: self.p[x], reverse=True)


        OUT.write('# P\n')
        np.savetxt(OUT, self.p[sortidx], fmt='%.16f', newline=' ')
        
        OUT.write('\n# P_uncertainty\n')
        np.savetxt(OUT, self.p_err[sortidx], fmt='%.16f', newline=' ')


        OUT.write('\n\n# Nt Mu ; Mu_err\n')
        # write out Mu with active and inactive info
        for i in xrange(self.mudim):
            
            if self.idxmap is not None:
                OUT.write('{0} '.format(self.idxmap[i]))
            else:
                OUT.write('{0} '.format(i))


            if i not in self.inactive_columns and i not in self.active_columns:
                OUT.write('nan '*self.pdim)
            else:
                np.savetxt(OUT, self.mu[sortidx,i], fmt='%.16f', newline=' ')
                
                OUT.write('; ')
                np.savetxt(OUT, self.mu_err[sortidx, i], fmt='%.4f', newline=' ')


                if i in self.inactive_columns:
                    OUT.write('i')

            OUT.write('\n')
    
        
        OUT.write('\n# Initial P\n')
        np.savetxt(OUT, self.p_initial[sortidx], fmt='%.16f', newline=' ')
        
        # write out full initial mu without worrying about active/inactive
        OUT.write('\n\n# Initial Mu\n')
        for i in xrange(self.mudim):
            if self.idxmap is not None:
                OUT.write('{0} '.format(self.idxmap[i]))
            else:
                OUT.write('{0} '.format(i))

            np.savetxt(OUT, self.mu_initial[sortidx,i], fmt='%.16f', newline=' ')
            OUT.write('\n')
    

        OUT.write('\n# PriorA\n')
        np.savetxt(OUT, self.priorA, fmt='%.16f', newline=' ')

        OUT.write('\n\n# PriorB\n')
        np.savetxt(OUT, self.priorB, fmt='%.16f', newline=' ')



    

    def writeModel(self, output):
        """Wrapper function for _writeModelParams to handle different types of output

        output can be:
            None   --> write to stdout
            Path   --> write to this path
            Obj    --> write to this file object
        """
        
    
        if output is None:
            self._writeModelParams(sys.stdout)
        elif hasattr(output, 'write'):
            self._writeModelParams(output)
        else:
            with open(output, 'w') as OUT:
                self._writeModelParams(OUT)



    def readModelFromFile(self, fname):
        """Read in BM model from file"""
            

        with open(fname) as inp:
            # pop off p header
            inp.readline()
            self.p = np.array(inp.readline().split(), dtype=float)
            self.pdim = len(self.p)
            


            # pop off mu header
            inp.readline()
            inp.readline()

            actives = []
            inactives = []
            mu = []
            idxmap = []
            
            i = -1
            read = True
            while read:
                
                i += 1
                spl = inp.readline().split()
                if len(spl) == 0 or spl[0][0]=='#':
                    break

                idxmap.append(int(spl[0]))
                
                vals = map(float, spl[1:1+self.pdim])
                
                if vals[0] != vals[0]:
                    mu.append([-999]*self.pdim)

                elif vals[0] == vals[0] and spl[-1] == 'i':
                    inactives.append(i)
                    mu.append(vals)
                else:
                    actives.append(i)
                    mu.append(vals)


            self.idxmap = np.array(idxmap)
            self.active_columns = np.array(actives, dtype=np.int32)
            self.inactive_columns = np.array(inactives, dtype=np.int32)

            mu = np.array(mu)
            self.mu = np.array(mu.transpose(), order='C')

            self.mudim = self.mu.shape[1]
            
            inp.readline()
            self.p_initial = np.array(inp.readline().split(), dtype=float)
            
            inp.readline()
            inp.readline()
        
            self.mu_initial = -1*np.ones(self.mu.shape)
            
            i = -1
            read = True

            while read:
                i += 1
                spl = inp.readline().split()
                if len(spl)==0 or spl[0][0]=='#':
                    break

                self.mu_initial[:, i] = map(float, spl[1:])

            inp.readline()
            self.priorA = np.array(inp.readline().split(), dtype=float)

            inp.readline()
            inp.readline()
            self.priorB = np.array(inp.readline().split(), dtype=float)

        

    def imputeInactiveParams(self, reads, mutations):
        """ Impute inactive Mu parameters using parameters of the base Bernoulli Mixture
        to assign component weights to each model.
        """
        
        if self.inactive_columns is None or len(self.inactive_columns)==0:
            return


        newp = np.copy(self.p)
        
        combined_columns = np.append(self.active_columns, self.inactive_columns)


        # get initial posterior probabilities
        W = self.computePosteriorProb(reads, mutations)
        
        for t in xrange(5):
            
            # update posterior probs based on inactive_column info
            if t > 0:
                accessoryFunctions.loglikelihoodmatrix(W, reads, mutations, combined_columns, self.mu, self.p)
                W = np.exp(W)
                W /= W.sum(axis=0)

            accessoryFunctions.maximizeP(newp, W) 
            accessoryFunctions.maximizeMu(self.mu, W, reads, mutations, self.inactive_columns, self.priorA, self.priorB)
        


        # look at whether the populations are shifting;
        # if they shift too much, there is a problem...
        pdiff = np.max(np.abs(newp - self.p))
        if pdiff > 0.01:
            raise ConvergenceError('Large P shift during inactive column imputation = {0}'.format(pdiff),'term')

        


    def refit_new_active(self, reads, mutations, **kwargs):
        """Refit a coverged BM with different active_columns
        Use prior converged p/mu parameters as a starting point
        """
        
        # set initial params to prior guess
        self.p_initial = np.copy(self.p)
        self.mu_initial = np.copy(self.mu)
        
        self.converged = False

        self.fitEM(reads, mutations, **kwargs)
        
        


    




# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
