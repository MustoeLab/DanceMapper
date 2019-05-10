
# branched from emtool3 to have new class structure


import numpy as np
import itertools, sys, copy
from collections import deque
import time
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
        # 10, 10000
        """Flexibly initialize BM object
        pdim             = dimension of the p vector -- i.e. number of model components
        mudim            = dimension of the mu vector -- i.e. number of data columns to fit
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
        

        self.setPriors(priorA, priorB)

        self.p = None
        self.mu = None
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

        mu = np.random.random((self.pdim, self.mudim))
        self.mu_initial = mu*(mu_upb-mu_lowb) + mu_lowb
        self.converged = False

    
    def setPriors(self, priorA, priorB):
        """set the beta priors
        priorA and priorB can be int or arraylike. If arraylike, must be 1D mudim array
        """
        
        if not isinstance(priorA, (float, int)):
            priorA = np.asarray(priorA)
            if priorA.size != self.mudim:
                raise IndexError("priorA size = {0} is not equal to mudim = {1}".format(priorA.size, self.mudim))
        
        else:
            priorA = priorA*np.ones(self.mudim)

        if not isinstance(priorB, (float, int)):
            priorB = np.asarray(priorB)
            if priorB.size != self.mudim:
                raise IndexError("priorB size = {0} is not equal to mudim = {1}".format(priorB.size, self.mudim))
        
        else:
            priorB = priorB*np.ones(self.mudim)


        self.priorA = priorA
        self.priorB = priorB
        self.converged = False

    

    def computePosteriorProb(self, reads, mutations):
        
        W = np.zeros((self.pdim, reads.shape[0]), dtype=np.float64)

        accessoryFunctions.loglikelihoodmatrix(W, reads, mutations, self.active_columns, self.mu, self.p)
        
        W = np.exp(W)

        W /= W.sum(axis=0)

        return W
 


    def maximization(self, reads, mutations, W):
        
        accessoryFunctions.maximization(self.p, self.mu, W, reads, mutations, self.active_columns, self.priorA, self.priorB)
        



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
        
        totaltime = []

        while CM.iterate: 
            
            # expectation step
            t = time.time()
            W = self.computePosteriorProb(reads, mutations)
            
            self.maximization(reads, mutations, W)

            # this will throw ConvergenceErrors if bad soln
            CM.update(self.p, self.mu)
            
            totaltime.append(time.time()-t)

        
        #print 'Total mean={0:.1f} std={1:.1f}'.format(np.mean(totaltime), np.std(totaltime))


        self.converged = CM.converged
        self.cError = CM.error
    
        # compute loglike and BIC        
        if self.converged:
            self.computeModelLikelihood(reads, mutations)

        # print outcome
        if verbal:
            if CM.converged:
                print 'EM converged in {0} steps'.format(CM.step)
            else:
                print self.cError.__str__(self.idxmap)

    
    
    
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
        npar = self.mudim*self.pdim + self.pdim-1
        
        # BIC = -2*ln(LL) + npar*ln(n)
        self.BIC = -2*self.loglike + npar*np.log(reads.shape[0])

        return self.loglike, self.BIC



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
        
        OUT.write('# P\n')
        np.savetxt(OUT, self.p, fmt='%.16f', newline=' ')
        
        OUT.write('\n\n# Mu\n')
        
        # write out Mu with active and inactive info
        for i in xrange(self.mudim):
            
            if self.idxmap is not None:
                OUT.write('{0} '.format(self.idxmap[i]))
            else:
                OUT.write('{0} '.format(i))


            if i not in self.inactive_columns and i not in self.active_columns:
                OUT.write('nan '*self.pdim)
            else:
                np.savetxt(OUT, self.mu[:,i], fmt='%.16f', newline=' ')
                
                if i in self.inactive_columns:
                    OUT.write('i')

            OUT.write('\n')
    
        
        OUT.write('\n# Initial P\n')
        np.savetxt(OUT, self.p_initial, fmt='%.16f', newline=' ')
        
        # write out full initial mu without worrying about active/inactive
        OUT.write('\n\n# Initial Mu\n')
        for i in xrange(self.mudim):
            if self.idxmap is not None:
                OUT.write('{0} '.format(self.idxmap[i]))
            else:
                OUT.write('{0} '.format(i))

            np.savetxt(OUT, self.mu_initial[:,i], fmt='%.16f', newline=' ')
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


        
    

    def imputeInactiveParams(self, reads, mutations):
        """ Impute inactive Mu parameters using parameters of the base Bernoulli Mixture
        to assign component weights to each model.
        """
        
        if self.inactive_columns is None or len(self.inactive_columns)==0:
            return


        originalp = np.copy(self.p)
        
        combined_columns = np.append(self.active_columns, self.inactive_columns)


        # get initial posterior probabilities
        W = self.computePosteriorProb(reads, mutations)
        
        for t in xrange(5):
            
            # update posterior probs based on inactive_column info
            if t > 0:
                accessoryFunctions.loglikelihoodmatrix(W, reads, mutations, combined_columns, self.mu, self.p)
                W = np.exp(W)
                W /= W.sum(axis=0)


            accessoryFunctions.maximization(self.p, self.mu, W, reads, mutations, self.inactive_columns, self.priorA, self.priorB)
        


        # look at whether the populations are shifting;
        # if they shift too much, there is a problem...
        pdiff = np.max(np.abs(self.p - originalp))
        if pdiff > 0.01:
            raise ConvergenceError('Large P shift during inactive column imputation = {0}'.format(popdiff),'term')

        


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
