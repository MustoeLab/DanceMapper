
import numpy as np
import itertools
import time
from collections import deque


class ConvergenceError(Exception):
    """Exception class for convergence errors encountered during EM fitting"""
    
    def __init__(self, msg, step, p, mu):
        """msg = identifying message
        step = abortion step
        p = population parameters at time of abortion
        mu = mu params at time of abortion
        """
        self.msg = msg
        self.step = step
        self.p = p
        self.mu = mu

    def __str__(self):
        return "{0} :: aborted at step {1}".format(self.msg, self.step)
        

class ConvergenceMonitor(object):
    """Object to monitor convergence of EM algorithm"""

    def __init__(self, convergeThresh=1e-4, maxsteps=1000, initsteps=21):

        self.step = 0
        self.p = None
        self.mu = None
        self.rms = 0
        self.rmshistory = deque()
        self.initsteps = initsteps
        self.convergeThresh = convergeThresh
        self.maxsteps = maxsteps
        self.converged = False

    def update(self, p, mu):
        
        self.step += 1
        if self.step < self.initsteps:
            self.checkConvergence(p,mu)
            return

        elif self.step > self.maxsteps:
            raise ConvergenceError('Maximum iterations exceeded', self.step, p, mu)
        
        else:
            self.checkParamBounds(p, mu)
            self.checkDegenerate(p,mu)
            self.checkConvergence(p, mu)
        
        
    def checkConvergence(self, p, mu):
        """Compare new params to prior params.
        If within tolerance, set converged to True
        """

        try:
            pd = np.abs(p - self.p) 
            md = np.abs(mu - self.mu) 
        except TypeError: # init the parameters 
            self.p = np.copy(p)
            self.mu = np.copy(mu)
            return
        
        
        if max(np.max(pd), np.max(md)) < self.convergeThresh:
            self.converged=True
        
        self.p[:] = p[:]
        self.mu[:] = mu[:]


    def checkParamBounds(self, p, mu):
        """Make sure p and mu params are within allowed bounds"""
        minp = np.min(p)
        if minp < 0.001:
            raise ConvergenceError('Low Population = {0}'.format(minp),self.step,p,mu)


        minmu = np.min(mu)
        if minmu < 1e-5:
            raise ConvergenceError('Low Mu = {0}'.format(minmu),self.step,p,mu)

        maxmu = np.max(mu)
        if maxmu > 0.3:
            raise ConvergenceError('High Mu = {0}'.format(maxmu),self.step,p,mu)


    def checkDegenerate(self, p, mu):
        """Check to see if converging to degenerate parameters
        Uses trend of rmsdiff to prematurely terminate 
        """

        rmsdiff = 10
        for i,j in itertools.combinations(range(mu.shape[0]), 2):
            d = (mu[i,:] - mu[j,:])**2
            # ignore 5 nts with highest diff
            p95 = np.percentile(d, 100*(mu.shape[1]-5.0)/mu.shape[1])
            diff = RMS(d[d<=p95])
            if diff < rmsdiff:
                rmsdiff = diff
        
        change = rmsdiff - self.rms

        self.rms = rmsdiff
        self.rmshistory.append(change)
        if len(self.rmshistory) > 50:
            self.rmshistory.popleft()
            
            if rmsdiff < 0.001 and np.sum(self.rmshistory)<0:
                #print p95, np.sum(d<=p95), len(d)

                raise ConvergenceError('Degenerate Mu: RMS diff={0:.4f}'.format(rmsdiff),self.step,p,mu)

        


class BernoulliMixture(object):

    def __init__(self, inpfile=None, bgfile=None, **kwargs):
        """Define important global parameters"""
    
        # primary data containing 'active' columns 
        # reads contains mutations, ireads is inverse (accounting for no-data)
        self.reads = None 
        self.ireads = None
        self.active_mask = None
        
        # contains inactive columns
        self.inactive_reads = None
        self.inactive_ireads = None
        self.inactive_mask = None
        
        # info about the reads
        self.columnindices = None
        self.seqlen = None
        self.numreads = None
        
        # parameter storage -- these have full dim
        self.p = None
        self.mu = None

        if inpfile is not None:
            self.readMatrix(inpfile, **kwargs)


 
    def readMatrix(self, fname, maxmissing=3, ignoreCols=[], **kwargs):
        """Read in primary data matrix and perfrom filtering
        Indices in ignoreCols are deleted from matrix.
        kwargs passed on to setActiveNts(minsig, bgarray, maxbg, minsig_bg)
        """
        
        rawreads = np.loadtxt(fname, dtype=np.int8)
        
        # delete unwanted columns
        ignoremask = np.ones(rawreads.shape[1], dtype=np.bool_)
        for x in ignoreCols:
            ignoremask[x] = False
        
        rawreads = rawreads[:,ignoremask]
        
        # now identify positions with no data and filter out incomplete reads
        nodata = (rawreads > 1)
        rowmask = (np.sum(nodata, axis=1) <= maxmissing)
        nodata = nodata[rowmask,:]
        
        # create reads and ireads arrays
        self.reads = np.array(rawreads[rowmask,:], dtype=np.bool_)
        self.reads[nodata] = False
        self.ireads = np.invert(self.reads)
        self.ireads[nodata] = False
        
        self.numreads = self.reads.shape[0]
        self.seqlen = self.reads.shape[1]
        self.active_mask = np.ones(self.seqlen, dtype=np.bool_)

        self.columnindices = np.arange(ignoremask.size)[ignoremask]
        
        # manually garbage collect big arrays just in case
        del nodata
        del rawreads
        
        self.setActiveNts(**kwargs)


    def setActiveNts(self, minsig=0.002, bgarray=None, maxbg=0.01, minsig_bg=0.002, **kwargs):
        """Apply quality filters to eliminate noisy nts from EM fitting
        minsig    = minimum signal (react. rate)
        bgarray   = array (or txt file of array) of background signal
        maxbg     = maximum allowable background signal
        minsig_bg = minimum signal above background
        """
        
        # comute reactivity rate
        sig = np.sum(self.reads, axis=0, dtype=np.float)
        sig = sig/(sig + np.sum(self.ireads, axis=0))
        
        active = sig >= minsig
        
        if bgarray is not None:
            if isinstance(bgarray, str):
                bgarray = np.loadtxt(bgarray)
            
            bg = bgarray[self.columnindices]
            
            active = active & (bg < maxbg)
            active = active & ( (sig-bg) >= minsig_bg)
        

        inactive = np.invert(active)

        if np.sum(active) != len(active):
            print 'Inactive nts :: {0}'.format(self.columnindices[inactive]+1)

        self.inactive_reads = self.reads[:,inactive]
        self.inactive_ireads = self.ireads[:,inactive]
        self.inactive_mask = inactive
        self.reads = self.reads[:,active]
        self.ireads = self.ireads[:,active]
        self.active_mask = active


    def mu2log(self, mu):
        """Compute log2 matrices for numerically safe probability calculations"""
        return np.log2(mu), np.log2(1-mu)


    def computeLogLike(self, p, mu):
        """Return the (natural) log-likelihood of the data given the Bernoulli mixture model"""

        logMu, logMuC = self.mu2log(mu)
        
        ll = np.zeros((self.numreads, p.size))

        for i in xrange(p.size):
            ll[:,i] += np.sum(logMu[i,:]*self.reads, axis=1) 
            ll[:,i] += np.sum(logMuC[i,:]*self.ireads, axis=1)
            ll[:,i] += np.log2(p[i])
    
        return np.sum( np.log( np.sum(np.exp2(ll), axis=1) ) )



    def computePosteriorProb(self, p, mu):
        """Compute Posterior Prob. of component membership for each read"""

        logMu, logMuC = self.mu2log(mu)

        W = np.zeros((self.numreads, p.size))

        for i in xrange(p.size):
            W[:,i] += np.sum(logMu[i,:]*self.reads, axis=1) 
            W[:,i] += np.sum(logMuC[i,:]*self.ireads, axis=1)
            W[:,i] += np.log2(p[i])
    
        W = np.exp2(W)
        W /= W.sum(axis=1).reshape((self.numreads, 1)) # note this could be numerically unstable
     
        return W
    


    def MAPClassification(self):

        W = self.computePosteriorProb(self.p, self.mu[:,self.active_mask])
    
        assignments = np.ones(self.numreads, dtype=np.int8)*-1
    
        for i in xrange(self.components):
            mask = (np.max(W, axis=1) == W[:,i])
            assignments[mask] = i
    
        assert np.sum(assignments<0) == 0

        return assignments


    def stochasticClassification(self):

        W = computePosteriorProb(self.p, self.mu[:, self.active_mask])
        
        cumW = np.zeros((W.shape[0], W.shape[1]+1)) 
        cumW[:,1:] = np.cumsum(W, axis=1)

        r = np.random.random(self.numreads)
        assignments = np.ones(self.numreads, dtype=np.int8)*-1
    
        for i in xrange(len(p)):
            mask = (cumW[:,i] <= r) & (r < cumW[:,i+1])
            assignments[mask] = i
        
        assert np.sum(assignments<0) == 0

        return assignments




    def fitEM(self, p_init, mu_init, maxiterations = 1000,
              convergeThresh=1e-4, a=None,b=None, verbal=False, **kwargs):
        """Fit model to data using EM 
        p_init = initial populations of each mixture model
                 Format is 1D float array, Dim = num_models
        mu_init = initial Bernoulli probabilities of each model
                  Format is 2D numpy array, Dim = (num_models, activents)
        
        maxiterations = maximum allowed iterations; if exceeded, raise ConvergenceError
        convergeThresh = terminate once maximum abs. change in params between iterations
                         falls below this threshold
        
        a = alpha parameter of beta prior on mu
        b = beta parameter of beta prior on mu
        """
        

        p_em = np.copy(p_init)
        mu_em = np.copy(mu_init)


        # place weak priors to guard against 0 counts that mess up probability calculations
        if a is None:
            a_arr = 2*np.ones(mu_em.shape)
        elif isinstance(a, (float, int)):
            a_arr = a*np.ones(mu_em.shape)
        else:
            a_arr = np.copy(a)

        if b is None:
            b_arr = 2*np.ones(mu_em.shape)
        elif isinstance(b, (float, int)):
            b_arr = b*np.ones(mu_em.shape)
        else:
            b_arr = np.copy(b)


        CM = ConvergenceMonitor(maxsteps=maxiterations, convergeThresh=convergeThresh)
        
        while not CM.converged: 

            # expectation step
            W = self.computePosteriorProb( p_em, mu_em)
        
            # M-step
            for i in xrange(p_em.size):
                ni = np.sum(W[:,i])
                p_em[i] = ni/self.numreads
                
                mu_em[i,:] = np.sum( self.reads * W[:,i].reshape((self.numreads, 1)) , axis=0) 
                mu_em[i,:] += a_arr[i,:]-1 # numerator contribution of prior
                mu_em[i,:] /= ni + a_arr[i,:] + b_arr[i,:] - 2  
        
            # compute updated reactivity profiles, accounting for missing data
            # Note :: this step is slow! contributes Nmod*(3*Nread*Seqlen+2*Nread) 
            #for i in xrange(num_models):
            #    Z_elem = X*Z[:,i].reshape((num_reads,1)) + cX*Z[:,i].reshape((num_reads,1))
            #    mu_em[i,:] = np.sum(X*Z_elem, axis=0) / np.sum(Z_elem, axis=0)
            
            # this will throw ConvergenceErrors if bad soln
            CM.update(p_em, mu_em)

        
        if verbal:
            print 'EM converged in {0} steps'.format(CM.step)

        return p_em, mu_em

    
    def imputeInactiveParams(self):
        """ Impute parameters of inactive columns
        self.p and self.mu must be defined"""
        
        if np.sum(self.inactive_mask) == 0:
            return

        logMu, logMuC = self.mu2log(self.mu[:,self.active_mask])
        Wactive = np.zeros((self.numreads, self.components))
        for i in xrange(self.components):
            Wactive[:,i] += np.sum(logMu[i,:]*self.reads, axis=1) 
            Wactive[:,i] += np.sum(logMuC[i,:]*self.ireads, axis=1)
            Wactive[:,i] += np.log2(self.p[i])
        
        
        W = np.empty(Wactive.shape)
        inactmu = np.zeros((self.components, self.inactive_reads.shape[1]))
        logMu = logMuC = np.zeros(inactmu.shape)
        newp = np.zeros(self.components)
        
        for t in xrange(10):
            
            if t>0: logMu, logMuC = self.mu2log(inactmu)
            
            # compute weights (for t=0. logM=logMuC=0, so W=Wactive)
            W[:] = Wactive 
            for i in xrange(self.components):
                W[:,i] += np.sum(logMu[i,:]*self.inactive_reads, axis=1) 
                W[:,i] += np.sum(logMuC[i,:]*self.inactive_ireads, axis=1)
            W = np.exp2(W)
            W /= W.sum(axis=1).reshape((self.numreads, 1))
            
            # update Mu pars
            for i in xrange(self.components):
                ni = np.sum(W[:,i])
                inactmu[i,:] = np.sum(self.inactive_reads*W[:,i].reshape((self.numreads, 1)), axis=0)
                inactmu[i,:] /= ni
                newp[i] = ni/self.numreads 
        
        popdiff = np.max(newp - self.p)
        if popdiff > 0.01:
            raise ConvergenceError('Significant Inactive P shift = {0}'.format(popdiff),'term', self.p, self.mu)
        
        self.mu[:, self.inactive_mask] = inactmu



    def random_init_params(self, components, mu_lowb = 0.005, mu_upb=0.15):
        """Compute initial starting conditions
        Components are assigned as equiprobable
        Mu randomly assigned, bounded by lowb and upb"""

        p = np.ones(components)/components
        mu = np.random.random((components, self.reads.shape[1]))*(mu_upb-mu_lowb) + mu_lowb

        return p, mu



    def bestFitEM(self, components, trials=5, termcount=3, verbal=False, assign=False, **kwargs):
        """Perform a EM-fitting from random initial conditions with error handling
        and selection of best fit

        components = number of model components to fit
        trials = number of fitting trials to run
        verbal = T/F on whether to print results of each trial
        
        additional kwargs are passed onto fitEM
        
        return bestFit, fitSummary if valid solution found
            bestFit = [ll, p, mu] of best model
            fitSummary = [nsolns, mean(ll), meanRMS(p), meanRMS(mu)]
            
        return None, None if no valid solution found
        """
        

        suboptimal = []
        minfit = None         
        mincount = 1

        for t in xrange(trials):
            
            if verbal:
                print 'Fitting {0} component model -- Trial {1}'.format(components, t+1)

            p_i, mu_i = self.random_init_params(components)

            try:
                p_em, mu_em = self.fitEM(p_i, mu_i, verbal=verbal, **kwargs)

            except ConvergenceError as e:
                if verbal:
                    print e
                continue

            fit = (self.computeLogLike(p_em, mu_em), p_em, mu_em)

            if verbal:
                print 'Trial {0} successful. ll={1:.0f}, p={2}'.format(t+1, fit[0], printParamLine(fit[1]))
        

            # add to appropriate containers
            if minfit is None:
                minfit = fit
                continue
            elif fit[0]>minfit[0]:
                suboptimal.append(minfit)
                minfit = fit
            else:
                suboptimal.append(fit)
            

            # determine if solution has been found previosuly;
            # terminate if sufficient counts
            diff = compareModels(suboptimal[-1][1], suboptimal[-1][2], minfit[1], minfit[2])
            if max(diff) < 0.001:
                mincount += 1
                if mincount == termcount:
                    break
            elif fit == minfit:
                mincount = 1


        if minfit is None:
            if verbal:
                print '**No fit found for {0}-component model**'.format(components)
            return None, None
        

        if assign:
            self.setModelComponents(components)
            self.p = minfit[1]
            self.mu[:,self.active_mask] = minfit[2]
            self.imputeInactiveParams()


        if len(suboptimal) > 0:
            s = [[], [], []]
            for m in suboptimal:
                pd,md = compareModels(m[1], m[2], minfit[1], minfit[2], func=RMS)
                s[0].append(m[0])
                s[1].append(pd)
                s[2].append(md)
        
            subsum = [len(suboptimal)+1] + [np.mean(x) for x in s]
        else:
            subsum = [1,0,0,0]
        
        
        return minfit, subsum



    def setModelComponents(self, components):
        """Initiliaze components, p, and mu"""
        self.components = components
        self.p = -1*np.ones(components)
        self.mu = -1*np.ones((components, self.seqlen))


    def computeNullModel(self):
        """Compute the Log likelihood for the null model (i.e. mixture of one)"""

        # compute 1D model
        m1 = np.sum(self.reads, axis=0, dtype=np.float_)
        mu = m1/self.numreads  #(m1 + np.sum(self.ireads, axis=0))
        mu = mu.reshape((1, self.reads.shape[1]))

        p = np.array([1.0])
        ll = self.computeLogLike(p, mu)

        return p, mu, ll



    def calc_model_BIC(self, loglike, components):
        """Compute the BIC for approximate model selection
        loglike = expected as the natural logarithm of the likelihood
        components = number of model components
        """
        npar = self.reads.shape[1]*components + components-1
        return -2*loglike + npar*np.log(self.numreads)



    def fitModel(self, maxcomponents=5, verbal=False, **kwargs):
        """Perform fits for different numbers of model components and select best model via BIC"""

        # compute 1-component model
        p,mu,ll = self.computeNullModel()
        bic = self.calc_model_BIC(ll, 1)
        fitlist = [(1, bic, p, mu)]

        if verbal:
            print "1-component BIC={0:.1f}".format(bic)
            print '*'*10

        
        for c in xrange(2,maxcomponents+1):
            
            fit, sumstats = self.bestFitEM(c, verbal=verbal, **kwargs)
            
            if fit is None: # no successful fit found
                fitlist.append((None, None, None, None))
                if verbal: print '*'*10
                break

            bic = self.calc_model_BIC(fit[0], c)
        
            if verbal:
                print "{0} fits found for {1}-component model".format(sumstats[0], c)
                print "Mean RMS error of suboptimal fits from best parameters:"
                print "\tp = {0:.3f}\n\tmu = {1:.3f}".format(sumstats[2], sumstats[3])
                print "Best Fit BIC = {0:.1f}".format(bic)
                #printParams(fit[1], fit[2], 
                #            idx = self.columnindices[self.active_mask]+1)
                print '*'*10
            
            fitlist.append((c, bic, fit[1], fit[2]))
            
            # terminate if model does not have better bic than lower dim model
            if bic-fitlist[-2][1] > -10:
                break
        
        if c==maxcomponents and bic-fitlist[-2][1]<=-10:
            fitlist.append(())

        if verbal:
            print '{0}-component model selected'.format(fitlist[-2][0])
    

        self.setModelComponents(fitlist[-2][0])
        self.p = fitlist[-2][2]
        self.mu[:,self.active_mask] = fitlist[-2][3]
        self.imputeInactiveParams()


    def compute_IM_Error(self):
        """
        Estimate error of MLE parameters for Bernoulli Mixture model from
        the observed information matrix
        Note -- only determines errors for active params

        Matrix is computed from the complete data likelihood.
        References:
        T. A. Louis, J. R. Statist. Soc. B (1982)
        M. J. Walsh, NUWC-NPT Technical Report 11768 (2006)
        McLachlan and Peel, Finite Mixture Models (2000)
        """
        
        activemu = self.mu[:,self.active_mask]

        W = self.computePosteriorProb(self.p, activemu)

        # compute size of I matrix
        seqdim = activemu.shape[1]
        pdim = self.components-1
        idim = pdim + seqdim*self.components
    
        Imat = np.zeros((idim, idim))
        eS = np.zeros(idim)
        eB = np.zeros(idim)
        diagidx = np.diag_indices(idim)   
        

        #iterate through the reads
        for i in xrange(self.numreads):
        
            # compute expectation of gradient
            # compute gradient for p variables (note last p is not included, since 
            # it can be expressed as 1-sum of other ps)
            ip = W[i,:]/self.p
            ip -= ip[-1]
            eS[:pdim] = ip[:-1]

            # compute gradient for mu variables
            idx = pdim
            for g in xrange(self.components):
            
                Wg = W[i,g]
                grad = self.reads[i,:]/activemu[g,:] - self.ireads[i,:]/(1-activemu[g,:])
                
                ss = Wg * np.dot( grad.reshape((seqdim, 1)), grad.reshape((1,seqdim)) )
                
                sele = slice(idx, idx+seqdim)
                Imat[sele, sele] -= ss
            
                eS[sele] = Wg*grad
                eB[sele] = Wg*( self.reads[i,:]/activemu[g,:]**2 + self.ireads[i,:]/(1-activemu[g,:])**2 )
            
                idx += seqdim
    
            Imat += np.dot( eS.reshape((idim, 1)), eS.reshape((1,idim)) )
            Imat[diagidx] += eB

    
        # take inverse
        validmat = True
        try:
            Imat = np.linalg.inv(Imat)
        except np.linalg.linalg.LinAlgError:
            validmat = False
    
        if validmat and np.min(np.diag(Imat)) < 0:
            validmat = False

        if not validmat:
            return None, None


        # convert to stderrs
        p_err = np.zeros(self.p.shape)
        p_err[:-1] = np.sqrt(np.diag(Imat[:pdim,:pdim]))
    
        # propagate p errors to calculate error of last population parameter
        a = -1* np.ones((1,pdim))
        p_err[-1] = np.sqrt(np.dot( np.dot(a, Imat[:pdim, :pdim]), a.transpose()))
    
        # fill mu_error array
        mu_err = np.zeros(activemu.shape)
        idx = pdim
        for g in xrange(self.components):
            mu_err[g,:] = np.sqrt( np.diag( Imat[idx:idx+seqdim, idx:idx+seqdim] ))
            idx += seqdim
        
        # convert to seqlen dimension; inactive nts given error = -1
        totmu = -1*np.ones(self.mu.shape)
        totmu[:,self.active_mask] = mu_err

        return p_err, totmu
    

    def bootstrap(self, n=100, randinit = False, verbal=False):
        
        # create new BM obj
        boot = BernoulliMixture()
        boot.numreads = self.numreads
        boot.seqlen = self.seqlen
        boot.active_mask = self.active_mask
        boot.inactive_mask = self.inactive_mask
        boot.setModelComponents(self.components)

        # containers for bootstrap samples
        psamp = np.zeros((n, self.components))
        musamp = np.zeros((n, self.components, self.seqlen))
        
        # indicator func indicating which samples converged
        converged = np.ones(n, dtype=bool)
        
        p_i = self.p
        mu_i = self.mu[:,self.active_mask]

        for i in xrange(n):
            
            s = np.random.choice(self.numreads, self.numreads)
            boot.reads = self.reads[s,:]
            boot.ireads = self.ireads[s,:]

            boot.inactive_reads = self.inactive_reads[s,:]
            boot.inactive_ireads = self.inactive_ireads[s,:]

            if randinit:
                p_i,mu_i = self.random_init_params(self.p.size)
             
            try:
                p, mu = boot.fitEM(p_i, mu_i, verbal=verbal)
            except ConvergenceError:
                converged[i]=False
                continue
            
            boot.p = p
            boot.mu[:, self.active_mask] = mu
            boot.imputeInactiveParams()
            
            idx = alignParams(boot.mu, self.mu)
            psamp[i,:] = boot.p[idx]
            musamp[i,:,:] = boot.mu[idx, :]
            

        p_err  = np.dstack((np.percentile(psamp[converged,:], 2.5, axis=0),
                            np.percentile(psamp[converged,:], 97.5, axis=0)))
        
        p_err = p_err.reshape((self.components, 2))

        mu_err = np.dstack((np.percentile(musamp[converged,:,:], 2.5, axis=0),
                            np.percentile(musamp[converged,:,:], 97.5, axis=0)))
        
        return p_err, mu_err
    

    def synbootstrap(self, n=100):
        
        raise AttributeError('Currently Deprecataed')

        boot = SynBernoulliMixture(p=self.p, mu=self.mu)
        psamp = np.zeros((n, self.components))
        musamp = np.zeros((n, self.components, self.seqlen))
        
        conv = np.ones(n, dtype=bool)

        for i in xrange(n):

            boot.generateReads(self.numreads)
            
            try:
                pi,mui = self.random_init_params(self.p.size)
                p,mu = boot.fitEM(pi, mui, verbal=True)
                idx = alignParams(mu, self.mu)

                psamp[i,:] = p[idx]
                musamp[i,:,:] = mu[idx, :]
            except ConvergenceError:
                conv[i]=False
                pass

        pstd = np.std(psamp[conv,:], axis=0)
        mustd = np.std(musamp[conv,:,:], axis=0)

        return  np.sum(conv), pstd, mustd
 

    def printParams(self):
        """Print out model parameters"""

        printParams(self.p, self.mu, self.columnindices+1)


####################################################################################


class SynBernoulliMixture(BernoulliMixture):
    """Class for synthetic Bernoulli Mixture dataset
    Inherits generic BernoulliMixture
    """

    def __init__(self, p=None, mu=None):
        """p is 1D array with population of each model
        mu is MxN 2D array with Bernoulli probs of each state
        """
        
        self.compiled = False

        defpar = int(p is None) + int(mu is None)
            
        if defpar == 1:
            raise ValueError("Illegal definition of p/mu without the other")

        elif defpar == 0:
            self._p_ = p
            self._mu_ = mu
            self.compileModel()
        else:
            self._p_ = []
            self._mu_ = []
    
    
    def compileModel(self):
        """Convert arrays to numpy arrays and check for errors"""
        
        if self.compiled:
            return
        
        # check to see if params are right type
        if not isinstance(self._mu_, np.ndarray) or self._mu_.dtype is np.dtype(object):
            self._mu_ = np.vstack(self._mu_)
            self._p_ = np.array(self._p_)
        
        if abs(self._p_.sum()-1.0) > 1e-10:
            raise AttributeError("Model populations don't sum to 1!")
        if self._p_.size != self._mu_.shape[0]:
            raise AttributeError('P and mu have inconsistent dimensions: p={0}, mu={1}'.format(self._p_.size, self._mu_.shape))
        
        # set the parameters
        self.seqlen = self._mu_.shape[1]
        self.columnindices = np.arange(self.seqlen)
        self.active_mask = np.ones(self.seqlen, dtype=np.bool_)
        self.inactive_mask = np.zeros(self.seqlen, dtype=np.bool_)
        self.compiled = True
           


    def addModel(self, mu, p):
        """Add model to the mixture.
        Mu can be either array/list or file
        p is its population"""

        if isinstance(mu, basestring):
            self._mu_.append( self.readParFile(mu) )
        
        else:
            self._mu_.append( np.array(mu) )

        self._p_.append(p)
        self.compiled = False


    def readParFile(self, inpfile):
        """Read model parameters from file"""

        data = []
        with open(inpfile) as inp:
            for line in inp:
                spl = line.split()
                data.append(float(spl[-1]))

        return np.array(data)

    

    def generateReads(self, num_reads):
        
        # finalize the model if not done so...
        self.compileModel()       
        
        # array for holding model assignments
        # Note max value of int8 == 127; this is fine since we never
        # expect this many models, but just be aware
        assignments = np.ones(num_reads, dtype=np.int8)*-1   
    
        # randomly sample different generating distributions
        rand = np.random.random(num_reads)
    
        low, up = 0.0, 0.0
        for i in xrange(self._p_.size):
            up += self._p_[i]
            readmask = (low <= rand) & (rand < up)
            assignments[readmask] = i
            low += self._p_[i]
        
        # make sure every 'read' has an assignment
        assert np.sum(assignments < 0) == 0


        reads = np.zeros((num_reads, self.seqlen), dtype=np.bool_)
        rand = np.random.random(reads.shape)

        for i in xrange(self._p_.size):
            readmask = (assignments == i)

            for j in xrange(self.seqlen):
                colmask = readmask & (rand[:, j] < self._mu_[i,j])
                reads[colmask, j] = True
        
        self.numreads = num_reads
        self.reads = reads
        self.ireads = np.invert(reads)
        self.inactive_reads = np.array([], dtype=np.bool_)
        self.inactive_ireads = np.array([], dtype=np.bool_)
        self.read_assignments = assignments


    def getSampleProfiles(self):
        """From the reads object, compute the model parameters and populations"""

        p = np.zeros(self._p_.shape)
        mu = np.zeros(self._mu_.shape)

        for i in xrange(len(self._p_)):
            
            mask = (self.read_assignments == i)
            
            p[i] = float(np.sum(mask))/len(mask)
            mu[i,:] = np.mean(self.reads[mask, :], axis=0)
        
        return p, mu
            

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
            prob = (alpha-1)*self._mu_[modelnum,n1]
            
            # select positions that should be mutated
            rmask = (np.random.random(len(asgn_mask)) < prob)
            # build total mask
            totmask = asgn_mask & rmask & self.reads[:,n2]
            
            # update the reads
            self.reads[totmask, n1] = 1





###################################################################################
# generic helper functions


def RMS(x):
    return np.sqrt(np.mean(x**2))


def compareModels(p1, mu1, p2, mu2, func=np.max):
    """Compute the error between two mixture model parameters"""
    
    minidx = alignParams(mu2, mu1)
    
    mu_diff = func( np.abs(mu1 - mu2[minidx,:]) )
    p_diff = func( np.abs(p1-p2[minidx]) )
    
    return p_diff, mu_diff


def alignParams(alg, ref):
    """Align two parameter sets to minimize RMS difference between them
    Return alignment vector for 2-->1"""
    
    mindiff=1000
    for i in itertools.permutations(range(alg.shape[0])):
        diff = RMS(np.abs(ref - alg[i,:]))
        if diff < mindiff:
            minidx = np.array(i)
            mindiff = diff
    
    return minidx



def printParamLine(params):
    out = ''
    for v in params:
        out+=' {0:.4f}'.format(v)
    return out

def printParams(p, mu, idx=None):
    
    if idx is None:
        idx = np.arange(1, mu.shape[1]+1)
    
    print 'P = '+printParamLine(p)
    print '---Mu---'
    for i in xrange(mu.shape[1]):
        print '{0} {1}'.format(idx[i], printParamLine(mu[:,i]))



