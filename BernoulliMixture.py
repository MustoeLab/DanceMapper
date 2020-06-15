

import numpy as np
import itertools, sys, copy, time
from collections import deque

try:
    import ringmapperpath
    sys.path.append(ringmapperpath.path())
    import accessoryFunctions
except:
    print('WARNING: Could not import accessoryFunctions!')



class ConvergenceError(Exception):
    """Exception class for convergence errors encountered during EM fitting"""
    
    def __init__(self, msg, step, badcolumns = []):
        """msg = identifying message
        step = abortion step
        badcolumns = list of failure columns
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

    def __init__(self, activecols, convergeThresh=1e-4, maxmu=0.5, maxsteps=1000, initsteps=51):

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
        
        self.maxmu = maxmu

        self.active_columns = activecols


    def update(self, p, mu):
        
        self.step += 1
        
        if self.step > self.maxsteps:
            self.error = ConvergenceError('Maximum iterations exceeded', self.step)
            self.iterate = False
            return
        
        # check convergence
        self.checkConvergence(p,mu)
        

        if self.step >= self.initsteps or self.converged:
            try:
                self.checkParamBounds(p, mu)
                self.checkMuRatio(mu)
                self.checkDegenerate(mu)

                if self.converged: # only do this check if converged
                    self.artifactCheck(mu)

            except ConvergenceError as e:
                self.error = e
                self.iterate = False
                self.converged = False

                    
        


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
            self.converged = True
            self.iterate = False
        

        self.lastp = np.copy(p)
        self.lastmu = np.copy(activemu)



    def checkParamBounds(self, p, mu):
        """Make sure p and mu params are within allowed bounds"""
        
        minp = np.min(p)
        if minp < 0.001:
            raise ConvergenceError('Low Population = {0}'.format(minp), self.step)

        # go through active columns and make sure params aren't too low
        activemu = mu[:, self.active_columns]
        
        minvals = np.min(activemu, axis=0)

        lowcolumns = self.active_columns[np.where(minvals < 1e-5)]

        if len(lowcolumns)>0:
            raise ConvergenceError('Columns with low Mu', self.step, lowcolumns)
        
        
        maxvals = np.max(activemu, axis=0)
        hicolumns = self.active_columns[np.where(maxvals > self.maxmu)]
        if len(hicolumns)>0:
            raise ConvergenceError('Columns with hi Mu', self.step, hicolumns)
        

    

    def checkMuRatio(self, mu, ratioCutoff=5.3):
        """Check to make sure that mu ratio does not exceed ratioCutoff
        ratioCutoff is ln-space; ln(200)=5.3; ln(100)=4.6"""
        
        activemu = mu[:, self.active_columns]
        
        hivalues = set()

        # iterate through all params
        for i,j in itertools.combinations(range(mu.shape[0]), 2):
            
            # compute ratio of reactivities
            ratio = np.abs(np.log(activemu[i]/activemu[j]))
            hivalues.update( np.where(ratio > ratioCutoff)[0] )
            
        hivalues = self.active_columns[sorted(hivalues)]
        
        if len(hivalues)>0:
            raise ConvergenceError('Columns with high mu ratio', self.step, hivalues)



    def checkDegenerate(self, mu, excludepercent=99):
        """Check to see if converging to degenerate parameters
        Uses trend of rmsdiff to prematurely terminate 
        """
        
        activemu = mu[:, self.active_columns]

        rmsdiff = 10 # initialize at high value

        for i,j in itertools.combinations(range(mu.shape[0]), 2):
            
            d = np.square(activemu[i,:] - activemu[j,:])
            
            excludevalue = np.percentile(d, excludepercent)
            diff = np.sqrt( np.mean( d[d<=excludevalue] ) )
            
            if diff < rmsdiff:
                rmsdiff = diff
        
        change = rmsdiff - self.rms

        self.rms = rmsdiff
        self.rmshistory.append(change)
        
        # look at trend over last 50 steps; if rmshistory is getting smaller
        # and rms is low, converging to degenerate soln...
        if len(self.rmshistory) > 50:
            self.rmshistory.popleft()
    
            if rmsdiff < 0.005 and np.sum(self.rmshistory)<0:
                raise ConvergenceError('Degenerate Mu: RMS diff={0:.4f}'.format(rmsdiff), self.step)
        
        # also do final check if converged
        elif self.converged and rmsdiff < 0.005:
            raise ConvergenceError('Degenerate Mu: RMS diff={0:.4f}'.format(rmsdiff), 'END')


            


    def artifactCheck(self, mu):
        """Check for artifacts where one highly reactive nt absorbs all mutation 
        signal, causing preceding 5 nts to have very low reactivity due to 
        alignment constraint. This gives characteristic anticorrelated local 
        reactivity profile. Will throw ConvergenceError if found
        """

        activemu = mu[:, self.active_columns]
        activecols = self.active_columns

        badidxs = set()

        # iterate through all params
        for i,j in itertools.permutations(range(mu.shape[0]), 2):
 
            # identify positions with high difference in reactivity
            ratio = activemu[i]/activemu[j] 
            hivalues = np.where(ratio > 10)[0]
            
            for idx in hivalues:
                
                colidx = activecols[idx]

                # scan 5' of each hi ratio value to see if artifact
                suppressed = True
                counted = False
                for m in range(1,6):
                    # if at least one nt is decently reactive, not an artifact
                    if (idx-m)>=0 and (activecols[idx]-activecols[idx-m])<6:
                        counted = True
                        if ratio[idx-m] > 0.1:
                            suppressed = False

                if counted and suppressed:
                    badidxs.add(idx)
                    continue


                # scan 3' of each hi ratio value to see if artifact
                suppressed = True
                counted = False
                for m in range(1,6):
                    # if at least one nt is decently reactive, not an artifact
                    if (idx+m)<len(activecols) and (activecols[idx+m]-activecols[idx])<6:
                        counted = True
                        if ratio[idx+m] > 0.1:
                            suppressed = False
                    
                if counted and suppressed:
                    badidxs.add(idx) 

        badidxs = self.active_columns[sorted(badidxs)]

        if len(badidxs) > 0:
            raise ConvergenceError('Potential anticorrelated Mu artifact', self.step, badidxs)



        
            
##############################################################################################



class BernoulliMixture(object):
    """This class contains parameters and fitting methods for a Bernoulli Mixture
    The number of model components is fixed
    """

    def __init__(self, pdim = None, mudim = None, p_initial=None, mu_initial=None,
                 active_columns = None, inactive_columns = None, idxmap = None,
                 priorA=1, priorB=1, fname=None):
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
                will override components, ncols parameters
        """
        
        
        self.pdim = pdim
        self.mudim = mudim
        self.p_initial  = p_initial
        self.mu_initial = mu_initial
        
        if self.p_initial is not None:
            self.pdim = self.p_initial.size
        
        if self.mu_initial is not None:
            self.mudim = self.mu_initial.shape[1]
        
        if self.pdim is not None and self.p_initial is None:
            self.initP()
        
        if self.mudim is not None and self.mu_initial is None:
            self.initMu()
        
        if self.pdim is not None and self.mudim is not None:
            self.setPriors(priorA, priorB)
        
        self.p = None
        self.mu = None
        self.p_err = None
        self.mu_err = None
        self.converged = False
        self.imputed = False

        self.cError = None
        self.loglike = None
        self.BIC = None
        
        self.idxmap = idxmap

        self.active_columns = None
        self.inactive_columns = np.array([], dtype=np.int32)

        if inactive_columns is not None:
            self.inactive_columns = np.array(inactive_columns, dtype=np.int32)
        
        if active_columns is not None or self.mudim is not None:
            self.set_active_columns(active_columns)

        if fname is not None:
            self.readModelFromFile(fname)


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
                cols = np.arange(self.mudim)
                mask = np.isin(cols, self.inactive_columns, invert=True)
                self.active_columns = np.array(cols[mask], dtype=np.int32)

            else:
                self.active_columns = np.array(cols, dtype=np.int32)    
                
                # make sure that no cols are double-listed
                self.inactive_columns = np.array([i for i in self.inactive_columns if i not in self.active_columns])



        except TypeError:
            
            if cols is None and self.mudim is None:
                raise TypeError("mudim is not defined")
            else:
                raise TypeError("{} is not a valid cols argument".format(cols))

    


    def initP(self):
        """Initialize p params to equiprobable"""
        if self.pdim is None:
            raise AttributeError("pdim is not defined")

        self.p_initial = np.ones(self.pdim) / self.pdim
        self.converged = False
    

    def initMu(self):
        """Compute random initial starting conditions for Mu, bounded by lowb and upb"""
        
        if self.pdim is None:
            raise AttributeError("pdim is not defined")
        if self.mudim is None:
            raise AttributeError("mudim is not defined")

        self.mu_initial = np.random.beta(1,40, (self.pdim, self.mudim))+0.001
        self.converged = False

    

    def setPriors(self, priorA, priorB):
        """set the beta priors
        priorA and priorB can be int or arraylike
        """
        
        # set priorA
        try:
            priorA = priorA*np.ones((self.pdim, self.mudim))
        except ValueError:
            raise ValueError('priorA shape={0} does not match mudim={1}'.format(np.asarray(priorA).shape, self.mudim))
        

        # set priorB
        try:
            priorB = priorB*np.ones((self.pdim, self.mudim))
        except ValueError:
            raise ValueError('priorB shape={0} does not match mudim={1}'.format(np.asarray(priorB).shape, self.mudim))
        

        self.priorA = priorA
        self.priorB = priorB
        self.converged = False

    

    def compute1ComponentModel(self, reads, mutations):
        
        if self.active_columns is None:
            self.set_active_columns()

        activecols = self.active_columns

        self.p = np.array([1.0])
        
        if self.mu_initial is None:
            self.initMu()
        
        if self.mu is None:
            self.mu = np.copy( self.mu_initial )

        # compute params
        mutsum = np.sum(mutations, axis=0, dtype=np.float_)
        readsum = np.sum(reads, axis=0, dtype=np.float_)
        
        self.mu[:,activecols] = mutsum[activecols]/readsum[activecols]
        
        self.converged = True
        
        self.loglike, self.BIC = self.computeModelLikelihood(reads, mutations)
        



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
 


    def maximization(self, reads, mutations, W):
        
        accessoryFunctions.maximizeP(self.p, W)
            
        accessoryFunctions.maximizeMu(self.mu, W, reads, mutations, 
                                      self.active_columns, self.priorA, self.priorB)
        
    

    def fitEM(self, reads, mutations, maxiterations = 1000, convergeThresh=1e-4, verbal=False, **kwargs):
        """Fit model to data using EM 
        
        maxiterations = maximum allowed iterations
        convergeThresh = terminate once maximum abs. change in params between iterations
                         falls below this threshold
        """
        

        if self.pdim == 1:
            self.compute1ComponentModel(reads, mutations)
            return


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
        
        
        with np.errstate(divide='ignore',invalid='ignore'):
            maxmu = np.sum(mutations, axis=0, dtype=float) / np.sum(reads, axis=0)
        
        maxmu = 1.5*np.max(maxmu[np.isfinite(maxmu)])

        CM = ConvergenceMonitor(self.active_columns, maxsteps=maxiterations, convergeThresh=convergeThresh,
                                maxmu = maxmu)
        
        timestart = time.time()

        while CM.iterate: 
            
            # expectation step
            W = self.computePosteriorProb(reads, mutations)
            
            self.maximization(reads, mutations, W)
            
            # this will throw ConvergenceErrors if bad soln
            CM.update(self.p, self.mu)
               
        
        self.converged = CM.converged
        self.cError = CM.error
        
        
        # make sure information matrix is defined
        if self.converged:
            try:
                self.computeUncertainty(reads, mutations, W)
            except ConvergenceError as e:
                self.converged = False
                self.cError = e


        # compute loglike and BIC        
        if self.converged:
            self.loglike, self.BIC = self.computeModelLikelihood(reads, mutations)

        # print outcome
        if verbal:
            if self.converged:
                
                print('\tValid solution!')
                msg = '\tP = ['
                for i in xrange(self.pdim):
                    msg += ' {0:.3f} +/- {1:.3f},'.format(self.p[i], self.p_err[i])
                print(msg[:-1]+' ]')
                print('\tEM converged in {0} steps ({1:.0f} seconds); BIC={2:.1f}'.format(CM.step, time.time()-timestart, self.BIC))
            
            else:
                print(self.cError.__str__(self.idxmap))
            
            
    
    
    def computeModelLikelihood(self, reads, mutations, active_columns=None):
        """Compute the (natural) log-likelihood of the data given the BM model
        and compute the BIC of the model
        
        if active_colums=None, use self.active_columns

        returns loglike, BIC
        """
        
        if active_columns is None:
            active_columns = self.active_columns

        llmat = np.zeros((self.pdim, reads.shape[0]), dtype=np.float64)
        
        accessoryFunctions.loglikelihoodmatrix(llmat, reads, mutations, active_columns, self.mu, self.p)
        
        # determine the likelihood of each read by summing over components
        readl = np.sum(np.exp(llmat), axis=0) 
    
        # total log-likelihood --> the product of individual read likelihoods
        loglike = np.sum( np.log( readl ) )

        # number of parameters
        npar = len(self.active_columns)*self.pdim + self.pdim-1
        
        # BIC = -2*ln(LL) + npar*ln(n)
        BIC = -2*loglike + npar*np.log(reads.shape[0])

        return loglike, BIC

   


    def computeUncertainty(self, reads, mutations, readWeights=None):
        """Compute the uncertainty of the model parameters from the information matrix
        
        Will raise ConvergenceError exception if information matrix is poorly defined
        """
       
        if readWeights is None:
            readWeights = self.computePosteriorProb(reads, mutations)    


        Imat = accessoryFunctions.computeInformationMatrix(self.p, self.mu, readWeights, reads, 
                                                           mutations, self.active_columns, 
                                                           self.priorA, self.priorB)
        

        # compute the inverse of the information matrix
        try:
            Imat = np.linalg.inv(Imat)
        except np.linalg.linalg.LinAlgError as e:
            raise ConvergenceError('Information matrix invalid: '+str(e), 'END')
        
                
        Imat_diag = np.diag(Imat)
        p1 = self.pdim-1
        
        # compute dim-1 p errors from Imat
        p_err = np.zeros(self.p.shape)
        p_err[:-1] = Imat_diag[:p1]
        
        # compute error for p[-1] via error propagation (p[-1] = 1 - p[0] - p[1] ...)
        a = -1* np.ones((1,p1))
        p_err[-1] = np.dot( np.dot(a, Imat[:p1, :p1]), a.transpose())
 
        if np.min(p_err) < 0:
            raise ConvergenceError('Information matrix has undefined p errors', 'END')
        
        p_err = np.sqrt(p_err)

        # reject if population errors are high
        #if np.max(p_err) > 0.1:
        #    print('\tSolution found:')
        #    msg = '\tP = ['
        #    for i in xrange(self.pdim):
        #        msg += ' {0:.3f} +/- {1:.3f},'.format(self.p[i], p_err[i])
        #    print(msg[:-1]+' ]')
        #    raise ConvergenceError('Solution is poorly defined: high P errors', 'END')


        if np.min(Imat_diag[p1:]) < 0:
            raise ConvergenceError('Information matrix has undefined mu errors', 'END')


        # compute mu errors
        mu_err = -1*np.ones(self.mu.shape) # initialize to -1, which will be value inactive/invalid
        
        for d in range(self.pdim):
            for i, col in enumerate(self.active_columns):
                idx = p1 + d*len(self.active_columns) + i
                mu_err[d, col] = np.sqrt(Imat[idx,idx])


        self.p_err = p_err
        self.mu_err = mu_err



    def alignModel(self, BM2):
        """Align BM2 to current BM
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

        return minidx



    def modelDifference(self, BM2, func=np.max, columns='active'):
        """compute the difference between two BM models. 
        The difference is evaluated using func
        columns can be active, inactive, both (both = active+inactive) 
        """
        
        if columns == 'active':
            if not np.array_equal(self.active_columns, BM2.active_columns) and \
                    len(self.active_columns) < len(BM2.active_columns):
                actlist = self.active_columns
            else:
                actlist = BM2.active_columns
        elif columns == 'inactive':
            if not np.array_equal(self.inactive_columns, BM2.inactive_columns) and \
                    len(self.inactive_columns) < len(BM2.inactive_columns):
                actlist = self.inactive_columns
            else:
                actlist = BM2.inactive_columns
        elif columns == 'both':
            actlist = np.append(self.active_columns, self.inactive_columns)
            actlist.sort()
        else:
            raise ValueError('Unknown column keyword: {}'.format(columns))



        idx = self.alignModel(BM2)
        
        d = np.abs(self.mu - BM2.mu[idx,])
        mudiff = func(d[:, actlist])
        
        pdiff = func(np.abs(self.p-BM2.p[idx,]))
        

        return pdiff, mudiff

    


    def _writeModelParams(self, OUT, sort_model=False):
        """Write out the params of the model to object OUT in a semi-human readable form"""
        
        if sort_model:
            self.sort_model()


        OUT.write('# P\n')
        np.savetxt(OUT, self.p, fmt='%.16f', newline=' ')
        
        OUT.write('\n# P_uncertainty\n')
        try:
            np.savetxt(OUT, self.p_err, fmt='%.16f', newline=' ')
        except (TypeError, ValueError):
            OUT.write('-- '*self.pdim)

        OUT.write('\n\n# Nt Mu ; Mu_err\n')
        # write out Mu with active and inactive info
        for i in xrange(self.mudim):
            
            if self.idxmap is not None:
                OUT.write('{0} '.format(self.idxmap[i]))
            else:
                OUT.write('{0} '.format(i))


            # write out nan for invalid columns
            if i not in self.inactive_columns and i not in self.active_columns:
                OUT.write('nan '*self.pdim)

            elif i in self.inactive_columns and not self.imputed:
                OUT.write('{0} ; {0} i'.format('-- '*self.pdim))

            else:
                np.savetxt(OUT, self.mu[:,i], fmt='%.16f', newline=' ')
                
                OUT.write('; ')
                try:
                    np.savetxt(OUT, self.mu_err[:, i], fmt='%.4f', newline=' ')
                except TypeError:
                    OUT.write('-- '*self.pdim)


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
        np.savetxt(OUT, self.priorA, fmt='%.16f')

        OUT.write('\n\n# PriorB\n')
        np.savetxt(OUT, self.priorB, fmt='%.16f')



    

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
            
            # pop off p error
            inp.readline()
            spl = inp.readline().split()
            if self.pdim > 1:
                self.p_err = np.array(spl, dtype=float)
            else:
                self.p_err = np.array([0])

            # pop off mu header
            inp.readline()
            inp.readline()

            actives = []
            inactives = []
            mu = []
            mu_err = []
            idxmap = []
            
            i = -1
            read = True
            while read:
                
                i += 1
                spl = inp.readline().split()
                if len(spl) == 0 or spl[0][0]=='#':
                    break

                idxmap.append(int(spl[0]))
                

                try:
                    vals = map(float, spl[1:1+self.pdim])
                    errs = map(float, spl[2+self.pdim:2+2*self.pdim])

                    # invalid nt
                    if vals[0] != vals[0]:
                        mu.append([np.nan]*self.pdim)
                        mu_err.append([-1]*self.pdim)
                        continue # skip to next position so not added to inactive/active list
                    else:
                        mu.append(vals)
                        mu_err.append(errs)
                
                except ValueError as e:
                    if '--' in spl[1:1+self.pdim]:                         
                        mu.append([-1]*self.pdim)
                        mu_err.append([-1]*self.pdim)
                    elif '--' in spl[2+self.pdim:2+2*self.pdim]:
                        mu.append(vals)
                        mu_err.append([-1]*self.pdim)
                    else:
                        raise e
                

                if spl[-1] == 'i':
                    inactives.append(i)
                else:
                    actives.append(i)


            self.idxmap = np.array(idxmap)
            self.active_columns = np.array(actives, dtype=np.int32)
            self.inactive_columns = np.array(inactives, dtype=np.int32)

            mu = np.array(mu)
            self.mu = np.array(mu.transpose(), order='C')
            self.mudim = self.mu.shape[1]
            
            self.mu_err = np.array(mu_err).transpose()

            # read in initial p
            inp.readline()
            self.p_initial = np.array(inp.readline().split(), dtype=float)
            
            inp.readline()
            inp.readline()
        
            self.mu_initial = -1*np.ones(self.mu.shape)
            
            i = -1
            read = True
            
            # read in initial mu
            while read:
                i += 1
                spl = inp.readline().split()
                if len(spl)==0 or spl[0][0]=='#':
                    break

                self.mu_initial[:, i] = map(float, spl[1:])
            
            return

            # read in priorA
            inp.readline()
            priorA = []
            for i in range(self.pdim):
                priorA.append(map(float, inp.readline().split()))
            self.priorA = priorA

            inp.readline()
            inp.readline()
            priorB = []
            for i in range(self.pdim):
                priorB.append(map(float, inp.readline().split()))
            self.priorB = priorB
        


    def imputeInactiveParams(self, reads, mutations):
        """Compute inactive Mu parameters
        EM is used to find optimal inactive mu, keeing p and active mus fixed
        """
        
        if len(self.inactive_columns)==0:
            return


       
        combined_columns = np.append(self.active_columns, self.inactive_columns)
        combined_columns.sort()

        # get initial posterior probabilities (uses only active columns)
        W = self.computePosteriorProb(reads, mutations)
        
        lastmu = np.copy(self.mu)
        converged = False
        nsteps = 0
        while not converged and nsteps<1000:
            
            # update posterior probs based on inactive_column info
            if nsteps > 0:
                accessoryFunctions.loglikelihoodmatrix(W, reads, mutations, combined_columns, self.mu, self.p)
                W = np.exp(W)
                W /= W.sum(axis=0)
            
            # update inactive mu
            accessoryFunctions.maximizeMu(self.mu, W, reads, mutations, self.inactive_columns, self.priorA, self.priorB)
            

            if np.max(np.abs(lastmu-self.mu)) < 1e-4:
                converged = True
            
            lastmu = np.copy(self.mu)
            nsteps+=1

        
        if not converged:
            print('WARNING: Inactive parameter imputation not converged!!!')
            return


        # update newp so that we can track implied population shift
        newp = np.copy(self.p)
        accessoryFunctions.maximizeP(newp, W) 

        print('Inactive parameters converged in {0} steps'.format(nsteps))
        print('\tPopulation computed using active+inactive = {0}'.format(newp))
        print('\tPopulation computed using only active     = {0}'.format(self.p))
        print('\t\t(the active-only populations are used)')
        
        if np.max(np.abs(newp - self.p)) > 0.01:
            print('WARNING: Imputed inactive parameters imply signification population shift')
        

        
        self.imputed = True



    def model_rms_diff(self, excludepercentile=None):
        """Return the minimum root-mean-square between model components
        excludepercentile will exclude the top indicated percentile from the calculation
        """

        minval = 1e5
        
        for i,j in itertools.combinations(range(self.pdim), 2):
            
            diff = np.square(self.mu[i]-self.mu[j])
            diff = diff[self.active_columns]
            
            if excludepercentile is not None:
                exvalue = np.percentile(diff, excludepercentile)
                rms = np.sqrt(np.mean( diff[diff<=exvalue] ))
            
            else:
                rms = np.sqrt(np.mean( diff ))

            if rms < minval:
                minval = rms

        return minval

    
    def model_absmean_diff(self, excludepercentile=None):
        """Return the minimum abs mean between model components
        excludepercentile will exclude the top indicated percentile from the calculation
        """

        minval = 1e5
        
        for i,j in itertools.combinations(range(self.pdim), 2):
            
            diff = np.abs(self.mu[i]-self.mu[j])
            diff = diff[self.active_columns]
            
            if excludepercentile is not None:
                exvalue = np.percentile(diff, excludepercentile)
                m = np.mean( diff[diff<=exvalue] )
            
            else:
                m = np.mean( diff )

            if m < minval:
                minval = m

        return minval


    def model_num_diff(self, diff_cutoff=0.01):
        """Return the minimum number of different mus between model components
        """

        minval = 1e5
        
        for i,j in itertools.combinations(range(self.pdim), 2):
            
            diff = np.abs(self.mu[i]-self.mu[j])
            diff = diff[self.active_columns] > diff_cutoff
                
            s = np.sum(diff)     

            if s < minval:
                minval = s

        return minval



    def sort_model(self):
        """Sort the model by population"""
        
        if len(self.p) == 1:
            return

        # sort model components by population
        sortidx = range(self.pdim)
        sortidx.sort(key=lambda x: self.p[x], reverse=True)

        
        self.p = self.p[sortidx]
        self.p_err = self.p_err[sortidx]
        self.mu = self.mu[sortidx]
        self.mu_err = self.mu_err[sortidx]

        self.p_initial = self.p_initial[sortidx]
        self.mu_initial = self.mu_initial[sortidx]
        
        self.priorA = self.priorA[sortidx]
        self.priorB = self.priorB[sortidx]







# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
