
import numpy as np
import itertools, argparse, warnings
from collections import deque

from ReactivityProfile import ReactivityProfile
from BernoulliMixture import *

import accessoryFunctions


class EnsembleMap(object):

    def __init__(self, inpfile=None, profilefile=None, **kwargs):
        """Define important global parameters"""
    
        # reads contains positions that are 'read'
        # mutations contains positions that are mutated
        self.reads = None 
        self.mutations = None
        
        # Affialiated ReactivitProfile object
        self.profile=None
    
        # np.array of sequence positions to actively cluster 
        self.active_columns=None
        
        # np.array of sequence positions to 'inactively' cluster
        self.inactive_columns=None

        # np.array of sequence positions that are 'invalid' -- don't cluster at all
        self.invalid_columns=None

        # info about the reads
        self.ntindices = None
        self.sequence = None
        self.seqlen = None
        self.numreads = None
        
        # contains final BMsolution, if fitting done
        self.BMsolution = None
        
        if profilefile is not None:
            self.profile = ReactivityProfile(profilefile)
            self.seqlen = len(self.profile.sequence)
            self.ntindices = self.profile.nts


        if inpfile is not None:
            self.readData(inpfile, **kwargs)

    

    def readData(self, fname, mincoverage=None, **kwargs):
        
        
        # Determine mincoverage quality filter
        if mincoverage is None:
            
            mincoverage = self.seqlen
            
            # if profile is defined, remove nan positions from calculation
            if self.profile is not None:
                mincoverage -= np.sum(np.isnan(self.profile.normprofile))

            mincoverage = int(round(mincoverage*0.95))
            print('No mincoverage specified. Using default 95% coverage = {} nts'.format(mincoverage))



        # read in the matrices
        reads, mutations = accessoryFunctions.fillReadMatrices(fname, self.seqlen, mincoverage)
        
        print('{} reads for clustering\n'.format(reads.shape[0]))

        self.reads = reads
        self.mutations = mutations
        
        self.numreads = self.reads.shape[0]
        
        self.initializeActiveCols(**kwargs)

 


    def initializeActiveCols(self, invalidcols=[], minrx=0.002, maxbg=0.01, minrxbg=0.002, verbal = True, **kwargs):
        """Apply quality filters to eliminate noisy nts from EM fitting
        invalidcols = list of columns to set to invalid
        minrx       = minimum signal (react. rate)
        bgarray     = array (or txt file of array) of background signal
        maxbg       = maximum allowable background signal
        minrxbg     = minimum signal above background
        verbal      = keyword to control printing of inactive nts
        """
        

        ###################################
        # initialize invalid positions
        ###################################

        if verbal:
            print("Nts {} set invalid by user".format(self.ntindices[invalidcols]))


        # supplement invalid cols from profile, checking for nans
        profilenan = []
        if self.profile is not None:
            for i, val in enumerate(self.profile.normprofile):
                if val != val and i not in invalidcols:
                    invalidcols.append(i)
                    profilenan.append(i)
        
        if verbal:
            print("Nts {} invalid due to masking in profile file".format(self.ntindices[profilenan]))


        # Double check data that we actually clustering to exclude very low rates
        signal = np.sum(self.mutations, axis=0, dtype=np.float)
        
        lowsignal = []         
        for i in np.where(signal < 0.0001*self.mutations.shape[0])[0]:
            if i not in invalidcols:
                lowsignal.append(i)
                invalidcols.append(i)

        if verbal:
            print("Nts {} set to invalid due to low mutation signal".format(self.ntindices[lowsignal]))
        
    
        invalidcols.sort()
        self.invalid_columns = np.array(invalidcols)

        
        ###################################
        # initialize inactive positions
        ###################################
        
        if 0: #self.profile is not None:

            with np.errstate(divide='ignore',invalid='ignore'):
                mod = self.profile.rawprofile
                unt = self.profile.backprofile
                diff = mod-unt
                
                inactive = (mod < minrx)
                inactive = inactive | (unt > maxbg)
                inactive = inactive | (diff < minrxbg)
            
            # invalid columns are distinct from inactive
            inactive[self.invalid_columns] = False
        
        else:
            inactive = np.zeros(self.reads.shape[1], dtype=bool)


        self.inactive_columns = np.where(inactive)[0]
            
        if verbal:
            print("Nts {} set to inactive due to low mutation signal".format(self.ntindices[self.inactive_columns]))

        
        ###################################
        # remaining nts are active!
        ###################################
        
        active = []
        for i in range(self.seqlen):
            if i not in self.invalid_columns and i not in self.inactive_columns:
                active.append(i)

        self.active_columns = np.array(active)
        
        if verbal:
            print("{} initial active columns".format(len(self.active_columns)))
        
        
    
                    
    def setColumns(self, activecols=None, inactivecols=None):
        """Set columns to specified values, if changed"""
        

        if activecols is not None and inactivecols is not None:
            
            # if the same, don't do anything
            if np.array_equal(activecols, self.active_columns) and \
                    np.array_equal(inactivecols, self.inactive_columns):
                return

            # make sure they aren't conflicting
            if np.sum(np.isin(activecols, inactivecols))>0 and \
                    np.sum(np.isin(activecols, inactivecols))>0:
                raise ValueError('activecols and inactivecols are conflicting')
            
            totc = len(activecols) + len(inactivecols) + len(self.invalid_columns)
            if totc != self.seqlen:
                raise ValueError('Not all columns assigned!')

            self.active_columns = np.array(activecols)
            self.inactive_columns = np.array(inactivecols)
        

        elif activecols is not None:
            
            if np.array_equal(activecols, self.active_columns):
                return

            self.active_columns = np.array(activecols)

            indices = np.arange(self.seqlen)
            
            mask = np.isin(indices, self.active_columns, invert=True)
            mask = mask | np.isin(indices, self.invalid_columns, invert=True)

            self.inactive_columns = np.array(indices[mask])
        

        elif inactivecols is not None:
            
            if np.array_equal(inactivecols, self.inactive_columns):
                return

            self.inactive_columns = np.array(inactivecols)

            indices = np.arange(self.seqlen)
            
            mask = np.isin(indices, self.inactive_columns, invert=True)
            mask = mask | np.isin(indices, self.invalid_columns, invert=True)

            self.active_columns = np.array(indices[mask])
 


    
    def setActiveColumnsInactive(self, columns, verbal=False):
        """Add currently active columns to the list of inactive columns
        -columns is a list of *active* column indices to set to inactive
        """
        
        if len(columns) == 0:
            return

        self.inactive_columns = np.append(self.inactive_columns, columns)
        self.inactive_columns.sort()

        self.active_columns = np.array([i for i in self.active_columns if i not in columns])

        if verbal:
            print("INACTIVE LIST UPDATE")
            print("\tNew inactive nts :: {}".format(self.ntindices[columns]))
            print("\tTotal inactive nts :: {}".format(self.ntindices[self.inactive_columns]))

    

        
    def MAPClassification(self):
        """NOT WORKING"""
        raise RuntimeError("Read classification not implemented")

        W = self.computePosteriorProb(self.p, self.mu[:,self.active_mask])
    
        assignments = np.ones(self.numreads, dtype=np.int8)*-1
    
        for i in xrange(self.components):
            mask = (np.max(W, axis=1) == W[:,i])
            assignments[mask] = i
    
        assert np.sum(assignments<0) == 0

        return assignments


    def stochasticClassification(self):
        """NOT WORKING"""
        raise RuntimeError("Read classification not implemented")


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

    
    def compute1ComponentModel(self):
        """Compute the null model (i.e. mixture of one)"""
        
        # create 1D BM object and assign its p/mu params
        mudim = self.seqlen
        BM = BernoulliMixture(pdim=1, mudim=mudim, 
                              active_columns=self.active_columns,
                              inactive_columns=self.inactive_columns,
                              idxmap=self.ntindices)
        
         
        BM.p = np.array([1.0])
        
        # compute 1D mu params
        mu = np.sum(self.mutations, axis=0, dtype=np.float_)
        
        musum = np.sum(self.reads, axis=0, dtype=np.float_)
        mask = np.where(musum > 0)[0]

        mu[mask] = mu[mask] / musum[mask]

        mask = np.where(musum==0)[0]
        mu[mask] = 0

        mu = mu.reshape((1, mudim))
        BM.mu = mu  
        BM.converged = True
        
        c = np.sum(self.reads, axis=0, dtype=np.float_)
        
        BM.computeModelLikelihood(self.reads, self.mutations)
        
        return BM



    def fitEM(self, components, trials=5, soln_termcount=3, badcolcount=2, verbal=False, 
            writeintermediate=None, **kwargs):
        """Fit Bernoulli Mixture model of a specified number of components.
        Trys a number of random starting conditions. Terminates after finding a 
        repeated valid solution, a repeated set of 'invalid' column solutions, 
        or after a maximum number of trials.

        components = number of model components to fit
        trials = max number of fitting trials to run
        soln_termcount = terminate after this many identical solutions founds
        badcolcount = set columns inactive after this many times of causing BM failure

        writeintermediate = write out each BM soln to specified prefix
        verbal = T/F on whether to print results of each trial
        
        additional kwargs are passed onto BernoulliMixture.fitEM
        
        returns:
            bestfit     = bestfit BernoulliMixture object
            fitlist     = list of other BernoulliMixture objects
        """
        
        
        # array for each col; incremented each time a col causes failure of BM soln
        badcolumns = np.zeros(self.seqlen, dtype=np.int32)
                
        
        # don't need to fit for c=1
        if components == 1:
            BM = self.compute1ComponentModel()

            if writeintermediate is not None:
                BM.writeModel('{}-1.txt'.format(writeintermediate))

            return BM, []


        bestfit = None
        fitlist = []
        bestfitcount = 1
        tt = 0 

        while tt < trials and bestfitcount < soln_termcount:
            
            tt += 1

            if verbal:
                print('Fitting {0} component model -- Trial {1}'.format(components, tt))
            

            # set up new BM
            BM = BernoulliMixture(pdim=components, mudim=self.seqlen,
                                  active_columns=self.active_columns, 
                                  inactive_columns=self.inactive_columns, 
                                  idxmap=self.ntindices)
            
            # fit the BM
            BM.fitEM(self.reads, self.mutations, verbal=verbal, **kwargs)
 
            
            if BM.converged:
                
                fitlist.append(BM)

                if writeintermediate is not None:
                    BM.writeModel('{0}-{1}-{2}.txt'.format(writeintermediate, components, tt))
                
                if bestfit is None:
                    bestfit = BM
                    continue
                
                
                # this func will refit best fit BM if columns aren't equal
                compareBM = self.compareBMs(BM, bestfit, verbal=verbal)
                
                # if bestfit couldn't be refit, then this indicates soln is unstable
                # Assign new solution as bestfit
                if compareBM is None:
                    bestfit = BM
                    continue
                

                pdiff, mudiff = BM.modelDifference(compareBM, func=np.max)

                if pdiff < 0.005 and mudiff < 0.005:
                    bestfitcount += 1
                    if BM.BIC < compareBM.BIC:
                        bestfit = BM

                elif BM.BIC < compareBM.BIC:
                    bestfitcount = 1
                    bestfit = BM
                
            
           
            else:  # solution did not coverge
                
                # increment bad cols 
                # (BM.cError.badcolumns will be empty if not badcolumn error)
                badcolumns[BM.cError.badcolumns] += 1
                
                bc = np.where(badcolumns>=badcolcount)[0]
                
                self.setActiveColumnsInactive(bc, verbal=verbal)
            
                # zero out columns that we've now set to inactive so won't be triggered in future
                badcolumns[bc] = 0




        if bestfitcount == soln_termcount and verbal:
            print('{0} identical fits found. Terminating trials'.format(bestfitcount))


        return bestfit, fitlist
        


    
    def compareBMs(self, BMnew, BMold, verbal=False):
        """Check if BMnew and BMold are the same"""

        # check to see if this solution has been reached already
        if np.array_equal(BMnew.active_columns, BMold.active_columns):
            return BMold
 

        if verbal:  
            print('Refitting prior {0}-component model with different inactive columns'.format(BMold.pdim))


        compareBM = BMold.copy()
        compareBM.set_active_columns(BMnew.active_columns)
        compareBM.refit_new_active(self.reads, self.mutations, verbal=verbal, maxiterations=20)
                    
        if not compareBM.converged:
            if verbal: print('\t{0}-component model could not be refit!'.format(BMold.pdim))
            return None
            
        # check that new solution hasn't changed too much!
        pdiff, mudiff = BMold.modelDifference(compareBM, func=np.max)
        if pdiff > 0.005 or mudiff > 0.005:
            if verbal: print('\tSignificant dP={0:.3f} or dMu={0:.3f} after refitting!'.format(pdiff, mudiff))
            return None

        return compareBM

                

    
    
    def findBestModel(self, maxcomponents=5, trials=5, dynamic_inactive_cutoff = 0.1, 
                            badcolcount=2, verbal=False, writeintermediate=None, **kwargs):
        """Fit BM model for progessively increasing number of model components 
        until model with best BIC is found. 
        Dynamically updates inactive list as problematic columns are identified

        maxcomponents           = maximum number of components to attempt fitting
        dynamic_inactive_cutoff = max fraction of nts allowed to dynamically set to inactive
        invalidtermcount        = number of times an invalid column must recur to terminate,
                                  and correspondingly set to inactive

        **kwargs passed onto fitEM method"""


        # best BernoulliMixture solution. Assign as 1-component solution to start
        overallBestBM = self.compute1ComponentModel()
        

        if verbal:  
            print("\n 1-component BIC = {0:.1f}".format(overallBestBM.BIC))
            print('*'*50+'\n')



        # iterate through each model size
        for c in xrange(2,maxcomponents+1):
            
            if verbal: print('\nAdvancing to {}-component model\n'.format(c))


            bestBM, fitlist = self.fitEM(c, trials=trials, badcolcount=badcolcount,
                                         verbal=verbal, writeintermediate=writeintermediate, **kwargs)
            
            # terminate if no valid solution found             
            if bestBM is None:
                if verbal:
                    print("No valid solution for {0}-component model".format(c))
                break
             
            elif verbal:
                self.printEMFitSummary(bestBM, fitlist)
             

            compareBM = self.compareBMs(bestBM, overallBestBM, verbal=verbal)

            if compareBM is None:
                warnings.warn("{0}-component model did not converge under new inactive list\nUnable to perform necessary {0} vs. {1} component comparison. Selecting prior {0}-component model.".format(c-1, c))
                break
            

            # if BIC is not better, terminate search
            if bestBM.BIC > compareBM.BIC-10:
                break
            else:
                overallBestBM = bestBM
                if verbal: 
                    print("{0}-component model assigned as new best model!".format(c))
                    
            

        if verbal:
            print('{0}-component model selected'.format(overallBestBM.pdim))
    
        
        # reset the active/inactive cols if necessary
        self.setColumns(activecols=overallBestBM.active_columns, inactivecols=overallBestBM.inactive_columns)    
        
        overallBestBM.imputeInactiveParams(self.reads, self.mutations)
        
        self.BMsolution = overallBestBM

        return self.BMsolution




    def printEMFitSummary(self, bestfit, fitlist):
         
        print "\n{0} fits found for {1}-component model".format(len(fitlist), bestfit.pdim)
        print "Best Fit BIC = {0:.1f}".format( bestfit.BIC )
        
        print "Deviation of fits from best parameters:"
        
        for i,f in enumerate(fitlist):

            pdiff, mudiff = bestfit.modelDifference(f, func=np.max)
                 
            print "\tModel {0}  BIC={1:.1f}  Max_dP={2:.4f}  Max_dMu={3:.4f}".format(i+1, f.BIC, pdiff, mudiff)

        print '*'*50, '\n'
 




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
        """
        REWRITE
        
        get it to return subsample from RINGexperiment
        have bootstrap calculations run in a different method
            --perhaps cluster.py?

        """
        # create new BM obj
        boot = RINGexperiment()
        boot.numreads = self.numreads
        boot.seqlen = self.seqlen
        boot.active_mask = self.active_mask
        boot.inactive_mask = self.inactive_mask

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
                #BM = BernoulliMixture(p_initial = BM.p, mu_initial )
                p_i, mu_i = self.random_init_params(self.p.size)
             
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
    
    

    def computeNormalizedReactivities(self):
        """From converged mu params and profile, compute normalized params"""

        model = self.BMsolution
        
        # eliminate invalid positions
        model.mu[:, self.invalid_columns] = np.nan

        # create temporary profile containing maxs at each position to normalize
        maxProfile = self.profile.copy()
        maxProfile.rawprofile = np.max(model.mu, axis=0)
        maxProfile.backgroundSubtract()
        normfactors = maxProfile.normalize(DMS=True)

        # now create new normalized profiles
        profiles = []

        for p in xrange(model.pdim):
            prof = self.profile.copy()
            prof.rawprofile = np.copy(model.mu[p,:])
            prof.backgroundSubtract()
            
            for i,nt in enumerate(prof.sequence):
                prof.normprofile[i] = prof.subprofile[i]/normfactors[nt]
        
            profiles.append(prof)


        return profiles




    def writeModelParams(self, output):
        """Print out model parameters"""
        
        model = self.BMsolution

        if self.profile is None:
            model.writeModelParams(output)
            return
        
        # compute normalized parameters
        profiles = self.computeNormalizedReactivities()
        
        # sort model components by population
        idx = range(model.pdim)
        idx.sort(key=lambda x: model.p[x], reverse=True)


        with open(output, 'w') as OUT:

            OUT.write("{0} components; BIC={1:.1f}\n".format(model.pdim, model.BIC))
            
            pline = 'p'
            for i in idx:
                pline += ' {0:.3f}'.format(model.p[i])
            OUT.write(pline+'\n')
            
            OUT.write('Nt\tSeq\t')
            OUT.write('nReact\tRaw\t\t'*model.pdim+'Background\n')
            
            
            for nt in xrange(model.mudim):
                
                muline = '{0}\t{1}\t'.format(self.ntindices[nt], self.profile.sequence[nt])
                
                if nt in self.invalid_columns:
                    muline += '{0}{1:.4f}'.format('nan\tnan\t\t'*model.pdim, self.profile.backprofile[nt])

                else:
                    for i in idx:
                        muline += '{0:.4f}\t{1:.4f}\t\t'.format(profiles[i].normprofile[nt], profiles[i].rawprofile[nt])
                    
                    muline += '{0:.4f}'.format(self.profile.backprofile[nt])


                    if nt in self.inactive_columns:
                        muline += ' i'

                OUT.write(muline + '\n')
            




####################################################################################


def parseArguments():

    parser = argparse.ArgumentParser(description = "Fit BM model to data")

    parser.add_argument('inputFile', type=str, help='Path to file')
    
    parser.add_argument('--profile', type=str, help='ShapeMapper profile')

    parser.add_argument('--outputFile', type=str, help='Path to output')
    
    parser.add_argument('--maxcomponents', type=int, default=5, help='Maximum number of components to fit (default=5)')
    
    parser.add_argument('--trials', type=int, default=10, help='Maximum number of fitting trials at each component number (default=10)')
    
    parser.add_argument('--badcol_cutoff', type=int, default=2, help='Inactivate column after it causes a bad solution X number of times (default=2)')

    parser.add_argument('--suppressVerbal', action='store_false', help='Suppress verbal output')

    parser.add_argument('--mincoverage', type=int, help='Minimum coverage (integer number of nts) required for read to be included in cacluations')
    
    parser.add_argument('--writeintermediates', type=str, help='Write each BM solution to file with specified prefix. Will be saved as prefix-1-1.bm, prefix-2-1.bm, ...')


    args = parser.parse_args()

    return args









if __name__=='__main__':
    
    args = parseArguments()
    
    EM = EnsembleMap(inpfile=args.inputFile, profilefile=args.profile, mincoverage=args.mincoverage,
                     verbal=args.suppressVerbal)
    

    EM.findBestModel(args.maxcomponents, trials=args.trials,
                     badcolcount = args.badcol_cutoff,
                     verbal=args.suppressVerbal,
                     writeintermediate = args.writeintermediates)
    

    EM.writeModelParams(args.outputFile)












# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
