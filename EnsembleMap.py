
import numpy as np
import sys, argparse, itertools

# get path to functions needed for mutstring I/O
import ringmapperpath
sys.path.append(ringmapperpath.path())


import accessoryFunctions as aFunc
from BernoulliMixture import BernoulliMixture



from ReactivityProfile import ReactivityProfile
from ringmapper import RINGexperiment
from pairmapper import PairMapper



class EnsembleMap(object):

    def __init__(self, modfile=None, untfile=None, profilefile=None, seqlen=None, **kwargs):
        """Define important global parameters"""
    
        # reads contains positions that are 'read'
        # mutations contains positions that are mutated
        self.reads = None 
        self.mutations = None
        
        # Affialiated ReactivitProfile object
        self.profile=None
    
        # np.array of sequence positions to actively cluster 
        self.active_columns=None
        
        self.initialActiveCount = 1

        # np.array of sequence positions to 'inactively' cluster
        self.inactive_columns=None

        # np.array of sequence positions that are 'invalid' -- don't cluster at all
        self.invalid_columns=None

        # info about the reads
        self.seqlen = None
        self.ntindices = None
        self.sequence = None
        self.numreads = None
        self.minreadcoverage = None
        
        # contains final BMsolution, if fitting done
        self.BMsolution = None
        
        
        if profilefile is not None:
            self.init_profile(profilefile)

        elif seqlen is not None:
            self.seqlen = seqlen
            self.ntindices = np.arange(1, seqlen+1, dtype=np.int32)

        
        # first initialize reads/mutations, which will set mincoverage
        if modfile is not None:
            self.readPrimaryData(modfile, **kwargs)
        
        # if provided, now update profile.backprofile with rate computed 
        # using reads filtered according to same criteria for primary data
        if untfile is not None:
            self.computeBGprofile(untfile, **kwargs)

        
        if self.reads is not None:
            self.initializeActiveCols(**kwargs)


    def init_profile(self, profilefile):

        self.profile = ReactivityProfile(profilefile)
        self.profile.normalize(DMS=True)
        self.sequence = ''.join(self.profile.sequence)
        self.seqlen = len(self.sequence)
        self.ntindices = self.profile.nts
        
    


    def readPrimaryData(self, modfilename, minreadcoverage=None, undersample=-1, **kwargs):
        
        # Determine mincoverage quality filter
        if minreadcoverage is None and self.minreadcoverage is None:
            
            minreadcoverage = self.seqlen
            
            # if profile is defined, remove nan positions from calculation
            if self.profile is not None:
                minreadcoverage -= np.sum(np.isnan(self.profile.normprofile))

            self.minreadcoverage = int(round(minreadcoverage*0.75))
            
            print('No mincoverage specified. Using default 75% coverage = {} nts'.format(self.minreadcoverage))
        

        elif minreadcoverage is not None and self.minreadcoverage is None:
            self.minreadcoverage = minreadcoverage

        elif minreadcoverage is not None and minreadcoverage != self.minreadcoverage:
            print('WARNING: resetting self.minreadcoverage to passed value {}'.format(minreadcoverage))
            self.minreadcoverage = minreadcoverage


        # read in the matrices
        reads, mutations = aFunc.fillReadMatrices(modfilename, self.seqlen, 
                                                  self.minreadcoverage, undersample=undersample)
        
        print('{} reads for clustering\n'.format(reads.shape[0]))

        self.reads = reads
        self.mutations = mutations
           
        self.numreads = self.reads.shape[0]
        
        self.checkDataIntegrity()

 
    def checkDataIntegrity(self):
        """Check the reads and mutations conform to expected format"""
        
        checksum = 0
        for n in xrange(self.numreads):
            mask = np.array(self.mutations[n,:], dtype=bool)
            if np.sum(self.reads[n,mask]) != np.sum(mask):
                raise ValueError('Data integrity failure! Read and mutation arrays do not agree at read {}'.format(n))
            else:
                checksum += np.sum(mask)


    def computeBGprofile(self, untfilename, verbal=True, **kwargs):
        """compute BG profile from raw mutation file"""
            
        if self.minreadcoverage is None:
            raise AttributeError('minreadcoverage not set!')


        bgrate, bgdepth = aFunc.compute1Dprofile(untfilename, self.seqlen, self.minreadcoverage)
        

        if self.profile is None:
            self.profile = ReactivityProfile()
        elif verbal:
            print('Overwriting bgrate from values computed from the raw mutation file {}'.format(untfilename))
        

        self.profile.backprofile = bgrate
        
        # reset rawprofile as well
        with np.errstate(divide='ignore',invalid='ignore'):
            self.profile.rawprofile = np.sum(self.mutations, axis=0, dtype=float)/np.sum(self.reads, axis=0)
            self.profile.backgroundSubtract(normalize=False)
            
 


    def initializeActiveCols(self, invalidcols=[], inactivecols=[], invalidrate=0.0001, 
                             maxbg=0.02, minrxbg=0.002, verbal = True, maskG=False,
                             maskU=False, **kwargs):
        """Apply quality filters to eliminate noisy nts from EM fitting
        invalidcols = list of columns to set to invalid
        inactivecols = list of columns to set inactive
        invalidrate = columns with rates below this value are set to invalid
        maxbg       = maximum allowable background signal
        minrxbg     = minimum signal above background
        verbal      = keyword to control printing of inactive nts
        """
        

        ###################################
        # initialize invalid positions
        ###################################
        
        # copy so we don't modify argument
        invalidcols = list(invalidcols[:])

        if verbal and len(invalidcols)>0:
            print("Nts {} set invalid by user".format(self.ntindices[invalidcols]))


        # supplement invalid cols from profile, checking for nans
        profilenan = []
        if self.profile is not None:
            for i, val in enumerate(self.profile.normprofile):
                if val != val and i not in invalidcols:
                    invalidcols.append(i)
                    profilenan.append(i)
        
        if verbal and len(profilenan)>0:
            print("Nts {} invalid due to masking in profile file".format(self.ntindices[profilenan]))


        # Check data to exclude very low rates
        signal = np.sum(self.mutations, axis=0, dtype=np.float)
        
        lowsignal = []         
        for i in np.where(signal < invalidrate*self.numreads)[0]:
            if i not in invalidcols:
                lowsignal.append(i)
                invalidcols.append(i)

        if verbal and len(lowsignal)>0:
            print("Nts {} set to invalid due to low mutation rate".format(self.ntindices[lowsignal]))
        
        
        highbg = []
        if self.profile is not None:
            for i, val in enumerate(self.profile.backprofile):
                if val > maxbg and i not in invalidcols:
                    invalidcols.append(i)
                    highbg.append(i)
        
        if verbal and len(highbg)>0:
            print("Nts {} set to invalid due to high untreated rate".format(self.ntindices[highbg]))
        
        
        invalidcols.sort()
        self.invalid_columns = np.array(invalidcols, dtype=int)
        
        if verbal:
            print("Total invalid nts: {}".format(self.ntindices[invalidcols]))

        
        ###################################
        # initialize inactive positions
        ###################################
        
        inactive = []

        for i in inactivecols:
            if i not in self.invalid_columns:
                inactive.append(i)

        if verbal and len(inactive) > 0:
            print("Nts set inactive by user: {}".format(self.ntindices[inactive]))
        

        lowrx = []
 
        if self.profile is not None:
            with np.errstate(invalid='ignore'):
                for i in np.where(self.profile.subprofile < minrxbg)[0]:
                    if i not in inactive and i not in self.invalid_columns:
                        lowrx.append(i)
                        inactive.append(i)


        if verbal and len(lowrx) > 0:
            print("Nts {} set to inactive due to low modification rate".format(self.ntindices[lowrx]))

        
        # handle G and U nts
        if maskG:
            gcols = []
            for i,s in enumerate(self.sequence):
                if s == 'G' and i not in self.invalid_columns and i not in inactive:
                    gcols.append(i)
                    inactive.append(i)
            
            print('G columns set inactive:')
            print gcols


        if maskU:
            ucols = []
            for i,s in enumerate(self.sequence):
                if s == 'U' and i not in self.invalid_columns and i not in inactive:
                    ucols.append(i)
                    inactive.append(i)

            print('U columns set inactive:')
            print ucols

       
        inactive.sort()
        self.inactive_columns = np.array(inactive, dtype=int)

        if verbal and len(inactive) > len(lowrx):
            print("Total inactive nts: {}".format(self.ntindices[inactive]))



        ###################################
        # remaining nts are active!
        ###################################
        
        active = []
        for i in range(self.seqlen):
            if i not in self.invalid_columns and i not in self.inactive_columns:
                active.append(i)

        self.active_columns = np.array(active)
        
        self.initialActiveCount = len(self.active_columns)

        if verbal:
            print("{} initial active columns".format(self.initialActiveCount))
        
        
    
                    
    def setColumns(self, activecols=None, inactivecols=None):
        """Set columns to specified values, if changed"""
        
        # invalid_columns is usually initialized, but if circumventing initializeActiveCols
        # via direct call to setColumns then need to init
        if self.invalid_columns is None:
            allcols = np.ones(self.seqlen, dtype=bool)
            allcols[activecols] = False
            allcols[inactivecols] = False
            invalid = np.where(allcols)[0]
            self.invalid_columns = np.array(invalid, dtype=int)
            if len(self.invalid_columns) > 0:
                print("Cols {} initialized to invalid in setColumns".format(self.invalid_columns))


        if inactivecols is not None and np.sum(np.isin(inactivecols, self.invalid_columns)) > 0:
            conflict = np.where(np.isin(inactivecols, self.invalid_columns))[0]
            print('WARNING! columns {} are invalid and cannot be set inactive'.format(inactivecols[conflict]))
            inactivecols = np.delete(inactivecols, conflict)
            
        if activecols is not None and np.sum(np.isin(activecols, self.invalid_columns)) > 0:
            conflict = np.where(np.isin(activecols, self.invalid_columns))[0]
            print('WARNING! columns {} are invalid and cannot be set active'.format(activecols[conflict]))
            activecols = np.delete(activecols, conflict)
         


        if activecols is not None and inactivecols is not None:
            
            # if the same, don't do anything
            if np.array_equal(activecols, self.active_columns) and \
                    np.array_equal(inactivecols, self.inactive_columns):
                return

            # make sure they aren't conflicting
            if np.sum(np.isin(activecols, inactivecols))>0 or \
                    np.sum(np.isin(inactivecols, activecols))>0:
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
            mask = mask & np.isin(indices, self.invalid_columns, invert=True)

            self.inactive_columns = np.array(indices[mask])
        

        elif inactivecols is not None:
            
            if np.array_equal(inactivecols, self.inactive_columns):
                return

            self.inactive_columns = np.array(inactivecols)

            indices = np.arange(self.seqlen)
            
            mask = np.isin(indices, self.inactive_columns, invert=True)
            mask = mask & np.isin(indices, self.invalid_columns, invert=True)

            self.active_columns = np.array(indices[mask])
 


    
    def setActiveColumnsInactive(self, columns, verbal=False):
        """Add currently active columns to the list of inactive columns
        -columns is a list of *active* column indices to set to inactive
        """
        
        if len(columns) == 0:
            return
        
        if len(self.active_columns)-len(columns) < 0.8*self.initialActiveCount:
            print("Call to setActiveColumnsInactive ignored -- maximum inactive exceeded")
            return

        self.inactive_columns = np.append(self.inactive_columns, columns)
        self.inactive_columns.sort()

        self.active_columns = np.array([i for i in self.active_columns if i not in columns])

        if verbal:
            print("INACTIVE LIST UPDATE")
            print("\tNew inactive nts :: {}".format(self.ntindices[columns]))
            print("\tTotal inactive nts :: {}".format(self.ntindices[self.inactive_columns]))

    

    
    def compute1ComponentModel(self):
        """Compute the null model (i.e. mixture of one)"""
        
        # create 1D BM object and assign its p/mu params
        BM = BernoulliMixture(pdim=1, mudim=self.seqlen, 
                              active_columns=self.active_columns,
                              inactive_columns=self.inactive_columns,
                              idxmap=self.ntindices)
        
        BM.fitEM(self.reads, self.mutations)
        
        return BM



    def fitEM(self, components, trials=5, soln_termcount=3, badcolcount0=2, badcolcount=5, 
              priorWeight=0.001, verbal=False, writeintermediate=None, **kwargs):
        """Fit Bernoulli Mixture model of a specified number of components.
        Trys a number of random starting conditions. Terminates after finding a 
        repeated valid solution, a repeated set of 'invalid' column solutions, 
        or after a maximum number of trials.

        components = number of model components to fit
        trials = max number of fitting trials to run
        soln_termcount = terminate after this many identical solutions founds
        badcolcount0 = inactivate columns after this many failures with no valid solns
        badcolcount  = inactivate columns after this many failures (with valid solns)
        
        priorWeight = weight of relative prior used during fitting. If -1, disable

        writeintermediate = write out each BM soln to specified prefix
        verbal = T/F on whether to print results of each trial
        
        additional kwargs are passed onto BernoulliMixture.fitEM
        
        returns:
            bestfit     = bestfit BernoulliMixture object
            fitlist     = list of other BernoulliMixture objects
        """
        

        try:
            if self.profile.backprofile is None:
                priorWeight = -1
        except AttributeError:
            priorWeight = -1


        if verbal and priorWeight>0:
            print('Using priorWeight={0}'.format(priorWeight))
        
        # array for each col; incremented when a col causes failure of BM soln
        badcolumns = np.zeros(self.seqlen, dtype=np.int32)
         
        
        if self.active_columns is None:
            print('active_columns not set... initializing')
            self.initializeActiveCols(verbal=True)

       
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
            
            # set the prior relative to expected background rate and read depth
            # leave priorB=1
            # (defaults used within BernoulliMixture are A=1, B=1)
            if priorWeight > 0:
                BM.setPriors(priorWeight*self.profile.backprofile*np.sum(self.reads, axis=0), 1)


            # fit the BM
            BM.fitEM(self.reads, self.mutations, verbal=verbal, **kwargs)
 
            
            if BM.converged:
                
                fitlist.append(BM)
                badcolumns[:] = 0 # reset badcolumns
                

                if writeintermediate is not None:
                    BM.writeModel('{0}-intermediate-{1}-{2}.bm'.format(writeintermediate, components, tt))
                
                if bestfit is None:
                    bestfit = BM
                    continue
                
                
                bestfitBIC = self.compareBIC(BM, bestfit, verbal=verbal)
                
                # doesn't matter if active_columns are different; diff computed
                # using the smallest list
                pdiff, mudiff = BM.modelDifference(bestfit, func=np.max)

                if pdiff < 0.03 and mudiff < 0.01:
                    bestfitcount += 1
                    if BM.BIC < bestfitBIC:
                        bestfit = BM
                
                # solution is different and better
                elif BM.BIC < bestfitBIC:
                    bestfitcount = 1
                    bestfit = BM
                
            
           
            else:  # solution did not coverge
                
                # increment bad cols 
                # (BM.cError.badcolumns will be empty if not badcolumn error)
                badcolumns[BM.cError.badcolumns] += 1
                
                if len(fitlist)==0:
                    bc = np.where(badcolumns>=badcolcount0)[0]
                else:
                    bc = np.where(badcolumns>=badcolcount)[0]

                self.setActiveColumnsInactive(bc, verbal=verbal)
               
                # zero out columns that we've now set to inactive so won't be triggered in future
                badcolumns[bc] = 0


        # END of while loop
        
        if verbal and bestfit is not None:
            self.printEMFitSummary(bestfit, fitlist)
            self.qualityCheck(bestfit)

        if bestfit is not None and bestfitcount != soln_termcount:
            bestfit = None
            if verbal:
                print('\nBestfit solution only found {} times -- unstable!!!\n'.format(bestfitcount))
        
        elif bestfit is not None and verbal:
            print('{0} identical fits found'.format(bestfitcount))
        


        # reset the active/inactive cols if necessary
        if bestfit is not None:
            self.setColumns(activecols=bestfit.active_columns, 
                            inactivecols=bestfit.inactive_columns)    
        

        return bestfit
        


    
    def compareBIC(self, BMnew, BMold, verbal=False):
        """Return BIC of BMold computed using the same set of active_columns as BMnew"""

        # check if BMold has the same active_columns
        if np.array_equal(BMnew.active_columns, BMold.active_columns):
            return BMold.BIC
        

        # if we get here then we need to refit
        if verbal:  
            print('\tRefitting prior {0}-component model with different inactive columns'.format(BMold.pdim))
        
        
        ll,bic = BMold.computeModelLikelihood(self.reads, self.mutations, 
                                              active_columns=BMnew.active_columns)

        return bic

                

    
    
    def findBestModel(self, maxcomponents=5, verbal=False, writeintermediate=None, **kwargs):
        """Fit BM model for progessively increasing number of model components 
        until model with best BIC is found. 
        Dynamically updates inactive list as problematic columns are identified

        maxcomponents = maximum number of components to attempt fitting

        **kwargs passed onto fitEM method"""


        if self.active_columns is None:
            print('active_columns not set... initializing')
            self.initializeActiveCols(verbal=True)


        # best BernoulliMixture solution. Assign as 1-component solution to start
        overallBestBM = self.compute1ComponentModel()
        

        if verbal:  
            print("\n 1-component BIC = {0:.1f}".format(overallBestBM.BIC))
            print('*'*50+'\n')
        

        if writeintermediate is not None:
            overallBestBM.writeModel('{0}-intermediate-1.bm'.format(writeintermediate))
 


        # iterate through each model size
        for c in xrange(2, maxcomponents+1):
            
            if verbal: print('\nAdvancing to {}-component model\n'.format(c))


            bestBM = self.fitEM(c, verbal=verbal, writeintermediate=writeintermediate, **kwargs)
             

            # terminate if no valid solution found             
            if bestBM is None:
                if verbal:  print("No valid solution for {0}-component model".format(c))
                break
             
       
            priorBIC = self.compareBIC(bestBM, overallBestBM, verbal=verbal)

            deltaBIC = bestBM.BIC-priorBIC
            
            if verbal:
                print("{0}-component model BIC={1:.1f} ==> dBIC={2:.1f}".format(c-1, priorBIC, deltaBIC))


            # if BIC is not better, terminate search
            if deltaBIC > -46: # p>1e10
                break
            else:
                overallBestBM = bestBM
                if verbal:
                    print("{0}-component model assigned as new best model!".format(c))
                    
            
        # End for loop

        if verbal:
            print('{0}-component model selected'.format(overallBestBM.pdim))
        

        # reset the active/inactive cols if necessary
        self.setColumns(activecols=overallBestBM.active_columns, 
                        inactivecols=overallBestBM.inactive_columns)    
        
        overallBestBM.imputeInactiveParams(self.reads, self.mutations)

        self.BMsolution = overallBestBM
        self.BMsolution.sort_model()

        self.qualityCheck()

            
        return self.BMsolution

    


    def printEMFitSummary(self, bestfit, fitlist):
         
        print "\n{0} fits found for {1}-component model".format(len(fitlist), bestfit.pdim)
        print "Best Fit BIC = {0:.1f}".format( bestfit.BIC )
        
        print("Deviation of fits from best parameters:")
        
        for i,f in enumerate(fitlist):

            pdiff, mudiff = bestfit.modelDifference(f, func=np.max)
            
            bmcode = ''
            if f is bestfit:
                bmcode='Best'

            print("\tModel {0}  Active={1}  BIC={2:.1f}  Max_dP={3:.4f}  Max_dMu={4:.4f} {5}".format(i+1, len(f.active_columns), f.BIC, pdiff, mudiff,bmcode))


        print('*'*65+'\n')
 

    
    def qualityCheck(self, bm=None):
        """Check that model is well defined"""
        
        if bm is None:
            bm = self.BMsolution

        if len(bm.p) == 1:
            return
        
        rms = bm.model_rms_diff()
        absmean = bm.model_absmean_diff()
        ndiff = bm.model_num_diff()
        p_err = max(bm.p_err)
       
        count = 0
    
        print('\n-----------------------------------------')
        print('Quality checks:')
        
        if rms > 0.01:
            print('Min Mu RMS Diff = {0:.3f}    PASSED'.format(rms))
        else:
            print('Min Mu RMS Diff = {0:.3f}    FAILED'.format(rms))
            count += 1

        if absmean > 0.01:
            print('Min Mu Mean Diff = {0:.3f}   PASSED'.format(absmean))
        else:
            print('Min Mu Mean Diff = {0:.3f}   FAILED'.format(absmean))
            count += 1

        if ndiff > 20:
            print('Min # Diff Mu = {}         PASSED'.format(ndiff))
        else:
            print('Min # Diff Mu = {}         FAILED'.format(ndiff))
            count += 1


        if p_err < 0.01:
            print('Max P error = {0:.3f}        PASSED'.format(p_err))
        else:
            print('Max P error = {0:.3f}        FAILED'.format(p_err))
            count += 1
        

        if count == 0:
            print('\nAll checks PASSED!')
        else:
            print('\nWARNING: {}/4 checks FAILED!'.format(count))
            print('\t\tSolution may be untrustworthy')
        
        print('-----------------------------------------\n')
       



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

######## confirm that normalization is being done correctly !

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




    def writeReactivities(self, output):
        """Print out model reactivities
        self.profile must be defined
        """
        
        model = self.BMsolution

        if self.profile is None:
            print('Cannot compute reactivities because profile was not provided')
            return
        
        # compute normalized parameters
        profiles = self.computeNormalizedReactivities()
        

        with open(output, 'w') as OUT:

            OUT.write("{0} components; BIC={1:.1f}\n".format(model.pdim, model.BIC))
            
            pline = 'p'
            for p in model.p:
                pline += ' {0:.3f}'.format(p)
            OUT.write(pline+'\n')
            
            OUT.write('Nt\tSeq\t')
            OUT.write('nReact\tRaw\t\t'*model.pdim+'Background\n')
            
            
            for nt in xrange(model.mudim):
                
                muline = '{0}\t{1}\t'.format(self.ntindices[nt], self.profile.sequence[nt])
                
                if nt in self.invalid_columns:
                    muline += '{0}{1:.4f}'.format('nan\tnan\t\t'*model.pdim, self.profile.backprofile[nt])

                else:
                    for prof in profiles:
                        muline += '{0:.4f}\t{1:.4f}\t\t'.format(prof.normprofile[nt], prof.rawprofile[nt])
                    
                    muline += '{0:.4f}'.format(self.profile.backprofile[nt])


                    if nt in self.inactive_columns:
                        muline += ' i'

                OUT.write(muline + '\n')
            


    def readModelFromFile(self, fname):
        """Read in BMsolution from BM file object"""
        

        self.BMsolution = BernoulliMixture()
        self.BMsolution.readModelFromFile(fname)

        # check to make sure the active, inactive are the same
        if not np.array_equal(self.active_columns, self.BMsolution.active_columns) or \
               np.array_equal(self.inactive_columns, self.BMsolution.inactive_columns):
            sys.stderr.write('active_columns in BMsolution and EnsembleMap object are different!\n')
            sys.stderr.write('Updating EnsembleMap columns to BMsolution values\n')
            
            self.setColumns(activecols=self.BMsolution.active_columns, 
                            inactivecols=self.BMsolution.inactive_columns)    
        


 

    def _sample_RINGs(self, window=1, corrtype='apc', bgfile=None, assignprob=0.9, 
                      subtractwindow=True, montecarlo=False, verbal=True):
        """Assign sample reads based on posterior prob and return list of RINGexperiment objs
        window     = correlation window
        corrtype   = metric for computing correlations
        bgfile     = parsed mutation file for bg sample (to filter out bg mutations)
        assignprob = posterior prob. used for assigning reads to models
        verbal     = verbal"""

        
        # setup the activestatus mask
        activestatus = np.zeros(self.seqlen, dtype=np.int8)
        activestatus[self.active_columns] = 1
        activestatus[self.inactive_columns] = 1
     
        
        # fill in the matrices
        if montecarlo:
            if verbal:
                print('Using MC for sample RING read assignment')
            
            read, comut, inotj = aFunc.fillRINGMatrix_montecarlo(self.reads, self.mutations, activestatus,
                                                                 self.BMsolution.mu, self.BMsolution.p, 
                                                                 window, self.reads.shape[0], subtractwindow)
        
        else:
            if verbal:
                print('Using {:.3f} as posterior prob for sample RING read assignment'.format(assignprob))

            read, comut, inotj = aFunc.fillRINGMatrix(self.reads, self.mutations, activestatus,
                                                      self.BMsolution.mu, self.BMsolution.p, window, assignprob, 
                                                      subtractwindow)
        


        relist = [] 
        
        # populate RINGexperiment objects
        for p in xrange(self.BMsolution.pdim):
            
            ring = RINGexperiment(arraysize=self.seqlen, corrtype=corrtype, verbal=verbal)

            ring.sequence = self.sequence

            ring.window = window
            ring.ex_readarr = read[p]
            ring.ex_comutarr = comut[p]
            ring.ex_inotjarr = inotj[p]
            
            # fill bg arrays (only need to do once; copy for >0 models)
            if bgfile is not None:
                if p==0:
                    ring.initDataMatrices('bg', bgfile, window=window, 
                                          mincoverage=self.minreadcoverage, verbal=verbal)
                else:
                    ring.bg_readarr = relist[0].bg_readarr
                    ring.bg_comutarr = relist[0].bg_comutarr
                    ring.bg_inotjarr = relist[0].bg_inotjarr
            
            ring.computeCorrelationMatrix(verbal=verbal, ignorents=self.invalid_columns)
            
            #ring.writeDataMatrices('ex', 't-{}-{}-{}-{}'.format(p,subtractwindow,assignprob,montecarlo))


            relist.append(ring)
        
        if verbal: print('\n')

        return relist

 

    def _null_RINGs(self, window=1, corrtype='g', assignprob=0.9, 
                    subtractwindow=True, montecarlo=False, verbal=True):
        """Create null (uncorrelated) model and assign reads based on posterior prob 
        to determine null-model correlations. 
        Returns list of RINGexperiment objs
        window     = correlation window
        corrtype   = metric for computing correlations
        bgfile     = parsed mutation file for bg sample (to filter out bg mutations)
        assignprob = posterior prob. used for assigning reads to models
        verbal     = verbal"""

       
        from SynBernoulliMixture import SynBernoulliMixture
        
        # initialize synthetic model, ensuring that invalid columns are masked out
        mu = np.copy(self.BMsolution.mu)
        mu[:,self.invalid_columns] = -1
        null_model = SynBernoulliMixture(p=self.BMsolution.p, mu=mu)
        
        # generate synthetic reads 
        nullEM = null_model.getEMobject(self.reads.shape[0], nodata=0.1, 
                                        invalidcols=self.invalid_columns,
                                        verbal=False)
 

        # setup the activestatus mask
        activestatus = np.zeros(self.seqlen, dtype=np.int8)
        activestatus[self.active_columns] = 1
        activestatus[self.inactive_columns] = 1
        
        
        # fill in the matrices
        if montecarlo:
            if verbal:
                print('Using MC for null RING read assignment')

            read, comut, inotj = aFunc.fillRINGMatrix_montecarlo(nullEM.reads, nullEM.mutations, activestatus,
                                                                 mu, self.BMsolution.p, 
                                                                 window, self.reads.shape[0], subtractwindow)
        else:
            if verbal:
                print('Using {:.3f} as posterior prob for null RING read assignment'.format(assignprob))

            read, comut, inotj = aFunc.fillRINGMatrix(nullEM.reads, nullEM.mutations, activestatus,
                                                      mu, self.BMsolution.p, window, assignprob, 
                                                      subtractwindow)
 

        relist = []

        # populate RINGexperiment objects
        for p in xrange(self.BMsolution.pdim):

            ring = RINGexperiment(arraysize=self.seqlen, corrtype=corrtype, verbal=False)

            ring.sequence = self.sequence

            ring.window = window
            ring.ex_readarr = read[p]
            ring.ex_comutarr = comut[p]
            ring.ex_inotjarr = inotj[p]
            
            ring.computeCorrelationMatrix(mincount=10, ignorents=self.invalid_columns)
            #ring.writeDataMatrices('ex', 'null-{}.txt'.format(p))
            relist.append(ring)
            
        return relist



    def computeRINGs(self, window=1, bgfile=None, assignprob=0.9,
                     subtractwindow=True, montecarlo=False, nulldifftest=True, verbal=True):
        """Compute RINGs based on posterior prob and mask out (i,j) pairs that are
        observed in null (uncorrelated) model. 
        Return list of RINGexperiment objects

        window     = correlation window
        corrtype   = metric for computing correlations
        bgfile     = parsed mutation file for bg sample (to filter out bg mutations)
        assignprob = posterior prob. used for assigning reads to models
        nulldifftest = perform difference G-test against null model as well
        verbal     = verbal"""
 

        # compute rings from experimental data
        sample = self._sample_RINGs(window=window, bgfile=bgfile, assignprob=assignprob, 
                                    subtractwindow=subtractwindow, montecarlo=montecarlo, verbal=verbal) 

        # compute correlations from null model (clustering only w/o correlations)
        null = self._null_RINGs(window=window, assignprob=assignprob, 
                                subtractwindow=subtractwindow, montecarlo=montecarlo, verbal=verbal) 


        # iterate through each model and mask out corrs present in the null model
        for p in range(self.BMsolution.pdim):
            
            sample_p = sample[p]
            null_p = null[p]
            
            for i,j in itertools.combinations(range(self.seqlen), 2):

                nullcorr = null_p.ex_correlations[i,j]
                nulldiff = sample_p.significantDifference(i,j, null_p.ex_readarr[i,j], null_p.ex_inotjarr[i,j],
                                                          null_p.ex_inotjarr[j,i], null_p.ex_comutarr[i,j])
                
                # null correlated @ p<0.001 level or not significantly different @ p<1e-6
                if nullcorr>10.83 or (nulldifftest and nulldiff<23.9):
                
                    if verbal and sample_p.ex_correlations[i,j]>23.9:
                        outstr='Model {}: Correlated pair ({},{}) w/ chi2={:.1f} ignored: NULL correlation chi2={:.1f} ; NULL difference chi2={:.1f}'.format(p,i+1,j+1, sample_p.ex_correlations[i,j], nullcorr, nulldiff)
                        print(outstr)

                    sample_p.ex_correlations[i,j] = np.ma.masked
                    sample_p.ex_correlations[j,i] = np.ma.masked
                    sample_p.ex_zscores[i,j] = np.ma.masked
                    sample_p.ex_zscores[j,i] = np.ma.masked
            
        return sample
                                     



    def writeReadsMutations(self, outputfile):

        with open(outputfile, 'w') as out:

            for i in range(self.reads.shape[0]):

                out.write('MERGED {0} {1} {2} INCLUDED 0 0 '.format(i, 0, self.reads.shape[1]))
                np.savetxt(out, self.reads[i,:], fmt='%d', newline='')
                out.write(' ')
                np.savetxt(out, self.mutations[i,:], fmt='%d', newline='')
                out.write('\n')



    
####################################################################################



def parseArguments():

    parser = argparse.ArgumentParser(description = "Fit BM model to data")
    
    optional = parser._action_groups.pop()
   

    ############################################################
    # required arguments

    required = parser.add_argument_group('required arguments')
    required.add_argument('--modified_parsed', help='Path to modified parsed.mut file')
    required.add_argument('--profile', help='Path to profile.txt file')
    

    ############################################################
    # Quality filtering arguments
    
    quality = parser.add_argument_group('quality filtering options')
    quality.add_argument('--mincoverage', type=int, help='Minimum coverage (integer number of nts) required for read to be included in cacluations')
    quality.add_argument('--minrxbg', type=float, default=0.002, help='Set nts with rx-bg less than this to inactive (default=0.002)')
    quality.add_argument('--undersample', type=int, default=-1, help='Only cluster with this number of reads. By default this option is disabled and all reads are used (default=-1).')
    

    ############################################################
    # Fitting options

    fitopt = parser.add_argument_group('options for fitting data')
    fitopt.add_argument('--fit', action='store_true', help='Flag specifying to fit data')
    fitopt.add_argument('--maxcomponents', type=int, default=5, help='Maximum number of components to fit (default=5)')
    fitopt.add_argument('--trials', type=int, default=50, help='Maximum number of fitting trials at each component number (default=50)')
    fitopt.add_argument('--badcol_cutoff', type=int, default=5, help='Inactivate column after it causes a failure X number of times *after* a valid soln has already been found (default=5)')
    fitopt.add_argument('--writeintermediates', action='store_true', help='Write each BM solution to file with specified prefix. Will be saved as prefix-intermediate-[component]-[trial].bm')

    fitopt.add_argument('--priorWeight', type=float, default=0.001, help='Weight of prior on Mu (default=0.001). Prior = priorWeight*readDepth*bgRate at each nt. Prior is disabled by passing -1, upon which a naive prior is used.')
    fitopt.add_argument('--maskG', action='store_true', help='set all Us to inactive')
    fitopt.add_argument('--maskU', action='store_true', help='set all Gs to inactive')



    ############################################################
    # RING options
    
    ringopt = parser.add_argument_group('options for performing RING analysis on clustered reads')
    ringopt.add_argument('--ring', action='store_true')
    ringopt.add_argument('--window', type=int, default=1, help='Window size for computing correlations (default=1)')

    ringopt.add_argument('--pairmap', action='store_true', help='Run PAIR-MaP analysis on clustered reads') 
    ringopt.add_argument('--readprob_cut', type=float, default=0.9, help='Posterior probability cutoff for assigning reads for inclusion in ring/pairmap analysis. Reads must have posterior prob greater than the cutoff (default=0.9)')
    ringopt.add_argument('--chisq_cut', type=float, default=23.9, help="Set chisq cutoff for RING/PAIR-MaP analysis (default = 23.9)")
    ringopt.add_argument('--nowindowsubtract', action='store_false')

    ############################################################
    # Other options
    
    optional.add_argument('--untreated_parsed', help='Path to modified parsed.mut file')

    optional.add_argument('--readfromfile', type=str, help='Read in model from BM file')
    optional.add_argument('--suppressverbal', action='store_false', help='Suppress verbal output')
    optional.add_argument('--outputprefix', type=str, default='emfit', help='Write output files with this prefix (default=emfit)')
    


    parser._action_groups.append(optional)

    args = parser.parse_args()
     
    
    ############################################################
    # Check that required arguments were passed

    if args.modified_parsed is None:
        sys.stderr.write("\nmodified_parsed argument not provided\n\n")
        sys.exit(1)
        
    if args.profile is None:
        sys.stderr.write("\nprofile argument not provided\n\n")
        sys.exit(1)
     
    
    if not args.fit and not args.ring and not args.pairmap:
        sys.stderr.write("\n Action argument [fit, ring, pairmap] not provided\n\n")
        sys.exit(1)
    

    if not 0<args.priorWeight<=1 and args.priorWeight != -1:
        sys.stderr.write('\npriorWeight value = {} is invalid!\n\n'.format(args.priorWeight))
        sys.exit(1)

    ############################################################
    # reformat arguments as necessary for downstream applications

    if args.writeintermediates:
        args.writeintermediates = args.outputprefix
    else:
        args.writeintermediates = None
    


    return args









if __name__=='__main__':
    
    args = parseArguments()
    
    EM = EnsembleMap(modfile=args.modified_parsed, untfile=args.untreated_parsed,
                     profilefile=args.profile, 
                     minrxbg = args.minrxbg,
                     maskG = args.maskG,
                     maskU = args.maskU,
                     minreadcoverage=args.mincoverage, 
                     undersample=args.undersample,
                     verbal=args.suppressverbal)
       
    if args.fit:
        
        EM.findBestModel(args.maxcomponents, trials=args.trials,
                         badcolcount = args.badcol_cutoff,
                         priorWeight = args.priorWeight,
                         verbal=args.suppressverbal,
                         writeintermediate = args.writeintermediates)

        EM.writeReactivities(args.outputprefix+'-reactivities.txt')
        EM.BMsolution.writeModel(args.outputprefix+'.bm')

    
    elif args.readfromfile is not None:

        EM.readModelFromFile(args.readfromfile)


    if args.ring:

        RE_list = EM.computeRINGs(window=args.window, bgfile=args.untreated_parsed, subtractwindow=args.nowindowsubtract,
                                  assignprob=args.readprob_cut, verbal=args.suppressverbal)

        for i,model in enumerate(RE_list):
            model.writeCorrelations('{0}-{1}-rings.txt'.format(args.outputprefix, i),
                                    chi2cut=args.chisq_cut)



    if args.pairmap:
        
        profiles = EM.computeNormalizedReactivities()

        RE_list = EM.computeRINGs(window=3, bgfile=args.untreated_parsed, subtractwindow=args.nowindowsubtract,
                                  assignprob=args.readprob_cut, verbal=args.suppressverbal)

        for i,model in enumerate(RE_list):
            
            if args.suppressverbal:
                print('--------------Computing PAIRs : Model {}--------------'.format(i))


            model.writeCorrelations('{0}-{1}-allcorrs.txt'.format(args.outputprefix,i), 
                                    chi2cut=args.chisq_cut)

            pairs = PairMapper(model, profiles[i])
            pairs.writePairs('{0}-{1}-pairmap.txt'.format(args.outputprefix, i))
            pairs.writePairBonusFile('{0}-{1}-pairmap.bp'.format(args.outputprefix, i))      


            


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
