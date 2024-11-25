import os
import numpy as np
import sys, argparse, itertools
import datetime
import warnings
#suppress warning from cython when using a different version of numpy vs what cython compiled
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

# get path to functions needed for mutstring I/O
import externalpaths
sys.path.append(externalpaths.structureanalysistools())
from ReactivityProfile import ReactivityProfile

sys.path.append(externalpaths.ringmapper())
from ringmapper import RINGexperiment
from pairmapper import PairMapper


import accessoryFunctions as aFunc
from BernoulliMixture import BernoulliMixture
import concat_profile_mut_N as concat_prof_mut
import de_cat_N7_N13 as de_cat





class DanceMap(object):

    def __init__(self, modfile=None, untfile=None, profilefile=None, seqlen=None, ignorebg=False, concat=False, **kwargs):
        """Define important global parameters"""
    
        # reads contains positions that are 'read'
        # mutations contains positions that are mutated
        self.reads = None 
        self.mutations = None
        
        # Affiliated ReactivityProfile object
        self.profile=None
    
        # np.array of sequence positions to actively cluster 
        self.active_columns=None
        
        # number of active columns at beginning of clustering
        self.initialActiveCount = 1

        # np.array of sequence positions to 'inactively' cluster
        self.inactive_columns=None

        # np.array of sequence positions that are 'invalid' -- don't cluster at all
        self.invalid_columns=None
        
        self.maxinactive = 0.8 # max fraction of initialActiveCols allowed to be inactivated

        # info about the reads
        self.seqlen = None
        self.ntindices = None
        self.sequence = None
        self.numreads = None
        self.minreadcoverage = None
        
        # contains final BMsolution, if fitting done
        self.BMsolution = None

        # Arrays for ring experiments
        self.ring_read = None
        self.ring_comut = None
        self.ring_inotj = None
        self.null_read = None
        self.null_comut = None
        self.null_inotj = None

        self.concat=concat
        if self.concat:
           self.inactivate_n7 = kwargs["inactivate_n7"]
           #print "Inactivate N7: {} ".format(self.inactivate_n7)
        else:
           self.inactivate_n7 = None
        
        if profilefile is not None:
            self.init_profile(profilefile, ignorebg)

        elif seqlen is not None:
            self.seqlen = seqlen
            self.ntindices = np.arange(1, seqlen+1, dtype=np.int32)

        
        # first initialize reads/mutations. 
        # Note this will set mincoverage (can be specified through kwargs)
        if modfile is not None:
            self.readPrimaryData(modfile, **kwargs)
        

        # if provided, now update self.profile.backprofile with rate computed 
        # using reads filtered according to same criteria for primary data
        if untfile is not None:
            self.computeBGprofile(untfile, **kwargs)

        
        if self.reads is not None:
            self.initializeActiveCols(**kwargs)




    def init_profile(self, profilefile, ignorebg=False):

        self.profile = ReactivityProfile(profilefile)
        
        if ignorebg or np.isnan(self.profile.backprofile).all():
            # this logic makes sure subprofile is computed only using modified sample, while also
            # filling backprofile with a baseline error rate for prior calculations
            self.profile.backprofile.fill(0)
            self.profile.subprofile = np.copy(self.profile.rawprofile)

        self.sequence = ''.join(self.profile.sequence)
        self.seqlen = len(self.sequence)
        self.ntindices = self.profile.nts
        
    


    def readPrimaryData(self, modfilename, minreadcoverage=None, undersample=-1, **kwargs):
        # Determine mincoverage quality filter


        #Hacky way to ensure primers are NaN'd out of the default read coverage. Before this the 
        #only options NaN'd out in the normProfile were Nts where the background was greater than
        # .02. All the rest, including primers, remained not NaN.
        if self.profile is not None:
           self.profile.normalize()

        if minreadcoverage is None and self.minreadcoverage is None:
            
            minreadcoverage = self.seqlen
            
            # if profile is defined, remove nan positions from calculation
            if self.profile is not None:
                minreadcoverage -= np.sum(np.isnan(self.profile.normprofile))



            self.minreadcoverage = int(round(minreadcoverage*0.75))

            # Avoid erroneous doubling of minreadcoverage due to concatenation
            if self.concat:
                self.minreadcoverage = int(round(self.minreadcoverage / 2))
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
        
        for n in xrange(self.numreads):
            mask = np.array(self.mutations[n,:], dtype=bool)
            if np.sum(self.reads[n,mask]) != np.sum(mask):
                raise ValueError('Data integrity failure! Read and mutation arrays do not agree at read {}'.format(n))



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
                             maskU=False, maskN=False, **kwargs):
        """Apply quality filters to eliminate noisy nts from EM fitting
        invalidcols = list of columns to set to invalid
        inactivecols = list of columns to set inactive
        invalidrate = columns with rates below this value are set to invalid
        maxbg       = maximum allowable background signal
        minrxbg     = minimum signal above background
        maskG       = set all G nts to inactive
        maskU       = set all U nts to inactive
        verbal      = keyword to control printing of inactive nts
        """
        

        ###################################
        # initialize invalid positions
        ###################################
        
        # copy so we don't modify argument
        invalidcols = list(invalidcols)
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
        
        # check backprofile to exclude high bg positions
        highbg = []
        if self.profile is not None and self.profile.backprofile is not None:
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
        
        # copy user specified inactivecols, making sure non are invalid
        for i in inactivecols:
            if i not in self.invalid_columns:
                inactive.append(i)


        if verbal and len(inactive) > 0:
            print("Nts set inactive by user: {}".format(self.ntindices[inactive]))
        
        
        # determine low reactivity cols
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
            
            print('Remaining G nts set inactive:'.format(self.ntindices[gcols]))


        if maskU:
            ucols = []
            for i,s in enumerate(self.sequence):
                if s == 'U' and i not in self.invalid_columns and i not in inactive:
                    ucols.append(i)
                    inactive.append(i)
            
            print('Remaining U nts set inactive:'.format(self.ntindices[ucols]))
        

        if maskN:
            ncols = []
            for i,s in enumerate(self.sequence):
                if s == 'N' and i not in self.invalid_columns and i not in inactive:
                    ncols.append(i)
                    inactive.append(i)
            
       

        #IF N7 and inactivate_n7 (check kwargs) then 
            #For column in N7 columns if column not in invalid and not in inactive, append
        if self.concat:
           if self.inactivate_n7:
              for i, nt in enumerate(self.sequence[len(self.sequence)/2:]):
                 
                 i_adjust = i + len(self.sequence) / 2
                 


                 if i_adjust not in self.invalid_columns and i_adjust not in inactive:
                    
                    inactive.append(i_adjust)

           else:
              for i, nt in enumerate(self.sequence[len(self.sequence)/2:]):
                 i_adjust = i + len(self.sequence) / 2

                 if i_adjust in inactive and i_adjust in self.invalid_columns:
                    print("Error! nt in both inactive and invalid")
                 #elif i_adjust not in self.invalid_columns and i_adjust not in inactive:
                 #   print("{} is active".format(i_adjust))

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

        # identify and delete any cols specified inactive that are invalid
        if inactivecols is not None and np.sum(np.isin(inactivecols, self.invalid_columns)) > 0:
            conflict = np.where(np.isin(inactivecols, self.invalid_columns))[0]
            print('WARNING! columns {} are invalid and cannot be set inactive'.format(inactivecols[conflict]))
            inactivecols = np.delete(inactivecols, conflict)
        
        # identify and delete any active cols that are invalid
        if activecols is not None and np.sum(np.isin(activecols, self.invalid_columns)) > 0:
            conflict = np.where(np.isin(activecols, self.invalid_columns))[0]
            print('WARNING! columns {} are invalid and cannot be set active'.format(activecols[conflict]))
            activecols = np.delete(activecols, conflict)
         

        # if both active and inactive are passed
        if activecols is not None and inactivecols is not None:
            
            # if active+inactive are same as self, don't do anything
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
        

        # if only activecols is passed
        elif activecols is not None:
            
            # don't do anything if activecols is unchanged from self
            if np.array_equal(activecols, self.active_columns):
                return

            self.active_columns = np.array(activecols)
            
            # determine by process of elimination the inactive columns
            indices = np.arange(self.seqlen)
            mask = np.isin(indices, self.active_columns, invert=True)
            mask = mask & np.isin(indices, self.invalid_columns, invert=True)

            self.inactive_columns = np.array(indices[mask])
        
        # if only inactivecols is passed
        elif inactivecols is not None:
            
            if np.array_equal(inactivecols, self.inactive_columns):
                return

            self.inactive_columns = np.array(inactivecols)
            
            # determine by process of elimination the active columns
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
        
        if len(self.active_columns)-len(columns) < self.maxinactive*self.initialActiveCount:
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



    def fitEM(self, components, trials=50, soln_termcount=3, badcolcount0=2, badcolcount=5, 
              priorWeight=0.01, verbal=False, writeintermediate=None, forcefit=False, **kwargs):
        """Fit Bernoulli Mixture model of a specified number of components.
        Trys a number of random starting conditions. Terminates after finding a 
        repeated valid solution or after a maximum number of trials.

        components = number of model components to fit
        trials = max number of fitting trials to run
        soln_termcount = terminate after this many identical solutions founds
        badcolcount0 = inactivate columns after this many failures when no valid soln has yet been found
        badcolcount  = inactivate columns after this many failures if valid soln already found
        
        priorWeight = weight of relative prior used during fitting. If -1, disable

        writeintermediate = write out each BM soln to specified prefix
        verbal = T/F on whether to print results of each trial
        forcefit = try extra hard to fit specified number of components by relaxing thresholds

        additional kwargs are passed onto BernoulliMixture.fitEM
        
        returns:
            bestfit     = bestfit BernoulliMixture object
            fitlist     = list of other BernoulliMixture objects
        """
        

        try:
            if self.profile.backprofile is None or np.all(self.profile.backprofile==0):
                priorWeight = -1
        except AttributeError:
            priorWeight = -1


        if verbal and priorWeight>0:
            print('Using priorWeight={0}'.format(priorWeight))
        

        # allow extra leniency in column inactivation to achieve fit
        if forcefit:
            self.maxinactive=0.5


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
            else:
                BM.setPriors(1e-4*np.sum(self.reads, axis=0), 1)


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
                
                # compute BIC of old bestfit with new active_columns if they have changed
                bestfitBIC = self.compareBIC(BM, bestfit, verbal=verbal)
                
                # compute the difference betwene models
                pdiff, mudiff = BM.modelDifference(bestfit)
                
                # if models are this close, then we have found the same soln
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
                
                # identify columns to inactivate: use different thresholds depending
                # whether or not we have found a soln (if soln already found, be more strict)
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
        

        if forcefit:
            bestfit.imputeInactiveParams(self.reads, self.mutations)
            self.BMsolution = bestfit
            self.BMsolution.sort_model()
            self.qualityCheck()
            
            bestfit = self.BMsolution
            

        return bestfit
        


    
    def compareBIC(self, BMnew, BMold, verbal=False):
        """Return BIC of BMold computed using the same set of active_columns as BMnew
        Note that for this method to work appropriately it requires that BMnew.active_columns
        is a subset of BMold.active_columns. 
            In the future, it might be good to update method so that it can take the
            intersection of active_columns if we want to be able to compare non-hierachically 
            solved models
        
        """

        # check if BMold has the same active_columns
        if np.array_equal(BMnew.active_columns, BMold.active_columns):
            return BMold.BIC
        
        elif not np.all(np.isin(BMnew.active_columns, BMold.active_columns)):
            raise ValueError('BMnew active_columns are not a subset of BMold active_columns')

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


        # Assign bestBM as 1-component solution to start
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

            pdiff, mudiff = bestfit.modelDifference(f)
            
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
       

    def splitProfile(self, maxProfile):
        """ Splits profile in two. Divides each attribute in half amongst
            two different profiles and returns the value.
        """
    
        sequence = maxProfile.sequence
        nts = maxProfile.nts
        rawprofile, rawerror = maxProfile.profile('raw', True)
        subprofile, suberror = maxProfile.profile('sub', True)
        backprofile, backerror = maxProfile.profile('back', True)
        normprofile, normerror = maxProfile.profile('norm', True)

        N13Profile = self.profile.copy()
        N7Profile = self.profile.copy()

        N13Profile.sequence = sequence[:len(sequence)/2]
        N13Profile.nts = nts[:len(nts)/2]
        N13Profile.rawprofile = rawprofile[:len(rawprofile)/2]
        N13Profile.rawerror = rawerror[:len(rawerror)/2]
        N13Profile.backprofile = backprofile[:len(backprofile)/2]
        N13Profile.backerror = backerror[:len(backerror)/2]
        N13Profile.normprofile = normprofile[:len(normprofile)/2]
        N13Profile.subprofile = subprofile[:len(subprofile)/2]
        N13Profile.suberror = suberror[:len(suberror)/2]


        N7Profile.sequence = sequence[len(sequence)/2:]
        N7Profile.nts = nts[len(nts)/2:]
        N7Profile.rawprofile = rawprofile[len(rawprofile)/2:]
        N7Profile.rawerror = rawerror[len(rawerror)/2:]
        N7Profile.backprofile = backprofile[len(backprofile)/2:]
        N7Profile.backerror = backerror[len(backerror)/2:]
        N7Profile.normprofile = normprofile[len(normprofile)/2:]
        N7Profile.subprofile = subprofile[len(subprofile)/2:]
        N7Profile.suberror = suberror[len(suberror)/2:]
        

        return N13Profile, N7Profile

    def joinProfile(self, N13prof, N7prof):
        """
            Joins two profiles together. Concatenates each attribute.
        """
        cat_prof = N13prof.copy() 
    
        sequenceN13 = N13prof.sequence
        ntsN13 = N13prof.nts
        rawprofileN13, rawerrorN13 = N13prof.profile('raw', True)
        subprofileN13, suberrorN13 = N13prof.profile('sub', True)
        backprofileN13, backerrorN13 = N13prof.profile('back', True)
        normprofileN13, normerrorN13 = N13prof.profile('norm', True)

        sequenceN7 = N7prof.sequence
        ntsN7 = N7prof.nts
        rawprofileN7, rawerrorN7 = N7prof.profile('raw', True)
        subprofileN7, suberrorN7 = N7prof.profile('sub', True)
        backprofileN7, backerrorN7 = N7prof.profile('back', True)
        normprofileN7, normerrorN7 = N7prof.profile('norm', True)


        cat_prof.sequence = np.concatenate((sequenceN13, sequenceN7))
        cat_prof.nts = np.concatenate((ntsN13, ntsN7))
        cat_prof.rawprofile = np.concatenate((rawprofileN13, rawprofileN7))
        cat_prof.rawerror = np.concatenate((rawerrorN13, rawerrorN7))
        cat_prof.subprofile = np.concatenate((subprofileN13, subprofileN7))
        cat_prof.suberror = np.concatenate((suberrorN13, suberrorN7))
        cat_prof.backprofile = np.concatenate((backprofileN13, backprofileN7))
        cat_prof.backerror = np.concatenate((backerrorN13, backerrorN7))
        cat_prof.normprofile = np.concatenate((normprofileN13, normprofileN7))
        cat_prof.normerror = np.concatenate((normerrorN13, normerrorN7))



        return cat_prof


    #Normalize profile
    def computeNormalizedReactivities(self, oldDMS=False, concat=False):
        """From converged mu params and profile, compute normalized params
           If the data run is concatenated - will normalize the first half 
           of the data in accordance with standard N1/3 normalization
           and the second half with N7 normalization. 
        """

        model = self.BMsolution
        
        # eliminate invalid positions
        model.mu[:, self.invalid_columns] = np.nan

        # create temporary profile containing maxs at each position to compute norm factors
        maxProfile = self.profile.copy()
        maxProfile.rawprofile = np.max(model.mu, axis=0)
        maxProfile.backgroundSubtract(normalize=False)
        

        if concat:

           #Split maxProfile into N1/3 half and N7 half
           N13maxProfile, N7maxProfile = self.splitProfile(maxProfile) 
           
           if not oldDMS:
               normfactors = N13maxProfile.normalize(eDMS=True)
               normFactorN7 = N7maxProfile.normalize(eDMS=True, N7=True)
               
           else:
               normfactors = maxProfile.normalize(oldDMS=True)
        


        else:
           if not oldDMS:
               normfactors = maxProfile.normalize(eDMS=True)
           else:
               normfactors = maxProfile.normalize(oldDMS=True)


        

        # now create new normalized profiles
        profiles = []




        
        for p in xrange(model.pdim):

            
            #If concat
            if concat:
                catprofile = self.profile.copy()
                N13prof, N7prof = self.splitProfile(catprofile)

                
                N13prof.rawprofile = np.copy(model.mu[p,:][:len(model.mu[p,:])/2])
                N7prof.rawprofile = np.copy(model.mu[p,:][len(model.mu[p,:])/2:])
                
                N13prof.backgroundSubtract(normalize=False)
                N7prof.backgroundSubtract(normalize=False)

                for i,nt in enumerate(N13prof.sequence):
                    try:
                        N13prof.normprofile[i] = N13prof.subprofile[i]/normfactors[nt]
                    except KeyError:
                        prof.normprofile[i] = np.nan


                for i, nt in enumerate(N7prof.sequence):
                    try:
                        if not np.isnan(N7prof.subprofile[i]):
                           if nt in normFactorN7 and np.isfinite(normFactorN7[nt]):
                               if i + 1 < len(N7prof.subprofile):
                                   if N7prof.subprofile[i] < 0:
                                       print("Setting NT {} to 3.32 since the background subtracted value is less than 0".format(i))
                                       N7prof.normprofile[i] = 3.32
                                   elif N7prof.sequence[i+1] == 'A' or N7prof.sequence[i+1] == 'N':
                                       N7prof.normprofile[i] = np.log2(normFactorN7['g_pur']/N7prof.subprofile[i])
                                   else:
                                       N7prof.normprofile[i] = np.log2(normFactorN7['g_pyr']/N7prof.subprofile[i])
                                

                        else: 
                            N7prof.normprofile[i] = np.nan


                


                    except Exception as e:
                        print("Error, exception e: ", e)


                #CHECK N7 and N13 outputs
            
                cat_profile = self.joinProfile(N13prof, N7prof)
                profiles.append(cat_profile)

            #Else
            else:
               prof = self.profile.copy()
               prof.rawprofile = np.copy(model.mu[p,:])
               prof.backgroundSubtract(normalize=False)
               
               for i,nt in enumerate(prof.sequence):
                   try:
                       prof.normprofile[i] = prof.subprofile[i]/normfactors[nt]
                   except KeyError:
                       prof.normprofile[i] = np.nan

            
            
            
        
            
               profiles.append(prof)


        

        
        return profiles




    def writeReactivities(self, output, oldDMS=False, concat=False):
        """Print out model reactivities
        self.profile must be defined
        """
        
        model = self.BMsolution

        if self.profile is None:
            print('Cannot compute reactivities because profile was not provided')
            return
        
        # compute normalized parameters
        profiles = self.computeNormalizedReactivities(oldDMS, concat)

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
            


    def readModelFromFile(self, fname, verbal=True):
        """Read in BMsolution from BM file object"""
        

        self.BMsolution = BernoulliMixture()
        self.BMsolution.readModelFromFile(fname)

        # check to make sure the active, inactive are the same
        if not np.array_equal(self.active_columns, self.BMsolution.active_columns) or \
               np.array_equal(self.inactive_columns, self.BMsolution.inactive_columns):

            if verbal:
                sys.stderr.write('active_columns in BMsolution and DanceMap object are different!\n')
                sys.stderr.write('Updating DanceMap columns to BMsolution values\n')
            
            invalids = np.where(self.BMsolution.mu[0,:] < 0)[0]
            self.invalid_columns = invalids

            self.setColumns(activecols=self.BMsolution.active_columns, 
                            inactivecols=self.BMsolution.inactive_columns)    
        


 

    def _sample_RINGs(self, window=1, corrtype='apc', bgfile=None, assignprob=0.9, mincount=10,
                      subtractwindow=True, montecarlo=False, verbal=True, N7=False, concat=False, fasta=None):
        """Assign sample reads based on posterior prob and return list of RINGexperiment objs
        window     = correlation window
        corrtype   = metric for computing correlations
        bgfile     = parsed mutation file for bg sample (to filter out bg mutations)
        assignprob = posterior prob. used for assigning reads to models. If -1, assign reads as MAP
        subtractwindow = exclude nt window when assigning read for that window
        montecarlo = sample reads using MC logic
        verbal     = verbal
        N7 = N7"""

        
        # setup the activestatus mask. Assign reads using both active & inactive cols
        activestatus = np.zeros(self.seqlen, dtype=np.int8)
        activestatus[self.active_columns] = 1
        activestatus[self.inactive_columns] = 1
     
        
        # fill in the matrices
        if montecarlo:
            #deprecated should probably remove
            if verbal:
                print('Using MC for sample RING read assignment')
            
            raise AttributeError('Monte Carlo option has been removed')
            
            read, comut, inotj = aFunc.fillRINGMatrix_montecarlo(self.reads, self.mutations, activestatus,
                                                                 self.BMsolution.mu, self.BMsolution.p, 
                                                                 window, self.reads.shape[0], subtractwindow)
            
        
        else:
            if verbal:
                print('Using {:.3f} as posterior prob for sample RING read assignment'.format(assignprob))
            #check to see if we've already filled the matrices for this window to prevent redundant calculation
            if self.ring_read is None and self.ring_comut is None and self.ring_inotj is None:
                self.ring_read, self.ring_comut, self.ring_inotj = aFunc.fillRINGMatrix(self.reads, self.mutations, activestatus,
                                                      self.BMsolution.mu, self.BMsolution.p, window, assignprob, 
                                                      subtractwindow)
                
        relist = [] 
        
        # populate RINGexperiment objects
        # handle N7 rings
        if self.concat:
          for p in xrange(self.BMsolution.pdim):
              
              ring = RINGexperiment(fasta=fasta, arraysize=self.seqlen, corrtype=corrtype, verbal=verbal, concat=concat, N7=N7)

              if N7:
                  indices_to_remove = [x for x in range(0, (len(self.sequence)/2))]
                  ring.sequence = self.sequence[:len(self.sequence)/2]
                  toignore = [x -( len(self.sequence) / 2) for x in self.invalid_columns if x >= len(self.sequence)/2]

              elif concat:
                  indices_to_remove = np.intp([])
                  ring.sequence = self.sequence[:len(self.sequence)/2]
                  toignore = self.invalid_columns

              else:
                  indices_to_remove = [x for x in range(len(self.sequence)/2, len(self.sequence)+1)]
                  ring.sequence = self.sequence[:len(self.sequence)/2]
                  toignore = [x for x in self.invalid_columns if x < len(self.sequence)/2]

              cur_read = np.copy(self.ring_read[p])
              cur_comut = np.copy(self.ring_comut[p])
              cur_inotj = np.copy(self.ring_inotj[p])
              
              for elem in [0, 1]:
                cur_read = np.delete(cur_read, indices_to_remove, axis=elem)
                cur_comut = np.delete(cur_comut, indices_to_remove, axis=elem)
                cur_inotj = np.delete(cur_inotj, indices_to_remove, axis=elem)
              
              
              ring.window = window
              ring.ex_readarr = cur_read
              ring.ex_comutarr = cur_comut
              ring.ex_inotjarr = cur_inotj
              
              # fill bg arrays (only need to do once; copy for >0 models)
              if bgfile is not None:
                  if p==0:
                      ring.initDataMatrices('bg', bgfile, window=window, 
                                            mincoverage=self.minreadcoverage, verbal=verbal)
                  else:
                      ring.bg_readarr = relist[0].bg_readarr
                      ring.bg_comutarr = relist[0].bg_comutarr
                      ring.bg_inotjarr = relist[0].bg_inotjarr
              
              ring.computeCorrelationMatrix(mincount=mincount, verbal=verbal, ignorents=toignore)
              
  
              relist.append(ring)
        
        else:
          for p in xrange(self.BMsolution.pdim):
              
              ring = RINGexperiment(arraysize=self.seqlen, corrtype=corrtype, verbal=verbal)
  
              ring.sequence = self.sequence
  
              ring.window = window
              ring.ex_readarr = self.ring_read[p]
              ring.ex_comutarr = self.ring_comut[p]
              ring.ex_inotjarr = self.ring_inotj[p]
              
              # fill bg arrays (only need to do once; copy for >0 models)
              if bgfile is not None:
                  if p==0:
                      ring.initDataMatrices('bg', bgfile, window=window, 
                                            mincoverage=self.minreadcoverage, verbal=verbal)
                  else:
                      ring.bg_readarr = relist[0].bg_readarr
                      ring.bg_comutarr = relist[0].bg_comutarr
                      ring.bg_inotjarr = relist[0].bg_inotjarr
              
              ring.computeCorrelationMatrix(mincount=mincount, verbal=verbal, ignorents=self.invalid_columns)
              
              relist.append(ring)

        if verbal: print('\n')

        return relist

 

    def _null_RINGs(self, window=1, corrtype='g', assignprob=0.9, mincount=10,
                    subtractwindow=True, montecarlo=False, verbal=False, N7=False, concat=False, fasta=None):
        """Create null (uncorrelated) model and assign reads based on posterior prob 
        to determine null-model correlations. 
        Returns list of RINGexperiment objs
        window     = correlation window
        corrtype   = metric for computing correlations
        assignprob = posterior prob. used for assigning reads to models. If -1, assign reads as MAP
        subtractwindow = exclude nt window when assigning read for that window
        montecarlo = sample reads using MC logic
        verbal     = verbal"""

       
        from SynBernoulliMixture import SynBernoulliMixture
        
        # initialize synthetic model, ensuring that invalid columns are masked out
        mu = np.copy(self.BMsolution.mu)
        mu[:,self.invalid_columns] = -1
        null_model = SynBernoulliMixture(p=self.BMsolution.p, mu=mu)
        
        # generate synthetic reads 
        nullEM = null_model.getEMobject(self.reads.shape[0], nodata_rate=0.1, 
                                        invalidcols=self.invalid_columns,
                                        verbal=False)
        
        # setup the activestatus mask. Assign reads using both active & inactive cols
        activestatus = np.zeros(self.seqlen, dtype=np.int8)
        activestatus[self.active_columns] = 1
        activestatus[self.inactive_columns] = 1
        
        
        # fill in the matrices
        if montecarlo:
            
            if verbal:
                print('Using MC for null RING read assignment')
            
            raise AttributeError('Monte Carlo option has been removed')
            read, comut, inotj = aFunc.fillRINGMatrix_montecarlo(nullEM.reads, nullEM.mutations, activestatus,
                                                                 mu, self.BMsolution.p, 
                                                                 window, self.reads.shape[0], subtractwindow)
            
        else:
            if verbal:
                print('Using {:.3f} as posterior prob for null RING read assignment'.format(assignprob))

            if self.null_read is None and self.null_comut is None and self.null_inotj is None:
                self.null_read, self.null_comut, self.null_inotj = aFunc.fillRINGMatrix(nullEM.reads, nullEM.mutations, activestatus,
                                                      mu, self.BMsolution.p, window, assignprob, 
                                                      subtractwindow)
                
 

        relist = []

        # populate RINGexperiment objects
        if self.concat:
          for p in xrange(self.BMsolution.pdim):
  
              ring = RINGexperiment(arraysize=self.seqlen, corrtype=corrtype, verbal=verbal)

              if N7:
                  indices_to_remove = [x for x in range(0, (len(self.sequence)/2))]
                  ring.sequence = self.sequence[:(len(self.sequence)/2)]
                  #get nts to ignore in N7 only, divide by 2 to adjust frame
                  toignore = [x - (len(self.sequence) / 2) for x in self.invalid_columns if x >= len(self.sequence)/2]
              elif concat:
                  indices_to_remove = np.intp([])
                  ring.sequence = self.sequence
                  toignore = self.invalid_columns
              else:
                  indices_to_remove = [x for x in range(len(self.sequence)/2, len(self.sequence)+1)]
                  ring.sequence = self.sequence[:(len(self.sequence)/2)]
                  toignore = [x for x in self.invalid_columns if x < len(self.sequence)/2]
                  
              cur_read = np.copy(self.null_read[p])
              cur_comut = np.copy(self.null_comut[p])
              cur_inotj = np.copy(self.null_inotj[p])
              
              
              for elem in [0, 1]:
                  cur_read = np.delete(cur_read, indices_to_remove, axis=elem)
                  cur_comut = np.delete(cur_comut, indices_to_remove, axis=elem)
                  cur_inotj = np.delete(cur_inotj, indices_to_remove, axis=elem)
              
              
              ring.window = window
              ring.ex_readarr = cur_read
              ring.ex_comutarr = cur_comut
              ring.ex_inotjarr = cur_inotj
  
               
              ring.computeCorrelationMatrix(mincount=mincount, ignorents=toignore, verbal=verbal)
              relist.append(ring)
        else:
          for p in xrange(self.BMsolution.pdim):
  
              ring = RINGexperiment(arraysize=self.seqlen, corrtype=corrtype, verbal=verbal)
  
              ring.sequence = self.sequence
              
              ring.window = window
              ring.ex_readarr = self.null_read[p]
              ring.ex_comutarr = self.null_comut[p]
              ring.ex_inotjarr = self.null_inotj[p]
              
              ring.computeCorrelationMatrix(mincount=mincount, ignorents=self.invalid_columns, verbal=verbal)
              relist.append(ring)
            
        return relist



    def computeRINGs(self, window=1, bgfile=None, assignprob=0.9, mincount=10,
                     subtractwindow=True, montecarlo=False, nulldifftest=True, verbal=True, N7=False, concat=False, fasta=None):
        """Compute RINGs based on posterior prob and mask out (i,j) pairs that are
        observed in null (uncorrelated) model. 
        Return list of RINGexperiment objects

        window     = correlation window
        bgfile     = parsed mutation file for bg sample (to filter out bg mutations)
        assignprob = posterior prob. used for assigning reads to models
        nulldifftest = perform difference G-test against null model as well
        subtractwindow = exclude nt window when assigning read for that window
        montecarlo = sample reads using MC logic
        verbal     = verbal"""
        if self.concat:
            length = len(self.sequence) / 2
        else:
            length = len(self.sequence)


        # compute rings from experimental data
        sample = self._sample_RINGs(window=window, bgfile=bgfile, assignprob=assignprob, mincount=mincount,
                                    subtractwindow=subtractwindow, montecarlo=montecarlo, verbal=verbal, N7=N7, concat=concat, fasta=fasta)
        
        if not nulldifftest:
            return sample


        # compute correlations from null model (clustering only w/o correlations)
        null = self._null_RINGs(window=window, assignprob=assignprob, mincount=10,
                                subtractwindow=subtractwindow, montecarlo=montecarlo, N7=N7, concat=concat, fasta=fasta) 



        # mask out corrs present in the null model
        for p in range(self.BMsolution.pdim):
            
            # p=0.001 for df=1
            for (i,j) in null[p].significantCorrelations('ex', 10.83):
                
                if verbal:
                    print('Pair ({},{}) ignored due to chi2={:.1f} correlation in null model={}'.format(i+1,j+1, null[p].ex_correlations[i,j], p))
                
                for k in range(self.BMsolution.pdim):
                    sample[k].ex_correlations[i,j] = np.ma.masked
                    sample[k].ex_correlations[j,i] = np.ma.masked
                    sample[k].ex_zscores[i,j] = np.ma.masked
                    sample[k].ex_zscores[j,i] = np.ma.masked
            


        # now test for significant difference in distributiosn
        for p in range(self.BMsolution.pdim):
            
            sample_p = sample[p]
            null_p = null[p]
            
            for i,j in itertools.combinations(range(length), 2):

                nulldiff = sample_p.significantDifference(i,j, null_p.ex_readarr[i,j], null_p.ex_inotjarr[i,j],
                                                          null_p.ex_inotjarr[j,i], null_p.ex_comutarr[i,j])
                
                # p=0.001 for df=3
                if nulldiff < 16.27:

                    if verbal and sample_p.ex_correlations[i,j]>23.93:
                        print('Model {}: Correlated pair ({},{}) w/ chi2={:.1f} ignored: NULL difference chi2={:.1f}'.format(p,i+1,j+1, sample_p.ex_correlations[i,j], nulldiff))
                
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

# Concat file helper functions
def cleanup(*args): 
    for f in args: os.remove(f)

def check_any_missing(*files):
    if not all(os.path.exists(f) for f in files):
        raise ValueError("Error: " + ", ".join(files) + " not located in the same directory.")

def insert_prefix(prefix, name):
        if "/" in prefix:
            spl_path = prefix.split("/")
            prefix = "/".join(spl_path[:-1])
            return prefix + "/" + name + spl_path[-1]
        else:
            return name + prefix

def make_fasta(profile, fasta, fa_name):
        with open(fasta, "w") as fasta_f:
            N1_profile = ReactivityProfile(profile)
            fasta_f.write('>{}\n'.format(fa_name))
            fasta_f.write(''.join(N1_profile.sequence))

def make_concatenated_N7_files(modified_parsed, untreated_parsed, profile):
    mod_mut = args.modified_parsed
    mod_mutga = mod_mut + "ga"
    prof_txt = args.profile
    prof_txtga = prof_txt + "ga"
    unt_mut = None
    unt_mutga = None
    

    # Ensure .mut and .mutga / .txt and .txtga are located in the same folder
    check_any_missing(mod_mut, mod_mutga)
    check_any_missing(prof_txt, prof_txtga)

    # Enables proper temp file production in cases where user passes the prefix as a path instead of
    # a simple string. ie "/path/prefix" instead of "prefix"
    mod_output = insert_prefix(args.outputprefix + ".mut", ".temp_mod_concat_")
    prof_output = insert_prefix(args.outputprefix + ".txt" , ".temp_prof_concat_")
    fasta_output = insert_prefix(args.outputprefix + ".fa" , ".temp_fasta_")
    

    # Concats the .mut and .mutga as well as .txt and .txtga files together into temp files informed
    # by user defined prefix
    concat_prof_mut.concat_mut(mod_mut[:mod_mut.index(".")], mod_output, prof_txt[:prof_txt.index(".")])
    concat_prof_mut.concat_profile(prof_txt[:prof_txt.index(".")], prof_output)

    # Create a temp fasta to from the profile provided to satisfy N7 ring dependency
    make_fasta(args.profile, fasta_output, prof_txt[:prof_txt.index(".")])


    # Create temp file for untreated .mut and .mutga if applicable
    if args.untreated_parsed:
        unt_output = insert_prefix(args.outputprefix + ".mut", ".temp_unt_concat_")
        unt_mut = args.untreated_parsed
        unt_mutga = unt_mut + "ga"

        check_any_missing(unt_mut, unt_mutga)
        concat_prof_mut.concat_mut(unt_mut[:unt_mut.index(".")], unt_output, prof_txt[:prof_txt.index(".")])


    return mod_output, unt_output, prof_output, fasta_output

    
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
    fitopt.add_argument('--forcefit', type=int, help='Force fit to specified number of components')
    fitopt.add_argument('--maxcomponents', type=int, default=5, help='Maximum number of components to fit (default=5)')
    fitopt.add_argument('--trials', type=int, default=50, help='Maximum number of fitting trials at each component number (default=50)')
    fitopt.add_argument('--badcol_cutoff', type=int, default=5, help='Inactivate column after it causes a failure X number of times *after* a valid soln has already been found (default=5)')
    fitopt.add_argument('--writeintermediates', action='store_true', help='Write each BM solution to file with specified prefix. Will be saved as prefix-intermediate-[component]-[trial].bm')

    fitopt.add_argument('--priorWeight', type=float, default=0.01, help='Weight of prior on Mu (default=0.01). Prior = priorWeight*readDepth*bgRate at each nt. Prior is disabled by passing -1, upon which a naive prior is used.')
    fitopt.add_argument('--maskG', action='store_true', help='set all Us to inactive')
    fitopt.add_argument('--maskU', action='store_true', help='set all Gs to inactive')
    fitopt.add_argument('--maskN', action='store_true', help='set all Ns to inactive')


    ############################################################
    # RING options
    
    ringopt = parser.add_argument_group('options for performing RING/PAIR analysis on clustered reads')
    ringopt.add_argument('--ring', action='store_true')
    ringopt.add_argument('--window', type=int, default=1, help='Window size for computing correlations (default=1)')

    ringopt.add_argument('--pairmap', action='store_true', help='Run PAIR-MaP analysis on clustered reads') 
    ringopt.add_argument('--readprob_cut', type=float, default=0.9, help='Posterior probability cutoff for assigning reads for inclusion in ring/pairmap analysis. Reads must have posterior prob greater than the cutoff (default=0.9). If set to -1, assign reads using maximum a posteriori (MAP) criteria')
    ringopt.add_argument('--chisq_cut', type=float, default=23.9, help="Set chisq cutoff for RING/PAIR-MaP analysis (default = 23.9)")
    ringopt.add_argument('--mincount', type=float, default=10, help="Set mincount cutoff for RING/PAIR-MaP analysis (default = 10)")
    ringopt.add_argument('--pm_secondary_reactivity', type=float, default=0.4, help="Set secondary_reactivity cutoff for pairmapper analysis (default = 0.4)")
    

    # note the below logic is a bit confusing because the internal variables are different than
    # external names. Internally, the variable is subtractWindow. By default this should be True.
    # InclWindow is basically the negation of subtractWindow, so passing --inclwindow flag will set
    # subtractWindow = False
    ringopt.add_argument('--inclwindow', action='store_false', help='Include considered windows in read probability calculation and assignment. By default windows are EXCLUDED from calculations')

    ############################################################
    # Other options
    
    optional.add_argument('--untreated_parsed', help='Path to untreated parsed.mut file')
    optional.add_argument('--readfromfile', type=str, help='Read in solved BM model from BM file')
    optional.add_argument('--ignore_untreated', action='store_true', help='Ignore untreated mutation rates from profile.txt file. If ShapeMapper was run without an untreated sample, this argument is superfluous. Untreated rates are used for establishing priors during fitting, and computing normalized rates)')
    optional.add_argument('--oldDMSnorm', action='store_true', help='Use old style (pre-eDMS) normalization')
    optional.add_argument('--suppressverbal', action='store_false', help='Suppress verbal output')
    optional.add_argument('--outputprefix', type=str, default='emfit', help='Write output files with this prefix (default=emfit)')

    optional.add_argument('--concat', action='store_true', default=False, help='Concatenate mut and mutga files together internally for processing. Must place .mut and .mutga files as well as profile.txt and .txtga files in same directory.')

    optional.add_argument('--activate_n7', action='store_true', default=False, help='Flag to activate N7 columns during DanceMap clustering. Default = False.')
    

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
     
    
    if not args.fit and not args.ring and not args.pairmap and not args.forcefit:
        sys.stderr.write("\n Action argument [fit, ring, pairmap, forcefit] not provided\n\n")
        sys.exit(1)
    

    if not 0<args.priorWeight<=1 and args.priorWeight != -1:
        sys.stderr.write('\npriorWeight value = {} is invalid!\n\n'.format(args.priorWeight))
        sys.exit(1)

    ############################################################
    # reformat writeintermediate bool flag as file prefix
    if args.writeintermediates:
        args.writeintermediates = args.outputprefix
    else:
        args.writeintermediates = None
    


    return args


if __name__=='__main__':
    
    # Log file messaging for keeping track of run info
    print(' '.join(sys.argv[:]))
    print('\nStarting up DANCE-Mapper pipeline')
    
    import version
    version.print_version()


    print('{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

    args = parseArguments()
    print('Arguments = {}\n\n'.format(args))

    if args.concat:
        mod_output, unt_output, prof_output, fasta_output = make_concatenated_N7_files(args.modified_parsed, args.untreated_parsed, args.profile)
        # Modify DM call to accomodate concatenated files
        DM = DanceMap(modfile=mod_output, untfile=unt_output,
                         profilefile=prof_output, 
                         minrxbg = args.minrxbg,
                         maskG = args.maskG,
                         maskU = args.maskU,
                         maskN = args.maskN,
                         concat=args.concat,
                         minreadcoverage=args.mincoverage, 
                         undersample=args.undersample,
                         ignorebg=args.ignore_untreated,
                         verbal=args.suppressverbal,
                         inactivate_n7= not args.activate_n7
                         )
    else:
       DM = DanceMap(modfile=args.modified_parsed, untfile=args.untreated_parsed,
                        profilefile=args.profile, 
                        minrxbg = args.minrxbg,
                        maskG = args.maskG,
                        maskU = args.maskU,
                        maskN = args.maskN,
                        minreadcoverage=args.mincoverage, 
                        undersample=args.undersample,
                        ignorebg=args.ignore_untreated,
                        verbal=args.suppressverbal)

    
    if args.fit:
        
        DM.findBestModel(args.maxcomponents, trials=args.trials,
                         badcolcount = args.badcol_cutoff,
                         priorWeight = args.priorWeight,
                         verbal=args.suppressverbal,
                         writeintermediate = args.writeintermediates)

        if args.concat:
           DM.BMsolution.writeModel(args.outputprefix+'-concat.bm')
           DM.writeReactivities(args.outputprefix+'-concat-reactivities.txt', oldDMS=args.oldDMSnorm, concat=args.concat)
           de_cat.decat_react(args.outputprefix+'-concat-reactivities.txt', args.outputprefix+'-N13-reactivities.txt', args.outputprefix+'-N7-reactivities.txt')
           de_cat.decat_bm(args.outputprefix+'-concat.bm', args.outputprefix+'-N13.bm', args.outputprefix+'-N7.bm')

        else:
           DM.writeReactivities(args.outputprefix+'-reactivities.txt', oldDMS=args.oldDMSnorm)
           DM.BMsolution.writeModel(args.outputprefix+'.bm')



    elif args.forcefit:

        bestBM = DM.fitEM(args.forcefit, trials=200, 
                          badcolcount = args.badcol_cutoff,
                          priorWeight = args.priorWeight,
                          verbal = args.suppressverbal, 
                          writeintermediate = args.writeintermediates,
                          forcefit = True)
    
        if args.concat:
           DM.BMsolution.writeModel(args.outputprefix+'-concat.bm')
           DM.writeReactivities(args.outputprefix+'-concat-reactivities.txt', oldDMS=args.oldDMSnorm, concat=args.concat)
           de_cat.decat_react(args.outputprefix+'-concat-reactivities.txt', args.outputprefix+'-N13-reactivities.txt', args.outputprefix+'-N7-reactivities.txt')
           de_cat.decat_bm(args.outputprefix+'-concat.bm', args.outputprefix+'-N13.bm', args.outputprefix+'-N7.bm')

        else:
           DM.writeReactivities(args.outputprefix+'-reactivities.txt', oldDMS=args.oldDMSnorm)
           DM.BMsolution.writeModel(args.outputprefix+'.bm')


    elif args.readfromfile is not None:

        DM.readModelFromFile(args.readfromfile)


    if args.ring:
        
        if args.suppressverbal:
                print('--------------Computing RINGs--------------')

        if args.concat:
            N1_RE_list = DM.computeRINGs(window=args.window, bgfile=args.untreated_parsed, subtractwindow=args.inclwindow, mincount=args.mincount, assignprob=args.readprob_cut, verbal=args.suppressverbal)
            N1N7_RE_list = DM.computeRINGs(fasta=fasta_output, bgfile=unt_output, subtractwindow=args.inclwindow, mincount=args.mincount, assignprob=args.readprob_cut, verbal=args.suppressverbal, concat=args.concat )
            N7_RE_list = DM.computeRINGs(bgfile=args.untreated_parsed + "ga", subtractwindow=args.inclwindow, mincount=args.mincount, assignprob=args.readprob_cut, verbal=args.suppressverbal, N7=True)

            for i,model in enumerate(N1_RE_list):
                model.writeCorrelations('{}-{}-N1rings.txt'.format(args.outputprefix, i), chi2cut=args.chisq_cut)
            for i,model in enumerate(N1N7_RE_list):
                model.writeCorrelations('{}-{}-N1N7rings.txt'.format(args.outputprefix, i), chi2cut=args.chisq_cut)
            for i,model in enumerate(N7_RE_list):
                model.writeCorrelations('{}-{}-N7rings.txt'.format(args.outputprefix, i), chi2cut=args.chisq_cut)

        else:
                RE_list = DM.computeRINGs(window=args.window, bgfile=args.untreated_parsed, subtractwindow=args.inclwindow, mincount=args.mincount,
                                  assignprob=args.readprob_cut, verbal=args.suppressverbal)

                for i,model in enumerate(RE_list):
                    model.writeCorrelations('{0}-{1}-rings.txt'.format(args.outputprefix, i),
                                    chi2cut=args.chisq_cut)

        #concat rings are computed in 3 separate ring exps N1/N17/N7

    if args.pairmap:
        
        profiles = DM.computeNormalizedReactivities(args.oldDMSnorm, concat=args.concat)
        
        if args.suppressverbal:
                print('--------------Computing PAIRs--------------')


        RE_list = DM.computeRINGs(window=3, bgfile=args.untreated_parsed, subtractwindow=args.inclwindow, mincount=args.mincount,
                                  assignprob=args.readprob_cut, verbal=args.suppressverbal)

        for i,model in enumerate(RE_list):
            
            if args.suppressverbal:
                print('*******Model {}*******'.format(i))


            model.writeCorrelations('{0}-{1}-allcorrs.txt'.format(args.outputprefix,i), 
                                    chi2cut=args.chisq_cut)

            pairs = PairMapper(model, profiles[i], secondary_reactivity=args.pm_secondary_reactivity)
            pairs.writePairs('{0}-{1}-pairmap.txt'.format(args.outputprefix, i))
            pairs.writePairBonusFile('{0}-{1}-pairmap.bp'.format(args.outputprefix, i))      


            
    if args.concat:
        cleanup(mod_output, prof_output, fasta_output)
        if args.untreated_parsed:
            cleanup(unt_output)
    

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
