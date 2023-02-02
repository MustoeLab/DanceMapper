
import numpy as np


import matplotlib 
matplotlib.use('Agg')

import matplotlib.pyplot as plot
import matplotlib.patches as mpatches
import sys, itertools

from scipy import stats

from BernoulliMixture import BernoulliMixture

import externalpaths
sys.path.append(externalpaths.structureanalysistools())
from ReactivityProfile import ReactivityProfile



class Cluster(object):

    def __init__(self, inpfile=None):
        
        self.p = None
        self.rawprofiles = None

        if inpfile is not None:
            self.readfile(inpfile)

    
    def readfile(self, inpfile):

        self._readBMfile(inpfile)

    def _readBMfile(self, inpfile):
        
        bm = BernoulliMixture()
        bm.readModelFromFile(inpfile, True)
        
        self.p = bm.p
        self.rawprofiles = bm.mu 
        self.inactive_columns = bm.inactive_columns
        self.invalid_columns = np.where(bm.mu[0,:]<0)[0]
        
        if self.inactive_columns is None:
            self.inactive_columns = []
        if self.invalid_columns is None:
            self.invalid_columns = []

        
        #for i in bm.inactive_columns:
        #    self.rawprofiles[:,i] = np.nan

        self.sortByPopulation()


    def sortByPopulation(self):

        idx = range(len(self.p))
        idx.sort(key=lambda x: self.p[x], reverse=True)
        
        self.p = self.p[idx]
        self.rawprofiles = self.rawprofiles[idx,:]

    

    def alignModel(self, clust2):
        
        if self.p.shape[0] > clust2.p.shape[0]:
            raise ValueError('Ref Cluster must have lower dimension than comparison Cluster')

        actlist = np.ones(self.rawprofiles.shape[1], dtype=bool)
        
        with np.errstate(invalid='ignore'):
            for i in range(len(self.p)):
                actlist = actlist & np.isfinite(self.rawprofiles[i,:]) & (self.rawprofiles[i,:]>-1)
            for i in range(len(clust2.p)):
                actlist = actlist & np.isfinite(clust2.rawprofiles[i,:]) & (clust2.rawprofiles[i,:]>-1)
            
        
        mindiff = 1000
        
        for idx in itertools.permutations(range(len(clust2.p))):
            
            ridx = idx[:self.p.shape[0]]
            d = self.rawprofiles - clust2.rawprofiles[ridx,]
            
            rmsdiff = np.square(d[:, actlist])
            rmsdiff = np.sqrt( np.mean(rmsdiff) )

            if rmsdiff < mindiff:
                minidx = idx
                mindiff = rmsdiff
        

        return minidx

    
    def returnMax(self):

        ave = np.zeros(self.rawprofiles.shape[1])

        for i in range(self.p.shape[0]):
            ave += self.p[i]*self.rawprofiles[i,:]
        

        p99 = np.percentile(ave[np.isfinite(ave)], 99)
        p100 = np.percentile(ave[np.isfinite(ave)], 100)

        return p100, p99, np.max(self.rawprofiles[np.isfinite(self.rawprofiles)])
        





class RPCluster(object):

    def __init__(self, fname=None):
        
        if fname is not None:
            self.readReactivities(fname)
    
    
    def readReactivities(self, fname):
    
        with open(fname) as inp:

            ncomp = int(inp.readline().split()[0])
            
            nt = []
            seq = []
            bg = []
            norm = [[] for x in range(ncomp)]
            raw = [[] for x in range(ncomp)]
            inactives = []
            
            population = np.array(map(float,inp.readline().split()[1:]))
            inp.readline()

            for line in inp:
                spl = line.split()
                
                nt.append(int(spl[0]))
                seq.append(spl[1])
        
                for i in range(ncomp):
                    norm[i].append(float(spl[2+2*i]))
                    raw[i].append(float(spl[3+2*i]))
                    
                    if spl[-1] == 'i':
                        inactives.append(int(spl[0])-1)

                bg.append(float(spl[4+2*i]))



        bg = np.array(bg)
        norm = np.array(norm)
        raw = np.array(raw)
        nts = np.array(nt)
        seq = np.array(seq)

        profiles = []
        for i in range(norm.shape[0]):

            p = ReactivityProfile()
            p.sequence = seq
            p.nts = nts
            p.normprofile = norm[i,:]
            p.rawprofile = raw[i,:]
            p.backprofile = bg
            p.backgroundSubtract(normalize=False)
            profiles.append(p)


        self.population = population
        self.profiles = profiles
        self.nclusts = self.population.shape[0]
        self.inactive_columns = inactives
        self.invalid_columns = []

    
    def renormalize(self):

        for i in range(len(self.profiles)):
            self.profiles[i].normalize(eDMS=True)




    def plotHist(self, name, nt, ax):
        
        data = []
        
        seqmask = self.profiles[0].sequence==nt
        
        for i in range(len(self.profiles)):
            react = self.profiles[i].profile(name)
            react = react[seqmask]
            react = react[np.isfinite(react)]
            data.append(react)
        
        ax.hist(data, label=range(len(data)), density=True)
        ax.legend()
        ax.set_title(nt)




    def alignModel(self, clust2):
        
        if self.nclusts > clust2.nclusts:
            raise ValueError('Ref Cluster must have lower dimension than comparison Cluster')

        actlist = np.ones(len(self.profiles[0].rawprofile), dtype=bool)
        
        with np.errstate(invalid='ignore'):
            for i in range(len(self.profiles)):
                actlist = actlist & np.isfinite(self.profiles[i].rawprofile) & (self.profiles[i].rawprofile>-1)
            for i in range(len(clust2.profiles)):
                actlist = actlist & np.isfinite(clust2.profiles[i].rawprofile) & (clust2.profiles[i].rawprofile>-1)
            
        
        mindiff = 1000
        
        for idx in itertools.permutations(range(len(clust2.profiles))):
            
            ridx = idx[:self.nclusts]
            
            rmsdiff = 0
            for i in range(self.nclusts):
                d = self.profiles[i].subprofile - clust2.profiles[ridx[i]].subprofile
                rmsdiff += np.sqrt(np.mean(np.square(d[actlist])))
            
            if rmsdiff < mindiff:
                minidx = idx
                mindiff = rmsdiff
        

        return minidx

 
    def computePearson(self, clust2, sidx, cidx, ptype=None):
        
        #idx = self.alignModel(clust2)

        mask = np.ones(len(self.profiles[0].rawprofile), dtype=bool)
        
        with np.errstate(invalid='ignore'):
            mask = mask & np.isfinite(self.profiles[sidx].rawprofile) & (self.profiles[sidx].rawprofile>-1)
            mask = mask & np.isfinite(clust2.profiles[sidx].rawprofile) & (clust2.profiles[cidx].rawprofile>-1)
        
        prof1 = self.profiles[sidx].profile(ptype)
        prof2 = clust2.profiles[cidx].profile(ptype)
        r,p = stats.pearsonr(prof1[mask], prof2[mask])

        return r    

    

    def printPearson(self, clust2, ptype=None):

        for i,pi in enumerate(self.population):
            for j,pj in enumerate(clust2.population):
                print("{0} {1:.3f}, {2} {3:.3f} : {4:.2f}".format(i,pi,j,pj, self.computePearson(clust2,i,j, ptype)))
   


 


def plotClusterProfile(clustobj, out=None, modelNums=None):

    xvals = 1+np.arange(clustobj.rawprofiles.shape[1])
    
    fig, ax = plot.subplots()
    for i,p in enumerate(clustobj.p):

        if modelNums is not None and i not in modelNums:
            continue
        
        with np.errstate(invalid='ignore'):
            mask = clustobj.rawprofiles[i]>-1

        ax.step(xvals[mask], clustobj.rawprofiles[i][mask], label='p={0:.2f}'.format(p), where='mid')
   
    for c in mergeColumns(clustobj.inactive_columns):
        ax.axvspan(c[0],c[1], color='gray', alpha=0.2)
    
    for c in mergeColumns(clustobj.invalid_columns):
        ax.axvspan(c[0],c[1], color='gray', alpha=0.6)

    print("Sample comparison:")
    print("-----------------------")
    printPearson(clustobj, clustobj)

    
    handles, labels = ax.get_legend_handles_labels()
    patch1 = mpatches.Patch(color='gray', alpha=0.2, label='Inactive Nts')
    patch2 = mpatches.Patch(color='gray', alpha=0.6, label='Invalid Nts')
    handles.extend([patch1, patch2]) 

    ax.legend(handles=handles)
    
    ax.set_xlabel('Nts')
    ax.set_ylabel('Mu (mutation rate)')

    if out is None:
        plot.show()
    else:
        fig.savefig(out)




def plotClusterComparison(clust1, clust2, name1='', name2='', out=None, align=False):

    
    xvals = 1+np.arange(clust1.rawprofiles.shape[1])
    
    fig, ax = plot.subplots(nrows=max(2,len(clust1.p)), ncols=1)
   
    if align:
        c2idx = clust1.alignModel(clust2)
    else:
        c2idx = np.arange(len(clust1.p))
    

    for i,p in enumerate(clust1.p):
        
        with np.errstate(invalid='ignore'):
            mask = (clust1.rawprofiles[i]>-1) & (clust2.rawprofiles[i]>-1)
        
        ax[i].step(xvals[mask], clust1.rawprofiles[i][mask], where='mid',
                   label='{0} p={1:.2f}'.format(name1, p))

        ax[i].step(xvals[mask], clust2.rawprofiles[c2idx[i]][mask], where='mid',
                   label='{0} p={1:.2f}'.format(name2, clust2.p[c2idx[i]]))
        

        for c in mergeColumns(clust1.inactive_columns):
            ax[i].axvspan(c[0],c[1], color='gray', alpha=0.2)
        
        for c in mergeColumns(clust1.invalid_columns):
            ax[i].axvspan(c[0],c[1], color='gray', alpha=0.6)
        

        corrcoef = stats.pearsonr(clust1.rawprofiles[i][mask], clust2.rawprofiles[c2idx[i]][mask])
        ax[i].text(0.02,0.9,'R={:.3f}'.format(corrcoef[0]), transform=ax[i].transAxes)
        
        ax[i].set_ylabel('Mu (mutation rate)')


        handles, labels = ax[i].get_legend_handles_labels()

        if i==0:
            patch1 = mpatches.Patch(color='gray', alpha=0.2, label='Inactive Nts ({})'.format(name1))
            patch2 = mpatches.Patch(color='gray', alpha=0.6, label='Invalid Nts ({})'.format(name1))
            handles.extend([patch1, patch2]) 

        ax[i].legend(handles=handles, loc='upper right')
    

    ax[-1].set_xlabel('Nts')


    print("Sample1 comparison:")
    print("-----------------------")
    printPearson(clust1, clust1)

    print("Sample2 comparison:")
    print("-----------------------")
    printPearson(clust2, clust2)

    print("Intersample comparison:")
    print("-----------------------")
    printPearson(clust1, clust2)

 
        
    if out is None:
        plot.show(fig)
    else:
        fig.savefig(out)



def printPearson(clust1, clust2):

    for i,pi in enumerate(clust1.p):
        for j,pj in enumerate(clust2.p):
            
            with np.errstate(invalid='ignore'):
                mask = (clust1.rawprofiles[i]>-1) & (clust2.rawprofiles[j]>-1)
     
            corrcoef = stats.pearsonr(clust1.rawprofiles[i][mask], clust2.rawprofiles[j][mask])

            print("{0} {1:.3f}, {2} {3:.3f} : {4[0]:.2f}".format(i,pi,j,pj,corrcoef))
   


   
def mergeColumns(columns):
    
    if columns is None or len(columns)==0:
        return []

    output = [ [columns[0], columns[0]+1] ]

    for i in range(1,len(columns)):
        
        if columns[i]==output[-1][1]:
            output[-1][1] += 1
        else:
            output.append([columns[i], columns[i]+1])


    for i in range(len(output)):
        output[i][0]+=0.5
        output[i][1]+=0.5

    return output



def plotReactProfile(rpclust, out=None, modelNums=None, ptype=None):

    fig, ax = plot.subplots()
    
    xvals = rpclust.profiles[0].nts 
    
    for i,p in enumerate(rpclust.population):
        
        if modelNums is not None and i not in modelNums:
            continue

        with np.errstate(invalid='ignore'):
            mask = rpclust.profiles[i].normprofile >-1
        
        prof = rpclust.profiles[i].profile(ptype)
        ax.step(xvals[mask], prof[mask], where='mid', label='p={:.2f}'.format(p))
    
    
    for c in mergeColumns(np.where(np.isnan(rpclust.profiles[0].normprofile))[0]):
        ax.axvspan(c[0],c[1], color='gray', alpha=0.6)
    
    for c in mergeColumns(rpclust.inactive_columns):
        ax.axvspan(c[0],c[1], color='gray', alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    patch1 = mpatches.Patch(color='gray', alpha=0.2, label='Inactive Nts')
    patch2 = mpatches.Patch(color='gray', alpha=0.6, label='Invalid Nts')
    handles.extend([patch1, patch2]) 

    ax.legend(handles=handles)
    

    #ax.set_ylim(-0.1, 2)
    ax.set_xticks(np.arange(0,xvals[-1],10), minor=True)
    ax.grid(lw=0.25, which='minor')
    

    ax.set_xlabel('Nts')
    ax.set_ylabel('Normalized Reactivity')


    if out is None:
        plot.show()
    else:
        fig.savefig(out)




def plotReactProfileComparison(rp1, rp2, name1='', name2='', ptype=None, out=None, align=False):

    
    xvals = rp1.profiles[0].nts
    
    fig, ax = plot.subplots(nrows=max(2,len(rp1.population)), ncols=1)
   
    if align:
        c2idx = rp1.alignModel(rp2)
    else:
        c2idx = np.arange(len(rp2.population))
    

    for i,p in enumerate(rp1.population):
        
        with np.errstate(invalid='ignore'):
            mask = (rp1.profiles[i].rawprofile>-1) & (rp2.profiles[i].rawprofile>-1)
        
        prof1 = rp1.profiles[i].profile(ptype)
        prof2 = rp2.profiles[i].profile(ptype)

        ax[i].step(xvals[mask], prof1[mask], where='mid', label='{0} p={1:.2f}'.format(name1, p))
        ax[i].step(xvals[mask], prof2[mask], where='mid', label='{0} p={1:.2f}'.format(name2, rp2.population[c2idx[i]]))
        

        for c in mergeColumns(rp1.inactive_columns):
            ax[i].axvspan(c[0],c[1], color='gray', alpha=0.2)
        
        for c in mergeColumns(rp1.invalid_columns):
            ax[i].axvspan(c[0],c[1], color='gray', alpha=0.6)
        

        corrcoef = rp1.computePearson(rp2, i, c2idx[i], ptype)
        ax[i].text(0.02,0.9,'R={:.3f}'.format(corrcoef), transform=ax[i].transAxes)
        
        ax[i].set_ylabel('Mu (mutation rate)')


        handles, labels = ax[i].get_legend_handles_labels()

        if i==0:
            patch1 = mpatches.Patch(color='gray', alpha=0.2, label='Inactive Nts ({})'.format(name1))
            patch2 = mpatches.Patch(color='gray', alpha=0.6, label='Invalid Nts ({})'.format(name1))
            handles.extend([patch1, patch2]) 

        ax[i].legend(handles=handles, loc='upper right')
    

    ax[-1].set_xlabel('Nts')


    print("Sample1 comparison:")
    print("-----------------------")
    rp1.printPearson(rp1)

    print("Sample2 comparison:")
    print("-----------------------")
    rp2.printPearson(rp2)

    print("Intersample comparison:")
    print("-----------------------")
    rp1.printPearson(rp2)

        
    if out is None:
        plot.show(fig)
    else:
        fig.savefig(out)





if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser(description='Plot and/or compare BM files')
    parser.add_argument('--bm1', help="Path of first bm file")
    parser.add_argument('--bm2', help="Path of second bm file (optional)")
    parser.add_argument('--react1', help='Plot -reactivities.txt file (instead of BM file)')
    parser.add_argument('--react2', help='Path of second reactivities file')
    parser.add_argument('--ptype', type=str, help='Which data to plot from reactivities file. Can be norm/raw')

    parser.add_argument('--out', help='Write figure to file. If --out flag not used, plot using interactive GUI')
    parser.add_argument('--align', action='store_true', help='Align bm1 and bm2 based on reactivity (rather than population)')
    parser.add_argument('--models', help='Only plot the specified models. E.g. --models 0,2 will plot models 0 and 2 for a 3 component model. Only works when visualizing single bm or react.')

    args = parser.parse_args()
    
    if args.models:
        args.models = map(int, args.models.split(','))


    if args.bm1 and not args.bm2:
        plotClusterProfile( Cluster(args.bm1), args.out, modelNums=args.models )


    if args.bm1 and args.bm2:

        plotClusterComparison( Cluster(args.bm1),  Cluster(args.bm2), 
                               name1 = args.bm1, name2= args.bm2, 
                               out = args.out, align=args.align)

    if args.react1 and not args.react2:
        plotReactProfile( RPCluster(args.react1), args.out, modelNums=args.models, ptype=args.ptype)
        
    elif args.react1 and args.react2:
        plotReactProfileComparison( RPCluster(args.react1), RPCluster(args.react2), 
                                    name1 = args.react1, name2= args.react2,
                                    out=args.out, align=args.align, ptype=args.ptype)
     

        
