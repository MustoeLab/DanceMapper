
import numpy as np
import matplotlib.pyplot as plot

import sys, itertools
from scipy import stats

from BernoulliMixture import BernoulliMixture
import externalpaths

sys.path.append(externalpaths.arcplot())
import arcPlot

sys.path.append(externalpaths.ringmapper())
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
        bm.readModelFromFile(inpfile)
        
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
            
            population = np.array(map(float,inp.readline().split()[1:]))
            inp.readline()

            for line in inp:
                spl = line.split()
                
                nt.append(int(spl[0]))
                seq.append(spl[1])
        
                for i in range(ncomp):
                    norm[i].append(float(spl[2+2*i]))
                    raw[i].append(float(spl[3+2*i]))
                
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

    
    def renormalize(self):

        for i in range(len(self.profiles)):
            self.profiles[i].normalize(DMS=True)




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

 
    def computePearson(self, clust2, sidx, cidx):
        
        #idx = self.alignModel(clust2)

        mask = np.ones(len(self.profiles[0].rawprofile), dtype=bool)
        
        with np.errstate(invalid='ignore'):
            mask = mask & np.isfinite(self.profiles[sidx].rawprofile) & (self.profiles[sidx].rawprofile>-1)
            mask = mask & np.isfinite(clust2.profiles[sidx].rawprofile) & (clust2.profiles[cidx].rawprofile>-1)
        
        r,p = stats.pearsonr(self.profiles[sidx].subprofile[mask], clust2.profiles[cidx].subprofile[mask])

        return r    




def plotClusterProfile(clustobj, out=None, modelNums=None):

    xvals = 1+np.arange(clustobj.rawprofiles.shape[1])
    
    fig, ax = plot.subplots()
    for i,p in enumerate(clustobj.p):

        if modelNums is not None and i not in modelNums:
            continue
        
        with np.errstate(invalid='ignore'):
            mask = clustobj.rawprofiles[i]>-1

        ax.step(xvals[mask], clustobj.rawprofiles[i][mask], label='{0:.2f}'.format(p), where='mid')
   
    for c in mergeColumns(clustobj.inactive_columns):
        ax.axvspan(c[0],c[1], color='gray', alpha=0.2)
    
    for c in mergeColumns(clustobj.invalid_columns):
        ax.axvspan(c[0],c[1], color='gray', alpha=0.4)

    print("Sample comparison:")
    print("-----------------------")
    printPearson(clustobj, clustobj)



    ax.legend()

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
            ax[i].axvspan(c[0],c[1], color='gray', alpha=0.4)
        

        corrcoef = stats.pearsonr(clust1.rawprofiles[i][mask], clust2.rawprofiles[c2idx[i]][mask])
        ax[i].text(0.02,0.9,'R={:.3f}'.format(corrcoef[0]), transform=ax[i].transAxes)

        ax[i].legend(loc='upper right')
        
    

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
            print("{0} {1:.3f}, {2} {3:.3f} : {4:.2f}".format(i,pi,j,pj,corrcoef[0]))
   


   
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



def plotProfileComparison(fname, out=None, modelNums=None):

    data = []
    seq = []
    with open(fname) as inp:
        inp.readline()
        p = map(float, inp.readline().split()[1:])
        inp.readline()

        for line in inp:
            spl = line.split()
            seq.append(spl[1])
            data.append([float(spl[2+2*i]) for i in range(len(p))])

    data = np.array(data)
    
    fig, ax = plot.subplots()
    
    xvals = 1+np.arange(len(seq))
    
    for i,p in enumerate(p):
        
        if modelNums is not None and i not in modelNums:
            continue

        with np.errstate(invalid='ignore'):
            mask = data[:,i] >-1
        
        ax.step(xvals[mask], data[mask,i], where='mid',
                label='p={:.2f}'.format(p))
    
    ax.legend()
    ax.set_ylim(-0.1, 2)
    ax.set_xticks(np.arange(0,xvals[-1],10), minor=True)
    ax.grid(lw=0.25, which='minor')

    if out is None:
        plot.show()
    else:
        fig.savefig(out)


def printratio(model):
    
    for k in range(model.rawprofiles.shape[1]):
        
        minval = np.min(model.rawprofiles[:,k])
        maxval = np.max(model.rawprofiles[:,k])
        print k+1, maxval/minval
        






if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref')
    parser.add_argument('--comp')
    parser.add_argument('--out')
    parser.add_argument('--react')
    parser.add_argument('--align', action='store_true')
    parser.add_argument('--models')
    parser.add_argument('--ratio', action='store_true')

    args = parser.parse_args()
    
    if args.models:
        args.models = map(int, args.models.split(','))


    if args.ratio:
        printratio(Cluster(args.ref))


    if args.ref and not args.comp:
        plotClusterProfile( Cluster(args.ref), args.out, modelNums=args.models )


    if args.ref and args.comp:

        plotClusterComparison( Cluster(args.ref),  Cluster(args.comp), 
                               name1 = args.ref, name2= args.comp, 
                               out = args.out, align=args.align)

    if args.react:
        
        plotProfileComparison( args.react, args.out, modelNums=args.models)
        


        
