

import DanceMapper, accessoryFunctions
import sys
import numpy as np


DM = DanceMapper.DanceMap(modfile=sys.argv[1], profilefile=sys.argv[2])
DM.readModelFromFile(sys.argv[3])


llmat = np.zeros((DM.BMsolution.pdim, DM.reads.shape[0]), dtype=np.float64)

accessoryFunctions.loglikelihoodmatrix(llmat, DM.reads, DM.mutations, np.array(DM.active_columns, dtype=np.int32), DM.BMsolution.mu, DM.BMsolution.p)

llmat = np.exp(llmat)
s = np.sum(llmat, axis=0)
llmat /= s

bins = np.linspace(0,1,11)

hist, edge = np.histogram(llmat[0,:], bins=bins)

hist = np.array(hist, dtype=float)/np.sum(hist)

for i in range(10):
    print("{} {:.3f}".format(i, hist[i]))



