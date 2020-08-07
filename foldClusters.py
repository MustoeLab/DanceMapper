

import sys, subprocess, argparse
import numpy as np
import externalpaths

from plotClusters import RPCluster


sys.path.append(externalpaths.arcplot())
from arcPlot import ArcPlot

sys.path.append(externalpaths.rnatools())
import RNAtools2 as RNAtools
import foldPK



def writeFiles(clusters, outprefix, nog=False, nou=False):

    seqfilename = '{0}.seq'.format(outprefix)

    with open(seqfilename, 'w') as seqfile:
        seqfile.write(';\nsequence\n')
        
        for n in clusters.profiles[0].sequence:
            seqfile.write(n)
        seqfile.write('1')

    
    dmsfilenames = []
    for i in range(len(clusters.profiles)):
        n = '{0}-{1}.dms'.format(outprefix, i)
        dmsfilenames.append(n)

        prof = clusters.profiles[i].normprofile
        nts = clusters.profiles[i].nts
        seq = clusters.profiles[i].sequence

        with open(n, 'w') as dmsfile:
            
            for j in range(len(prof)):
                if np.isnan(prof[j]) or (nog and seq[j]=='G') or (nou and seq[j]=='U'):
                    dmsfile.write('{} -999\n'.format(nts[j]))
                else:
                    dmsfile.write('{} {:.4f}\n'.format(nts[j], prof[j]))


    return seqfilename, dmsfilenames




def parseArgs():

    prs = argparse.ArgumentParser()

    prs.add_argument('inputReactivity',type=str, help="Path of clustering reactivity file")
    prs.add_argument('output', type=str, help='Prefix for writing files')
    prs.add_argument('--bp', type=str, help='Prefix for bp files (i.e. for test-0-pairmap.txt pass test)')
    prs.add_argument('--pk', action='store_true', help='fold using ShapeKnots')
    prs.add_argument('--nog', action='store_true', help='mask gs for folding')
    prs.add_argument('--nou', action='store_true', help='mask us for folding')
    prs.add_argument('--renorm', action='store_true', help='renormalize reactivities')

    return prs.parse_args()



if __name__=='__main__':


    args = parseArgs()

    foldpath= '/Users/anthonymustoe/Code/RNAstructure/exe/fold'

    clusters = RPCluster(args.inputReactivity)   

    if args.renorm:
        clusters.renormalize()


    seqfile, dmsfiles = writeFiles(clusters, args.output, nog=args.nog, nou=args.nou)


    for i,dfile in enumerate(dmsfiles):

        if not args.pk:
            command = [foldpath, seqfile, dfile[:-4]+'.ct', '--dmsnt', dfile]

            if args.bp:
                command.extend(('-x', args.bp+'-{}-pairmap.bp'.format(i)))
            print command
            subprocess.call(command)

        else:
            if args.bp:
                foldPK.iterativeShapeKnots(seqfile, dfile[:-4], dmsfile=dfile, bpfile=args.bp+'-{}-pairmap.bp'.format(i))
            else:
                foldPK.iterativeShapeKnots(seqfile, dfile[:-4], dmsfile=dfile)


        aplot = ArcPlot(title = '{} P={:.3f}'.format(args.output, clusters.population[i]))

        if args.pk:
            finalname = dfile[:-4]+'.f.ct'
        else:
            finalname = dfile[:-4]+'.ct'

        aplot.addCT( RNAtools.CT(finalname) )
        aplot.readDMSProfile(dfile)
        aplot.writePlot( dfile[:-4]+'.pdf')












