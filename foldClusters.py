

import sys, subprocess, argparse, os
import numpy as np
import externalpaths

from plotClusters import RPCluster


sys.path.append(externalpaths.arcplot())
from arcPlot import ArcPlot
from pmanalysis import PairMap

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

    prs = argparse.ArgumentParser(description='Perform RNAstructure modeling and plot results for all models from a BM')

    prs.add_argument('inputReactivity',type=str, help="Path of clustering reactivity file (-reactivities.txt)")
    prs.add_argument('output', type=str, help='Prefix for writing output files')
    prs.add_argument('--bp', type=str, help="Prefix for bp files. I.e. if your pairmap files are test-[]-pairmap.bp, you should pass '--bp test'")
    prs.add_argument('--pk', action='store_true', help='fold using ShapeKnots (default is to use Fold)')
    prs.add_argument('--nog', action='store_true', help='mask Gs (set to no-data)')
    prs.add_argument('--nou', action='store_true', help='mask Us (set to no-data)')

    return prs.parse_args()



if __name__=='__main__':


    args = parseArgs()

    foldpath = externalpaths.rnastructure()+'/fold'
    skpath = externalpaths.rnastructure()+'/ShapeKnots'

    if not args.pk and not os.path.isfile(foldpath):
        exit('Path to RNAstructure:fold is invalid! Check directory path in externalpaths!')

    elif args.pk and not os.path.isfile(skpath):
        exit('Path to RNAstructure:ShapeKnots is invalid! Check directory path in externalpaths!')



    clusters = RPCluster(args.inputReactivity)   

    seqfile, dmsfiles = writeFiles(clusters, args.output, nog=args.nog, nou=args.nou)


    for i,dfile in enumerate(dmsfiles):

        if not args.pk:
            command = [foldpath, seqfile, dfile[:-4]+'.ct', '--dmsnt', dfile]

            if args.bp:
                command.extend(('-x', args.bp+'-{}-pairmap.bp'.format(i)))
            print(command)
            subprocess.call(command)

        else:
            if args.bp:
                foldPK.iterativeShapeKnots(skpath, seqfile, dfile[:-4], dmsfile=dfile, 
                                           bpfile=args.bp+'-{}-pairmap.bp'.format(i))
            else:
                foldPK.iterativeShapeKnots(skpath, seqfile, dfile[:-4], dmsfile=dfile)


        aplot = ArcPlot(title = '{} P={:.3f}'.format(args.output, clusters.population[i]))

        if args.pk:
            finalname = dfile[:-4]+'.f.ct'
        else:
            finalname = dfile[:-4]+'.ct'

        aplot.addCT( RNAtools.CT(finalname) )
        aplot.readDMSProfile(dfile)

        if args.bp:
            aplot.addPairMap( PairMap(args.bp+'-{}-pairmap.txt'.format(i)), panel=-1)

        aplot.writePlot( dfile[:-4]+'.pdf')












