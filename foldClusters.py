

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

    seqfilename = '{0}.fa'.format(outprefix)

    with open(seqfilename, 'w') as seqfile:
        seqfile.write('>sequence\n')
        
        for n in clusters.profiles[0].sequence:
            seqfile.write(n)

    
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
    prs.add_argument('--prob', action='store_true', help='compute pairing probability (partition)')
    prs.add_argument('--pk', action='store_true', help='fold using ShapeKnots (default is to use Fold)')
    prs.add_argument('--nog', action='store_true', help='mask Gs (set to no-data)')
    prs.add_argument('--nou', action='store_true', help='mask Us (set to no-data)')
    prs.add_argument('--notDMS', action='store_true', default=False, help='Turn off any assumptions that DMS was used (default=False)')

    return prs.parse_args()



if __name__=='__main__':


    args = parseArgs()

    foldpath = externalpaths.rnastructure()+'/Fold'
    skpath = externalpaths.rnastructure()+'/ShapeKnots'
    pfunpath = externalpaths.rnastructure()+'/partition'
    pplotpath = externalpaths.rnastructure()+'/ProbabilityPlot'
 
    if not args.pk and not os.path.isfile(foldpath):
        exit('Path to RNAstructure:fold is invalid! Check directory path in externalpaths!')

    elif args.pk and not os.path.isfile(skpath):
        exit('Path to RNAstructure:ShapeKnots is invalid! Check directory path in externalpaths!')
    
    elif args.prob:
        if not os.path.isfile(pfunpath):
            exit('Path to RNAstructure:partition is invalid! Check directory path in externalpaths!')
        elif not os.path.isfile(pplotpath):
            exit('Path to RNAstructure:ProbabilityPlot is invalid! Check directory path in externalpaths!')


    clusters = RPCluster(args.inputReactivity, args.notDMS)   

    seqfile, dmsfiles = writeFiles(clusters, args.output, nog=args.nog, nou=args.nou)


    for i,dfile in enumerate(dmsfiles):
            
        if args.prob:
            command = [pfunpath, seqfile, dfile[:-4]+'.pfs',
                       ['--dmsnt', '--SHAPE'][args.notDMS], dfile]
            if args.bp:
                command.extend(('-x', args.bp+'-{}-pairmap.bp'.format(i)))
            print(command)
            subprocess.call(command)
            
            command = [pplotpath, '-t', dfile[:-4]+'.pfs', dfile[:-4]+'.dp']
            print(command)
            subprocess.call(command)
            

        elif not args.pk:
            command = [foldpath, seqfile, dfile[:-4]+'.ct',
                       ['--dmsnt', '--SHAPE'][args.notDMS], dfile]

            if args.bp:
                command.extend(('-x', args.bp+'-{}-pairmap.bp'.format(i)))
            print(command)
            subprocess.call(command)

        else:
            foldPKargs = {'ShapeKnotsPath':skpath,
                          'seqfile':seqfile,
                          'outprefix':dfile[:-4],
                          ['dmsfile', 'shapefile'][args.notDMS]: dfile}
            if args.bp:
                foldPKargs['bpfile'] = args.bp+'-{}-pairmap.bp'.format(i)
            foldPK.iterativeShapeKnots(**foldPKargs)


        aplot = ArcPlot(title = '{} P={:.3f}'.format(args.output, clusters.population[i]), fasta=seqfile)
        
        if args.prob:
            aplot.addPairProb( RNAtools.DotPlot(dfile[:-4]+'.dp') )
        elif args.pk:
            aplot.addCT( RNAtools.CT(dfile[:-4]+'.f.ct') )
        else:
            aplot.addCT( RNAtools.CT(dfile[:-4]+'.ct') )

        if args.notDMS:
            aplot.readProfile(dfile)
        else:
            aplot.readDMSProfile(dfile)

        if args.bp:
            aplot.addPairMap( PairMap(args.bp+'-{}-pairmap.txt'.format(i)), panel=-1)

        aplot.writePlot( dfile[:-4]+'.pdf')












