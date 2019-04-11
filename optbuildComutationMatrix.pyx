
########################################################
#   optbuildComutationMatrix contains optimized
#   cython functions for parsing mutation string files
#
#   Anthony Mustoe
#   Weeks Lab, UNC
#   July 2016
#
#   Version 2.0
########################################################



#cython: boundscheck=False
#cython: wraparound=False

from libc.stdio cimport *

import numpy as np
cimport numpy as np


###########################################################################
# Function and object definitions

cdef extern from "string.h":
    char *strtok(char*, const char*)


cdef extern from "stdlib.h":
    int atoi(char*)


cdef extern from "stdio.h":
    
    FILE *fopen(const char*, const char*)
    int fclose(FILE*)
    ssize_t getline(char**, size_t*, FILE*)


cdef struct READ:
    int start
    int stop
    char* read
    char* qual
 


###########################################################################

def findMaxSeqLen(str inputFile):
    # use c functions to scan through mutation file and determine max read index

    cdef FILE* cfile
    cdef size_t lsize = 0
    cdef ssize_t endfile
    cdef char* line = NULL
    
    cdef READ r
    cdef int maxstop = 0
    
    # open the file
    cfile = fopen(inputFile, "r")
    if cfile == NULL:
        raise IOError(2, "No such file or directory: '{0}'".format(inputFile))

    while True:
        
        # get the the line of text using c-function
        endfile = getline(&line, &lsize, cfile)
        if endfile == -1:
            break
        
        # parse line 
        r = parseLine(line)
        if r.read == NULL:
            continue

        if r.stop > maxstop:
            maxstop = r.stop
            
    fclose(cfile)
    return maxstop




###########################################################################
#
# Fill mutation matrices with read filtering
#
###########################################################################



def read_data(str inputFile, int[:,::1] read_arr, int[:, ::1] comut_arr, int[:, ::1] inotj_arr,
              int window, int phred_cutoff, char* accepted_events, int mutseparation, int maxdel):
    
    # Note that window parameter is only included to maintain argument similarity with
    # with read_dataWindow. It is not used in the method.
    if window != 1:
        exit("read_data is incomptable with window != 1; passed {0}".format(window))



    cdef int maxindex = read_arr.shape[0]

    # working arrays storing read events
    # first position contains element counter
    cdef int[:] readnts = np.zeros(maxindex+1, dtype=np.int32)
    cdef int[:] mutnts  = np.zeros(maxindex+1, dtype=np.int32)
    

    # mem alloc for reading the file using c
    cdef FILE* cfile
    cdef size_t lsize = 0
    cdef ssize_t endfile
    cdef char* line = NULL
    
    
    cdef READ r
    cdef int i, j, i_index

    
    # open the file
    cfile = fopen(inputFile, "r")
    if cfile == NULL:
        raise IOError(2, "No such file or directory: '{0}'".format(inputFile))


    cdef int linenum = -1

    # iterate through lines
    while True:
        
        linenum += 1 

        # get the the line of text using c-function
        endfile = getline(&line, &lsize, cfile)
        if endfile == -1:
            break
        

        # parse line into individual values, reset and fill readnts/mutnts
        try:
            r = parseLine(line)
            if r.read == NULL:
                raise IndexError()
            elif r.stop > maxindex:
                print "Line {0} outside array bounds :: read bounds = ({1}, {2})".format(linenum, r.start, r.stop)
                continue

            fillReadMut(readnts, mutnts, r, phred_cutoff, accepted_events, mutseparation, maxdel)
 
        except:
            print "Skipping incorrectly formatted line {0}".format(linenum)
            continue
 
   
        # fill the read count matrix. Want to fill upper-right of matrix,
        # so traverse through arrays in reverse
        for i in xrange(readnts[0], 0, -1):
        
            i_index = readnts[i]
            read_arr[ i_index, i_index ] += 1
            
            for j in xrange(i-1, 0, -1):
                read_arr[ i_index, readnts[j] ] += 1
               
        
        
        # fill in the mut matrices
        for i in xrange(mutnts[0], 0, -1):
        
            i_index = mutnts[i]
            comut_arr[ i_index, i_index ] += 1
            
            for j in xrange(i-1, 0, -1):
                comut_arr[i_index, mutnts[j] ] += 1


            # fill in inotj
            # Note this loop overcounts for j=mutated
            # diagnol is not used, so don't worry about i=j case
            for j in xrange(readnts[0], 0, -1): 
                inotj_arr[ i_index, readnts[j] ] += 1

            # correct for the over addition in inotj in above loop
            for j in xrange(mutnts[0], 0, -1):
                inotj_arr[ i_index, mutnts[j] ] -= 1
        


    fclose(cfile)

    return 0





cdef int fillReadMut(int[:] readnts, int[:] mutnts, READ r, int phred_cut, char* accepted_events, int mutdist_cut, int maxdelsize) except -1:
    ## parse the read for mut events
    # readnts and mutnts contain seq indices of valid reads and muts, respectively
    # Element 0 or readnts and mutnts is used as a counter...
    # Mutations within mutdist_cut are ignored
    # Reads with deletions larger than maxdelsize are ignored

    # NOTE:: Reads are iterated 3'->5', and nts are added to arrays accordingly
    #        Downstream iterations should be aware of this...
    

    cdef char a, mutcode
    cdef int i, seqpos, qscore

    cdef int readindex = 1 
    cdef int mutindex = 1 
    cdef int mutdist = mutdist_cut
    
    cdef int readlen = len(r.read) - 1
    cdef int lastvalid = 1

    cdef int delsize = 0

    # traverse read 3' -> 5'
    for i, mutcode in enumerate(r.read[::-1]):
        
        qscore = r.qual[readlen-i] - 33 
        seqpos = r.stop - i

        # Prob of miscalled match is very low at any qscore,
        # thus use permissive cutoff
        if mutcode == '|' and qscore >= 10:
            readnts[readindex] = seqpos
            readindex += 1
            mutdist += 1
            delsize = 0
        
        elif mutcode == '~':
            mutdist = 0
            delsize += 1
            
            # ignore reads that have too long mutations
            if delsize > maxdelsize:
                readindex = 1
                mutindex = 1
                break

        elif mutcode in accepted_events:
            
            if mutdist >= mutdist_cut and qscore >= phred_cut:
                mutnts[mutindex] = seqpos
                mutindex += 1
                readnts[readindex] = seqpos
                readindex += 1
            
            # otherwise, retroactively erase intervening 'read nts'
            # accomplish by resetting readindex to position where the initiating
            # mutation occured
            elif mutdist < mutdist_cut:
                readindex = lastvalid
            
            lastvalid = readindex
            mutdist = 0 
        


    # adjust index so that it points to last element, rather than forward
    # note this for looping in read_data, since we are looping in reverse
    mutnts[0] = mutindex-1
    readnts[0] = readindex-1

    return 1


#############################################################################

def filter_write(str inputFile, str outputFile, int seqlen, int phred_cutoff, 
        char* accepted_events, int mutseparation, int maxdel):
    
    # working arrays storing read events
    # first position contains element counter
    cdef int[:] readnts = np.zeros(2*seqlen, dtype=np.int32)
    cdef int[:] mutnts  = np.zeros(2*seqlen, dtype=np.int32)
    cdef int[:] outval  = np.empty(seqlen, dtype=np.int32)

    # mem alloc for reading the file using c
    cdef FILE* cfile
    cdef size_t lsize = 0
    cdef ssize_t endfile
    cdef char* line = NULL
    
    cdef READ r
    cdef int i, j

   
    # open the file
    cfile = fopen(inputFile, "r")
    if cfile == NULL:
        raise IOError(2, "No such file or directory: '{0}'".format(inputFile))


    outobj = open(outputFile, "w")
    
    # iterate through lines
    while True:
        
        # get the the line of text using c-function
        endfile = getline(&line, &lsize, cfile)
        if endfile == -1:
            break

        # parse line into individual values, reset and fill readnts/mutnts
        try:
            r = parseLine(line)
            if r.read == NULL:
                raise IndexError()
            #elif r.stop > seqlen:
            #    print "Line outside array bounds :: read bounds = ({0}, {1})".format(r.start, r.stop)
            #    continue
            
            fillReadMut(readnts, mutnts, r, phred_cutoff, accepted_events, mutseparation, maxdel)
            
            if readnts[1] > seqlen:
                print 'Read outside of bounds :: {0}'.format(readnts[1])
                raise IndexError()

        except:
            continue

        # initialize every value to 'no read'
        outval[:] = 9

        for i in xrange(readnts[0], 0, -1):
            j = readnts[i]
            outval[j] = 0
        
        for i in xrange(mutnts[0], 0, -1):
            j = mutnts[i]
            outval[j] = 1
        
        outobj.write(' '.join(map(str, outval))+'\n')


    fclose(cfile)
    outobj.close()


###########################################################################
###########################################################################
###########################################################################


cdef READ parseLine(char* line):
    # uses c functions to read through line; avoids having to use python split
    # which outputs python array and slows things down

    cdef READ r

    cdef char* token = strtok(line, " \t");
    token = strtok(NULL, " \t") #pop off the read name
    
    cdef int tokindex = 1

    while token:
        
        if tokindex == 1:
            r.start = atoi(token) - 1 # correct for 1-indexing 

        elif tokindex == 2:
            r.stop = atoi(token) - 1

        elif tokindex == 3:
            r.read = token

        elif tokindex == 5:
            r.qual = token
            break
        
        token = strtok(NULL, " \t")
        tokindex+=1
    
    # perform quaility checks...
    # Note r.qual is expected to have \n as last character;
    # hence the -1 in the length comparison...
    if r.start > r.stop or len(r.read) != len(r.qual)-1:
        r.read = NULL

    return r




###########################################################################
###########################################################################
###########################################################################



def read_dataWindow(str inputFile, int[:,::1] read_arr, int[:, ::1] comut_arr, int[:, ::1] inotj_arr,
                    int window, int phred_cutoff, char* accepted_events, int mutseparation, int maxdel):
 

    cdef int maxindex = read_arr.shape[0]

    # working arrays storing read events
    # first position contains element counter
    cdef int[:] readnts = np.zeros(maxindex+1, dtype=np.int32)
    cdef int[:] mutnts  = np.zeros(maxindex+1, dtype=np.int32)
    
    # mem alloc for reading the file using c
    cdef FILE* cfile
    cdef size_t lsize = 0
    cdef ssize_t endfile
    cdef char* line = NULL
    
    
    cdef READ r
    cdef int i, j, i_index

    
    # open the file
    cfile = fopen(inputFile, "r")
    if cfile == NULL:
        raise IOError(2, "No such file or directory: '{0}'".format(inputFile))


    cdef int linenum = -1

    # iterate through lines
    while True:
        
        linenum += 1 
        # get the the line of text using c-function
        endfile = getline(&line, &lsize, cfile)
        if endfile == -1:
            break
        

        # parse line into individual values, reset and fill readnts/mutnts
        try:
            r = parseLine(line)
            if r.read == NULL:
                raise IndexError()
            elif r.stop > maxindex:
                print "Line {0} outside array bounds :: read bounds = ({1}, {2})".format(linenum, r.start, r.stop)
                continue
            
            fillReadMutWindow(readnts, mutnts, r, window, phred_cutoff, accepted_events, mutseparation, maxdel)
 
        except:
            print "Skipping incorrectly formatted line {0}".format(linenum)
            continue
        
        #print np.asarray(readnts)
        #print np.asarray(mutnts)
   
        # fill the read count matrix. Want to fill upper-right of matrix,
        # so traverse through arrays in reverse
        for i in xrange(readnts[0], 0, -1):
        
            i_index = readnts[i]
            read_arr[ i_index, i_index ] += 1
            
            for j in xrange(i-1, 0, -1):
                read_arr[ i_index, readnts[j] ] += 1
               
        
        
        # fill in the mut matrices
        for i in xrange(mutnts[0], 0, -1):
        
            i_index = mutnts[i]
            comut_arr[ i_index, i_index ] += 1
            
            for j in xrange(i-1, 0, -1):
                comut_arr[i_index, mutnts[j] ] += 1


            # fill in inotj
            # Note this loop overcounts for j=mutated
            # diagnol is not used, so don't worry about i=j case
            for j in xrange(readnts[0], 0, -1): 
                inotj_arr[ i_index, readnts[j] ] += 1

            # correct for the over addition in inotj in above loop
            for j in xrange(mutnts[0], 0, -1):
                inotj_arr[ i_index, mutnts[j] ] -= 1
        


    fclose(cfile)

    return 0





cdef int fillReadMutWindow(int[:] readnts, int[:] mutnts, READ r, int window, int phred_cut, char* accepted_events, int mutdist_cut, int maxdelsize) except -1:
    ## parse the read for mut events
    # readnts and mutnts contain seq indices of valid reads and muts, respectively
    # Element 0 or readnts and mutnts is used as a counter...
    # Mutations within mutdist_cut are ignored
    # Reads with deletions larger than maxdelsize are ignored
    # Mutations are considered over the window

    # NOTE:: Reads are iterated 3'->5', and nts are added to arrays accordingly
    #        Downstream iterations should be aware of this...
 

    cdef char mutcode
    cdef int i,j, seqpos, qscore, counter, end

    cdef int readindex = 1 
    cdef int mutindex = 1 
    cdef int mutdist = mutdist_cut
    
    cdef int readlen = len(r.read) - 1
    cdef int lastmut = readlen+1

    cdef int delsize = 0
    cdef int boolval = 0

    cdef int[:] rmut = np.zeros(readlen+1, dtype=np.int32)
    cdef int[:] rread = np.zeros(readlen+1, dtype=np.int32)

    
    # first quality filter read 3' -> 5'
    for i, mutcode in enumerate(r.read[::-1]):
        
        seqpos = readlen-i
        qscore = r.qual[seqpos] - 33 
        
        # Prob of match actually being mismatch is very low at any qscore,
        # thus use permissive cutoff
        if mutcode == '|' and qscore >= 10:
            mutdist += 1
            delsize = 0
            rread[seqpos] = 1
        
        elif mutcode == '~':
            mutdist = 0
            delsize += 1
            
            # ignore reads that have too long mutations
            if delsize > maxdelsize:
                boolval = 1
                break

        elif mutcode in accepted_events:
                        
            if mutdist >= mutdist_cut and qscore >= phred_cut:
                rmut[seqpos] = 1
                rread[seqpos] = 1
            elif mutdist < mutdist_cut:
                rread[seqpos:lastmut] = 0
            
            lastmut = seqpos
            mutdist = 0 
    

    if boolval:
        mutnts[0] = 0
        readnts[0] = 0
        return 1    
    
    end = -1
    if r.start < 1:
        end = -r.start -1 # -1 here corrects for 1-indexing correction in parseLine
    
    for i in xrange(readlen-window+1, end, -1):
        # if there is mutation in the window, assume its good
        # regardless if some nucs are missing data...
        
        boolval = 0
        counter = 0
        for j in xrange(i, i+window):
            if rmut[j]:
                boolval = 1
                break
            if rread[j]:
                counter+=1

        if boolval:
            mutnts[mutindex] = r.start+i
            readnts[readindex] = r.start+i
            mutindex += 1
            readindex += 1

        elif counter == window:
            readnts[readindex] = r.start+i
            readindex += 1
            

    # adjust index so that it points to last element, rather than forward
    # note this for looping in read_data, since we are looping in reverse
    mutnts[0] = mutindex-1
    readnts[0] = readindex-1

    return 1











# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
