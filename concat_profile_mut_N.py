import numpy as np
import sys


#This function opens the .mut and .mutga strings and just concats
#them together. 
def concat_mut(inputmut, output, profile):
   read_length = 0
   with open("{}.txt".format(profile), 'r') as prof:
      lines = prof.readlines()
      last_line = lines[-1].split()
      read_length = int(last_line[0])
      


   with open("{}.mut".format(inputmut), 'r') as mut:
      with open("{}.mutga".format(inputmut), 'r') as mutGA:
         #with open("{}.mut".format(output), 'w') as concat:
         with open("{}".format(output), 'w') as concat:
            parsedLines = mut.read().splitlines()
            parsedLinesGA = mutGA.read().splitlines()
            length = len(parsedLines)
            for i in range(length):
               splGA = parsedLinesGA[i].split()
               spl = parsedLines[i].split()
               if(splGA[4] == "INCLUDED" and spl[4] == "INCLUDED"):
                  to_add = []
                  to_add += (read_length - (int(spl[3]) + 1)) * ["0"]
                  to_add += (int(spl[2])) * ["0"]
                  
                  
                  if(len(to_add) > 0):         
                     spl[8] = spl[8] + "".join(to_add) + splGA[8]
                     spl[6] = spl[6] + "".join(to_add) + splGA[6]
                     spl[7] = spl[7] + "".join(to_add) + splGA[7]
                     spl[3] = str(int(spl[3]) + read_length)
                  else:
                     spl[8] = spl[8] + splGA[8]
                     spl[6] = spl[6] + splGA[6]
                     spl[7] = spl[7] + splGA[7]
                     spl[3] = str(int(spl[3]) + read_length)

                  newline = " ".join(spl)
                  concat.write(newline + "\n")

# Concatenates the N1/3 and N7 profile again for further downstream processing
def concat_profile(inputProf, profOut):
   with open("{}.txt".format(inputProf), 'r') as N1:
      with open("{}.txtga".format(inputProf), 'r') as N7:
         #with open("{}.txt".format(profOut), 'w') as out:
         with open("{}".format(profOut), 'w') as out:
            linesN1 = [line.rstrip() for line in N1]
            linesN7 = [line.rstrip() for line in N7][1:]

             
            nt_length = (len(linesN1) - 1)

            for i in range(len(linesN7)):
               spl_line = linesN7[i].split()
               newnum = int(spl_line[0]) + nt_length
               newnum = str(newnum)
               spl_line[0] = newnum
               if spl_line[1] == 'G':
                  spl_line[1] = 'N'
               elif spl_line[1] == 'g':
                  spl_line[1] = 'n'

               newline = " ".join(spl_line)
               linesN1.append(newline)


            for line in linesN1:
               out.write(line + "\n")

      




if(__name__ == "__main__"):
   #Rudimentary arg parsing. Basically do -m for for concat of muts
   if(sys.argv[1] == "-h"):
      print("USAGE: concat_profile_mut.py -m Mut Output Profile")
      print("USAGE: concat_profile_mut.py -p Profile Output")
   elif(sys.argv[1] == "-m"):
      concat_mut(sys.argv[2], sys.argv[3], sys.argv[4])
   #-p for concat of profile
   elif(sys.argv[1] == "-p"):
      concat_profile(sys.argv[2], sys.argv[3])


