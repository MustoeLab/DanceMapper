import argparse
import numpy as np
from numpy import isnan, nan, sqrt

################################################################################
#Processes an N1/3-N7 concatenated  dance-reactivities.txt file and produces
#a .txt .txtga profile file for each predicted model in the ensemble. These .txtga files #may be fed into arcplot etc.
################################################################################

#Parser function. Finds total number of models, length of sequence,
#and extracts N7 lines.
def parseLines(filename):
   num_models = 0
   Nt_Length = 0
   reactFile = open(filename, 'r')
   lines = [line.rstrip() for line in reactFile]
   fline = lines[0]
   num_models  = int(fline.split()[0])
   lline = lines[len(lines) - 1]
   Nt_Length =  int(lline.split()[0]) / 2
   reactFile.close()


   N7index = int(Nt_Length) + 3
   lines = lines[3:]
   return num_models, Nt_Length, lines

#Extracts a two dimensional list of the raw and normalized reactivities as well
#as the sequence.
def findReactivities(lines, num_Models, Nt_Length):


   seq = []
   norm = []

   for i in range(num_Models):
      norm.append([])


   for i  in range(len(lines)):
      splLine = lines[i].split()
      index = 2
      lIndex = 0 
      seq.append(splLine[1])
      for j in range(len(norm)):
         norm[lIndex].append(splLine[index]) 
         lIndex += 1
         index += 2
   
   norm_N13 = []
   norm_N7 = []

   for i in range(len(norm)):
      norm_N13.append(norm[i][:int(Nt_Length)])
      norm_N7.append(norm[i][int(Nt_Length):])

   return norm_N13, norm_N7, seq


#Iterates through extracted reactivities and generates .txtga profiles.
#def make_txtga(seq, raw, norm, output):
def make_txtga(norm_N13, norm_N7, seq, output):
   for i in range(len(norm_N13)):
      lines_N13 = []
      lines_N7 = []
      lines_N13.append("Nucleotide  Sequence Modified_mutations   Modified_read_depth  Modified_effective_depth   Modified_rate  Modified_off_target_mapped_depth Modified_low_mapq_mapped_depth   Modified_mapped_depth   Untreated_mutations  Untreated_read_depth Untreated_effective_depth  Untreated_rate     Untreated_off_target_mapped_depth   Untreated_low_mapq_mapped_depth  Untreated_mapped_depth      Denatured_mutations  Denatured_read_depth Denatured_effective_depth  Denatured_rate Denatured_off_target_mapped_depth   Denatured_low_mapq_mapped_depth  Denatured_mapped_depth  Reactivity_profile   Std_err  HQ_profile  HQ_stderr   Norm_profile   Norm_stderr")
      lines_N7.append("Nucleotide  Sequence Modified_mutations   Modified_read_depth  Modified_effective_depth   Modified_rate  Modified_off_target_mapped_depth Modified_low_mapq_mapped_depth   Modified_mapped_depth   Untreated_mutations  Untreated_read_depth Untreated_effective_depth  Untreated_rate     Untreated_off_target_mapped_depth   Untreated_low_mapq_mapped_depth  Untreated_mapped_depth      Denatured_mutations  Denatured_read_depth Denatured_effective_depth  Denatured_rate Denatured_off_target_mapped_depth   Denatured_low_mapq_mapped_depth  Denatured_mapped_depth  Reactivity_profile   Std_err  HQ_profile  HQ_stderr   Norm_profile   Norm_stderr")
      for j in range(len(norm_N13[i])):
         line_N13 = []
         line_N13.append(str(j + 1))
         line_N13.append(" {} ".format(seq[j]))
         line_N13.append(" 0 " * 25)
         line_N13.append(str(norm_N13[i][j]))
         line_N13.append(" 0")
        
         line_N13 = ''.join(line_N13)

         line_N7 = []
         line_N7.append(str(j + 1))
         if seq[j] == "N":
            seq[j] = "G"
         line_N7.append(" {} ".format(seq[j]))
         line_N7.append(" 0 " * 25)
         line_N7.append(str(norm_N7[i][j]))
         line_N7.append(" 0")
         line_N7 = ''.join(line_N7)




         lines_N13.append(line_N13)
         lines_N7.append(line_N7)

      if output == None:
         oFile_N13 = open("dance_map_model_{}_profile.txt".format(i), "w")
         oFile_N7 = open("dance_map_model_{}_profile.txtga".format(i), "w")
         for line in lines_N13:
            oFile_N13.write(line + "\n")
         for line in lines_N7:
            oFile_N7.write(line + "\n")

      else:
         oFile_N13 = open("{}_{}_profile.txt".format(output, i), "w")
         oFile_N7 = open("{}_{}_profile.txtga".format(output, i), "w")
         for line in lines_N13:
            oFile_N13.write(line + "\n")
         for line in lines_N7:
            oFile_N7.write(line + "\n")

      oFile_N13.close()
      oFile_N7.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reactivities", help = "Path to the concatenated dance-reactivities.txt")
    parser.add_argument("--output", help = "Output file(s) prefix", default=None)
    args=parser.parse_args()
    num_Models, Nt_Length, lines = parseLines(args.reactivities)
    norm_N13, norm_N7, seq = findReactivities(lines, num_Models, Nt_Length) 
    make_txtga(norm_N13, norm_N7, seq, args.output)
