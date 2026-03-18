import sys
import re

####################
#Script that deconcatenates N13/N7 for bm, reactivity, and ring files.
####################
def decat_bm(filename, N13output, N7output):
   bmfile = open(filename, "r")
   eIndex = 0
   lines = bmfile.readlines()
   for i in range(len(lines)):
      if lines[i] == "# Initial P\n":
         eIndex = i
         eIndex = i - 1

   bmfile.close()
   
   header = lines[0:6]

   react_lines = lines[6:eIndex]
   #for line in react_lines:
   #   print(line)

   midpt = int(len(react_lines) / 2 ) 
   N13 = react_lines[0:midpt]
   N7 = react_lines[midpt:]


   
   #print("Length: ", int(len(react_lines) / 2))

   N13_file = open(N13output, "w")
   for line in header:
      N13_file.write(line)
   for line in N13:
      N13_file.write(line)

   N13_file.close()

   N7_file = open(N7output, "w")
   for line in header:
      N7_file.write(line)
   for line in N7:
      N7_file.write(line)


def decat_react(filename, N13output, N7output):
   reactfile = open(filename, "r")
   lines = reactfile.readlines()
   reactfile.close()
   
   header = lines[0:3]


   react_lines = lines[3:]
   midpt = int(len(react_lines) / 2 ) 
   N13 = react_lines[0:midpt]
   N7 = react_lines[midpt:]

   N13_file = open(N13output, "w")
   for line in header:
      N13_file.write(line)
   for line in N13:
      N13_file.write(line)
   N13_file.close()

   N7_file = open(N7output, "w")
   for line in header:
      N7_file.write(line)
   for line in N7:
      N7_file.write(line)
   N7_file.close()


def decat_rings(filename, output):
   ringFile = open(filename, "r")
   lines = ringFile.readlines()
   
   concatLength = int(lines[0].split()[0])
   ntLength = int(concatLength / 2)

   splFLine = lines[0].split()
   splFLine[0] = str(ntLength)
   splFLine.append("\n")
   splFLine = " ".join(splFLine)
   toAdd = []
   toAdd.append(splFLine)
   toAdd.append(lines[1])

   for index in range(2, len(lines)):
      splLine = lines[index].split()
      i = int(splLine[0])
      j = int(splLine[1])
      if ( i < ntLength and j < ntLength):
         toAdd.append(lines[index])

   ringFile.close()
      
   ringoutput = open( output + ".txt", "w")
   for line in toAdd:
      ringoutput.write(line)

   ringoutput.close()


if __name__=="__main__":
   if (sys.argv[1] == "-h"):
      print("Usage: input N13_output, N7_output")

   else:
      if (re.search('.reactivities\.txt', sys.argv[1]) != None):
         #print("reactivities.txt file input, de - concatenating reactivities")
         decat_react(sys.argv[1], sys.argv[2], sys.argv[3])
      elif (re.search('.\.bm', sys.argv[1]) != None):
         #print(".bm file input, de - concatenating bm")
         decat_bm(sys.argv[1], sys.argv[2], sys.argv[3])
      elif (re.search('.rings\.txt', sys.argv[1]) != None or re.search('.ringmap\.txt', sys.argv[1])):
         #print("rings.txt or ringmap.txt file input, de - concatenating rings")
         decat_rings(sys.argv[1], sys.argv[2])
