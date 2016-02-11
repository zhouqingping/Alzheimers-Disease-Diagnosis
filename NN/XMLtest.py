import XMLReader as xmlR
import os
from nifti import *

#neptune is a local user account
os.chdir("/DeepLearningAD/ADNI")
all = os.listdir(os.getcwd())
xmls =[]
passed = 0
fail = 0
print os.getcwd()
for xml in all:
    temp = xml.split(".")
    if temp[len(temp)-1] == "xml":
        xmls.append(xml)
for file in xmls:
    reader = xmlR.XMLReader(file)
    try:
        path = reader.path_to_scan(os.getcwd())
        print "passed"
        passed += 1
    except OSError:
        print "FAILED"
        fail +=1
print passed
print fail

#Checks to see what the max size of each NiFTi image is
def maxsizeofNIFTI(xmls):
    os.chdir("/DeepLearningAD/ADNI")
    all = os.listdir(os.getcwd())
    maxsize = [0,0,0]
    for file in xmls:
        reader = xmlR.XMLReader(file)
        try:
            path = reader.path_to_scan(os.getcwd())
            nim = NiftiImage(path) #tests to see if the file exists
            for i in range(3):
                if maxsize[i] < nim.extent[i]:
                    maxsize[i] = nim.extent[i]
        except OSError:
            print "FAILED - OSError"
            print file
        except RuntimeError:
            print "FAILED - Runtime Error"
            print file
    print maxsize

maxsizeofNIFTI(xmls)