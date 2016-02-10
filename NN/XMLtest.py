import XMLReader as xmlR
import os

#neptune is a local user account
os.chdir("/home/neptune/temp_data/ADNI")
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