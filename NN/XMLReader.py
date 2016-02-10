"""
A class to read the XML doc with each MRI scan. Each scan is included with an XML with various information.
"""

# parser
import xml.etree.ElementTree as ET
import os


class XMLReader:
    # Instantiates
    def __init__(self, path):
        # Safety check to make sure the file path is valid
        try:
            self.tree = ET.parse(path)
            self.root = self.tree.getroot()
        except IOError as ioerr:
            print "Failed to parse\n"
            print ioerr
            print "\n"

    # Returns 0 for NC, 1 for MCI, 2 for AD
    def subject_status(self):
        for i in self.root.iter("subjectInfo"):
            if i.text == "Normal":
                return 0
            elif i.text == "MCI":
                return 1
            elif i.text == "AD":
                return 2
            else:
                continue  # multiple subjectInfo fields in  xml doc, makes all of them are iterated through

    # Returns patient number
    def subject_identifier(self):
        for i in self.root.iter("subjectIdentifier"):
            return i.text

    # checks to see if the current XML doc is for an MRI
    def is_mri(self):
        for i in self.root.iter("modality"):
            if i.text == "MRI":
                return True

    # finds the image ID
    def getderiveduid(self):
        for i in self.root.iter("imageUID"):
            return i.text

    # finds a path to the respective scan from current directory
    def path_to_scan(self, origin):
        # first folder, patient number
        id = self.subject_identifier()
        path = "/" + id + "/"

        # second folder, scan label
        for i in self.root.iter("processedDataLabel"):
            label = i.text.split(";")
            break
        firstItem = True
        for i in label:
            if firstItem == True:
                path = path + i.replace(" ", "")
                firstItem = False
                continue
            path = path + "__" + i.strip().replace(" ", "_")

        # third folder scan date
        for i in self.root.iter("dateAcquired"):
            split = i.text.split(" ")
            lhs = split[0]
            rhs = split[1]
            break
        path = path + "/" + lhs
        rhsplit = rhs.split(":")
        for i in rhsplit:
            path = path + "_" + i
        # fourth folder, series number
        for i in self.root.iter("seriesIdentifier"):
            sid = i.text
            break
        path = path + "/" + "S" + sid

        # finally finds the scan, checks to see if its a .nii file
        items = os.listdir(origin + path)
        scans = 0
        scan = []
        for i in items:
            curr = i.split(".")
            if curr[len(curr) - 1] == "nii":
                scan.append(i)
                scans += 1
        if scans > 1:
            imageid = self.getderiveduid()
            for i in scan:
                parsed = i.replace(".nii", "").split("_")
                for x in parsed:
                    if x == imageid:
                        return path + "/" + i
        return path + "/" + scan[0]
