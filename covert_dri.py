import sys, glob
import xml.etree.ElementTree as et

"""
This script creates a clean version of the Dr. Inventor corpus.

Clean version looks like this (label \t sentence):

Background  Some sentece.
Background  This is another sentence.
Result  Last sentence of this document

Method  First sentence of the next document.
Method  Another sentence.
Result  Last sentence again

...

"""

#Get input and output path
in_path = sys.argv[1]
out_path = sys.argv[2]

if (in_path[-1] != '/'):
    in_path += '/'
if(out_path[-1] != '/'):
    out_path += '/'

files = glob.glob(in_path + '**/*RHETORICAL*.xml')

f_output = open(out_path + 'full_clean.txt', 'w', encoding="utf-8")
ignore = ['DRI_Unspecified', 'Sentence']

def map_labels(label):
    x = {"DRI_Challenge": "Challenge",
         "DRI_Challenge_Goal": "Challenge",
         "DRI_Approach": "Approach",
         "DRI_Outcome": "Outcome",
         "DRI_Outcome_Contribution": "Outcome",
         "DRI_Background": "Background",
         "DRI_FutureWork": "FutureWork",
         "DRI_Challenge_Hypothesis": "Challenge"}
    return x[label]

for filename in files:
    tree = et.parse(filename)
    root = tree.getroot()
    sentences = root.iter('Sentence')
    for sentence in sentences:
        label = sentence.attrib['rhetoricalClass']
        text = "".join(sentence.itertext())

        if label != None and text != "" and label not in ignore:
            text = text.replace('\n', '')
            f_output.write(map_labels(label) + "\t" + text + "\n")

    f_output.write("\n")

f_output.close()
