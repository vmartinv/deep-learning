import re
import os
import sys
import h5py

def list_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))]

def filter_line(line):
    return "\t" in line or ";" in line or "<" in line  or ">" in line or len(re.sub(' +',' ',line)) < 32

def load_data(path, max_lines = None):
    files = list_files(path)
    print files
    data = []

    for f in files:
        print "processing",f
        with open(f) as infile:
            i = 0
            for line in infile:
                if i == max_lines:
                    break
                if not filter_line(line):
                    x = re.sub(' +',' ',line)[:-1]
                    x = x.decode("ISO-8859-1","decode")
                    data.append(x)
                    #alphabet.update(set)
                    i = i + 1
                    del x

    return ' '.join(data)

text = load_data("rawdata/spanish", 100)
for letter in sorted(set(text)):
    if not letter.isalpha():
        p=text.find(letter)
        print letter,'=',text[p-20:p+20]
for l in sorted(set(text)):
    print l,


H5FOLDER = 'dataseth5'
if not os.path.exists(H5FOLDER):
    os.makedirs(H5FOLDER)
# Create a new file using defaut properties.
dataset = h5py.File(os.path.join(H5FOLDER, 'lesslines.h5'),'w')

# Create a dataset under the Root group.
text = text.lower().encode("unicode_escape")

text = dataset.create_dataset("dataset", data=[text])

# Close the file before exiting
dataset.close()

