import re
import os
import sys


def list_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))]

def filter_line(line):
    return "\t" in line or ";" in line or "<" in line  or ">" in line or len(re.sub(' +',' ',line)) < 32

def load_data(path, max_lines = 100):
    files = list_files(path)
    print files
    data = ""
    #alphabet = set()

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
                    data = data + " " + x
                    #alphabet.update(set)))
                    i = i + 1
                    del x

    return (data)

#texts,letters = load_data("data")

#for letter in sorted(letters):
#    print letter,
