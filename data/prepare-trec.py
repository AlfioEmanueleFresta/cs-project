import os

TREC_FILE = 'raw/trec.txt'

with open(TREC_FILE, 'rt') as f:
    lines = f.readlines()
    lines = [line.rstrip("\n") for line in lines]

prepared = {}

for line in lines:
    space = line.index(" ")

    label = line[0:space]
    sentence = line[space + 1:]

    if label in prepared:
        prepared[label].append(sentence)

    else:
        prepared[label] = [sentence]

for label in prepared.keys():
    for sentence in prepared[label]:
        print("Q: %s" % sentence)
    print("A: %s" % label)
    print("")

