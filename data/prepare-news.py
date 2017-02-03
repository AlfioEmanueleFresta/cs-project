import os

NEWS_FILE = 'raw/en09062011.news'

prepared = {}

with open(NEWS_FILE, 'rt') as f:

    while True:

        title = f.readline().rstrip("\n")
        subtitle = f.readline().rstrip("\n")
        url = f.readline().rstrip("\n")
        id = f.readline().rstrip("\n")
        date = f.readline().rstrip("\n")
        source = f.readline().rstrip("\n")
        category = f.readline().rstrip("\n")
        f.readline()

        if title == "":
            break

        #sentence = "%s. %s" % (title, subtitle)
        sentence = "%s" % (title,)
        label = category

        if label in prepared:
            prepared[label].append(sentence)

        else:
            prepared[label] = [sentence]

for label in prepared.keys():
    for sentence in prepared[label]:
        print("Q: %s" % sentence)
    print("A: %s" % label)
    print("")
print("")
