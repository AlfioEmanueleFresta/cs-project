import gzip


posts = []

with gzip.open("raw/snippets.txt.gz", "rt") as f:
    while True:
        line = f.readline().rstrip("\n")
        if not line:
            break
        line = line.split(' ')
        title = ' '.join(line[:-1])
        label = line[-1]
        posts.append((title, label))


# Prepare a dictionary by answer
processed = {}
for title, label in posts:
    if label in processed:
        processed[label].append(title)
    else:
        processed[label] = [title]

for label in processed.keys():
    for question in processed[label]:
        print("Q: %s" % question)
    print("A: %s" % label)
    print("")

