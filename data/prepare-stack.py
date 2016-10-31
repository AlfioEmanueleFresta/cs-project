import xml.etree.ElementTree as ET


tree = ET.parse('raw/stackoverflow-cs.xml')
root = tree.getroot()

i = 0
min_title_length = 25
max_title_length = 100
ratio_labels_to_samples = .001  # .005 for CS

posts = []


def _count(posts):
    d = {}
    for title, tags in posts:
        if not isinstance(tags, list):
            tags = [tags]
        for tag in tags:
            if tag in d:
                d[tag] += 1
            else:
                d[tag] = 1
    return d


for post in root.findall('row'):

    # Skip all non-posts
    if post.attrib['PostTypeId'] != "1":
        continue

    title = post.attrib['Title'].strip()

    tags = post.attrib['Tags']
    tags = tags.rstrip('>').lstrip('<')
    tags = tags.split('><')

    # Skip posts with short titles
    if len(title) < min_title_length:
        continue

    # Skip posts with verbose titles
    if len(title) > max_title_length:
        continue

    posts.append((title, tags))

tags_to_occurrences = _count(posts)


def _best_label(list_of_labels):
    # Choose the best label from a list of labels
    best_label = None
    for label in list_of_labels:
        try:
            # The best is the least common -- assumption is, it is more specific. We remove macro categories.
            if best_label is None or tags_to_occurrences[label] < tags_to_occurrences[best_label]:
                best_label = label
        except KeyError:
            continue
    assert best_label is not None
    return best_label

# Process post and only keep the best label
posts = [(title, _best_label(labels)) for title, labels in posts if any([label in tags_to_occurrences for label in labels])]
tags_to_occurrences = _count(posts)

# Now, remove rarely occurring tags
posts = [(title, label) for title, label in posts if tags_to_occurrences[label] > len(posts) * ratio_labels_to_samples]
tags_to_occurrences = _count(posts)

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

