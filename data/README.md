# Datasets

* The `raw` folder contains a few raw compressed datasets that can be used in
  conjunction with a preparation script to prepare a well-formatted dataset.
  These are compressed using Gzip and should be decompressed prior to use, e.g.
  ```bash
  $ cd data/raw/
  $ gzip -d trec.txt.gz
  ```

* The `prepare-*.py` scripts can be used to convert the raw datasets to the
  prepared versions. The scripts output to stdout, therefore the output should
  be saved to a file, e.g.
  ```bash
  $ cd data/
  $ python prepare-trec.py > trec.txt

  # Count the number of questions and answers generated.
  $ grep "Q:" trec.txt | wc -l
    5452
  $ grep "A:" trec.txt | wc -l
    50
  ```

## Dataset table

| Dataset                 | Raw file                         | Script name           | Source                                       |
|-------------------------|----------------------------------|-----------------------|----------------------------------------------|
| TREC                    | data/raw/trec.txt.gz             | data/prepare-trec.py  | Text REtrieval Conference Dataset            |
| Stack Overflow CS Posts | data/raw/stackoverflow-cs.xml.gz | data/prepare-stack.py | Stack Overflow XML database dump, CS subsite |
|                         |                                  |                       |                                              |