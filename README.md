# CS Project

## Installation

The software can be configured on a modern Ubuntu 16.04+ machine by running:

```bash
$ cd cs-project/                                # cd into this directory
$ virtualenv -p python3 .venv                   # start a new python virtual environment
$ source .venv/bin/activate                     # activate the newly-created virtual environment
$ pip install -r requirements.txt               # install the requirements in the virtual environment (1)
$ pip install -r requirements-processing.txt    # install the requirements in the virtual environment (2)
```

## How to run the experiment

The experiment that has been included with the report can be run either in a single thread/worker, or in N threads/workers.
Please note that the experiment would likely take weeks to run in the first case, so it is highly recommended
 to run it in parallel on multiple machines (e.g. a cluster).

A modern NVIDIA GPU configured to work with the Theano installation would also significantly speed up the
experimentation process.


### Single machine

Simply run:

```
$ python experiment.py
```

The training process is shown on stdout. The results are saved to the `output.csv` file.

### Multiple workers

Example with 5 workers:

```
machine1$ python experiment.py --workers 5 --worker-id 1
machine2$ python experiment.py --workers 5 --worker-id 2
...
machine5$ python experiment.py --workers 5 --worker-id 5
```

The workload is split across all workers. The results are saved to the `output.csv` file. Note that the workers will try
to append text to this file atomically, as soon as each line is ready, so this file can be on a shared file-systems, and
all the results saved to the same file.


## Process results and prepare graphs

The following script can be launched to prepare all graphs, and generate the Latex source code for the result tables:

```
$ python process-results.py
```


## `example.py` usage -- Interactive shell

| Option         | Description                                                                                                                   |
|----------------|-------------------------------------------------------------------------------------------------------------------------------|
| `-v`           | Verbose mode. Print useful information, and progress.                                                                         |
| `--no-display` | Don't show a graphical interface when training (loss and accuracy plot).                                                      |
| `--no-train`   | Skip the training phase. Loads the last saved model and launches the interactive prediction shell immediately.                |
| `--resume`     | Load the last saved model before starting training, effectively resuming the training phase (e.g. if terminated prematurely). |
| `--transient`  | After training, don't save the model to disk. Any existing model will be preserved.                                           |

### Example

#### Training

```bash
$ example.py -v

Using gpu device 0: GeForce GTX 1070 (CNMeM is disabled, cuDNN 5005)
Loading Word Embedding data/embeddings/glove.6B.50d.txt (compressed=0)...
Loading search trees from cache file (data/embeddings/glove.6B.50d.txt.cache)...
Loaded 400003 words with vector length=52
Compiling functions...
Preparing training data... OK
Starting training. You can use CTRL-C to stop the training process.
Epoch      Time        Tr. loss   Val. loss  Val. acc.   B  Best acc.
---------  ----------  ---------  ---------  ----------  -  ----------
   1/1500   2.093092s   3.430102   3.156474   17.31092%  *   17.31092%
   2/1500   2.177848s   3.219012   3.008631   21.31092%  *   21.31092%
   3/1500   2.139112s   2.990131   2.814335   31.93277%  *   31.93277%
   4/1500   2.105939s   2.816056   2.672511   35.24370%  *   35.24370%
   5/1500   2.110484s   2.645123   2.506764   38.89076%  *   38.89076%
   6/1500   2.094425s   2.484004   2.363108   41.47899%  *   41.47899%
   7/1500   2.307011s   2.338798   2.253465   45.84874%  *   45.84874%
   8/1500   2.464797s   2.210065   2.157949   45.36134%  *   45.36134%
   9/1500   2.429513s   2.123823   2.086996   46.89076%  *   46.89076%
  10/1500   2.237623s   2.029540   2.014093   48.89076%  *   48.89076%
  ...
  30/1500   2.116877s   1.151394   1.380362   63.36134%  *   63.36134%
  31/1500   2.114621s   1.119197   1.364482   63.00840%  *   63.00840%
  32/1500   2.112367s   1.080961   1.350141   63.59664%  *   63.59664%
  33/1500   2.154481s   1.059438   1.345990   63.94958%  *   63.94958%
  34/1500   2.152469s   1.032259   1.330365   64.65546%  *   64.65546%
  35/1500   2.114789s   1.016570   1.309091   65.61345%  *   65.61345%
  36/1500   2.123771s   0.978632   1.291949   66.06723%  *   66.06723%
  37/1500   2.123244s   0.960823   1.296887   66.30252%      66.06723%
  38/1500   2.112140s   0.942375   1.279238   66.18487%  *   66.18487%
  39/1500   2.130267s   0.922863   1.293013   65.24370%      66.18487%
  ...
 124/1500   2.182609s   0.136449   1.523257   72.31933%      72.57143%
Training interrupted at epoch 124.
Best result (epoch=77, loss= 1.163830, accuracy= 72.57143%)
Testing network... DONE
Test results (loss= 1.245674, accuracy= 72.35294%)
Saving model to file (data/model.cache.npz)... OK
Interactive shell ready. Type 'exit' to close.
>> where are you from?
where {in outside now from .}  are {other these those have many}  you
  {'ll ? 'd me n't}  from {in while on . which}  ? {you maybe n't `` know}
  SYMEND {the , . of to}
[  2.60805461e-12   3.63053090e-08   5.51215784e-09   6.98147389e-08
   ...
   1.63617440e-08   6.38746513e-08]
LOC:other
predicted in 0.0144 seconds
>> exit
Bye!
```

### Prediction only

```bash
$ example.py -v --no-train

Using gpu device 0: GeForce GTX 1070 (CNMeM is disabled, cuDNN 5005)
Loading Word Embedding data/embeddings/glove.6B.50d.txt (compressed=0)...
Loading search trees from cache file (data/embeddings/glove.6B.50d.txt.cache)...
Loaded 400003 words with vector length=52
Compiling functions...
Loading model from file (data/model.cache.npz)... OK
Interactive shell ready. Type 'exit' to close.
>> who is the current us president
who {whom young him he had}  is {this it as same example}  the {which part in of on}
  current {term due terms change future}  us {u.s. about will _ take}
  president {vice met secretary presidency chairman}  SYMEND {the , . of to}
[  6.67965724e-05   1.51979915e-12   8.63229343e-06   2.17978493e-03
   ...
   1.00515853e-08   3.73741371e-10]
HUM:ind
predicted in 0.0211 seconds
>>
```


### Prediction only (quiet mode)

```bash
$ example.py -v --no-train

>> what is your name?
DESC:def
>> where are you from?
LOC:other
>>
```