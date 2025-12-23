# Downloaded Data

The following datasets need to be downloaded and placed into this directory.

To download from the command line, use the instructions at the end of this page.

## SemEval-2017 Datasets

Download and extract the following two files from the SemEval-2017 Task 8 page:

- Training and development data: [semeval2017-task8-dataset.tar.bz2](https://s3-eu-west-1.amazonaws.com/downloads.gate.ac.uk/pheme/semeval2017-task8-dataset.tar.bz2)
- Test data: [rumoureval2017-test.tar.bz2](http://alt.qcri.org/semeval2017/task8/data/uploads/rumoureval2017-test.tar.bz2)

The training and development data are from Zubiaga A, Liakata M, Procter R, Wong Sak Hoi G, Tolmie P (2016) [*Analysing How People Orient to and Spread Rumours in Social Media by Looking at Conversational Threads.*](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0150989) PLoS ONE 11(3): e0150989.

For more information on the task and data, see the [SemEval-2017 Task 8](http://alt.qcri.org/semeval2017/task8/) webpage.

## To download via the command line

### Download

```
cd <your-directory>/downloaded_data
wget https://s3-eu-west-1.amazonaws.com/downloads.gate.ac.uk/pheme/semeval2017-task8-dataset.tar.bz2
wget http://alt.qcri.org/semeval2017/task8/data/uploads/rumoureval2017-test.tar.bz2
```

### Extract and tidy up

```
tar -xf semeval2017-task8-dataset.tar.bz2
tar -xf rumoureval2017-test.tar.bz2
rm semeval2017-task8-dataset.tar.bz2
rm rumoureval2017-test.tar.bz2
```