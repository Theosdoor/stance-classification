# README.md
<!-- README.md with commands, include random seeds, package versions, and the exact SemEval data files used. -->


## Data Loading Usage

`data_loader.py` is used to handle data loading / formatting.

### Loading the Dataset

To load the train, dev, and test splits, use the `load_dataset` function:

```python
from data_loader import load_dataset

train_df, dev_df, test_df = load_dataset()
```

### Preprocessing

`format_input_with_context` formats a target tweet (row) for classifier / prompt input:

```python
from data_loader import format_input_with_context

formatted_text = format_input_with_context(
    row,
    df,
    use_features=True, # include metadata features
    use_context=True, # include conversation context
    max_tokens=256,
    tokenizer=tokenizer # for max_tokens
)
```

## Random Seeds

I use `RAND_SEED = 42` in all experiments.

## Package Versions

Package versions pinned in `requirements.txt`.

## SemEval-2017 Datasets

The following datasets need to be downloaded and placed into the `downloaded_data` directory:

- **Training/Dev Data Root**: `downloaded_data/semeval2017-task8-dataset/rumoureval-data`
- **Train Labels**: `downloaded_data/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-train.json`
- **Dev Labels**: `downloaded_data/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-dev.json`
- **Test Data Root**: `downloaded_data/semeval2017-task8-test-data`
- **Test Labels**: `downloaded_data/subtaska.json`

> Below is adapted from [branchLSTM README](https://github.com/kochkinaelena/branchLSTM/blob/master/downloaded_data/README.md)

### Download online

Download and extract the following two files from the SemEval-2017 Task 8 page:

- Training and development data: [semeval2017-task8-dataset.tar.bz2](https://s3-eu-west-1.amazonaws.com/downloads.gate.ac.uk/pheme/semeval2017-task8-dataset.tar.bz2)
- Test data: [rumoureval2017-test.tar.bz2](http://alt.qcri.org/semeval2017/task8/data/uploads/rumoureval2017-test.tar.bz2)

For more information on the task and data, see the [SemEval-2017 Task 8](http://alt.qcri.org/semeval2017/task8/) webpage.

### Download via the command line

```
cd downloaded_data
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
