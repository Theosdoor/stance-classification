# Towards Automated Rumour Stance Classification
<!-- README.md with commands, include random seeds, package versions, and the exact SemEval data files used. -->

![Comparison of all 5 classification methods on the test set](results/exp4_test_final.png)
*Figure 14: Comparison of all 5 classification methods on the test set. The classifier score is the test set performance of the best model for validation macro-F1. Prompting scores are averaged over 3 runs on the test set.*

## Tweet Viewer

An interactive web-based viewer for exploring the SemEval 2017 Task 8 dataset is available at:

🔗 **[https://theosdoor.github.io/stance-classification/](https://theosdoor.github.io/stance-classification/)**

The viewer displays Twitter conversation threads with their stance labels:
- **Support (S)** – replies supporting the rumour claim
- **Deny (D)** – replies denying or refuting the claim  
- **Query (Q)** – replies questioning the claim's veracity
- **Comment (C)** – neutral comments without clear stance

## Quickstart

### 1. Set up data

Create the `downloaded_data` directory and download the [SemEval-2017 Task 8](https://alt.qcri.org/semeval2017/task8/index.php?id=data-and-tools) datasets:

> **Note:** On mac/linux, use `curl -LO` instead of `wget`.

```bash
# Create directory and navigate into it
mkdir -p downloaded_data && cd downloaded_data

# download training/dev data and test tweets
wget https://s3-eu-west-1.amazonaws.com/downloads.gate.ac.uk/pheme/semeval2017-task8-dataset.tar.bz2 # train/dev data

wget http://alt.qcri.org/semeval2017/task8/data/uploads/rumoureval2017-test.tar.bz2 # test

# download test labels (subtaskA.json)
wget http://alt.qcri.org/semeval2017/task8/data/uploads/subtaska.json

# Extract and clean up
tar -xf semeval2017-task8-dataset.tar.bz2
tar -xf rumoureval2017-test.tar.bz2
rm semeval2017-task8-dataset.tar.bz2 rumoureval2017-test.tar.bz2

cd .. # back to root
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run notebooks

- `nb_1_analytics.ipynb` - code for the analytics part of the report (section 2).
- `nb_2_classifier.ipynb` - code for the fine-tuning part of the report (section 3.1).
- `nb_3_prompting.ipynb` - code for the prompting part of the report (section 3.2 and 3.3).
- `data_loader.py` is used to handle data loading / formatting.

#### Note - using data_loader.py

To load the train, dev, and test splits, use the `load_dataset` function:

```python
from data_loader import load_dataset
train_df, dev_df, test_df = load_dataset()
```

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
