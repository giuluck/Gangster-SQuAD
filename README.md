# Gangster-SQuAD

NLP system trained on the Stanford Question Answering Dataset (SQuAD) that, given a paragraph and a question regarding it, provides a single answer, which is obtained selecting a span of text from the paragraph.

## Requirements

In order to install all the necessary libraries it suffices to run

```bash
pip install -r requirements.txt
```

We suggest the use of `venv` if you are running in local.

## Running

Be sure to download the weights of our model from [this](https://drive.google.com/drive/folders/15NSBlS5Dk8zX6TX81yI3xxH3jMdc5wbV?usp=sharing) link and save them in the same directory of the `compute_answers.py` script.  
The `predictions.json` file can be obtained with the following command

```bash
python3 compute_answers.py <dataset.json>
```
