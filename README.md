# :arrows_clockwise: Multitasking Transformers :arrows_clockwise:
training nlp models that can perform multiple tasks with the same set of representations.

pre-trained models are currently available that multitask over eight clinical note tasks.

This codebase can be utilized to replicate results for a currently in-review AMIA paper. See the Replication section
for details.
# Installation

Install with

```
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.0/en_core_sci_sm-0.2.0.tar.gz
```
```
pip install git+https://github.com/AndriyMulyar/multitasking_transformers
```

# Use
[Examples](/examples) are available for training, evaluation and text prediction.

Running the script [predict_ner.py](/examples/predict_ner.py) will automatically
download a pre-trained clinical note multi-tasking model, run the model through a de-identified
clinical note snippet and display the entity tags in your browser with displacy.

# Replication
See the directory [/examples/experiment_replication](/examples/experiment_replication).

# Preprint
https://arxiv.org/abs/2004.10220

# Acknowledgement
Implementation, development and training in this project were supported by funding from the McInnes NLP Lab at Virginia Commonwealth University.


-------------
Todo:
- [ ] Upload single task finetuned pre-trained models.
- [ ] Wrap in API and release API in docker container.
