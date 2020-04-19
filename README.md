# :arrows_clockwise: Multitasking Transformers :arrows_clockwise:
training nlp models that can perform multiple tasks with the same set of representations.

pre-trained models are currently available that multitask over eight clinical note tasks.

This codebase can be utilized to replicate results for a currently in-review AMIA paper. See the Replication section
for details.
# Installation

Install with

```
pip install git+https://github.com/AndriyMulyar/multitasking_transformers
```

# Use
[Examples](/examples) are available for training, evaluation and text prediction.

Running the script [predict_ner.py](/examples/predict_ner.py) will automatically
download a pre-trained clinical note multi-tasking model, run the model through a de-identified
clinical note snippet and display the results in browser.


# Replication
See the directory [/examples/experiment_replication](/examples/experiment_replication).


# Acknowledgement
Implementation, development and training in this project were supported by funding from the McInnes NLP Lab at Virginia Commonwealth University.


-------------
Todo:
- [x] Individual finetuning training example.
- [x] Finish MT replication README.
- [ ] Upload single task finetuned pre-trained models.
- [x] Incorporate non-preprocessing prediction code.
- [ ] Wrap in API and release API in docker container.
