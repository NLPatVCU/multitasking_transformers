# :arrows_clockwise: Multitasking Transformers :arrows_clockwise:
training nlp models that can perform multiple tasks with the same set of representations.

pre-trained models are currently available that multitask over eight clinical note tasks.

# Installation

Install with

```
pip install git+https://github.com/AndriyMulyar/multitasking_transformers
```

# Use
The current state of this codebase only supports training and validation. Check back later for open-ended prediction.
More [examples](/examples).


# Replication
See the directory [/examples/clinical/mt](/examples/clinical/mt).

# Notes
- For training you will need a GPU.
- For bulk inference where speed is not of concern lots of available memory and CPU cores will likely work.
- Model downloads are cached in `~/.cache/torch/bert_document_classification/`. Try clearing this folder if you have issues.



# Acknowledgement
Implementation, development and training in this project were supported by funding from McInnes NLP Lab at Virginia Commonwealth University.


-------------
Todo:
- [ ] Individual model training example.
- [ ] Finish MT replication README.
- [ ] Incorporate non-preprocessing prediction code.
- [ ] Wrap in API and release API in docker container.
