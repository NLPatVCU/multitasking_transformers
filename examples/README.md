## Loading the data.
Unfortunately, PHI laws and agreements make distributing clinical datasets difficult. To ease the friction required to
replicate our experiments, we provide a preprocessing script that takes as input the un-zipped dataset download from
the data distributor and caches the tensors required to train our multitasking models.


Datasets should be un-zipped into the appropriate sub-directory of raw_datasets. Once un-zipped in the
correct directory, `preprocess_datasets.py` will handle the rest.