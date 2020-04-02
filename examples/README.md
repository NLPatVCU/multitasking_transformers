## Loading the data.
Unfortunately, PHI laws and agreements make distributing clinical datasets difficult. To ease the friction required to
replicate our experiments, we provide a preprocessing script that takes as input the un-zipped dataset downloads from
the data distributor and caches the tensors required to train our multitasking models.

Fortunately, two datasets are under no PHI license and so are included in the the package:
1) [quaero-2014](https://quaerofrenchmed.limsi.fr/)
2) [med-rqe]()

3) 

The rest of the datasets can be found at the [DBMI Data Portal](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/).

Datasets should be un-zipped into the appropriate sub-directory of raw_datasets. Once un-zipped in the
correct directory, `preprocess_datasets.py` will handle the rest.