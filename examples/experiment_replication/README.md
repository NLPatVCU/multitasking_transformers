## Loading the data.
Unfortunately, PHI laws and agreements make distributing clinical datasets difficult. To ease the friction required to
replicate our experiments, we provide a preprocessing script that takes as input the un-zipped dataset downloads from
the data distributor and caches the tensors required to train our multitasking models.

Fortunately, two datasets are under no PHI license and so are included in the the package:
1) [quaero-2014](https://quaerofrenchmed.limsi.fr/)
2) [med-rqe](https://github.com/abachaa/MEDIQA2019/tree/master/MEDIQA_Task2_RQE)

To run `preprocess_datasets.py`, you will need to set the directory of your installed BERT model (or alternatively a
transformers library compatible alias). Set this by updating `bert_weight_directory` in `preprocess_datasets.py`.
Running `preprocess_datasets.py` with
the included datasets will work and you will be able to multitask train on the the two tasks (quaero_2014 and med_rqe).

To fully replicate the MT-Clinical-BERT training, you must download the remaining dataset's yourself.
Med-NLI can be found here: [med-nli](https://physionet.org/content/mednli-bionlp19/1.0.1/) and the rest of the datasets can be found at the [DBMI Data Portal](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/).

Datasets should be un-zipped into the appropriate sub-directory of raw_datasets. Once un-zipped in the
correct directory, `preprocess_datasets.py` will handle the rest.

For reference, your `raw_datasets` directory should look roughly like this:

```bash
├── __init__.py
├── language.py
├── ner
│   ├── data.py
│   ├── i2b2_2010
│   │   ├── concept_assertion_relation_training_data
│   │   ├── concept_assertion_relation_training_data.tar.gz
│   │   ├── reference_standard_for_test_data
│   │   ├── reference_standard_for_test_data.tar.gz
│   │   ├── test_data
│   │   └── test_data.tar.gz
│   ├── i2b2_2012
│   │   ├── 2012-07-15.original-annotation.release
│   │   ├── 2012-07-15.original-annotation.release.tar.gz
│   │   ├── 2012-08-23.test-data.groundtruth.tar.gz
│   │   └── ground_truth
│   ├── i2b2_2014
│   │   ├── 2014_training-PHI-Gold-Set1.tar.gz
│   │   ├── testing-PHI-Gold-fixed
│   │   ├── testing-PHI-Gold-fixed.tar.gz
│   │   ├── training-PHI-Gold-Set1
│   │   ├── training-PHI-Gold-Set2
│   │   └── training-PHI-Gold-Set2.tar.gz
│   ├── __init__.py
│   ├── n2c2_2018
│   │   ├── gold-standard-test-data
│   │   ├── gold-standard-test-data.zip
│   │   ├── training_20180910
│   │   └── training_20180910.zip
│   └── quaero_frenchmed_2014
│       ├── test
│       └── train
├── nli
│   ├── data.py
│   ├── __init__.py
│   ├── med_nli
│   │   ├── mli_test_v1.jsonl
│   │   └── mli_train_v1.jsonl
│   ├── med_rqe
│       ├── Readme.txt
│       ├── RQE_test_pairs_AMIA2016.xml
│       └── RQE_train_pairs_AMIA2016.xml
└── similarity
    ├── data.py
    ├── __init__.py
    ├── n2c2_2019
    │   ├── clinicalSTS2019.test.labels.txt
    │   ├── clinicalSTS2019.test.txt
    │   └── clinicalSTS2019.train.txt

```