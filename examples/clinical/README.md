#### Replicating the MT-Clinical-BERT model.

To replicate results in the MT-Clinical-BERT model:
 
1) First follow the data pre-processing instructions found in the README of
the examples directly.
2) Everything can be controlled from the gin config file `config.gin`. Update this with:

- multi_tasking_train.model_storage_directory : the directory to store your models
- multi_tasking_train.ml_flow_directory : the directory for mlflow to log to. This requires setting up an mlflow tracker instance.
- multi_tasking_train.transformer_weights : the directory containing your pre-trained BERT instance. This should be the same as you set in `preprocess_datasets.py`.
- multi_tasking_train.device : 'cuda' or 'cpu'. Note the current implementation supports only single device training.

3) Running the bash script `multitasking_transformer_train.sh` with the desired gpu set will begin training.