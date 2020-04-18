from multitasking_transformers.data import NERDataset, SentencePairRegressionDataset, SentencePairClassificationDataset
from multitasking_transformers.heads import TransformerHead, SubwordClassificationHead,\
    CLSRegressionHead, CLSClassificationHead
from multitasking_transformers.multitaskers import MultiTaskingBert
from multitasking_transformers.dataloaders import RoundRobinDataLoader
from torch.utils.data import DataLoader
import logging, os, torch, socket, time
from pprint import pprint

log = logging.getLogger('root')




def setup_logger():

    #Set run specific envirorment configurations
    timestamp = time.strftime("run_%Y_%m_%d_%H_%M_%S") + "_{machine}".format(machine=socket.gethostname())


    log.handlers.clear()
    formatter = logging.Formatter('%(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.setLevel(logging.INFO)
    log.addHandler(ch)

    #Set global GPU state
    # if torch.cuda.is_available() and gin.query_parameter('multi_tasking_train.device') == 'cuda':
    #     log.info("Using CUDA device:{0}".format(torch.cuda.current_device()))
    # else:
    #     if gin.query_parameter('multi_tasking_train.device') == 'cpu':
    #         log.info("Utilizing CPU")
    #     else:
    #         raise Exception(f"Unrecognized device: {gin.query_parameter('multi_tasking_train.device')}")

    #ML-Flow
    # mlflow.set_tracking_uri(f"{gin.query_parameter('multi_tasking_train.ml_flow_directory')}")
    # mlflow.set_experiment(f"/{gin.query_parameter('multi_tasking_train.experiment_name')}")

    # mlflow.start_run()
    # gin_parameters = gin.config._CONFIG.get(list(gin.config._CONFIG.keys())[0])
    #gin_parameters['tasks'] = TASKS

    # mlflow.log_params(gin_parameters)


HEADS = {
    'cls_classification': CLSClassificationHead,
    'cls_regression': CLSRegressionHead,
    'subword_classification': SubwordClassificationHead
}


DATASETS = {
    'ner': NERDataset,
    'sts': SentencePairRegressionDataset,
    'nli': SentencePairClassificationDataset
}

clinical_ner = ['i2b2_2010', 'i2b2_2012', 'i2b2_2014', 'n2c2_2018', 'quaero_2014']
clinical_sts = ['n2c2_2019']
clinical_nli = ['medrqe_2016', 'mednli_2018']

TASKS = {
    'ner': { }
    ,
    'sts': { }
    ,
    'nli': { }
}

def load_clinical_configured_tasks(preprocessed_directory=os.path.join(os.getcwd(), '..', '..', 'data')):

    for dataset in clinical_ner:
        if os.path.exists(f"{preprocessed_directory}/{dataset}/ner/train"):
            if dataset not in TASKS['ner']:
                TASKS['ner'][dataset] = {}
            TASKS['ner'][dataset].update(
                {
                'head': 'subword_classification',
                'batch_size': 25,
                'train': f"{preprocessed_directory}/{dataset}/ner/train",
                'test': f"{preprocessed_directory}/{dataset}/ner/test"
                }
            )

    for dataset in clinical_sts:
        if os.path.exists(f"{preprocessed_directory}/{dataset}/similarity/train"):
            if dataset not in TASKS['sts']:
                TASKS['sts'][dataset] = {}
            TASKS['sts'][dataset].update(
                {
                'head': 'cls_regression',
                'batch_size': 40,
                'train': f"{preprocessed_directory}/{dataset}/similarity/train",
                'test': f"{preprocessed_directory}/{dataset}/similarity/test"
                }
            )
    for dataset in clinical_nli:
        if os.path.exists(f"{preprocessed_directory}/{dataset}/nli/train"):
            if dataset not in TASKS['nli']:
                TASKS['nli'][dataset] = {}
            TASKS['nli'][dataset].update(
                {
                'head': 'cls_classification',
                'batch_size': 40,
                'train': f"{preprocessed_directory}/{dataset}/nli/train",
                'test': f"{preprocessed_directory}/{dataset}/nli/test"
                }
            )

    return TASKS


def predict(
          transformer_weights='mt_clinical_bert_8_tasks',
          model_storage_directory='',
          device='cuda',
          learning_rate=5e-5,
          seed=5,
          shuffle=True,
          num_workers=1,
          transformer_hidden_size = 768,
          transformer_dropout_prob = .1
          ):

    # log.info(gin.config_str())

    torch.random.manual_seed(seed)
    heads_and_datasets = []
    load_clinical_configured_tasks()

    print("MT Prediction over the following tasks:")
    pprint(TASKS)

    for task in TASKS:
        for dataset in TASKS[task]:
            train_dataset = DATASETS[task](TASKS[task][dataset]['train'])
            test_dataset = DATASETS[task](TASKS[task][dataset]['test'])

            labels = train_dataset.entity_labels if hasattr(train_dataset, 'entity_labels') else None
            if hasattr(train_dataset, 'class_labels'):
                labels = train_dataset.class_labels

            head = HEADS[TASKS[task][dataset]['head']](dataset,
                                                       labels=labels,
                                                       hidden_size=transformer_hidden_size,
                                                       hidden_dropout_prob=transformer_dropout_prob
                                                       )

            if TASKS[task][dataset]['head'] == 'subword_classification':
                if 'evaluate_biluo' in TASKS[task][dataset]:
                    head.config.evaluate_biluo = TASKS[task][dataset]['evaluate_biluo']
                else:
                    head.config.evaluate_biluo = False
            heads_and_datasets.append((head,
                                       DataLoader(test_dataset,
                                                  batch_size = TASKS[task][dataset]['batch_size'],
                                                  shuffle = shuffle,
                                                  num_workers=num_workers
                                                  )
                                       )
                                      )



    heads = [head for head, _ in heads_and_datasets]
    # mlflow.set_tag('number_tasks', str(len(heads)))
    mtb = MultiTaskingBert(heads,
                           model_storage_directory=model_storage_directory,
                           transformer_weights=transformer_weights,
                           device=device,
                           learning_rate=learning_rate
                           )

    mtb.predict(heads_and_datasets)



if __name__ == "__main__":
    setup_logger()
    predict()
    # mlflow.end_run()












