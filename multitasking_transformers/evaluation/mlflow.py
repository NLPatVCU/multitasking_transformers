import numpy as np

class MLFlowRun:
    """
    A custom data structure for extracting and analyzing experiments from MLFlow.
    """

    def __init__(self,tracking_uri=None, experiment_name=None, run_id=None):
        """
        Pulls all statistics from tracking servers in an organized manner.
        :param tracking_uri: the url of the tracking server
        :param experiment_name: the name of the tracking server experiment
        :param run_id: the uuid of the mlflow run
        :return: a dictionary organized by task -> dataset -> task specific metrics
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_id = run_id

        self.metrics = self.compute_run_statistics()

    def compute_run_statistics(self):
        from mlflow.tracking.client import MlflowClient

        client = MlflowClient(tracking_uri=self.tracking_uri)
        experiments = [str(exp.experiment_id) for exp in client.list_experiments() if exp.name == f"/{self.experiment_name}"]

        runs = client.search_runs(
          experiment_ids=experiments
        )
        try:
            run = [run for run in runs if run.info.run_id == self.run_id][0]
        except Exception as e:
            raise Exception(f"No run with id '{self.run_id}' under experiment '{self.experiment_name}'")
        metrics = list(run.data.metrics.keys())

        METRICS = {

        }
        for metric in metrics:
            if metric.split('/')[-1].startswith('SubwordClassificationHead'):
                if 'ner' not in METRICS:
                    METRICS['ner'] = {}
                if len(metric.split('/')) == 2: #cross entity level metric
                    dataset_name = '_'.join(metric.split('/')[1].split('_')[1:])
                    metric_name = metric.split('/')[0]
                    if dataset_name not in METRICS['ner']:
                        METRICS['ner'][dataset_name] = {'GLOBAL_METRICS':{}, 'ENTITY_METRICS':{}}
                    METRICS['ner'][dataset_name]['GLOBAL_METRICS'][metric_name] = []
                else:
                    dataset_name = '_'.join(metric.split('/')[-1].split('_')[1:])
                    if dataset_name not in METRICS['ner']:
                        METRICS['ner'][dataset_name] = {'GLOBAL_METRICS':{}, 'ENTITY_METRICS':{}}
                    entity_name = metric.split('/')[1]
                    metric_name = metric.split('/')[0]
                    if entity_name not in METRICS['ner'][dataset_name]['ENTITY_METRICS']:
                        METRICS['ner'][dataset_name]['ENTITY_METRICS'][entity_name] = {}
                    if metric_name not in METRICS['ner'][dataset_name]['ENTITY_METRICS'][entity_name]:
                        METRICS['ner'][dataset_name]['ENTITY_METRICS'][entity_name][metric_name] = []
            if metric.split('/')[-1].startswith('CLSRegressionHead'):
                if 'sts' not in METRICS:
                    METRICS['sts'] = {}
                dataset_name = '_'.join(metric.split('/')[-1].split('_')[1:])
                metric_name = metric.split('/')[0]
                if dataset_name not in METRICS['sts']:
                    METRICS['sts'][dataset_name] = {}
                if metric_name not in METRICS['sts'][dataset_name]:
                    METRICS['sts'][dataset_name][metric_name] = []
            if metric.split('/')[-1].startswith('CLSClassificationHead'):
                if 'nli' not in METRICS:
                    METRICS['nli'] = {}
                dataset_name = '_'.join(metric.split('/')[-1].split('_')[1:])
                metric_name = metric.split('/')[0]
                if dataset_name not in METRICS['nli']:
                    METRICS['nli'][dataset_name] = {}
                if metric_name not in METRICS['nli'][dataset_name]:
                    METRICS['nli'][dataset_name][metric_name] = []


        metric_history = {metric:{} for metric in metrics}

        for metric in metrics:
            time_steps = client.get_metric_history(run_id=self.run_id, key=metric)

            # metric_by_time = {metric.step: {'value': metric.value, 'timestamp': metric.timestamp} for metric in time_steps}
            metric_by_time = [metric.value for metric in time_steps]
            metric_history[metric] = metric_by_time

            if metric.split('/')[-1].startswith('SubwordClassificationHead'):
                if len(metric.split('/')) == 2:  # cross entity level metric
                    metric_name = metric.split('/')[0]
                    dataset_name = '_'.join(metric.split('/')[1].split('_')[1:])
                    METRICS['ner'][dataset_name]['GLOBAL_METRICS'][metric_name] = metric_by_time
                else:
                    dataset_name = '_'.join(metric.split('/')[-1].split('_')[1:])
                    entity_name = metric.split('/')[1]
                    metric_name = metric.split('/')[0]
                    METRICS['ner'][dataset_name]['ENTITY_METRICS'][entity_name][metric_name] = metric_by_time
            if metric.split('/')[-1].startswith('CLSRegressionHead'):
                dataset_name = '_'.join(metric.split('/')[-1].split('_')[1:])
                metric_name = metric.split('/')[0]
                METRICS['sts'][dataset_name][metric_name] = metric_by_time

            if metric.split('/')[-1].startswith('CLSClassificationHead'):
                dataset_name = '_'.join(metric.split('/')[-1].split('_')[1:])
                metric_name = metric.split('/')[0]
                METRICS['nli'][dataset_name][metric_name] = metric_by_time
        return METRICS

    def compare_against_run(self, run, task):
        """
        Compares this run with 'run' on the given (task,dataset) along the given metric (or all).
        For NER:
            - Single class tasks will be evaluated with F1
            - Multi-class tasks with micro-averaged F1
        :param run: an instance of a processed MLFlowRun
        :param task: the task to compare on
        :param dataset: the dataset name
        :param metric: the metric to compare along.
        :return: a list [dataset, task metric, my_performance, other_performance]
        """
        my_metrics = self.metrics[task]
        other_metrics = run.metrics[task]

        single_entity_datasets = {}
        ignored_datasets = []

        #ignore un-matching datasets, single out single label ner TASKS
        for dataset in sorted(list(my_metrics.keys())):
            if dataset not in other_metrics:
                ignored_datasets.append(dataset)
                print(f"Ignoring metrics from: {dataset}")
            if task == 'ner':
                if len(my_metrics[dataset]['ENTITY_METRICS'].keys()) == 1:
                    single_entity_datasets[dataset] = list(my_metrics[dataset]['ENTITY_METRICS'].keys())[0]


        comparison = []

        for dataset in my_metrics:
            if dataset in ignored_datasets:
                continue
            if task == 'ner':
                my_performance = None
                other_performance = None
                if dataset in single_entity_datasets:
                    metric = 'F1'
                    my_performance = my_metrics[dataset]['ENTITY_METRICS'][single_entity_datasets[dataset]][metric]
                    other_performance = other_metrics[dataset]['ENTITY_METRICS'][single_entity_datasets[dataset]][metric]
                    #print(dataset, metric, single_entity_datasets[dataset])
                else:
                    metric = 'Micro-F1'
                    #print(dataset, metric)
                    my_performance = my_metrics[dataset]['GLOBAL_METRICS'][metric]
                    other_performance = other_metrics[dataset]['GLOBAL_METRICS'][metric]

                comparison.append((dataset, task, metric, my_performance, other_performance))
            if task == 'nli':
                metric = 'Accuracy'
                my_performance = my_metrics[dataset][metric]
                other_performance = other_metrics[dataset][metric]
                comparison.append((dataset, task, metric, my_performance, other_performance))
            if task == 'sts':
                metric = 'PearsonRho'
                my_performance = my_metrics[dataset][metric]
                other_performance = other_metrics[dataset][metric]
                comparison.append((dataset, task, metric, my_performance, other_performance))



        return comparison
