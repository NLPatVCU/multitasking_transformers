from typing import List, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import logging, mlflow
from scipy.stats import pearsonr


log = logging.getLogger('root')

def evaluate_ner(predicted_and_correct: List[Tuple[List,List]],
                 bilou_labels: List,
                 head_identifier,
                 evaluate_bilou = False,
                 step=1):
    """

    :param predicted_and_correct:
    :param bilou_labels: list of bilou labels
    :param head_identifier: head name
    :param evaluate_bilou: If true, will evaluate over bilou labels, otherwise will strip prefixes and eval over original.
    :param step: current epoch for logging
    :return:
    """
    flattened_predictions = [int(x) for tuple in predicted_and_correct for x in tuple[0]]
    flattened_gold_labels = [x for tuple in predicted_and_correct for x in tuple[1]]

    assert len(flattened_predictions) == len(flattened_gold_labels)


    #convert BILOU labels back to original by mapping all BILOU tags to original entity tags by stripping prefixes.
    entity_labels = []
    bilou_to_entity = {}  # maps biluou indices to entity indices.
    if evaluate_bilou:
        entity_labels = bilou_labels
        for idx, label in enumerate(bilou_labels):
            bilou_to_entity[idx] = entity_labels.index(label)
    else:
        for idx, label in enumerate(bilou_labels):
            if label == 'O' or label == 'BERT_TOKEN':
                entity_labels.append(label)
                bilou_to_entity[idx] = entity_labels.index(label)
            else:
                if label[2:] not in entity_labels: # remove prefix
                    entity_labels.append(label[2:])
                bilou_to_entity[idx] = entity_labels.index(label[2:])


    filtered_labels_indices = []
    for idx, label in enumerate(entity_labels):
        if not label in ['BERT_TOKEN', 'O']:
            filtered_labels_indices.append(idx)


    #map back to original
    flattened_predictions = [bilou_to_entity[label_idx] for label_idx in flattened_predictions]
    flattened_gold_labels = [bilou_to_entity[label_idx] for label_idx in flattened_gold_labels]


    #Compute metrics
    f_measures = f1_score(flattened_gold_labels, flattened_predictions, average=None,
                          labels=[idx for idx, label in enumerate(entity_labels)])
    precisions = precision_score(flattened_gold_labels, flattened_predictions, average=None,
                                 labels=[idx for idx, label in enumerate(entity_labels)])
    recalls = recall_score(flattened_gold_labels, flattened_predictions, average=None,
                           labels=[idx for idx, label in enumerate(entity_labels)])

    for label_idx in filtered_labels_indices:
        mlflow.log_metric(f"Precision/{entity_labels[label_idx].replace(' ', '_')}/{head_identifier}",precisions[label_idx], step = step)
        mlflow.log_metric(f"Recall/{entity_labels[label_idx].replace(' ', '_')}/{head_identifier}", recalls[label_idx], step=step)
        mlflow.log_metric(f"F1/{entity_labels[label_idx].replace(' ', '_')}/{head_identifier}", f_measures[label_idx], step=step)

    # exclude bert_token, other in macro/micro calculation
    micro_f1 = f1_score(flattened_gold_labels, flattened_predictions, average='micro', labels=filtered_labels_indices)
    macro_f1 = f1_score(flattened_gold_labels, flattened_predictions, average='macro', labels=filtered_labels_indices)

    mlflow.log_metric(f"Macro-F1/{head_identifier}", float(macro_f1), step=step)
    mlflow.log_metric(f"Micro-F1/{head_identifier}", float(micro_f1), step=step)
    log.info("Micro-F1: " + str(micro_f1))
    log.info("Macro-F1: " + str(macro_f1))


def evaluate_sts(predicted_and_correct: Tuple[List,List], head_identifier, step=1):
    correlation = pearsonr(predicted_and_correct[0], predicted_and_correct[1])[0]
    mlflow.log_metric(f"PearsonRho/{head_identifier}", float(correlation), step=step)
    log.info(f"Correlation: {correlation}")

def evaluate_classification(predicted_and_correct: Tuple[List,List], labels: List,  head_identifier, step=1):

    label_indices = [idx for idx, label in enumerate(labels)]

    f_measures = f1_score(predicted_and_correct[1], predicted_and_correct[0], average=None,
                          labels=[idx for idx, label in enumerate(labels)])
    precisions = precision_score(predicted_and_correct[1], predicted_and_correct[0], average=None,
                                 labels=[idx for idx, label in enumerate(labels)])
    recalls = recall_score(predicted_and_correct[1], predicted_and_correct[0], average=None,
                           labels=[idx for idx, label in enumerate(labels)])


    for label_idx in label_indices:
        mlflow.log_metric(f"Precision/{labels[label_idx].replace(' ', '_')}/{head_identifier}", precisions[label_idx],
                          step=step)
        mlflow.log_metric(f"Recall/{labels[label_idx].replace(' ', '_')}/{head_identifier}", recalls[label_idx],
                          step=step)
        mlflow.log_metric(f"F1/{labels[label_idx].replace(' ', '_')}/{head_identifier}", f_measures[label_idx],
                          step=step)

    # exclude bert_token, other in macro/micro calculation
    micro_f1 = f1_score(predicted_and_correct[1], predicted_and_correct[0], average='micro', labels=label_indices)
    macro_f1 = f1_score(predicted_and_correct[1], predicted_and_correct[0], average='macro', labels=label_indices)
    accuracy = accuracy_score(predicted_and_correct[1], predicted_and_correct[0])

    mlflow.log_metric(f"Macro-F1/{head_identifier}", float(macro_f1), step=step)
    mlflow.log_metric(f"Micro-F1/{head_identifier}", float(micro_f1), step=step)
    mlflow.log_metric(f"Accuracy/{head_identifier}", float(accuracy), step=step)
    log.info("Micro-F1: " + str(micro_f1))
    log.info("Macro-F1: " + str(macro_f1))
    log.info("Accuracy: " + str(accuracy))






