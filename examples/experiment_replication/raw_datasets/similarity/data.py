from pkg_resources import resource_exists, resource_listdir, resource_string, resource_stream,resource_filename
from ..language import get_language
import random, os

base_path = os.path.join(os.getcwd(), 'raw_datasets')

def load_n2c2_2019_train_dev(hold_out=.2):
    data = load_n2c2_2019(partition='train')
    random.Random(1).shuffle(data)  # shuffle files with random seed 1
    return data[:int(len(data) * (1 - hold_out))], data[int(len(data) * (1 - hold_out)):]


def load_n2c2_2019(partition='train'):
    """

    :param partition:
    :return:
    """
    language = get_language()

    language.remove_pipe('parser')
    language.remove_pipe('_set_sentence_parse_exceptions')

    assert partition in ['train', 'test']

    annotation_dir = os.path.join(base_path, 'similarity', 'n2c2_2019')

    if not os.path.exists(os.path.join(annotation_dir, f'clinicalSTS2019.{partition}.txt')):
        raise FileNotFoundError('Cannot find %s data' % partition)

    lines = open(os.path.join(annotation_dir, f'clinicalSTS2019.{partition}.txt')).read().strip()

    test_labels_lines = open(os.path.join(annotation_dir, f'clinicalSTS2019.test.labels.txt')).read().strip().split('\n')

    data = []
    for idx,line in enumerate(lines.split('\n')):
        if partition == 'train':
            sentence1, sentence2, score = tuple(line.split("\t"))
            data.append({
                'index': idx,
                'sentence_1': sentence1,
                'sentence_2': sentence2,
                'similarity': float(score)
            })
        elif partition == 'test':
            sentence1, sentence2 = tuple(line.split("\t"))
            data.append({
                'index': idx,
                'sentence_1': sentence1,
                'sentence_2': sentence2,
                'similarity': float(test_labels_lines[idx])
            })

    return data


# def load_sts_b_data():
#     """
#     Loads the STS-B dataset found here: http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark
#     """
#
#     if not resource_exists('n2c2_2019', 'clinical_semantic_similarity/data/sts_b/sts-train.csv'):
#         raise FileNotFoundError('Cannot find STS-B dataset')
#
#     train = resource_string('n2c2_2019', 'clinical_semantic_similarity/data/sts_b/sts-train.csv').decode('utf-8').strip()
#     dev = resource_string('n2c2_2019', 'clinical_semantic_similarity/data/sts_b/sts-dev.csv').decode('utf-8').strip()
#     test = resource_string('n2c2_2019', 'clinical_semantic_similarity/data/sts_b/sts-dev.csv').decode('utf-8').strip()
#
#     def yielder():
#         for partition in (train, dev, test):
#             data = []
#             for idx,line in enumerate(partition.split('\n')):
#                 #print(tuple(line.split("\t")))
#                 #_, _, _,_, score, sentence1, sentence2 = tuple(line.split("\t"))
#                 line = tuple(line.split("\t"))
#                 data.append({
#                     'index': idx,
#                     'sentence_1': line[5],
#                     'sentence_2': line[6],
#                     'similarity': float(line[4])
#                 })
#             yield data
#
#     return tuple([dataset for dataset in yielder()])
#
#
# def load_cui_semantic_srs(dataset='mayo'):
#     assert dataset in ['mayo', 'mini-mayo', 'umn']
#
#     mapping = {'mayo': 'MayoSRS.gold.txt', 'mini-mayo': 'MiniMayoSRS.coders.txt', 'umn': 'UMNSRS_reduced_sim.gold.txt'}
#
#     mayo = resource_string('n2c2_2019', 'clinical_semantic_similarity/data/mayo_umn_srs/%s' % mapping[dataset]).decode('utf-8').strip()
#
#     data = []
#     for idx, line in enumerate(mayo.split('n')):
#         score, cui1, cui2 = line.split('<>')
#         data.append({'index': idx, 'cui_1': cui1, 'cui_2': cui2, 'score': float(score)})
#     return data



