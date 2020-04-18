from pkg_resources import resource_exists, resource_listdir, resource_string, resource_stream,resource_filename
from ..language import get_language
import random, json, os
import xml.etree.ElementTree as ET

MEDNLI_LABELS = sorted(['entailment', 'contradiction', 'neutral'])
MEDRQE_LABELS = sorted(['false', 'true'])
base_path = os.path.join(os.getcwd(), 'raw_datasets')


def load_mednli_2018(partition='train'):
    """

    :param partition:
    :return:
    """
    assert partition in ['train', 'test']
    language = get_language()

    language.remove_pipe('parser')
    language.remove_pipe('_set_sentence_parse_exceptions')

    annotation_dir = os.path.join(base_path, 'nli', 'med_nli')

    if not os.path.exists(os.path.join(annotation_dir, f'mli_{partition}_v1.jsonl')):
        raise FileNotFoundError('Cannot find %s data' % partition)


    file = open(os.path.join(annotation_dir, f'mli_{partition}_v1.jsonl'), 'r').read().strip()

    instances = []
    for idx, line in enumerate(file.split('\n')):
        instance = json.loads(line)
        label = MEDNLI_LABELS.index(instance['gold_label'])
        instances.append((idx, instance['sentence1'], instance['sentence2'], label))

    return instances, MEDNLI_LABELS




def load_medrqe_2016(partition='train'):
    """

    :param partition:
    :return:
    """
    assert partition in ['train', 'test']
    language = get_language()

    language.remove_pipe('parser')
    language.remove_pipe('_set_sentence_parse_exceptions')

    annotation_dir = os.path.join(base_path, 'nli', 'med_rqe')

    if not os.path.exists(os.path.join(annotation_dir, f'RQE_{partition}_pairs_AMIA2016.xml')):
        raise FileNotFoundError('Cannot find %s data' % partition)


    # if not resource_exists('clinical_data', f"nli/med_rqe/RQE_{partition}_pairs_AMIA2016.xml"):
    #     raise FileNotFoundError(f"Cannot find {partition} data")

    #file = resource_string('clinical_data', f"nli/med_rqe/RQE_{partition}_pairs_AMIA2016.xml").decode('utf-8').strip()
    file = open(os.path.join(annotation_dir, f'RQE_{partition}_pairs_AMIA2016.xml')).read().strip()
    root = ET.fromstring(file)

    pairs = root.findall("./pair")

    instances = []
    for instance in pairs:
        id = instance.attrib['pid']
        label = MEDRQE_LABELS.index(instance.attrib['value'])
        instances.append( (id, instance.findall('./chq')[0].text, instance.findall('./faq')[0].text, label) )

    return instances, MEDRQE_LABELS



