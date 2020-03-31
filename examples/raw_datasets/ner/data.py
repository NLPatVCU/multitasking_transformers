from pkg_resources import resource_exists, resource_listdir, resource_string, resource_stream, resource_filename, os
from ..language import get_language
import random, spacy
import xml.etree.ElementTree as ET

N2C2_2018_NER_LABELS = sorted(['Drug', 'Frequency', 'Reason', 'ADE', 'Dosage', 'Duration', 'Form', 'Route', 'Strength'])
N2C2_2018_RELATION_LABELS = sorted(['Frequency-Drug', 'Strength-Drug', 'Route-Drug', 'Dosage-Drug', 'ADE-Drug',
                                    'Reason-Drug', 'Duration-Drug', 'Form-Drug'])

I2B2_2010_NER_LABELS = sorted(['problem', 'treatment', 'test'])
I2B2_2010_RELATION_LABELS = sorted([])

TAC_2018_NER_LABELS = sorted(['CellLine', 'Dose', 'DoseDuration', 'DoseDurationUnits', 'DoseFrequency', 'DoseRoute',
                               'DoseUnits', 'Endpoint', 'EndpointUnitOfMeasure', 'GroupName', 'GroupSize', 'SampleSize',
                               'Sex', 'Species', 'Strain', 'TestArticle', 'TestArticlePurity',
                               'TestArticleVerification', 'TimeAtDose', 'TimeAtFirstDose', 'TimeAtLastDose',
                               'TimeEndpointAssessed', 'TimeUnits', 'Vehicle'])

END_2017_NER_LABELS = sorted(['activeingredient', 'adversereaction', 'auc', 'chemoclass', 'clearance', 'cmax',
                              'co-administereddrug', 'company', 'corecomposition', 'dose', 'duration',
                              'eliminationhalflife', 'fdaapprovaldate', 'frequency', 'inactiveingredient',
                              'indication', 'molecularweight', 'nanoparticle', 'particlediameter', 'plasmahalflife',
                              'pre-existingdisease', 'routeofadministration', 'surfacecharge', 'surfacecoating',
                              'tmax', 'tradename', 'u.s.patent', 'volumeofdistribution'])
END_2017_RELATION_LABELS = sorted([])

I2B2_2014_NER_LABELS = sorted(['DEVICE', 'MEDICALRECORD', 'ORGANIZATION', 'ZIP', 'DATE', 'PROFESSION', 'HOSPITAL',
                               'EMAIL', 'CITY', 'LOCATION-OTHER', 'STATE', 'COUNTRY', 'PHONE', 'USERNAME', 'AGE',
                               'FAX', 'PATIENT', 'DOCTOR', 'IDNUM', 'HEALTHPLAN', 'STREET', 'BIOID', 'URL'])
I2B2_2014_RELATION_LABELS = sorted([])

I2B2_2012_NER_LABELS = sorted(['OCCURRENCE', 'EVIDENTIAL', 'TREATMENT', 'TEST', 'CLINICAL_DEPT', 'PROBLEM'])
I2B2_2012_RELATION_LABELS = sorted([])

QUAERO_FRENCHMED_2014_NER_LABELS = sorted(['DEVI', 'CHEM', 'LIVB', 'PHEN', 'ANAT', 'PROC', 'PHYS', 'GEOG', 'OBJC', 'DISO'])
QUAERO_FRENCHMED_2014_RELATION_LABELS = sorted([])

base_path = os.path.join(os.getcwd(), 'raw_datasets')

def load_n2c2_2018(partition='train'):
    assert partition in ['train', 'test']
    language = get_language()

    if partition == 'train':
        annotation_dir = os.path.join(base_path, 'ner', 'n2c2_2018', 'training_20180910', 'training_20180910')
    else:
        annotation_dir = os.path.join(base_path, 'ner', 'n2c2_2018', 'gold-standard-test-data', 'test')


    file_list = os.listdir(annotation_dir)
    file_ids = sorted(set(file[:-4] for file in file_list)) #remove ending to get unique files

    annotations = []
    raw_text_files = []

    unique_entity_labels = set()
    unique_relation_labels = set()
    for id in file_ids:
        raw_text_files.append(open(os.path.join(annotation_dir, '%s.txt' % id), 'r').read().strip())
        annotation_file = open(os.path.join(annotation_dir, '%s.ann' % id), 'r').read().strip()

        annotation = {
            'entities': {

            },
            'relations': []
        }
        for line in annotation_file.split('\n'):

            line = line.split('\t')
            if line[0][0] == "T":
                annotation['entities'][line[0]] = []
                label = line[1].split(' ')[0]
                unique_entity_labels.add(label)
                spans = [int(index) for x in line[1].split(' ')[1:] for index in x.split(';')]
                if len(spans) == 4 and (spans[1] == spans[2] or spans[1]+1 == spans[2]):
                    annotation['entities'][line[0]].append(tuple((spans[0], spans[3], label)))
                else:
                    for idx in range(0, len(spans), 2):
                        annotation['entities'][line[0]].append(tuple((spans[idx], spans[idx+1], label)))
                #print(annotation['entities'][line[0]])
            if line[0][0] == "R":
                relation, source, target = line[1].split(' ')[0], line[1].split(' ')[1].split(':')[1], line[1].split(' ')[2].split(':')[1]
                unique_relation_labels.add(relation)
                annotation['relations'].append(tuple((source, target, relation)))
                #print(annotation['relations'][-1])

        annotations.append(annotation)
    raw_text_files = list(language.pipe(raw_text_files, batch_size=50))

    #Assures that the annotated dataset labels align with the tokenization.
    for idx, (id, doc, annotation) in enumerate(zip(file_ids, raw_text_files, annotations)):
        #doc = language(raw_text)

        fixed_annotation = {
            'entities':{

            },
            'relations': annotation['relations']
        }

        for idx, key in enumerate(annotation['entities']):
            instance = annotation['entities'][key]
            fixed_annotation['entities'][key] = []
            for span in instance:
                char_span = doc.char_span(span[0], span[1])
                if char_span is None:
                    if doc.char_span(span[0]-1, span[1]) is not None:
                        char_span = doc.char_span(span[0]-1, span[1])
                    elif doc.char_span(span[0]-1, span[1]-1) is not None:
                        char_span = doc.char_span(span[0]-1, span[1]-1)
                    elif doc.char_span(span[0], span[1]+1) is not None:
                        char_span = doc.char_span(span[0], span[1]+1)
                    elif doc.char_span(span[0]+5, span[1]+5) is not None:
                        char_span = doc.char_span(span[0]+5, span[1]+5)



                if char_span is None:
                    # for token in doc:
                    #     print(token)
                    print(str(doc)[span[0]- 20:span[1] + 20])
                    print(id, span, str(doc)[span[0]:span[1]])
                    continue
                    # raise RuntimeError(
                    #     'Could not load mention span from %s as it does not align with tokenization. Add \'%s\' to tokenization exceptions.'
                    #     % (id, str(doc)[int(span[0]):int(span[1])]))

                #checks if this span overlaps with any other
                overlapping = False
                for idx2, key2 in enumerate(annotation['entities']):
                    for s2 in annotation['entities'][key2]:
                        if char_span.start_char <= s2[1] and s2[0] <= char_span.end_char:
                            if idx == idx2:
                                pass
                            else:
                                # overlapping span, ignore the occurence.
                                overlapping = True
                if not overlapping:
                    fixed_annotation['entities'][key].append(tuple((char_span.start_char, char_span.end_char, span[2])))

        fixed_annotation['entity_labels'] = N2C2_2018_NER_LABELS
        fixed_annotation['relation_labels'] = N2C2_2018_RELATION_LABELS

        doc._.id = id
        yield doc, fixed_annotation