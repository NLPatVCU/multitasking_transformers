import os, time, json, sys
#import from non-standard path module, do not remove.
csfp = os.path.abspath(os.path.join(os.path.dirname(__file__), 'experiment_replication'))
if csfp not in sys.path:
    sys.path.insert(0, csfp)
import torch
from multitasking_transformers.heads import SubwordClassificationHead
from multitasking_transformers.multitaskers.util import get_model_path
from transformers import BertConfig, BertForTokenClassification, BertModel
from tokenizers import BertWordPieceTokenizer
from pprint import pprint
from experiment_replication.raw_datasets.language import get_language

def visualize(data_generator):
    from spacy import displacy
    from spacy.gold import biluo_tags_from_offsets
    from spacy.tokens import Span
    language = get_language()
    ner = language.create_pipe("ner")
    # language.add_pipe(ner, last=True)
    docs = []
    print(data_generator)
    for text, annotation in data_generator:
        doc = language(text)
        for label in annotation['entity_labels']:
            ner.add_label(label)

        spans = []
        for key in annotation['entities']:
            for start, stop, label in annotation['entities'][key]:
                span = doc.char_span(start, stop, label=label)
                if span is None:
                    continue
                spans.append(span)

        doc.ents = spans
        docs.append(doc)

    displacy.serve(docs, style="ent")

device='cpu'
clinical_ner_tasks = ['i2b2_2010', 'i2b2_2012', 'i2b2_2014', 'n2c2_2018', 'quaero_2014']
model_path = get_model_path('mt_clinical_bert_8_tasks')
tokenizer = BertWordPieceTokenizer(os.path.join(model_path, 'vocab.txt'), lowercase=True)

#initialize finetuned stacked transformer
bert = BertModel.from_pretrained(model_path)
bert.eval()

heads = {}
#initialize pre-trained heads
for task in clinical_ner_tasks:
    config = json.load(open(os.path.join(model_path, f"SubwordClassificationHead_{task}.json"), 'rb'))
    heads[task] = SubwordClassificationHead(task, labels=config['labels'],
                                            hidden_size=config['hidden_size'],
                                            hidden_dropout_prob=config['hidden_dropout_prob'])
    heads[task].from_pretrained(model_path)

encoding = tokenizer.encode("""Admission Date:  [**2109-7-21**]       Discharge Date: [**2109-8-13**]

Date of Birth:   [**2053-6-5**]       Sex:  F

Service:  [**Doctor Last Name 1181**] MEDICINE
HISTORY OF PRESENT ILLNESS:  This is a 56-year-old white
female with a history of right frontal craniotomy on [**2109-7-1**], for a dysembryoplastic angioneural epithelial lesion
with features of an oligodendroglioma who was started on
Dilantin postoperatively for seizure prophylaxis and was
subsequently developed eye discharge and was seen by an
optometrist who treated it with sulfate ophthalmic drops.
The patient then developed oral sores and rash in the chest
the night before admission which rapidly spread to the face,
trunk, and upper extremities within the last 24 hours.  The
patient was unable to eat secondary to mouth pain.  She had
fevers, weakness, and diarrhea.  There were no genital
the morning of [**7-20**].

PAST MEDICAL HISTORY:  1.  Hypercholesterolemia.  2.  Benign
right frontal cystic tumor status post right frontal
craniotomy on [**2109-7-1**].

ALLERGIES:  NO KNOWN DRUG ALLERGIES.

MEDICATIONS:  Lipitor, Tylenol with Codeine, Dilantin,
previously on Decadron q.i.d. tapered over one week and""")

input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)
attention_mask = torch.tensor([encoding.attention_mask], dtype=torch.long, device=device)
token_type_ids = torch.tensor([encoding.type_ids], dtype=torch.long, device=device)

#Encode token representations
token_representations = bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]


head_annotations = []
#Get head predictions
for task, head in heads.items():
    subword_scores = head(token_representations)[0]
    predicted_labels = subword_scores.max(2)[1].tolist()
    predicted_labels = list(map(lambda x : x[2:] if '-' in x else x.replace('BERT_TOKEN', 'O'),
                           [head.config['labels'][label_key] for label_key in predicted_labels[0]]))

    offsets = encoding.offsets[1:-1]
    predicted_labels = predicted_labels[1:-1]

    spans = []
    i = 0
    prev_label = 'O'
    while i < len(predicted_labels):
        if predicted_labels[i] == 'O':
            i += 1
            continue
        label_start = i
        while i+1 != len(predicted_labels) and predicted_labels[i] == predicted_labels[i+1]:
            i+=1
        label_end = i
        spans.append((offsets[label_start:label_end+1][0][0], offsets[label_start:label_end+1][-1][1], predicted_labels[i]))
        i+=1

    print(task)
    print(spans)
    annotation = {'entities':{f"T{i}": [span] for i, span in enumerate(spans)},
                  'entity_labels': list(map(lambda x : x[2:] if '-' in x else x, head.config['labels']))}
    head_annotations.append( tuple((str(encoding.original_str), annotation)))
    for start, end, label in spans:
        print((start, end, label), encoding.original_str[start:end])
    print(encoding.original_str)
    print()
visualize(head_annotations)
