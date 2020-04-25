import os, time, json, sys
#import from non-standard path module, do not remove.
csfp = os.path.abspath(os.path.join(os.path.dirname(__file__), 'experiment_replication'))
if csfp not in sys.path:
    sys.path.insert(0, csfp)
import torch
from multitasking_transformers.heads import SubwordClassificationHead
from multitasking_transformers.multitaskers.util import get_model_path
from transformers import BertConfig, BertForTokenClassification, BertModel
from tokenizers import BertWordPieceTokenizer, Encoding
from pprint import pprint
from experiment_replication.raw_datasets.language import get_language


text = """Admission Date:  [**2109-7-21**]       Discharge Date: [**2109-8-13**]

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
"""
batch_size = 25
#Defines the maximum number of subwords per sequence during chunking.
#Smaller values result in faster per instance computations, larger values are faster for longer chunks of text
max_sequence_length = 512
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
clinical_ner_tasks = ['i2b2_2010','n2c2_2018', 'i2b2_2012', 'i2b2_2014', 'quaero_2014']
model_path = get_model_path('mt_clinical_bert_8_tasks')
tokenizer = BertWordPieceTokenizer(os.path.join(model_path, 'vocab.txt'), lowercase=True, add_special_tokens=False)

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

encoding = tokenizer.encode(text)

def prepare_encoding(encoding: Encoding):
    """
    Given a arbitrarily long text (>512 subwords), chunks it into the BERT context window.
    :param encoding:
    :return:
    """
    def chunk_encoding(tensor : torch.Tensor):
        chunks = tensor.split(max_sequence_length)
        batch = torch.zeros(size=(len(chunks), max_sequence_length), dtype=torch.long)
        #we don't include special tokens during prediction (empirically, doesn't look like it hurts!)
        for index, chunk in enumerate(chunks):
            batch[index][0:len(chunk)] = torch.clone(chunk)
            # batch[index][0] = tokenizer.cls_token
            # batch[index][chunk.shape[0] + 1] = tokenizer.sep_token

        return batch, [len(chunk) for chunk in chunks]

    input_ids, num_tokens_in_instance = chunk_encoding(torch.tensor(encoding.ids, dtype=torch.long))
    attention_mask, _ = chunk_encoding(torch.tensor(encoding.attention_mask, dtype=torch.long))
    token_type_ids, _ = chunk_encoding(torch.tensor(encoding.type_ids, dtype=torch.long))


    return (input_ids, attention_mask, token_type_ids),\
           [encoding.offsets[i:i+max_sequence_length] for i in range(0, len(encoding.offsets) ,max_sequence_length)],\
           num_tokens_in_instance

(input_ids, attention_mask, token_type_ids), offsets, num_tokens_in_instance  = prepare_encoding(encoding)

token_representations = bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]


head_annotations = []
#Get head predictions
for task, head in heads.items():
    print(f"Predicting head: {head}")
    batch_subword_scores = head(token_representations)[0]

    batch_predicted_labels = batch_subword_scores.max(2)[1].tolist()
    # print(len(batch_predicted_labels))
    spans = []
    for idx, (predicted_labels, sequence_offsets) in enumerate(zip(batch_predicted_labels, offsets)):

        #print(predicted_labels)
        #merge multiple spans together into final annotation.
        predicted_labels = list(map(lambda x : x[2:] if '-' in x else x.replace('BERT_TOKEN', 'O'),
                               [head.config['labels'][label_key] for label_key in predicted_labels]))

        sequence_offsets = sequence_offsets
        predicted_labels = predicted_labels
        # print(sequence_offsets)
        # print(predicted_labels)
        # print(f"Num tokens in instance: {num_tokens_in_instance[idx]}")
        i = 0
        prev_label = 'O'

        #Group together tokens tagged with entities (post-processing heuristic)
        while i < num_tokens_in_instance[idx]:
            if predicted_labels[i] == 'O':
                i += 1
                continue
            label_start = i
            while i+1 != num_tokens_in_instance[idx] and predicted_labels[i] == predicted_labels[i+1]:
                i+=1
            label_end = i

            spans.append((sequence_offsets[label_start:label_end+1][0][0],
                          sequence_offsets[label_start:label_end+1][-1][1],
                          predicted_labels[i]))

            i+=1


    # print(task)
    # print(spans)
    annotation = {'entities':{f"T{i}": [span] for i, span in enumerate(spans)},
                  'entity_labels': list(map(lambda x : x[2:] if '-' in x else x, head.config['labels']))}
    head_annotations.append( tuple((str(encoding.original_str), annotation)))

visualize(head_annotations)
