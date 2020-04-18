import os, time, json
import torch
from multitasking_transformers.heads import SubwordClassificationHead
from multitasking_transformers.multitaskers.util import get_model_path
from transformers import BertConfig, BertForTokenClassification, BertModel
from tokenizers import BertWordPieceTokenizer
from pprint import pprint

device='cpu'
clinical_ner_tasks = ['i2b2_2010', 'i2b2_2012', 'i2b2_2014', 'n2c2_2018', 'quaero_2014']
model_path = get_model_path('mt_clinical_bert_8_tasks')
tokenizer = BertWordPieceTokenizer(os.path.join(model_path, 'vocab.txt'), lowercase=True)
# tokenizer.enable_padding(direction='right',
#                                       pad_id=0,
#                                       max_length=512)

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

#Get head predictions
for task, head in heads.items():
    subword_scores = head(token_representations)[0]
    predicted_labels = subword_scores.max(2)[1].tolist()
    predicted_labels = [head.config['labels'][label_key] for label_key in predicted_labels[0]]
    print(task)
    print(encoding.offsets)
    print(encoding.tokens)
    print(predicted_labels)
    print()

