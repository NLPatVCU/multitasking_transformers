import unittest, spacy, sys
from spacy.gold import biluo_tags_from_offsets

from . import NERDataset
from transformers import BertTokenizer, AlbertTokenizer

CLINICAL_DATA_PRESENT = 'clinical_data' in sys.modules

class TestCreateNERDataset(unittest.TestCase):

    @unittest.skipUnless(CLINICAL_DATA_PRESENT, "Clinical data package not installed (a private preprocessing package)")
    def test_n2c2_2018_ner(self):
        from clinical_data.ner import load_n2c2_2018_train_dev, load_n2c2_2018
        n2c2_train = list(load_n2c2_2018(partition='train'))
        bert_inputs, bert_sequence_lengths, bert_labels, spacy_labels, alignments, biluo_ordered_labels, class_counts \
            = NERDataset.create_ner_dataset(n2c2_train, BertTokenizer.from_pretrained('bert-base-uncased'))
        bert_input_ids, bert_token_masks, bert_attention_masks = bert_inputs
        print(bert_labels[1])
        print(alignments)
        print(class_counts)
        self.assertEqual(biluo_ordered_labels.index('BERT_TOKEN'), bert_labels[0][0])




    # @unittest.skipIf(CLINICAL_DATA_PRESENT, "Clinical data package is installed (a private preprocessing package)")
    # def test_ner(self):
    #     n2c2_train = list(load_n2c2_2018(partition='train'))
    #     bert_input_ids, bert_attention_masks, bert_sequence_lengths, bert_labels, biluo_ordered_labels \
    #         = create_ner_dataset(n2c2_train, BertTokenizer.from_pretrained('bert-base-uncased'))
    #     self.assertEqual(biluo_ordered_labels.index['BERT_TOKEN'], bert_labels, bert_labels[0][0])