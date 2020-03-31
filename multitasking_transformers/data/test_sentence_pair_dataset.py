import unittest, spacy, sys
from spacy.gold import biluo_tags_from_offsets

from . import SentencePairRegressionDataset, SentencePairClassificationDataset
from transformers import BertTokenizer, AlbertTokenizer

CLINICAL_DATA_PRESENT = 'clinical_data' in sys.modules

class TestCreateSentencePairRegressionDataset(unittest.TestCase):

    # @unittest.skipUnless(CLINICAL_DATA_PRESENT, "Clinical data package not installed (a private preprocessing  package)")
    # def test_n2c2_2019_sts(self):
    #     from clinical_data.similarity import load_n2c2_2019_train_dev, load_n2c2_2019
    #     n2c2_test = list(load_n2c2_2019(partition='test'))
    #     bert_inputs, bert_labels \
    #         = SentencePairRegressionDataset.create_sentence_pair_dataset(n2c2_test, BertTokenizer.from_pretrained('bert-base-uncased'))
    #     bert_input_ids, bert_token_type_ids, bert_attention_masks = bert_inputs
    #     print(bert_labels)

    @unittest.skipUnless(CLINICAL_DATA_PRESENT, "Clinical data package not installed (a private preprocessing package)")
    def test_medreq_2016_test(self):
        from clinical_data.entailment import load_medrqe_2016
        medreq_2016 = list(load_medrqe_2016(partition='train'))
        bert_inputs, bert_labels, class_labels \
            = SentencePairClassificationDataset.create_sentence_pair_dataset(medreq_2016, BertTokenizer.from_pretrained('bert-base-uncased'))
        bert_input_ids, bert_token_type_ids, bert_attention_masks = bert_inputs
        print(bert_labels)
        print(class_labels)


    @unittest.skipIf(CLINICAL_DATA_PRESENT, "Clinical data package is installed (a private preprocessing package)")
    def test_ner(self):
        n2c2_train = list(load_n2c2_2019(partition='train'))
        bert_input_ids, bert_attention_masks, bert_sequence_lengths, bert_labels, biluo_ordered_labels \
            = SentencePairRegressionDataset.create_sentence_pair_dataset(n2c2_train, BertTokenizer.from_pretrained('bert-base-uncased'))
        self.assertEqual(biluo_ordered_labels.index['BERT_TOKEN'], bert_labels, bert_labels[0][0])