import os, torch, sys
from multitasking_transformers.data import NERDataset, \
    SentencePairRegressionDataset, SentencePairClassificationDataset
from transformers import BertTokenizer

#import from non-standard path module, do not remove.
csfp = os.path.abspath(os.path.join(os.path.dirname(__file__), "raw_datasets"))
if csfp not in sys.path:
    sys.path.insert(0, csfp)

#bert_weight_directory = '/home/rodriguezne2/models/biobert_large'

#bert_weight_directory = '/home/aymulyar/models/biobert_pretrain_output_all_notes_150000'
bert_weight_directory = '/media/andriy/Samsung_T5/research/clinical/models/bert/biobert_pretrain_output_all_notes_150000'

N2C2_2018_NER_TRAIN_PATH = os.path.join(os.getcwd(), 'data', 'n2c2_2018', 'ner', 'train')
N2C2_2018_NER_TEST_PATH = os.path.join(os.getcwd(), 'data', 'n2c2_2018','ner', 'test')

I2B2_2010_NER_TRAIN_PATH = os.path.join(os.getcwd(), 'data', 'i2b2_2010', 'ner', 'train')
I2B2_2010_NER_TEST_PATH = os.path.join(os.getcwd(), 'data', 'i2b2_2010','ner', 'test')

END_2017_NER_TRAIN_PATH = os.path.join(os.getcwd(), 'data', 'end_2017', 'ner', 'train')
END_2017_NER_TEST_PATH = os.path.join(os.getcwd(), 'data', 'end_2017','ner', 'test')

TAC_2018_NER_TRAIN_PATH = os.path.join(os.getcwd(), 'data', 'tac_2018', 'ner', 'train')
TAC_2018_NER_TEST_PATH = os.path.join(os.getcwd(), 'data', 'tac_2018', 'ner', 'test')

I2B2_2014_NER_TRAIN_PATH = os.path.join(os.getcwd(), 'data', 'i2b2_2014', 'ner', 'train')
I2B2_2014_NER_TEST_PATH = os.path.join(os.getcwd(), 'data', 'i2b2_2014', 'ner', 'test')

I2B2_2012_NER_TRAIN_PATH = os.path.join(os.getcwd(), 'data', 'i2b2_2012', 'ner', 'train')
I2B2_2012_NER_TEST_PATH = os.path.join(os.getcwd(), 'data', 'i2b2_2012', 'ner', 'test')

QUAERO_2014_NER_TRAIN_PATH = os.path.join(os.getcwd(), 'data', 'quaero_2014', 'ner', 'train')
QUAERO_2014_NER_TEST_PATH = os.path.join(os.getcwd(), 'data', 'quaero_2014', 'ner', 'test')

N2C2_2019_SIMILARITY_TRAIN_PATH = os.path.join(os.getcwd(), 'data', 'n2c2_2019', 'similarity', 'train')
N2C2_2019_SIMILARITY_TEST_PATH = os.path.join(os.getcwd(), 'data', 'n2c2_2019', 'similarity', 'test')

MEDRQE_2016_ENTAILMENT_TRAIN_PATH = os.path.join(os.getcwd(), 'data', 'medrqe_2016', 'nli', 'train')
MEDRQE_2016_ENTAILMENT_TEST_PATH = os.path.join(os.getcwd(), 'data', 'medrqe_2016', 'nli', 'test')

MEDNLI_2018_NLI_TRAIN_PATH = os.path.join(os.getcwd(), 'data', 'mednli_2018', 'nli', 'train')
MEDNLI_2018_NLI_TEST_PATH = os.path.join(os.getcwd(), 'data', 'mednli_2018', 'nli', 'test')





#START N2C2 2018 NER
from raw_datasets.ner import load_n2c2_2018

save_dir = N2C2_2018_NER_TRAIN_PATH
if not os.path.isdir(save_dir):
    print(f"Saving N2C2 2018 NER Train: {save_dir}")
    os.makedirs(save_dir)
    inputs = NERDataset.create_ner_dataset(list(load_n2c2_2018(partition='train')),
                                                       BertTokenizer.from_pretrained(bert_weight_directory),
                                                       save_directory=save_dir)
save_dir = N2C2_2018_NER_TEST_PATH
if not os.path.isdir(save_dir):
    print(f"Saving N2C2 2018 NER Test: {save_dir}")
    os.makedirs(save_dir)
    inputs = NERDataset.create_ner_dataset(list(load_n2c2_2018(partition='test')),
                                                       BertTokenizer.from_pretrained(bert_weight_directory),
                                                       save_directory=save_dir)

#END N2C2 2018 NER

# #START TAC 2018 NER
# from clinical_data.ner import load_tac_2018
#
# save_dir = TAC_2018_NER_TRAIN_PATH
# if not os.path.isdir(save_dir):
#     print(f"Saving TAC 2018 NER Train: {save_dir}")
#     os.makedirs(save_dir)
#     inputs = NERDataset.create_ner_dataset(list(load_tac_2018(partition='train')),
#                                                        BertTokenizer.from_pretrained(bert_weight_directory),
#                                                        save_directory=save_dir)
#
# save_dir = TAC_2018_NER_TEST_PATH
# if not os.path.isdir(save_dir):
#     print(f"Saving TAC 2018 NER Test: {save_dir}")
#     os.makedirs(save_dir)
#     inputs = NERDataset.create_ner_dataset(list(load_tac_2018(partition='test')),
#                                                        BertTokenizer.from_pretrained(bert_weight_directory),
#                                                        save_directory=save_dir)
#
# #END TAC 2018 NER
#
#START I2B2 2010 NER
from raw_datasets.ner import load_i2b2_2010

save_dir = I2B2_2010_NER_TRAIN_PATH
if not os.path.isdir(save_dir):
    print(f"Saving I2B2 2010 Train: {save_dir}")
    os.makedirs(save_dir)
    inputs = NERDataset.create_ner_dataset(list(load_i2b2_2010(partition='train')),
                                                       BertTokenizer.from_pretrained(bert_weight_directory),
                                                       save_directory=save_dir)


save_dir = I2B2_2010_NER_TEST_PATH
if not os.path.isdir(save_dir):
    print(f"Saving I2B2 2010 NER Test: {save_dir}")
    os.makedirs(save_dir)
    inputs = NERDataset.create_ner_dataset(list(load_i2b2_2010(partition='test')),
                                                       BertTokenizer.from_pretrained(bert_weight_directory),
                                                       save_directory=save_dir)

#END I2B2 2010 NER
#
# #START END 2017 NER
# from clinical_data.ner import load_end_2017
#
# save_dir = END_2017_NER_TRAIN_PATH
# if not os.path.isdir(save_dir):
#     print(f"Saving END 2017 Train: {save_dir}")
#     os.makedirs(save_dir)
#     inputs = NERDataset.create_ner_dataset(list(load_end_2017(partition='train')),
#                                            BertTokenizer.from_pretrained(bert_weight_directory),
#                                            save_directory=save_dir)
#
# save_dir = END_2017_NER_TEST_PATH
# if not os.path.isdir(save_dir):
#     print(f"Saving END 2017 NER Test: {save_dir}")
#     os.makedirs(save_dir)
#     inputs = NERDataset.create_ner_dataset(list(load_end_2017(partition='test')),
#                                            BertTokenizer.from_pretrained(bert_weight_directory),
#                                            save_directory=save_dir)
# #END END 2017 NER
#
# #START I2b2 2014 NER
# from clinical_data.ner import load_i2b2_2014
#
# save_dir = I2B2_2014_NER_TRAIN_PATH
# if not os.path.isdir(save_dir):
#     print(f"Saving I2B2 2014 Train: {save_dir}")
#     os.makedirs(save_dir)
#     inputs = NERDataset.create_ner_dataset(list(load_i2b2_2014(partition='train')),
#                                            BertTokenizer.from_pretrained(bert_weight_directory),
#                                            save_directory=save_dir)
#
# save_dir = I2B2_2014_NER_TEST_PATH
# if not os.path.isdir(save_dir):
#     print(f"Saving I2B2 2014 NER Test: {save_dir}")
#     os.makedirs(save_dir)
#     inputs = NERDataset.create_ner_dataset(list(load_i2b2_2014(partition='test')),
#                                            BertTokenizer.from_pretrained(bert_weight_directory),
#                                            save_directory=save_dir)
# #END I2b2 2014 NER
#
# #START I2b2 2012 NER
# from clinical_data.ner import load_i2b2_2012
#
# save_dir = I2B2_2012_NER_TRAIN_PATH
# if not os.path.isdir(save_dir):
#     print(f"Saving I2B2 2012 Train: {save_dir}")
#     os.makedirs(save_dir)
#     inputs = NERDataset.create_ner_dataset(list(load_i2b2_2012(partition='train')),
#                                            BertTokenizer.from_pretrained(bert_weight_directory),
#                                            save_directory=save_dir)
#
# save_dir = I2B2_2012_NER_TEST_PATH
# if not os.path.isdir(save_dir):
#     print(f"Saving I2B2 2012 NER Test: {save_dir}")
#     os.makedirs(save_dir)
#     inputs = NERDataset.create_ner_dataset(list(load_i2b2_2012(partition='test')),
#                                            BertTokenizer.from_pretrained(bert_weight_directory),
#                                            save_directory=save_dir)
# #END I2b2 2012 NER
#
#
# #START FRENCH  NER
# from clinical_data.ner import load_quaero_frenchmed
#
# save_dir = QUAERO_2014_NER_TRAIN_PATH
# if not os.path.isdir(save_dir):
#     print(f"Saving QUAERO 2014 Train: {save_dir}")
#     os.makedirs(save_dir)
#     inputs = NERDataset.create_ner_dataset(list(load_quaero_frenchmed(partition='train')),
#                                            BertTokenizer.from_pretrained(bert_weight_directory),
#                                            save_directory=save_dir)
#
# save_dir = QUAERO_2014_NER_TEST_PATH
# if not os.path.isdir(save_dir):
#     print(f"Saving QUAERO 2014 NER Test: {save_dir}")
#     os.makedirs(save_dir)
#     inputs = NERDataset.create_ner_dataset(list(load_quaero_frenchmed(partition='test')),
#                                            BertTokenizer.from_pretrained(bert_weight_directory),
#                                            save_directory=save_dir)
# #END I2b2 2012 NER
#
#
# #START N2C2 2019 Similarity
# from clinical_data.similarity import load_n2c2_2019
#
# save_dir = N2C2_2019_SIMILARITY_TRAIN_PATH
# if not os.path.isdir(save_dir):
#     print(f"Saving N2C2 2019 Similarity Train Data: {save_dir}")
#     os.makedirs(save_dir)
#     bert_inputs, labels = SentencePairRegressionDataset.create_sentence_pair_dataset(list(load_n2c2_2019(partition='train')),
#                                                        BertTokenizer.from_pretrained(bert_weight_directory),
#                                                        save_directory=save_dir)
# save_dir = N2C2_2019_SIMILARITY_TEST_PATH
# if not os.path.isdir(save_dir):
#     print(f"Saving N2C2 2019 Similarity Test Data: {save_dir}")
#     os.makedirs(save_dir)
#     bert_inputs, labels = SentencePairRegressionDataset.create_sentence_pair_dataset(list(load_n2c2_2019(partition='test')),
#                                                        BertTokenizer.from_pretrained(bert_weight_directory),
#                                                        save_directory=save_dir)
#
# #END N2C2 2019 Similarity
#
# #START MEDRQE 2016 Entailment
# from clinical_data.nli import load_medrqe_2016
#
# save_dir = MEDRQE_2016_ENTAILMENT_TRAIN_PATH
# if not os.path.isdir(save_dir):
#     print(f"Saving MedRQE 2016 Entailment Train Data: {save_dir}")
#     os.makedirs(save_dir)
#     bert_inputs, labels, class_labels = SentencePairClassificationDataset.create_sentence_pair_dataset(list(load_medrqe_2016(partition='train')),
#                                                        BertTokenizer.from_pretrained(bert_weight_directory),
#                                                        save_directory=save_dir)
# save_dir = MEDRQE_2016_ENTAILMENT_TEST_PATH
# if not os.path.isdir(save_dir):
#     print(f"Saving MedRQE 2016 Entailment Test Data: {save_dir}")
#     os.makedirs(save_dir)
#     bert_inputs, labels, class_labels = SentencePairClassificationDataset.create_sentence_pair_dataset(list(load_medrqe_2016(partition='test')),
#                                                        BertTokenizer.from_pretrained(bert_weight_directory),
#                                                        save_directory=save_dir)
# #END MEDRQE 2016 Entailment
#
# #START MEDNLI 2018 NLI
# from clinical_data.nli import load_medrqe_2016
# from clinical_data.nli import load_mednli_2018
#
# save_dir = MEDNLI_2018_NLI_TRAIN_PATH
# if not os.path.isdir(save_dir):
#     print(f"Saving MedNLI 2018 NLI Train Data: {save_dir}")
#     os.makedirs(save_dir)
#     bert_inputs, labels, class_labels = SentencePairClassificationDataset.create_sentence_pair_dataset(list(load_mednli_2018(partition='train')),
#                                                        BertTokenizer.from_pretrained(bert_weight_directory),
#                                                        save_directory=save_dir)
# save_dir = MEDNLI_2018_NLI_TEST_PATH
# if not os.path.isdir(save_dir):
#     print(f"Saving MedNLI 2018 NLI Test Data: {save_dir}")
#     os.makedirs(save_dir)
#     bert_inputs, labels, class_labels = SentencePairClassificationDataset.create_sentence_pair_dataset(list(load_mednli_2018(partition='test')),
#                                                        BertTokenizer.from_pretrained(bert_weight_directory),
#                                                        save_directory=save_dir)
# #END MEDNLI 2018 NLI
#
#
#
# """
# Biomedical Datasets
# """
#
# all_biomedical_datasets = ["AnatEM", "BC5CDR-disease", "BioNLP11ID-chem", "BioNLP13CG-cc", "BioNLP13CG",
#                 "BioNLP13PC-chem", "CRAFT-cell", "CRAFT-species", "NCBI-disease", "BC2GM", "BC5CDR",
#                 "BioNLP11ID-ggp", "BioNLP13CG-cell", "BioNLP13CG-species", "BioNLP13PC-ggp",
#                 "CRAFT-chem", "Ex-PTM", "BC4CHEMD", "BioNLP09", "BioNLP11ID", "BioNLP13CG-chem",
#                 "BioNLP13GE", "BioNLP13PC", "CRAFT-ggp", "JNLPBA", "BC5CDR-chem", "BioNLP11EPI",
#                 "BioNLP11ID-species", "BioNLP13CG-ggp", "BioNLP13PC-cc", "CRAFT-cc", "CRAFT", "linnaeus"]
#
# #all_biomedical_datasets = ["NCBI-disease"]
# # BC2GM_NER_TRAIN_PATH = os.path.join(os.getcwd(), 'data', 'ner', 'biomedical', 'BC2GM', 'train')
# # BC2GM_NER_TEST_PATH = os.path.join(os.getcwd(),  'data', 'ner', 'biomedical', 'BC2GM', 'test')
# #
# # BC4CHEMD_NER_TRAIN_PATH = os.path.join(os.getcwd(), 'data', 'ner', 'biomedical', 'BC4CHEMD', 'train')
# # BC4CHEMD_NER_TEST_PATH = os.path.join(os.getcwd(), 'data', 'ner', 'biomedical', 'BC4CHEMD', 'test')
#
# from clinical_data.ner import load_biomedical_dataset
#
# for dataset in all_biomedical_datasets:
#     for partition in ['train', 'test']:
#         save_dir = os.path.join(os.getcwd(), 'data','biomedical', 'ner', dataset, partition)
#         if not os.path.isdir(save_dir):
#             print(f"Saving {dataset} {partition.capitalize()}: {save_dir}")
#             os.makedirs(save_dir)
#
#             try:
#                 inputs = NERDataset.create_ner_dataset(load_biomedical_dataset(dataset, partition),
#                                                        BertTokenizer.from_pretrained(bert_weight_directory),
#                                                        save_directory=save_dir,
#                                                        conll_format=True
#                                                        )
#             except BaseException as e:
#                 import shutil
#                 shutil.rmtree(os.path.join(os.getcwd(), 'data','biomedical', 'ner', dataset))
#                 raise e
#
# from biomedical_datasets.huner import load_huner_dataset
#
# huner_datasets = {
#     'cellline': ['cellfinder', 'cll', 'gellus', 'jnlpba'],
#     'chemical': ['biosemantics', 'cdr', 'cemp', 'chebi', 'chemdner', 'scai_chemicals'],
#     'disease': ['biosemantics', 'cdr', 'miRNA', 'ncbi', 'scai_diseases'],  # TODO add arizona
#     'gene': ['bc2gm', 'bioinfer', 'cellfinder', 'deca', 'fsu', 'gpro', 'iepa', 'jnlpba', 'miRNA',
#              'osiris', 'variome'],
#     'species': ['cellfinder', 'linneaus', 'miRNA', 's800', 'variome']
# }
#
# for type in huner_datasets:
#     for dataset in huner_datasets[type]:
#             for partition in ['train', 'test']:
#                 if dataset in ['cellfinder', 'biosemantics']:
#                     continue
#                 save_dir = os.path.join(os.getcwd(), 'data','biomedical', 'huner', dataset, partition)
#                 if not os.path.isdir(save_dir):
#                     print(f"Saving {dataset} {partition.capitalize()}: {save_dir}")
#                     os.makedirs(save_dir)
#                     input_document_sequence_batch_size = 200
#                     if dataset == 'cellfinder' and partition == 'train':
#                         input_document_sequence_batch_size = 15
#                     if dataset == 'cellfinder' and partition == 'test':
#                         input_document_sequence_batch_size = 28
#                     if dataset == 'gellus' and partition == 'train':
#                         input_document_sequence_batch_size = 70
#                     if dataset == 'cemp':
#                         input_document_sequence_batch_size = 50
#                     if dataset == 'chebi':
#                         input_document_sequence_batch_size = 30
#                     if dataset == 'chemdner':
#                         input_document_sequence_batch_size = 20
#                     if dataset == 'fsu':
#                         input_document_sequence_batch_size = 150
#                     if dataset == 'gpro':
#                         input_document_sequence_batch_size = 50
#                     if dataset == 'variome':
#                         input_document_sequence_batch_size = 20
#                     if dataset == 'linneaus':
#                         input_document_sequence_batch_size = 100
#
#
#                     try:
#                         inputs = NERDataset.create_ner_dataset(load_huner_dataset(dataset, type, partition=partition),
#                                                                BertTokenizer.from_pretrained(bert_weight_directory),
#                                                                save_directory=save_dir,
#                                                                input_document_sequence_batch_size=input_document_sequence_batch_size,
#                                                                conll_format=True
#                                                                )
#                     except BaseException as e:
#                         import shutil
#                         shutil.rmtree(os.path.join(os.getcwd(), 'data','biomedical', 'huner', dataset))
#                         raise e
