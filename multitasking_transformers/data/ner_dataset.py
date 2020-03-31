from builtins import print

import torch, os, pickle, warnings
from typing import Tuple, List, Union
from spacy.gold import biluo_tags_from_offsets
from transformers import BertTokenizer, AlbertTokenizer


"""
Given a dictionary of NER labels, creates a pytorch Dataset of BERT-style input tensors.
"""

#TODO How will we handle a training run where we combine both train and dev?
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_directory: str, ):
        self.dataset_directory = dataset_directory
        self.entity_labels = torch.load(os.path.join(self.dataset_directory, 'entity_names.pl'))

        self.bert_input_ids = torch.load(os.path.join(self.dataset_directory, f"bert_input.pt"))
        self.bert_attention_masks = torch.load(os.path.join(self.dataset_directory, f"bert_attention_mask.pt"))
        self.bert_sequence_lengths = torch.load(os.path.join(self.dataset_directory, f"bert_sequence_length.pt"))
        self.bert_labels = torch.load(os.path.join(self.dataset_directory, f"bert_labels.pt"))
        self.spacy_labels = torch.load(os.path.join(self.dataset_directory, f"spacy_labels.pt"))
        self.bert_alignment = torch.load(os.path.join(self.dataset_directory, f"subword_to_spacy_alignment.pt"))
        self.loss_weights = torch.load(os.path.join(self.dataset_directory, f"loss_weights.pt"))

    def __len__(self):
        'Denotes the total number of samples'
        return self.bert_input_ids.shape[0]

    def __getitem__(self, index: int):
        # Load data and get label, pad with [] (cannot use None) to align with LM inputs (we don't need token types for subword classification).
        return self.bert_input_ids[index], [], self.bert_attention_masks[index],\
               self.bert_sequence_lengths[index], self.bert_labels[index], self.spacy_labels[index],\
               self.bert_alignment[index], self.loss_weights



    @staticmethod
    def create_ner_dataset(data,
                       tokenizer: Union[BertTokenizer, AlbertTokenizer],
                       save_directory=None,
                       max_sequence_length=512,
                       conll_format=False,) -> Tuple[Tuple[torch.LongTensor, torch.LongTensor,
                                                                        torch.LongTensor], torch.LongTensor, List[str]]:
        """
            Given a list of tuples of document with span level annotations, saves bert input and labels onto disk.
            This method is designed as a pre-processing step to be utilized with a pytorch Dataset and Dataloader.

            :param data:  a list of tuples relating a document to its set of annotations.
            :param tokenizer: the transformers tokenizer to utilize.
            :param conll_format: set true if data is a tuple containing parallel arrays of tokens and labels and list of entities
            :return the location the dataset was saved
            """
        # TODO insure sequences are not split on token boundaries.
        if conll_format:
            assert len(data) == 3, "Should contain list of tokens, tags and list of bilou entities"
            token_sequences = []
            label_sequences = []
            token_sequence = data[0]
            label_sequence = data[1]
            token_sequences.append(token_sequence)
            label_sequences.append(label_sequence)
            biluo_ordered_labels = sorted([entity_label for entity_label in data[2] if entity_label != 'O'] + ['O', 'BERT_TOKEN'])
            tags_from_annotations = biluo_ordered_labels

        else: #custom spacy format
            assert len(data) > 1
            assert 'entities' in data[0][1]
            assert 'entity_labels' in data[0][1]
            token_sequences = []
            label_sequences = []

            entity_labels = set()
            tags_from_annotations = set()

            for doc, annotations in data:
                for label in annotations['entity_labels']:
                    entity_labels.add(label)
                offsets = [offset for annotation in annotations['entities'].values() for offset in annotation]
                tags = biluo_tags_from_offsets(doc, offsets)
                for tag in tags:
                    tags_from_annotations.add(tag)

                token_sequences.append([x for x in doc])
                label_sequences.append(tags)

            biluo_ordered_labels = sorted([f"{prefix}-{entity_label}" for prefix in ['B', 'I', 'L', 'U']
                                           for entity_label in entity_labels if entity_label != 'O'] + ['O', 'BERT_TOKEN'])
            tags_from_annotations = sorted(list(tags_from_annotations) + ['BERT_TOKEN'])


        # convert each string label to a unique id with respect to the biluo_labels of the tokenization
        encoded_label_sequences = [[biluo_ordered_labels.index(label) for label in seq] for seq in label_sequences]

        class_counts = [0] * len(biluo_ordered_labels)

        for seq in encoded_label_sequences:
            for id in seq:
                class_counts[id]+=1

        class_counts = torch.FloatTensor(class_counts)
        loss_weights = torch.abs( 1- (class_counts / len([x for x in seq for seq in encoded_label_sequences])) )
        # Assert that all labels appear in the annotations. This could occur if annotation processing could not align
        # all annotations into the defined spacy tokenization.
        if biluo_ordered_labels != tags_from_annotations:
            warnings.warn("Processed dataset does not contain instances from all labels when converted to BILOU scheme.")

        # Now generate bert input tensors
        all_bert_sequence_alignments, all_bert_subword_sequences, all_bert_label_sequences, original_tokenization_labels = [], [], [], []


        for sequence, labels in zip(token_sequences, encoded_label_sequences):

            # alignment from the bert tokenization to spaCy tokenization
            assert len(sequence) == len(labels)

            #maps each original token to it's subwords
            token_idx_to_subwords = []
            for token in sequence:
                token_idx_to_subwords.append([subword for subword in tokenizer.tokenize(str(token))])

            #token_idx_to_subwords = [seq for seq in token_idx_to_subwords if seq]
            bert_subwords = ['[CLS]', '[SEP]']
            bert_subword_labels = [biluo_ordered_labels.index('BERT_TOKEN'), biluo_ordered_labels.index('BERT_TOKEN')]
            bert_subword_to_original_tokenization_alignment = [-1,-1]
            original_tokens_processed = []

            # print(token_idx_to_subwords[:10])
            # print([str(token) for token in sequence][:10])
            # exit()
            idx = 0
            chunk_start = 0
            while idx < len(sequence):

                start_next_buffer = False
                token_in_buffer_size = len(bert_subwords) + len(token_idx_to_subwords[idx]) <= max_sequence_length

                if token_in_buffer_size:
                    #build a sequence
                    bert_subwords[-1:-1] = [subword for subword in token_idx_to_subwords[idx]]
                    bert_subword_labels[-1:-1] = [labels[idx] for _ in token_idx_to_subwords[idx]]
                    bert_subword_to_original_tokenization_alignment[-1:-1] = [idx-chunk_start for _ in token_idx_to_subwords[idx]]
                    original_tokens_processed.append(idx)
                    idx+=1

                #Insure we aren't splitting on a label by greedily splitting on 'O' labels once the buffer gets very full (>500 subwords)
                if len(bert_subwords) > 500 and labels[idx-1] == biluo_ordered_labels.index('O'):
                    start_next_buffer = True

                if not token_in_buffer_size or start_next_buffer:
                    all_bert_subword_sequences.append(bert_subwords)
                    all_bert_label_sequences.append(bert_subword_labels)
                    all_bert_sequence_alignments.append(bert_subword_to_original_tokenization_alignment)

                    original_tokenization_labels.append([labels[i] for i in original_tokens_processed])

                    #reset sequence builders
                    bert_subwords = ['[CLS]', '[SEP]']
                    bert_subword_labels = [biluo_ordered_labels.index('BERT_TOKEN'), biluo_ordered_labels.index('BERT_TOKEN')]
                    bert_subword_to_original_tokenization_alignment = [-1, -1]
                    original_tokens_processed = []
                    chunk_start = idx

            if bert_subwords != ['[CLS]', '[SEP]']:
                #Add the remaining
                all_bert_subword_sequences.append(bert_subwords)
                all_bert_label_sequences.append(bert_subword_labels)
                all_bert_sequence_alignments.append(bert_subword_to_original_tokenization_alignment)
                original_tokenization_labels.append([labels[i] for i in original_tokens_processed])

        for seq in original_tokenization_labels:
            for label in seq:
                assert label != -1

        max_num_spacy_labels = max([len(seq) for seq in original_tokenization_labels])

        bert_input_ids = torch.zeros(size=(len(all_bert_subword_sequences), max_sequence_length), dtype=torch.long)
        bert_attention_masks = torch.zeros_like(bert_input_ids)
        bert_sequence_lengths = torch.zeros(size=(len(all_bert_subword_sequences), 1))

        bert_labels = torch.zeros_like(bert_input_ids)
        bert_alignment = torch.zeros_like(bert_input_ids)
        gold_original_token_labels = torch.zeros(size=(len(all_bert_subword_sequences), max_num_spacy_labels), dtype=torch.long)

        for idx, (bert_subword_sequence, bert_label_sequence, alignment, original_tokenization_label) \
                in enumerate(zip(all_bert_subword_sequences, all_bert_label_sequences, all_bert_sequence_alignments, original_tokenization_labels)):
            if len(bert_subword_sequence) > 512:
                raise BaseException("Error sequence at index %i as it is to long (%i tokens)" % (idx, len(bert_subword_sequence)))
            input_ids = tokenizer.convert_tokens_to_ids(bert_subword_sequence)
            attention_masks = [1] * len(input_ids)

            while len(input_ids) < max_sequence_length: #pad bert aligned input until max length
                input_ids.append(0)
                attention_masks.append(0)
                bert_label_sequence.append(0)
                alignment.append(-1)
            while len(original_tokenization_label) < max_num_spacy_labels: #pad spacy aligned input with -1
                original_tokenization_label.append(-1)

            bert_input_ids[idx] = torch.tensor(input_ids, dtype=torch.long)
            bert_attention_masks[idx] = torch.tensor(attention_masks, dtype=torch.long)
            bert_alignment[idx] = torch.tensor(alignment, dtype=torch.long)
            bert_sequence_lengths[idx] = torch.tensor(sum([1 for x in input_ids if x != 0]), dtype=torch.long)
            gold_original_token_labels[idx] = torch.tensor(original_tokenization_label, dtype=torch.long)
            bert_labels[idx] = torch.tensor(bert_label_sequence, dtype=torch.long)


            for i in range(1, len(bert_labels[idx]) - 1):
                # print()
                # print(f"Bert Labels | {i} | {bert_labels[idx][i]}")
                # print(f"Correct Original Labels | {i} | {gold_original_token_labels[idx][bert_alignment[idx][i]]}")
                # print(f"Bert Labels: {bert_labels[idx]}")
                # print(f"Spacy Labels: {gold_original_token_labels[idx]}")
                # print(f"Bert Alignment: {bert_alignment[idx]}")
                try:
                    assert bert_labels[idx][i] == gold_original_token_labels[idx][bert_alignment[idx][i]]
                except BaseException:
                    pass



        if save_directory:
            torch.save(bert_input_ids, os.path.join(save_directory, f"bert_input.pt")) #bert input ids
            torch.save(bert_attention_masks, os.path.join(save_directory, f"bert_attention_mask.pt")) #bert attention masks
            torch.save(bert_sequence_lengths, os.path.join(save_directory, f"bert_sequence_length.pt")) #length of actual bert sequence
            torch.save(bert_labels, os.path.join(save_directory, f"bert_labels.pt")) #correct labels relative to bert tokenization
            torch.save(gold_original_token_labels, os.path.join(save_directory, f"spacy_labels.pt")) #correct labels relative to spacy tokenization
            torch.save(bert_alignment, os.path.join(save_directory, f"subword_to_spacy_alignment.pt")) #alignment between bert and spacy sequences
            torch.save(biluo_ordered_labels, os.path.join(save_directory, 'entity_names.pl')) #entity labels
            torch.save(loss_weights, os.path.join(save_directory, 'loss_weights.pt')) #global entity class counts

        return (bert_input_ids, None, bert_attention_masks), bert_sequence_lengths, bert_labels, original_tokenization_labels, \
               bert_alignment, biluo_ordered_labels, loss_weights









