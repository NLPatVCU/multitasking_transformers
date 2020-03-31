import torch, os, pickle
from typing import Tuple, List, Union
from transformers import BertTokenizer, AlbertTokenizer

def _truncate_seq_pair(tokens_a, tokens_b, max_length: int):
    """
    Copied exactly from: https://github.com/huggingface/pytorch-pretrained-BERT/blob/78462aad6113d50063d8251e27dbaadb7f44fbf0/examples/extract_features.py#L150
    Truncates a sequence pair in place to the maximum length.
    """

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class SentencePairClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_directory: str):
        self.dataset_directory = dataset_directory
        self.class_labels = sorted(torch.load(os.path.join(self.dataset_directory, 'class_labels.pt')))

        self.bert_input_ids = torch.load(os.path.join(self.dataset_directory, f"bert_input_ids.pt"))
        self.bert_token_type_ids = torch.load(os.path.join(self.dataset_directory, f"bert_token_type_ids.pt"))
        self.bert_attention_masks = torch.load(os.path.join(self.dataset_directory, f"bert_attention_masks.pt"))
        self.labels = torch.load(os.path.join(self.dataset_directory, f"labels.pt"))

    def __len__(self):
        'Denotes the total number of samples'
        return self.bert_input_ids.shape[0]

    def __getitem__(self, index: int):
        # Load data and get label
        return self.bert_input_ids[index], self.bert_token_type_ids[index],\
               self.bert_attention_masks[index], self.labels[index]

    def create_sentence_pair_dataset(data: List[Tuple],
                          tokenizer: Union[BertTokenizer, AlbertTokenizer],
                          save_directory=None,
                          max_sequence_length=128):
        """
        data is tuple where 1st element is dataset and second is a sorted list of label strings
        dataset is a list of instances (tuples) where first index is id, sent1, sent2, label_index
        :param tokenizer: 
        :param save_directory:
        :param max_sequence_length:
        :return:
        """
        max_bert_input_length = 0

        data, class_labels = data[0], data[1]

        #label is index of label in class_labels or if single label 1 if true or 0 if false.
        for id, sent1, sent2, label in data:
            sentence_1_tokenized, sentence_2_tokenized = tokenizer.tokenize(sent1), tokenizer.tokenize(sent2)
            _truncate_seq_pair(sentence_1_tokenized, sentence_2_tokenized, max_sequence_length - 3)  # accounting for positioning tokens

            max_bert_input_length = max(max_bert_input_length,
                                        len(sentence_1_tokenized) + len(sentence_2_tokenized) + 3)

        bert_input_ids = torch.empty((len(data), max_bert_input_length), dtype=torch.long)
        bert_token_type_ids = torch.empty((len(data), max_bert_input_length), dtype=torch.long)
        bert_attention_masks = torch.empty((len(data), max_bert_input_length), dtype=torch.long)
        labels = torch.empty((len(data), len(class_labels)), dtype=torch.long)

        for idx, (id, sent1, sent2, label) in enumerate(data):
            tokens = []
            input_type_ids = []

            tokens.append("[CLS]")
            input_type_ids.append(0)
            sentence_1_tokenized = tokenizer.tokenize(sent1)
            sentence_2_tokenized = tokenizer.tokenize(sent2)
            _truncate_seq_pair(sentence_1_tokenized, sentence_2_tokenized, max_sequence_length - 3)
            for token in sentence_1_tokenized:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            for token in sentence_2_tokenized:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            attention_masks = [1] * len(input_ids)
            while len(input_ids) < max_bert_input_length:
                input_ids.append(0)
                attention_masks.append(0)
                input_type_ids.append(0)

            bert_input_ids[idx] = torch.tensor(input_ids, dtype=torch.long)
            bert_token_type_ids[idx] = torch.tensor(input_type_ids, dtype=torch.long)
            bert_attention_masks[idx] = torch.tensor(attention_masks, dtype=torch.long)

            one_hot = [0] * len(class_labels)
            one_hot[label] = 1

            labels[idx] = torch.tensor(one_hot, dtype=torch.long)

        if save_directory:
            torch.save(bert_input_ids, os.path.join(save_directory, f"bert_input_ids.pt"))
            torch.save(bert_token_type_ids, os.path.join(save_directory, f"bert_token_type_ids.pt"))
            torch.save(bert_attention_masks, os.path.join(save_directory, f"bert_attention_masks.pt"))
            torch.save(labels, os.path.join(save_directory, f"labels.pt"))
            torch.save(sorted(class_labels), os.path.join(save_directory, f"class_labels.pt"))

        return (bert_input_ids, bert_token_type_ids, bert_attention_masks), labels, sorted(class_labels)
