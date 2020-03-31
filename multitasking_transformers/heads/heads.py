import torch, json, os
from typing import List
from torch import nn
from transformers import BertPreTrainedModel
from transformers.modeling_bert import BertOnlyMLMHead
from torch.nn import CrossEntropyLoss, Linear, Dropout, Module
import gin, logging

log = logging.getLogger('root')



class TransformerHeadConfig(dict):
    """
    Config class following the Transformers saving and loading conventions:
    Idea: https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
    Saving/loading: https://github.com/huggingface/transformers/blob/bfec203d4ed95255619e7e2f28c9040744a16232/src/transformers/configuration_utils.py#L241
    """

    def __init__(self, *args, **kwargs):
        super(TransformerHeadConfig, self).__init__(*args, **kwargs)
        self.__dict__ = self
        assert hasattr(self, 'head_name'), "TransformerHeads require names."
        assert hasattr(self, 'head_task'), "TransformerHeads require tasks."

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `Config` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        dict_obj = json.loads(text)
        return cls(**dict_obj)

    def __str__(self):
        return f"{self.head_name}_{self.head_task}".replace(' ', '_')

class TransformerHead(Module):

    def __init__(self, head_name, head_task, hidden_size=None, hidden_dropout_prob=None, labels=None):
        """

        :param hidden_size: the dimension of the hidden state of the transformer.
        :param hidden_dropout_prob: dropout rate during training
        """
        super(TransformerHead, self).__init__()
        self.config = TransformerHeadConfig({
            'head_name': head_name,
            'head_task': str(head_task),
            'labels': labels,
            'hidden_size': hidden_size,
            'hidden_dropout_prob': hidden_dropout_prob
        })



    def from_pretrained(self, head_directory: str):
        """
        Loads a transformer head from pretrained weights
        :param head_directory: the directory of the head.
        :param head_name: the name of the head.
        :param head_task: the task of the head.
        :return: a pretrained head.
        """
        config_filepath = os.path.join(head_directory, f"{self.config.head_name}_{self.config.head_task}.json".replace(' ', '_'))
        weight_filepath = os.path.join(head_directory, f"{self.config.head_name}_{self.config.head_task}.pt".replace(' ', '_'))

        try:
            loaded_config = TransformerHeadConfig.from_json_file(config_filepath)
        except FileNotFoundError:
            return False

        assert loaded_config.head_name == self.config.head_name
        assert loaded_config.head_task == self.config.head_task

        self.config.update(loaded_config.__dict__)
        self.load_state_dict(torch.load(weight_filepath))
        return True

    def save(self, head_directory: str):
        """
        Saves a head to a given directory.
        :param head_directory:
        :return:
        """
        config_filepath = os.path.join(head_directory, f"{self.config.head_name}_{self.config.head_task}.json".replace(' ', '_'))
        weight_filepath = os.path.join(head_directory, f"{self.config.head_name}_{self.config.head_task}.pt".replace(' ', '_'))

        self.config.to_json_file(config_filepath)
        torch.save(self.state_dict(), weight_filepath)


    def __str__(self):
        return self.config.__str__()
    def __repr__(self):
        return self.config.__repr__()

class SubwordClassificationHead(TransformerHead):

    #TODO needs testing
    """
    This head defines a subword classification architecture
    """
    def __init__(self, head_task, labels=None, hidden_size=768, hidden_dropout_prob=.1):
        super(SubwordClassificationHead, self).__init__(type(self).__name__,
                                                        head_task,
                                                        labels=labels,
                                                        hidden_size=hidden_size,
                                                        hidden_dropout_prob=hidden_dropout_prob)
        self.entity_labels = self.config.labels
        self.config.evaluate_biluo = False
        self.classifier = nn.Linear(hidden_size, len(self.entity_labels))


    def forward(self, hidden_states: torch.Tensor, attention_mask = None, labels=None, loss_weight=None):
        logits = self.classifier(hidden_states)

        outputs = (logits,)  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=loss_weight)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                #print(active_loss)
                #print(active_loss.shape)
                #print("Sequence Lengths: " + str(sequence_lengths))
                active_logits = logits.view(-1, len(self.entity_labels))[active_loss]
                #print(active_logits.shape)
                active_labels = labels.view(-1)[active_loss]
                #print(active_labels.shape)
                #print(active_labels)
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, len(self.entity_labels)), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs



class BertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.cls = BertOnlyMLMHead(config)

        #self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
    ):

        sequence_output = hidden_states
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)


class MaskedLMHead(TransformerHead):
    """
    This head performs masked language modeling.
    """
    def __init__(self, head_task, labels=None, hidden_size=768, hidden_dropout_prob=.1):
        super(MaskedLMHead, self).__init__(type(self).__name__,
                                                        head_task,
                                                        labels=labels,
                                                        hidden_size=hidden_size,
                                                        hidden_dropout_prob=hidden_dropout_prob)
        self.masked_lm = None


    def _init_mlm_head(self, transformer_config):
        self.masked_lm = BertForMaskedLM(transformer_config)

    def forward(self, hidden_states: torch.Tensor, attention_mask = None, labels=None, loss_weight=None):
        if not self.masked_lm:
            raise Exception("Masked Language Model Head is not initialized.")
        return self.masked_lm(hidden_states, masked_lm_labels=labels)


class CLSRegressionHead(TransformerHead):
    def __init__(self, head_task, hidden_size=768, hidden_dropout_prob=.1, labels=None):
        super(CLSRegressionHead, self).__init__(type(self).__name__,
                                                head_task,
                                                hidden_size=hidden_size,
                                                hidden_dropout_prob=hidden_dropout_prob)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, labels=None, loss_weight=None):

        logits = self.classifier(hidden_states[:,0,:]) #take CLS token

        outputs = (logits,)  # add hidden states and attention if they are here
        if labels is not None:
            loss_function = torch.nn.MSELoss()

            loss = loss_function(logits.view(-1), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

class CLSClassificationHead(TransformerHead):
    def __init__(self, head_task, hidden_size=768, hidden_dropout_prob=.1, labels=None):
        super(CLSClassificationHead, self).__init__(type(self).__name__,
                                                head_task,
                                                labels=labels,
                                                hidden_size=hidden_size,
                                                hidden_dropout_prob=hidden_dropout_prob)

        self.class_labels = self.config.labels

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, len(self.class_labels))

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, labels=None, loss_weight=None):
        logits = self.classifier(hidden_states[:, 0, :])  # take CLS token

        outputs = (logits,)  # add hidden states and attention if they are here
        if labels is not None:
            loss_function = torch.nn.CrossEntropyLoss()
            # log.info(logits.view(-1, len(self.class_labels)).shape, labels.view(-1, len(self.class_labels)).shape)
            # exit()

            # log.info(labels.view(-1, len(self.class_labels)))
            # log.info(labels.view(-1, len(self.class_labels)).max(1)[1])
            # exit()
            loss = loss_function(logits.view(-1, len(self.class_labels)), labels.view(-1, len(self.class_labels)).max(1)[1])
            outputs = (loss,) + outputs

        return outputs



