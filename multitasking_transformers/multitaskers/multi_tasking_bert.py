import os, torch, logging, mlflow, numpy as np
from typing import List, Tuple
from transformers import BertModel, BertConfig, BertTokenizer, CONFIG_NAME, WEIGHTS_NAME
from multitasking_transformers.dataloaders import RoundRobinDataLoader
from multitasking_transformers.heads import TransformerHead, SubwordClassificationHead,\
    CLSRegressionHead, CLSClassificationHead

from multitasking_transformers.evaluation import evaluate_ner, evaluate_sts, evaluate_classification
from .util import get_model_path
from torch.utils.data import DataLoader

log = logging.getLogger('root')

class MultiTaskingBert():

    def __init__(self,
                 heads: List[TransformerHead],
                 transformer_weights: str,
                 model_storage_directory=None,
                 device='cpu',
                 learning_rate=5e-5,
                 transformer_layers=12,
                 use_pretrained_heads=True):
        """
        
        :param model_directory: a directory path to the multi-tasking model. This contains bert weights and head weights.
        """
        self.transformer_weights = get_model_path(transformer_weights)
        self.heads = heads
        self.model_storage_directory = model_storage_directory
        self.transformer_layers = transformer_layers

        self.device = device

        self.bert_tokenizer = BertTokenizer.from_pretrained(self.transformer_weights)
        if os.path.exists(self.transformer_weights):
            if os.path.exists(os.path.join(self.transformer_weights, CONFIG_NAME)):
                config = BertConfig.from_json_file(os.path.join(self.transformer_weights, CONFIG_NAME))
            elif os.path.exists(os.path.join(self.transformer_weights, 'bert_config.json')):
                config = BertConfig.from_json_file(os.path.join(self.transformer_weights, 'bert_config.json'))
            else:
                raise ValueError("Cannot find a configuration for the BERT based model you are attempting to load.")
        else:
            config = BertConfig.from_pretrained(self.transformer_weights)

        config.output_hidden_states = True

        use_tf_model = 'biobert_v1' in self.transformer_weights
        self.bert = BertModel.from_pretrained(self.transformer_weights, config=config, from_tf=use_tf_model)

        for head in heads:
            if use_pretrained_heads:
                if head.from_pretrained(self.transformer_weights):
                    log.info(f"Loading pretrained head: {head}")
                else:
                    log.info(f"Training new head: {head}")
                if getattr(head, '_init_mlm_head', None): #lm heads required bert model configurations.
                    head._init_mlm_head(config)
            else:
                log.info(f"Training new head: {head}")

        if not hasattr(self, 'epoch'):
            self.epoch = 0

        self.optimizer = torch.optim.Adam(
            self.bert.parameters(),
            weight_decay=0,
            lr=learning_rate
        )


    def save_checkpoint(self, checkpoint_path: str):
        """
        Saves a checkpoint of the multi-tasking model.

        A check point includes:
        Base model weights, tokenizer and configuration.
        Head weights and configurations.

        :param checkpoint_path: the directory to save the checkpoint
        :return: The directory containing the saved model.
        """
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        else:
            raise ValueError("Attempting to save checkpoint to an existing directory")
        log.info(f"Saving checkpoint: {checkpoint_path}")
        base = self.bert

        #Save base model
        torch.save(base.state_dict(), os.path.join(checkpoint_path, WEIGHTS_NAME))
        base.config.to_json_file(os.path.join(checkpoint_path, CONFIG_NAME))
        self.bert_tokenizer.save_vocabulary(checkpoint_path)

        #Save heads
        for head in self.heads:
            head.save(os.path.join(checkpoint_path))



    def fit(self,
            heads_and_dataloaders: List[Tuple[TransformerHead, DataLoader, DataLoader]],
            num_epochs=1,
            evaluation_interval = 1,
            checkpoint_interval = 1,
            repeat_in_epoch_sampling=True,
            use_loss_weight=False,
            in_epoch_logging_and_saving=False):
        """
        Fits a multi tasking model on the given heads and dataloaders
        :param datasets:
        """

        train_loader = \
            RoundRobinDataLoader([(head, train_loader) for head, train_loader, _ in heads_and_dataloaders],
                                 repeat_in_epoch_sampling=repeat_in_epoch_sampling)

        self.bert.train()
        self.bert.to(device=self.device)
        for head in self.heads:
            head.to(device=self.device)
            head.train()


        for epoch in range(1,num_epochs+1):
            self.epoch+=1

            task_epoch_loss = {str(head): 0.0 for head in self.heads}
            # depending on round robin scheme, some tasks maybe passed more than once. we must keep count.
            task_batches = {str(head): 0 for head in self.heads}

            for training_batch_idx, (head, dataset_batch_idx, batch) in enumerate(train_loader):

                if head.__class__.__name__ == "SubwordClassificationHead":
                    bert_input_ids, bert_token_type_ids, bert_attention_masks, \
                    bert_sequence_lengths, labels, _, _, loss_weights = batch

                    if use_loss_weight:
                        loss_weights = loss_weights[0].to(device=self.device)
                        #log.info(loss_weights)
                if head.__class__.__name__ == "MaskedLMHead":
                    bert_input_ids, labels = batch
                    bert_token_type_ids = None
                    bert_attention_masks = None

                if head.__class__.__name__ in ["CLSRegressionHead", "CLSClassificationHead"]:
                    bert_input_ids, bert_token_type_ids, bert_attention_masks, labels = batch

                if not use_loss_weight:
                    loss_weights = None

                bert_input_ids = bert_input_ids.to(device=self.device)

                if isinstance(bert_attention_masks, torch.Tensor): #some tasks (ie. NER) do not need token types.
                    bert_attention_masks = bert_attention_masks.to(device=self.device)
                else:
                    bert_attention_masks = None

                if isinstance(bert_token_type_ids, torch.Tensor): #some tasks (ie. NER) do not need token types.
                    bert_token_type_ids = bert_token_type_ids.to(device=self.device)
                else:
                    bert_token_type_ids = None

                labels = labels.to(device=self.device)


                hidden_states, _ , all_hidden = self.bert(bert_input_ids,
                                    attention_mask = bert_attention_masks,
                                    token_type_ids = bert_token_type_ids
                                    )

                loss, _ = head(hidden_states, labels=labels, loss_weight=loss_weights)

                if dataset_batch_idx % 5 == 0:
                    log.info(f"|{training_batch_idx}|{dataset_batch_idx}|: {head}, {loss}")

                if in_epoch_logging_and_saving: #useful for language modeling.
                    if (training_batch_idx+1) % 100 == 0:
                        mlflow.log_metric(f"{head}/epoch_train_loss", float(task_epoch_loss[str(head)])/task_batches[str(head)], step=training_batch_idx)

                    if (training_batch_idx+1)  % 16000 == 0:
                        self.save_checkpoint(os.path.join(self.model_storage_directory, f'checkpoint_{epoch}_{training_batch_idx+1}'))


                task_epoch_loss[str(head)] += float(loss.item())
                task_batches[str(head)] += 1

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            for head in self.heads:
                task_epoch_loss[str(head)] /= task_batches[str(head)]
                mlflow.log_metric(f"{head}/train_loss", float(task_epoch_loss[str(head)]), step=epoch)
            log.info(f"Epoch {self.epoch} Loss: {task_epoch_loss}")

            if epoch % evaluation_interval == 0:
                self.predict([(head, test_loader) for head,_ , test_loader in heads_and_dataloaders])
            if epoch % checkpoint_interval == 0:
                self.save_checkpoint(os.path.join(self.model_storage_directory, f'checkpoint_{epoch}'))

    def predict(self, heads_and_dataloaders: List[Tuple[TransformerHead, DataLoader]]):
        """
        Predicts over the heads and dataloaders present.
        :param heads_and_dataloaders:
        :return:
        """

        self.bert.eval()
        self.bert.to(device=self.device)

        for head in self.heads:
            head.eval()
        with torch.no_grad():
            for head, dataloader in heads_and_dataloaders:
                log.info(f"Predicting {head}")
                if isinstance(head, SubwordClassificationHead):
                    spacy_predictions_and_correct_labels = []
                    for idx, batch in enumerate(dataloader):
                        bert_input_ids, bert_token_type_ids, bert_attention_masks,\
                        bert_sequence_lengths, correct_bert_labels, correct_spacy_labels, alignments, _ = batch
                        print("Here")
                        bert_input_ids = bert_input_ids.to(device=self.device)
                        bert_attention_masks = bert_attention_masks.to(device=self.device)

                        if isinstance(bert_token_type_ids, torch.Tensor):  # some tasks (ie. NER) do not need token types.
                            bert_token_type_ids = bert_token_type_ids.to(device=self.device)
                        else:
                            bert_token_type_ids = None

                        hidden_states, _, all_hidden = self.bert(bert_input_ids,
                                                     attention_mask=bert_attention_masks,
                                                     token_type_ids=bert_token_type_ids
                                                     )

                        subword_scores = head(all_hidden[self.transformer_layers])[0]
                        batch_sequence_predictions = subword_scores.max(2)[1] #subword predictions


                        for j in range(batch_sequence_predictions.shape[0]):
                            # retrieve only tokens from the sequence we care about
                            bert_sequence_predictions = batch_sequence_predictions[j][:int(bert_sequence_lengths[j].item())].cpu().numpy()

                            # array to store predictions relative to initial (spaCy) tokenization
                            spacy_sequence_predictions = [head.config.labels.index('O')] * (alignments[j].max().item()+1)


                            for token_index in range(1, len(bert_sequence_predictions) - 1):  # account for bert padding tokens
                                spacy_sequence_predictions[alignments[j][token_index]] = bert_sequence_predictions[token_index]


                            spacy_predictions_and_correct_labels.append(
                                (spacy_sequence_predictions, correct_spacy_labels[j][:alignments[j].max().item()+1].tolist())
                            )
                            assert(len(spacy_sequence_predictions) == len(correct_spacy_labels[j][:alignments[j].max().item()+1].tolist()))

                    evaluate_ner(spacy_predictions_and_correct_labels,
                                 head.config.labels, str(head), step=self.epoch, evaluate_bilou=head.config.evaluate_biluo)

                if isinstance(head, CLSRegressionHead):

                    predicted_scores = []
                    correct_scores = []
                    for idx, batch in enumerate(dataloader):
                        bert_input_ids, bert_token_type_ids, bert_attention_masks, labels = batch

                        bert_input_ids = bert_input_ids.to(device=self.device)
                        bert_attention_masks = bert_attention_masks.to(device=self.device)

                        if isinstance(bert_token_type_ids, torch.Tensor):  # some tasks (ie. NER) do not need token types.
                            bert_token_type_ids = bert_token_type_ids.to(device=self.device)
                        else:
                            bert_token_type_ids = None

                        hidden_states, _, all_hidden = self.bert(bert_input_ids,
                                                     attention_mask=bert_attention_masks,
                                                     token_type_ids=bert_token_type_ids
                                                     )
                        scores = head(all_hidden[self.transformer_layers])[0]

                        predicted_scores = predicted_scores + scores.cpu().squeeze().tolist()
                        correct_scores = correct_scores + labels.squeeze().tolist()

                    evaluate_sts((np.asarray(predicted_scores, dtype=np.float),
                                  np.asarray(correct_scores, dtype=np.float)),
                                 str(head), step=self.epoch)

                if isinstance(head, CLSClassificationHead):

                    predicted_labels = []
                    correct_labels = []
                    for idx, batch in enumerate(dataloader):
                        bert_input_ids, bert_token_type_ids, bert_attention_masks, labels = batch

                        bert_input_ids = bert_input_ids.to(device=self.device)
                        bert_attention_masks = bert_attention_masks.to(device=self.device)

                        if isinstance(bert_token_type_ids, torch.Tensor):  # some tasks (ie. NER) do not need token types.
                            bert_token_type_ids = bert_token_type_ids.to(device=self.device)
                        else:
                            bert_token_type_ids = None

                        hidden_states, _, all_hidden = self.bert(bert_input_ids,
                                                     attention_mask=bert_attention_masks,
                                                     token_type_ids=bert_token_type_ids
                                                     )
                        predictions = head(hidden_states)[0]

                        #.max will select the label index we want.
                        predicted_labels = predicted_labels + predictions.max(1)[1].cpu().squeeze().tolist()
                        correct_labels = correct_labels + labels.max(1)[1].squeeze().tolist()

                    evaluate_classification((np.asarray(predicted_labels, dtype=np.int),
                                  np.asarray(correct_labels, dtype=np.int)), head.config.labels,
                                 str(head), step=self.epoch)

        self.bert.train()
        for head in self.heads:
            head.train()















