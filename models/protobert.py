from typing import Optional, Union, Tuple, List

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput


class ProtoBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        # classifier_dropout = (
        #     config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        # )
        # self.dropout = nn.Dropout(classifier_dropout)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        # input_ids: Optional[torch.Tensor] = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # token_type_ids: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.Tensor] = None,
        # head_mask: Optional[torch.Tensor] = None,
        # inputs_embeds: Optional[torch.Tensor] = None,
        # labels: Optional[torch.Tensor] = None,
        task_batch: List[Tuple[torch.Tensor]],
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        loss_batch = []
        logits_batch = []
        for task in task_batch:
            support, labels_support, query, labels_query = task

            support = {k: v for k, v in support.items()}
            query = {k: v for k, v in query.items()}
            labels_support = labels_support
            labels_query = labels_query

            n = 2
            k = labels_support.shape[0] // n
            # (nk, dim)
            support_representations = self.bert(
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **support
            )[1]

            # (nq, dim)
            query_representations = self.bert(
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **query
            )[1]

            # (n, dim)
            prototypes = support_representations.view(n, k, -1).mean(dim=1)

            # (nq, n) 
            query_distances = torch.cdist(query_representations, prototypes) 
            query_logits = F.softmax(-query_distances, dim=1)

            loss = F.cross_entropy(query_logits, labels_query)

            logits_batch.append(query_logits)
            loss_batch.append(loss)
    
        logits = torch.stack(logits_batch)
        loss = torch.mean(torch.stack(loss_batch))

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )
