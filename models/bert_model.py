import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel, BertForSequenceClassification
from torch.nn import CrossEntropyLoss, MSELoss
import math
import main_HiURE

class BertForHiURE(BertPreTrainedModel):
    """
    Bert Model for Contextualized Relation Representation Encoder
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        args = main_HiURE.parser.parse_args()
        self.last_fc = nn.Linear((2 + args.add_word_num) * config.hidden_size, self.config.num_labels)

        # self.classifier = nn.Linear(2 * config.hidden_size, config.hidden_size)
        # self.fc = nn.Linear(config.hidden_size, self.config.num_labels)
        #self.classifier_3 = nn.Linear(config.hidden_size//2, self.config.num_labels)
        #self.classifier = nn.Linear(2 * config.hidden_size, self.config.num_labels)
        self.init_weights()
        self.output_emebedding = None

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None, e1_pos=None, e2_pos=None, aug_pos=None, sentence_data=None):

        if sentence_data is not None:
            device = torch.device("cuda")
            input_ids = sentence_data[0]#.to(device)
            attention_mask = sentence_data[1]#.to(device)
            labels = sentence_data[2]#.to(device)
            e1_pos = sentence_data[3]#.to(device)
            e2_pos = sentence_data[4]#.to(device)
            aug_pos = sentence_data[5]#.to(device)

        # print('sen_data: ',sentence_data)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        # print('outputs: ',outputs)
        e_pos_outputs = []
        sequence_output = outputs[0]
        for i in range(0, len(e1_pos)):
            e1_pos_output_i = sequence_output[i, e1_pos[i].item(), :]
            e2_pos_output_i = sequence_output[i, e2_pos[i].item(), :]
            e_pos_output_i = torch.cat((e1_pos_output_i, e2_pos_output_i), dim=0)
            # data augmentation part
            for aug_index in aug_pos[i]:
                aug_pos_output_i = sequence_output[i, aug_index.item(), :]
                e_pos_output_i = torch.cat((e_pos_output_i, aug_pos_output_i), dim=0)
                pass
            e_pos_outputs.append(e_pos_output_i)
        e_pos_output = torch.stack(e_pos_outputs)
        # return e_pos_output


        self.output_emebedding = e_pos_output #e1&e2 cancat output

        e_pos_output = self.dropout(e_pos_output)
        return e_pos_output


# f_theta1
class RelationClassification(BertForHiURE):
    def __init__(self, config):
        super().__init__(config)


# g_theta2
class LabelGeneration(BertForHiURE):
    def __init__(self, config):
        super().__init__(config)
