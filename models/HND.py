import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from . import gnn_iclr
import pdb
import copy
from transformers.models.bert.modeling_bert import ACT2FN
import math
import random
from transformers import AdamW
import numpy as np


def KL_loss(p, q, T=1.0):
    p = p/T + 1e-4
    q = q/T + 1e-4
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    p_loss = p_loss.sum(-1)
    q_loss = q_loss.sum(-1)
    loss = (p_loss + q_loss) / 2.0

    return loss.mean()

def compute_kl_loss(p, q,T=1.0):
    p = p/T
    q = q/T
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q.detach(), dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p.detach(), dim=-1), reduction='none')
    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum(-1)
    q_loss = q_loss.sum(-1)
    loss = 0.8 * p_loss + 0.2 * q_loss
    return loss.mean()

def contrast_loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def Cosime(X):
    x = F.normalize(X, dim=-1, p=2)
    S = torch.matmul(x,x.transpose(-1,-2))
    d = S.size(-1)
    Mask = (1.0 - torch.eye(d).unsqueeze(0).expand(S.size())).to(S)
    Loss = (Mask * S).mean()
    return Loss


class Focal_loss(nn.Module):
    def __init__(self,gamma=2.0,alpha=-1):
        super().__init__()
        self.alpha=alpha
        self.gamma=gamma


    def forward(self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ):
      """
      Args:
          inputs: A float tensor of arbitrary shape.
                  The predictions for each example.
          targets: A float tensor with the same shape as inputs. Stores the binary
                  classification label for each element in inputs
                  (0 for the negative class and 1 for the positive class).
          mask:
          alpha: (optional) Weighting factor in range (0,1) to balance
                  positive vs negative examples or -1 for ignore. Default = 0.25
          gamma: Exponent of the modulating factor (1 - p_t) to
                 balance easy vs hard examples.
          reduction: 'none' | 'mean' | 'sum'
                   'none': No reduction will be applied to the output.
                   'mean': The output will be averaged.
                   'sum': The output will be summed.
      Returns:
          Loss tensor with the reduction option applied.
      """

      ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
      p = torch.sigmoid(inputs)
      p_t = p * targets + (1 - p) * (1 - targets)
      loss = ce_loss * ((1 - p_t) ** self.gamma)

      if self.alpha >= 0:
        alpha_t = (1 - self.alpha) * targets + self.alpha * (1 - targets)
        loss = alpha_t * loss
      return loss.mean()

class Focal_loss1(nn.Module):
    def __init__(self,gamma=2.0,alpha=1):
        super().__init__()
        self.alpha=alpha
        self.gamma=gamma

    def forward(self,
        inputs: torch.Tensor,   #[b,N,label_num]
        targets: torch.Tensor,  #[b,N,label_num]
        step=0
    ):
        log_p = F.log_softmax(inputs, dim=-1)
        p = torch.exp(log_p)
        # p = F.softmax(inputs, dim=-1)

        loss_list = -(self.alpha * ((1 - p) ** self.gamma) * log_p)
        loss = (loss_list * targets).sum()
        # if(step>10005):
        #     pdb.set_trace()
       
        return loss



class MultiHeadAttention(nn.Module):
    def __init__(self, config, num_attention_heads=4):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads

        self.query = nn.Sequential(
            nn.Linear(config.hidden_size,  self.hidden_size//3),
            # nn.LeakyReLU(0.1)
            # nn.ReLU(inplace=True),
        )
        self.key = nn.Sequential(
            nn.Linear(config.hidden_size,  self.hidden_size//3),
            # nn.LeakyReLU(0.1)
            # nn.ReLU(inplace=True),
        )
        self.value = nn.Sequential(
            nn.Linear(config.hidden_size, self.hidden_size),
            # nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.out_put0 = nn.Linear(config.hidden_size, self.hidden_size)
        self.out_put1 = nn.Linear(config.hidden_size, self.hidden_size)
        self.act = ACT2FN["gelu"]
        # self.out_put = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size),
        #     # nn.ReLU(inplace=True),
        #     ACT2FN["gelu"],
        #     nn.Dropout(config.attention_probs_dropout_prob),
        # )
        self.Layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def transpose_for_scores(self, x):
        attention_head_size = x.size(-1) // self.num_attention_heads
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, samples,samples1):
        # samples => [B,N*K,d] => [B,h,N*K,d]
        query_layer = self.transpose_for_scores(self.query(samples))
        key_layer = self.transpose_for_scores(self.key(samples))
        value_layer = self.transpose_for_scores(self.value(samples))

        # [B,h,N*K,N*K]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # [B,h,N*K,d]
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        #[B,N*K,d]
        # samples10 = torch.matmul(context_layer, self.out_put0)
        samples11 = self.dropout(self.act(self.out_put0(context_layer)))
        # samples12 = torch.matmul(samples11, self.out_put1.transpose(-1, -2))
        samples_out = self.Layer_norm(samples11 + samples1)
        return samples_out


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, config, num_attention_heads=4):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads     #每个head的维度

        # self.query = nn.Sequential(
        #     nn.Linear(config.hidden_size,  self.hidden_size//3),
        #     # nn.LeakyReLU(0.1)
        #     # nn.ReLU(inplace=True),
        # )
        # self.key = nn.Sequential(
        #     nn.Linear(config.hidden_size,  self.hidden_size//3),
        #     # nn.LeakyReLU(0.1)
        #     # nn.ReLU(inplace=True),
        # )
        # self.value = nn.Sequential(
        #     nn.Linear(config.hidden_size, self.hidden_size),
        #     # nn.ReLU(inplace=True),
        # )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # self.out_put = nn.Linear(self.hidden_size, self.hidden_size)
        # self.act = ACT2FN["gelu"]
        # self.out_put = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size),
        #     # nn.ReLU(inplace=True),
        #     ACT2FN["gelu"],
        #     nn.Dropout(config.attention_probs_dropout_prob),
        # )
        # self.Layer_norm = nn.LayerNorm(config.hidden_size)
        self.Layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        # self.out_put1 = nn.Linear(self.hidden_size, self.hidden_size)

    def transpose_for_scores(self, x):
        attention_head_size = x.size(-1) // self.num_attention_heads
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, relation, samples, W):
        # samples => [B,N*K,d] => [B,h,N*K,d]
        # query_layer = self.transpose_for_scores(self.query(samples))
        # key_layer = self.transpose_for_scores(self.key(relation))
        # value_layer = self.transpose_for_scores(self.value(relation))
        query_layer = self.transpose_for_scores(samples * W[:, :1] + W[:, 1:2])
        key_layer = self.transpose_for_scores(relation * W[:, 2:3] + W[:, 3:4])
        value_layer = self.transpose_for_scores(relation * W[:, 4:5] + W[:, 5:6])

        # [B,h,N*K,N*K]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # [B,h,N*K,d]
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        #[B,N*K,d]
        samples1 = self.dropout(self.act(context_layer) + samples)
        samples1 = self.Layer_norm(samples1) * W[:, 6:7] + W[:, 7:8]
        # samples1 = (samples1 + samples)/2.0
        return samples1

class Task_Specific_Net(nn.Module):
    def __init__(self, config, class_num=5, mid_dim=256):
        super().__init__()
        self.hidden_size = int((config.hidden_size // class_num) * class_num)
        self.class_num = class_num
        self.Super_net1 = nn.Sequential(
            # nn.Linear(class_num, self.hidden_size // 2),
            nn.Linear(self.hidden_size, self.hidden_size//2),
            # nn.ReLU(inplace=True),
            # nn.Tanh(),
            # nn.Dropout(config.attention_probs_dropout_prob),
            nn.LayerNorm(self.hidden_size//2, eps=config.layer_norm_eps),
            # nn.Linear(self.hidden_size // 2, 2),
            nn.Linear(self.hidden_size//2, self.hidden_size // class_num),
        )
        self.Super_net11 = nn.Sequential(
            # nn.Linear(class_num, self.hidden_size // 2),
            nn.Linear(self.hidden_size, self.hidden_size//2),
            # nn.ReLU(inplace=True),
            # nn.Tanh(),
            # nn.Dropout(config.attention_probs_dropout_prob),
            nn.LayerNorm(self.hidden_size//2, eps=config.layer_norm_eps),
            # nn.Linear(self.hidden_size // 2, 2),
            nn.Linear(self.hidden_size//2, self.hidden_size // class_num),
        )

        self.Super_net2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            # nn.ReLU(inplace=True),
            # nn.Tanh(),
            # nn.Dropout(config.attention_probs_dropout_prob),
            nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(self.hidden_size, self.hidden_size * 2),
        )

        self.Super_net3 = nn.Sequential(
            # nn.Linear(self.hidden_size//8, self.hidden_size // 2),
            nn.Linear(1, self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            # nn.ReLU(inplace=True),
            # nn.Tanh(),
            # nn.Dropout(config.attention_probs_dropout_prob),
            nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps),
            # nn.Linear(self.hidden_size // 2, 64),
            nn.Linear(self.hidden_size, self.hidden_size//class_num),
        )

        self.Super_Project = MultiHeadCrossAttention(config)

        self.Drop = nn.Dropout(config.attention_probs_dropout_prob)
        # self.act = nn.ReLU(inplace=True)
        self.act = ACT2FN["gelu"]
        self.Layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        # self.Super_Project1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.Task_net = nn.Linear(self.hidden_size, self.class_num, bias=False)


    def initialize_task_net(self, relations):
        B, N, d = relations.size()
        self.Weight11 = self.Super_net1(relations).view(B, 1, -1)
        self.Weight12 = self.Super_net11(relations).view(B, 1, -1)
        self.Weight1 = torch.cat((self.Weight11, self.Weight12), dim=-2)
        # self.L2 = torch.sum(self.Weight1.pow(2)) / 2
        W = torch.tensor(self.Weight1.tolist()).to(self.Weight1)
        self.Task_net1 = torch.nn.Parameter(W.float())


        self.Weight2 = self.Super_net2(relations)
        self.L2 = torch.sum(self.Weight2.pow(2)) / 2
        W = torch.tensor(self.Weight2.tolist()).to(self.Weight2)
        self.Task_net2 = torch.nn.Parameter(W.float())

        relations_temp = relations.unsqueeze(-2)
        self.Weight3 = self.Super_net3(relations_temp.transpose(-1, -2)).transpose(1, 2).contiguous().view(B, d, -1)
        self.L2 += torch.sum(self.Weight3.pow(2)) / 2
        W = torch.tensor(self.Weight3.tolist()).to(self.Weight3)
        self.Task_net3 = torch.nn.Parameter(W.float())


    def Encoder_Sample(self, samples, W, W1):
        # Output = self.Layer_norm(self.Drop(self.act(torch.matmul(samples, W))) + samples) * W1[:, :1] + W1[:, 1:2]
        Output = torch.cat((self.Drop(self.act(torch.matmul(samples, W))), samples), dim=-1) * torch.cat((W1[:, :1], W1[:, 1:2]), dim=-1)
        return Output

    def forward(self, Sup_samples, Que_samples=None, Sup_samples_o=None, Que_samples_o=None, Task=True, step=0):
        if(Task):
            relation = self.Task_net3
            Imput_sample = torch.cat((Sup_samples, Que_samples), dim=-2)
            Output_sample = self.Encoder_Sample(Imput_sample, relation, self.Task_net1)

            N = Sup_samples.size(-2)
            Rel = self.Task_net2
            Sup = Output_sample[:, :N]
            Que = Output_sample[:, N:]

            predict1 = torch.matmul(Sup, Rel.transpose(-1, -2))
            predict11 = torch.matmul(Que, Rel.transpose(-1, -2))

            predict2 = torch.matmul(Sup, Sup.transpose(-1, -2))
            predict3 = torch.matmul(Rel, Rel.transpose(-1, -2))

            predict = (predict1, predict11, predict2, predict3)
        else:
            detal1  = (self.Task_net1 - self.Weight1).detach()
            W1 = (self.Weight1 + detal1)
            detal2  = (self.Task_net2 - self.Weight2).detach()
            W2 = (self.Weight2 + detal2)
            detal3  = (self.Task_net3 - self.Weight3).detach()
            W3 = (self.Weight3 + detal3)
            relation = W3
            Rel = W2

            B, N, d = Sup_samples.size()
            samples_sup = Sup_samples.view(-1, d).unsqueeze(0).expand(B, B * N, d)
            samples_que = Que_samples.view(-1, d).unsqueeze(0).expand(B, B * N, d)

            samples_input = torch.cat((Sup_samples, Que_samples, samples_sup, samples_que), dim=-2)
            samples_out = self.Encoder_Sample(samples_input, relation, W1)
            Sup_prompt = samples_out[:, :N]
            Que_prompt = samples_out[:, N:2 * N]

            Proto = (Sup_prompt + Rel) / 2.0
            Sup = torch.cat((Proto, Sup_samples_o), dim=-1)
            Que = torch.cat((Que_prompt, Que_samples_o), dim=-1)
            predict1 = torch.matmul(Que, Sup.transpose(-1, -2))

            Predict = torch.matmul(samples_out, Rel.transpose(-1, -2))
            predict2 = Predict[:, : N]
            predict21 = Predict[:, N:2 * N]

            predict2 = torch.matmul(Que_prompt, Sup_prompt.transpose(-1, -2))
            predict4 = torch.matmul(Sup_prompt, Sup_prompt.transpose(-1, -2))
            predict5 = torch.matmul(Que_prompt, Que_prompt.transpose(-1, -2))
            predict6 = torch.matmul(Rel, Rel.transpose(-1, -2))

            predict = (predict1, predict2, predict21, predict4, predict5, predict6)

        return predict

def get_task_op(model,Train=False):
    if(Train):
        lr1 = 2e-5
        lr2 = 0.01
    else:
        lr1 = 2e-5
        lr2 = 0.02

    parameters_to_optimize = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight','Task_net1']
    component = ["MLP"]
    component1 = ['Task_net3', 'Task_net1','Task_net2']
    parameters = [
        {'params': [p for n, p in parameters_to_optimize
                    if not any(nd in n for nd in no_decay) and any(nd in n for nd in component)],
         'weight_decay': 1e-4,
         'lr': lr1
         },
        {'params': [p for n, p in parameters_to_optimize
                    if any(nd in n for nd in no_decay) and any(nd in n for nd in component)],
         'weight_decay': 0.0,
         'lr': lr1
         },
        {'params': [p for n, p in parameters_to_optimize
                    if not any(nd in n for nd in no_decay) and any(nd in n for nd in component1)],
         'weight_decay': 1e-4,
         'lr': lr2
         },
        {'params': [p for n, p in parameters_to_optimize
                    if any(nd in n for nd in no_decay) and any(nd in n for nd in component1)],
         'weight_decay': 0.0,
         'lr': lr2
         }
    ]
    optimizer = AdamW(parameters, lr=lr2)

    return optimizer

class HND(fewshot_re_kit.framework.FewShotREModel):
    def __init__(self, sentence_encoder, N, hidden_size=768):
        '''
        N: Num of classes
        '''
        self.config = sentence_encoder.config
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)

        self.loss_fnt = Focal_loss()
        self.loss_fnt_c = Focal_loss1()
        
        self.hidden_size = int(self.config.hidden_size // N * N)
        self.Input_example = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, self.hidden_size),
            # nn.ReLU(inplace=True),
            # nn.Dropout(self.config.attention_probs_dropout_prob),
            # nn.LayerNorm(self.hidden_size, eps=self.config.layer_norm_eps)
            # nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.Input_example1 = nn.Sequential(
            nn.Linear(self.config.hidden_size*2, self.config.hidden_size),
            nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps),
            # nn.BatchNorm1d(self.config.hidden_size),
            nn.ReLU(inplace=True),
            # nn.Dropout(self.config.attention_probs_dropout_prob),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
        )
        self.MLP = nn.Sequential(
            # nn.Linear(self.config.hidden_size, self.config.hidden_size),
            # nn.ReLU(inplace=True),
            nn.Dropout(self.config.attention_probs_dropout_prob),
            nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps),
            nn.Linear(self.config.hidden_size, self.config.hidden_size * 2),
        )

        # self.bn = nn.BatchNorm1d(self.config.hidden_size, affine=False)

        self.Task_Model = Task_Specific_Net(config=self.config,class_num=N)

        self.beta = 0.99
        self._get_target_encoder()
    def set_requires_grad(self,model, val):
        for p in model.parameters():
            p.requires_grad = val

    def _get_target_encoder(self):
        self.target_encoder = copy.deepcopy(self.sentence_encoder)
        self.set_requires_grad(self.target_encoder, False)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def update_moving_average(self):
        model_pair = [[self.sentence_encoder, self.target_encoder]]
        for Online,target in model_pair:
            for current_params, ma_params in zip(Online.parameters(), target.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.update_average(old_weight, up_weight)

    def get_task_loss(self,step = 0, Train=True):
        predicts = self.Task_Model(self.sup_feature.detach(), self.query_feature.detach())
        Label1 = torch.zeros_like(predicts[-1])
        Mask1 = torch.eye(Label1.size(-1)).unsqueeze(0).expand(Label1.size()).to(Label1)
        Mask = (Mask1 < 0.5)
        Label = Label1[Mask]

        Loss = self.loss_fnt_c(predicts[0], self.sup_label_one)
        Loss10 = 0.1 * self.loss_fnt(predicts[-1][Mask], Label)
        Loss11 = 0.1 * self.loss_fnt(predicts[-2][Mask], Label)
        Loss_out = Loss + Loss10 + Loss11

        B, N = self.sup_label.size()
        AC1 = 0
        if (Train):
            Sup = self.MLP(self.Input_example1(self.sup_feature_o.detach()))
            logits = torch.matmul(Sup, self.query_feature_targ.transpose(-1, -2))
            AC1 = (logits.max(-1)[-1] == self.sup_label).sum()
            if(AC1<B * N):
                Loss_out += self.loss(logits, self.query_label)


        AC = (predicts[0].max(-1)[-1] == self.sup_label).sum()
        AC2 = (predicts[1].max(-1)[-1] == self.sup_label).sum()
        Flag = (AC > (B * N - 1))
        if(Flag):
            pro = F.softmax(predicts[1],dim=-1)
            values, indices = pro.topk(3, dim=-1, largest=True, sorted=True)
            Mask_index = (pro < values[:, :, -1:])

            Predict_que = predicts[1][Mask_index]
            Label = torch.zeros_like(Predict_que)

            Loss2 = self.loss_fnt(Predict_que, Label)
            Loss_out += Loss2

        Ac_list = [int(AC), int(AC1), int(AC2)]

        if (torch.isnan(Loss_out).any() or torch.isinf(Loss_out).any()):
            pdb.set_trace()

        return Flag,Loss_out,AC

    def get_gen_loss(self, step=0, Train=True):
        predicts = self.Task_Model(self.sup_feature, self.query_feature, self.sup_feature_o, self.query_feature_o,
                                   Task=False, step=step)
        Label1 = torch.zeros_like(predicts[-1])
        Mask1 = torch.eye(Label1.size(-1)).unsqueeze(0).expand(Label1.size()).to(Label1)
        Mask = (Mask1 < 0.5)
        Label = Label1[Mask]

        Loss1 = self.loss_fnt_c(predicts[0], self.query_label_one)
        Loss2 = 0.0 * self.loss_fnt_c(predicts[1], self.sup_label_one) + self.loss_fnt_c(predicts[2], self.query_label_one)
        Loss4 = self.loss_fnt(predicts[-1][Mask], Label) + self.loss_fnt(predicts[-2][Mask], Label) \
                + self.loss_fnt(predicts[-3][Mask], Label)

        Loss_out = Loss1 + Loss2 + 0.1 *Loss4 + 0.0001 * self.Task_Model.L2

        label0 =  self.query_label.flatten()  #.max(-1)[-1].flatten()
        predict_new = predicts[0].max(-1)[-1].flatten()

        if(Train):
            Que = self.MLP(self.Input_example1(self.query_feature_o))
            logits = torch.matmul(Que, self.sup_feature_targ.transpose(-1, -2))
            Loss_out += self.loss(logits, self.query_label)


        if (torch.isnan(predicts[0]).any() or torch.isinf(predicts[0]).any()):
            pdb.set_trace()
        return predict_new, label0, Loss_out

    def forward(self, Samples, Relation, relation_examples, relation_examples1, label, N, K, Q, Train=True):
        if(Train):
            relation,relation_p = self.sentence_encoder(Relation, example=False)  # [b*r_n,d]
            query_feature,_ = self.sentence_encoder(Samples)  # [b*r_n,d]
            sup_feature, sup_feature1 = self.sentence_encoder(relation_examples)  # [b*r_n,d]

            D = query_feature.size(-1)
            self.query_feature_o = query_feature.view(-1, N * Q, D)  #[b,N*Q,d]
            self.sup_feature_o = sup_feature.view(-1, N, K, D).mean(-2)  #[b,N*K,d]
            self.query_feature = self.Input_example(self.query_feature_o)  # [b,N*Q,d]
            self.sup_feature = self.Input_example(self.sup_feature_o)  # [b,N*K,d]

            relation = relation.view(-1, N, D//2)  #[b,N,d]
            relation_p = relation_p.view(-1, N, D // 2)  # [b,N,d]
            # sup_feature1 = sup_feature1.view(-1, N, K, D//2).mean(-2)
            sup_feature1 = torch.logsumexp(sup_feature1.view(-1, N, K, D // 2), dim=-2)
            self.relation_o = torch.cat((relation,sup_feature1),dim=-1)
            self.relation = self.Input_example(self.relation_o)

            with torch.no_grad():
                relation_targ, _ = self.target_encoder(Relation, example=False)  # [b*r_n,d]
                query_feature_targ, _ = self.target_encoder(Samples)  # [b*r_n,d]
                sup_feature_targ, sup_feature1_targ = self.target_encoder(relation_examples)  # [b*r_n,d]

                self.query_feature_targ = query_feature_targ.view(-1, N * Q, D)  # [b,N*Q,d]
                self.sup_feature_targ = sup_feature_targ.view(-1, N, K, D).mean(-2)  # [b,N*K,d]

                relation_targ = relation_targ.view(-1, N, D // 2)  # [b,N,d]
                sup_feature1_targ = sup_feature1_targ.view(-1, N, K, D // 2).mean(-2)
                self.relation_targ = torch.cat((relation_targ, sup_feature1_targ), dim=-1)
        else:
            with torch.no_grad():
                relation, relation_p = self.sentence_encoder(Relation, example=False)  # [b*r_n,d]
                query_feature, _ = self.sentence_encoder(Samples)  # [b*r_n,d]
                sup_feature, sup_feature1 = self.sentence_encoder(relation_examples)  # [b*r_n,d]

                D = query_feature.size(-1)
                self.query_feature_o = query_feature.view(-1, N * Q, D)  # [b,N*Q,d]
                self.sup_feature_o = sup_feature.view(-1, N, K, D).mean(-2)  # [b,N*K,d]
                self.query_feature = self.Input_example(self.query_feature_o)  # [b,N*Q,d]
                self.sup_feature = self.Input_example(self.sup_feature_o)  # [b,N*K,d]

                relation = relation.view(-1, N, D // 2)  # [b,N,d]
                relation_p = relation_p.view(-1, N, D // 2)  # [b,N,d]
                # sup_feature1 = sup_feature1.view(-1, N, K, D // 2).mean(-2)
                sup_feature1 = torch.logsumexp(sup_feature1.view(-1, N, K, D // 2), dim=-2)
                self.relation_o = torch.cat((relation, sup_feature1), dim=-1)
                self.relation = self.Input_example(self.relation_o)


        B = self.query_feature.size(0)
        label_one_hot = label.view(B, -1, N).float()
        label_index = label_one_hot.max(-1)[-1]

        # self.sup_label_one = label_one_hot[:, :N , K].long()
        self.query_label_one = label_one_hot[:, :N * Q].long()
        self.sup_label_one = self.query_label_one
        # self.sup_label = label_index[:, :N * K].long()
        self.query_label = label_index[:, :N * Q].long()
        self.sup_label = self.query_label

        self.Task_Model.initialize_task_net(self.relation)
        self.Task_Model.MLP = self.MLP
        self.optimizer = get_task_op(self.Task_Model, Train=Train)


