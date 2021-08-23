#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: naraysa & akshitac8


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import torchvision
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

random.seed(3483)
np.random.seed(3483)
torch.manual_seed(3483)
torch.cuda.manual_seed(3483)
torch.cuda.manual_seed_all(3483)

def tensordot(x,y):
    return torch.einsum("abc,cd->abd", (x, y))

def matmul(x,y):
    return torch.einsum("ab,bc->ac", (x, y))

class CONV3_3(nn.Module):
    def __init__(self, num_in=512,num_out=512,kernel=3):
        super(CONV3_3, self).__init__()
        self.body = nn.Conv2d(num_in, num_out, kernel, padding=int((kernel-1)/2), dilation=1)
        self.bn = nn.BatchNorm2d(num_out, affine=True, eps=0.001, momentum=0.99)
        self.relu = nn.ReLU(True)
    def forward(self, x):
        x = self.body(x)
        x = self.relu(x)
        x = self.bn(x) 
        return x

class CONV1_1(nn.Module):
    def __init__(self, num_in=512,num_out=512,kernel=1):
        super(CONV1_1, self).__init__()
        self.body = nn.Conv2d(num_in, num_out, kernel, padding=int((kernel-1)/2), dilation=1)
    def forward(self, x):
        x = self.body(x)
        return x

class vgg_net(nn.Module):
    def __init__(self):
        super(vgg_net, self).__init__()
        vgg19 = torchvision.models.vgg19(True)
        self.features = nn.Sequential(*list(vgg19.features[-1:]))
        self.fc = nn.Sequential(*list(vgg19.classifier[0:-2]))

    def forward(self, x):
        x = x.view([-1,512,14,14])
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

class RCB(nn.Module):
    """
    Region contextualized block
    """
    def __init__(self, heads=8, d_model=512, d_ff=1024, dropout = 0.1):
        super(RCB, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.w_q = nn.Conv2d(in_channels = d_model , out_channels = d_model , kernel_size=1, bias=True)
        self.w_k = nn.Conv2d(in_channels = d_model , out_channels = d_model , kernel_size=1, bias=True)
        self.w_v = nn.Conv2d(in_channels = d_model, out_channels = d_model, kernel_size=1, bias=True)
        self.w_o = nn.Conv2d(in_channels = d_model , out_channels = d_model , kernel_size=1, bias=True)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.sub_network = C_R(d_model, d_ff)

    def F_R(self, q, k, v, d_k, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        scores = scores.masked_fill(scores == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores) 
        return scores

    def forward(self, q_feat, k_feat, v_feat):
        if k_feat is None:
            k_feat = q_feat
        bs = q_feat.size(0)
        spa = q_feat.size(-1)
        residual = q_feat
        k_h_r = self.w_k(k_feat).view(bs, self.h, self.d_k, spa*spa).transpose(3,2)
        q_h_r = self.w_q(q_feat).view(bs, self.h, self.d_k, spa*spa).transpose(3,2)
        v_h_r = self.w_v(v_feat).view(bs, self.h, self.d_k, spa*spa).transpose(3,2)
        r_h = self.F_R(q_h_r, k_h_r, v_h_r, self.d_k, self.dropout_1)
        alpha_h = torch.matmul(r_h, v_h_r)
        o_r = alpha_h.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        o_r = o_r.permute(0,2,1)
        o_r = o_r.view(-1,self.d_model,spa,spa)
        o_r = self.dropout_2(self.w_o(o_r))
        o_r += residual
        input_o_r = o_r
        e_r = self.sub_network(o_r)
        e_r += input_o_r
        return e_r

class C_R(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = d_model , out_channels = d_ff , kernel_size= 1, bias=True)
        self.conv2 = nn.Conv2d(in_channels = d_ff , out_channels = d_model , kernel_size= 1, bias=True)
    def forward(self, x):
        x_out = self.conv2(F.relu(self.conv1(x), True))
        return x_out

class SCB(nn.Module):
    """
    scene contextualized block
    """
    def __init__(self, opt, D):
        super(SCB, self).__init__()
        self.channel_dim = opt.channel_dim
        self.sigmoid = nn.Sigmoid()
        self.gcdropout = nn.Dropout(0.2)
        self.lrelu = nn.LeakyReLU(0.2, False)
        self.w_g = nn.Conv2d(in_channels=4096,out_channels=self.channel_dim,kernel_size=1,bias=True) #nn.Linear(4096, self.channel_dim, bias=False) # 
        self.gcff = CONV3_3(num_in=self.channel_dim, num_out=self.channel_dim)
        self.channel_conv = CONV1_1(num_in=self.channel_dim, num_out=self.channel_dim)

    def F_G(self, q , k):
        r_g = q * k
        r_g = self.sigmoid(r_g)     
        r_g = r_g.view(-1,self.channel_dim,1)
        return r_g

    def forward(self, h_r, vecs, x_g):
        # import pdb;pdb.set_trace()
        q_g = self.lrelu(self.channel_conv(h_r))
        v_g =  self.lrelu(self.channel_conv(h_r))
        k_g = self.w_g(self.gcdropout(x_g).view(-1,4096,1,1))
        # k_g = self.w_g(self.gcdropout(x_g))
        q_g_value = q_g.view(-1,self.channel_dim,196).mean(-1).repeat(1,1,1).view(-1,self.channel_dim)
        r_g = self.F_G(q_g_value,k_g.view(-1,self.channel_dim))
        # r_g = self.F_G(q_g_value,k_g)
        c_g = r_g.unsqueeze(3).unsqueeze(4) * v_g.unsqueeze(2)
        c_g = c_g.view(-1,self.channel_dim,14,14)
        e_g = c_g + self.gcff(c_g)
        return e_g

class BiAM(nn.Module):
    def __init__(self, opt, dim_w2v=300, dim_feature=[196,512]):
        super(BiAM, self).__init__()
        D = dim_feature[1]     #### D is the feature dimension of attention windows
        self.channel_dim = opt.channel_dim
        self.conv_3X3 = CONV3_3(num_out=self.channel_dim)
        self.region_context_block = RCB(heads=opt.heads, d_model=self.channel_dim, d_ff=self.channel_dim*2, dropout = 0.1)
        self.scene_context_block = SCB(opt, D)
        self.W = nn.Linear(dim_w2v,D, bias=True)
        self.conv_1X1 = CONV1_1(num_in=self.channel_dim*2, num_out=D)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def predict(self, e_f, vecs, W):
        classifiers = W(vecs)                                 
        m = tensordot(e_f, classifiers.t())                                   
        logits = torch.topk(m,k=6,dim=1)[0].mean(dim=1)
        return logits
        
    def forward(self, features, vecs, x_g):
        # import pdb;pdb.set_trace()
        x_r = features.view([-1,512,14,14])
        h_r = self.conv_3X3(x_r)
        e_r = self.region_context_block(h_r,h_r,h_r)
        e_g = self.scene_context_block(h_r, vecs, x_g)
        e_f = torch.cat([e_r, e_g], dim=1)
        e_f = self.lrelu(self.conv_1X1(e_f))
        e_f = e_f.permute(0,2,3,1)
        e_f = e_f.view(-1,196,512)
        logits = self.predict(e_f, vecs, self.W)
        return logits

def ranking_lossT(logitsT, labelsT):
    eps = 1e-8
    subset_idxT = torch.sum(torch.abs(labelsT),dim=0)
    subset_idxT = (subset_idxT>0).nonzero().view(-1).long().cuda()
    sub_labelsT = labelsT[:,subset_idxT]
    sub_logitsT = logitsT[:,subset_idxT]    
    positive_tagsT = torch.clamp(sub_labelsT,0.,1.)
    negative_tagsT = torch.clamp(-sub_labelsT,0.,1.)
    maskT = positive_tagsT.unsqueeze(1) * negative_tagsT.unsqueeze(-1)
    pos_score_matT = sub_logitsT * positive_tagsT
    neg_score_matT = sub_logitsT * negative_tagsT
    IW_pos3T = pos_score_matT.unsqueeze(1)
    IW_neg3T = neg_score_matT.unsqueeze(-1)
    OT = 1 + IW_neg3T - IW_pos3T
    O_maskT = maskT * OT
    diffT = torch.clamp(O_maskT, 0)
    violationT = torch.sign(diffT).sum(1).sum(1) 
    diffT = diffT.sum(1).sum(1) 
    lossT =  torch.mean(diffT / (violationT+eps))
    return lossT