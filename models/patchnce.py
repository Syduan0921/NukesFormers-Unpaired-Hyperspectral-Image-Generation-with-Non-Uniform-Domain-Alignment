from packaging import version
import torch
from torch import nn
import numpy as np

import random



class PatchNCELoss(nn.Module):
    def __init__(self, opt, spat=False):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.spat = spat

    def sigmoid(self, x):
        return 1 / (1+torch.exp(-x))


    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        if self.spat:
            l_pos = torch.bmm(feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
            #l_pos = compute_sam_torch(
                #feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        else:
            l_pos = torch.bmm(feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
            #l_pos = compute_sam_torch(
                #feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit

        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batchSize

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        if self.spat:
            l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))
            #l_neg_curbatch = compute_sam_torch(feat_q, feat_k.transpose(2, 1))
        else:
            l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))
            #l_neg_curbatch = compute_sam_torch(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, 0)
        l_neg = l_neg_curbatch.view(-1, npatches)
        #l_neg, _ = l_neg.sort(dim=0, descending=True)

          
        l_neg = l_neg[:, :self.opt.choose_patch]
        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T
        target = torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device)
        #out = (out- out.min()) / (out.max()-out.min()) + 1e-4
        loss = self.cross_entropy_loss(out, target)

        return loss

def compute_sam_torch(label,output):
    # assert self.label.ndim == 3 and self.label.shape == self.label.shape

    b, c, p = label.shape
    if b == 1:
        x_true = label.reshape(b, c, -1)
        x_pred = output.reshape(b, -1, c)
        # x_pred[np.where((np.linalg.norm(x_pred, 2, 1)) == 0),] += 0.0001
        sam = torch.bmm(x_true, x_pred)
        sam = torch.arccos(sam)
        sam = torch.unsqueeze(sam, dim=0)
    else:
        x_true = label.reshape(b, c, -1)
        x_pred = output.reshape(b, -1, c)

        # x_pred[np.where((np.linalg.norm(x_pred, 2, 1)) == 0),] += 0.0001
        sam = torch.bmm(x_true, x_pred)
        sam = torch.arccos(sam)
    return sam

def sam_sim(x, y):
    s = torch.bmm(x, y)
    t = torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2))
    th = torch.arccos(s / t)
    return th

def sam_calu(x, y):
    s = np.sum(np.dot(x, y), axis=-2)
    t = np.sqrt(np.sum(x ** 2)) * np.sqrt(np.sum(y ** 2))
    th = np.arccos(s / t)
    # print(s,t)
    return th

class PatchNCELossold(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.similarity_function = self._get_similarity_function()
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def _get_similarity_function(self):

        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        return self._cosine_simililarity

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, M, C)
        # v shape: (N, M)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, feat_q, feat_k):
        batchSize = feat_q.shape[0]
        feat_k = feat_k.detach()
        l_pos = self.cos(feat_q,feat_k)
        l_pos = l_pos.view(batchSize, 1)
        l_neg_curbatch = self.similarity_function(feat_q.view(batchSize,1,-1),feat_k.view(1,batchSize,-1))
        l_neg_curbatch = l_neg_curbatch.view(1,batchSize,-1)
        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(batchSize, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, batchSize)
        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        return loss