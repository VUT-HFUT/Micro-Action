import torch.nn.functional as F
import torch
import torch.nn as nn
from ..builder import LOSSES

# def nll_loss(output, target):
#     return F.nll_loss(output, target)


# def bce_loss(output, target):

#     t = target.clone().detach()

#     t[t >= 0.5] = 1  # threshold to get binary labels
#     t[t < 0.5] = 0

#     loss = F.binary_cross_entropy_with_logits(output, t)
#     return loss

# def combined_loss(output, target):
#     l = F.mse_loss(output, target)

#     l += bce_loss(output, target)

#     return l

# def mse_loss(output, target):
# 	return F.mse_loss(output, target)




# def mse_center_loss(output, target, labels):
#     t = labels.clone().detach()
#     t[t >= 0.5] = 1  # threshold to get binary labels
#     t[t < 0.5] = 0
#     #output (8,300)
#     target = target[0,:26] #(26,300)

#     positive_centers = []
#     for i in range(output.size(0)):
#         p = target[t[i, :] == 1]
#         if p.size(0) == 0:
#             positive_center = torch.zeros(300).cuda()
#         else:
#             positive_center = torch.mean(p, dim=0)

#         positive_centers.append(positive_center)

#     positive_centers = torch.stack(positive_centers,dim=0)
#     loss = F.mse_loss(output, positive_centers)
  
#     return loss





def mse_center_loss(output, target, labels):
    #(out['embed'], embeddings, target)
    t = labels.clone().detach()
    t[t >= 0.5] = 1  # threshold to get binary labels
    t[t < 0.5] = 0


    positive_centers = []
    for i in range(output.size(0)):
        p = target[i]
        if p.size(0) == 0:
            positive_center = torch.zeros(300).cuda()
        else:
            # positive_center = torch.mean(p, dim=0)
            positive_center = p
        positive_centers.append(positive_center)

    positive_centers = torch.stack(positive_centers,dim=0)
    #在dim=0上堆积
    loss = F.mse_loss(output, positive_centers)
  
    return loss


@LOSSES.register_module()
class MseLoss(nn.Module):
    def forward(self, output, target, labels):
        loss = mse_center_loss(output, target, labels)
        return loss