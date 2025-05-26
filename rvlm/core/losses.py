import torch
from torch import nn
from torch.nn import functional as F

def js_divergence(x, y, tau=0.01, eps=1e-8):
    p = (x + 1) / 2
    q = (y + 1) / 2
    p = p / (p.sum(dim=-1, keepdim=True) + eps)
    q = q / (q.sum(dim=-1, keepdim=True) + eps)

    # p = F.softmax(x / tau, dim=-1)
    # q = F.softmax(y / tau, dim=-1)
    m = 0.5 * (p + q)
    return - (0.5 * F.kl_div(m.log(), p, reduction='batchmean') \
         + 0.5 * F.kl_div(m.log(), q, reduction='batchmean'))
    
def compute_entropy(x, tau=1.0, eps=1e-8):
    p = (x + 1) / 2
    p = p / (p.sum(dim=-1, keepdim=True) + eps)
    #p = F.softmax(x / tau, dim=-1)
    return -(p * p.log()).sum(dim=-1).mean()

class OrthoFeatLoss(nn.Module):
    def __init__(self, ortho_pretrained=False):
        super().__init__()
        self.ortho_pretrained = ortho_pretrained

    def forward(self, features):

        keys = list(features.keys())
        keys = [k for k in features if k in {'out', 'out_lora'}]

        cos_sim = F.cosine_similarity(features['out'], features['out_lora'], dim=-1)
        return 1.0 - cos_sim.mean()

# class OrthoFeatLoss(nn.Module):
#     def __init__(self, ortho_pretrained=False):
#         super().__init__()
#         self.ortho_pretrained = ortho_pretrained

#     def forward(self, features):

#         keys = list(features.keys())


#         keys.remove('out')
#         if not self.ortho_pretrained:
#             keys.remove('org')
#         num_keys = len(keys)

#         feature_tensors = torch.stack([features[k] for k in keys]) # num_keys, token, batch, feat_dim
#         cosine_sim_matrix = F.cosine_similarity(feature_tensors.unsqueeze(1), feature_tensors.unsqueeze(0), dim=-1).pow(2)
#         i, j = torch.triu_indices(num_keys, num_keys, offset=1)
#         loss = cosine_sim_matrix[i, j].mean()
        
#         return loss

# class OrthoFeatLoss(nn.Module):
#     def __init__(self, pairs, args, save_dist=False):
#         super().__init__()
#         if args.kl: self.loss_fn = F.kl_div
#         else: self.loss_fn = F.cosine_similarity
#         self.pairs = pairs
#         self.tau = 1.0
#         self.eps = 1e-8
#         self.args = args
#         self.save_dist = save_dist
#         self.dists = []

#     def forward(self, features):
#         if self.save_dist: self.dists.append({k:v.detach().cpu() for k,v in features.items()})
        

#         if self.args.loss_gram:
#             features = [v for k,v in features.items()]
#             features = torch.stack(features, dim=-2)

#             if features.dim() == 4:  # q,k,v (Token, batch, Lora, D)
#                 T, B, L, D = features.shape
#                 features = features.view(-1, L, D)  # Shape: (Token x batch, Lora, D)
#             elif features.dim() == 3:  # out (Token x batch, Lora, D)
#                 _, L, D = features.shape
#                 pass
#             else:
#                 raise ValueError(f"Unexpected shape for features: {features.shape}")

#             features = features / (torch.norm(features, dim=-1, keepdim=True) + 1e-8)
            
#             gram_matrices = torch.bmm(features, features.transpose(1, 2))  # Shape: (batch, N, N)
#             gram_matrices_lower = torch.tril(gram_matrices)

#             identity = torch.eye(L, device=features.device).unsqueeze(0)  # Shape: (1, L, L)
#             identity_lower = torch.tril(identity)

#             loss = torch.norm(gram_matrices_lower - identity_lower, dim=(1, 2))**2  # Frobenius norm for each sample

#         else:
#             features1 = torch.stack([features[idx1] for idx1, _ in self.pairs])
#             features2 = torch.stack([features[idx2] for _, idx2 in self.pairs])
#             all_features = torch.stack([features[i] for i in range(self.args.num_lora)])

#             if self.args.dot:
#                 loss = (features1 * features2).sum(dim=-1).mean(dim=-1)
#             elif self.args.kl:
#                 loss = js_divergence(features1, features2, self.tau)
#                 #loss = (-1) * self.loss_fn(F.log_softmax(features1/self.tau, dim=-1), F.softmax(features2/self.tau, dim=-1), reduction="batchmean")
#             elif self.args.feat_kk:
#                 features1 = features1.view(-1, features1.shape[-1])
#                 features2 = features2.view(-1, features1.shape[-1])
#                 normed1 = F.normalize(features1, p=2, dim=1)
#                 normed2 = F.normalize(features2, p=2, dim=1)
#                 cross_mat = normed1.transpose(0, 1) @ normed2 # shape (k, k)
#                 # batch mean
#                 cross_mat = cross_mat / normed1.shape[0]
#                 loss = (cross_mat ** 2).sum()          
#             else:
#                 loss = F.cosine_similarity(features1, features2, dim=-1).pow(2)

#         loss = loss.mean()
        
#         if self.args.entropy:
#             loss += self.args.lambda_reg * compute_entropy(all_features, tau=self.tau)
#         elif self.args.l1:
#             loss += self.args.lambda_reg * all_features.abs().sum(dim=-1).mean()
        
#         return loss

class OrthoParamLoss(nn.Module):
    def __init__(self, pairs, compare_org=False):
        super().__init__()
        self.pairs = pairs
        self.compare_org = compare_org
    
    def forward(self, params, org_params=None):
        loss = 0
        for idx1, idx2 in self.pairs:
            params1, params2 = params[idx1], params[idx2]
            for param1, param2 in zip(params1, params2):
                loss += torch.abs(torch.mm(param1, param2.T)).sum()
        if self.compare_org:
            for lora_params in params.values():
                for lora_param, org_param in zip(lora_params, org_params):
                    loss += torch.abs(torch.mm(org_param, lora_param.T)).sum()
        return loss
