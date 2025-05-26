from torch import nn
from tqdm import tqdm

from .clip import *
from .model import *
from data.templates import imagenet_templates
from ..losses import OrthoFeatLoss
from .. import loralib

class CLIP_FT(nn.Module):
    def __init__(self, model_arch, device, classnames, n_cls=2, text_cls=False, freeze_encoder=True, cl=False):
        super().__init__()
        
        self.model, self.preprocess, self.val_preprocess = load(model_arch, device=device)
        self.model.float()      # mixed precision -> underflow/overflow in optimizer.step()
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
        self.n_layers = self.model.visual.transformer.layers
        self.cl = cl


        self.model.eval()
        if not text_cls or not self.cl:
            logit_scale = self.model.logit_scale
            with torch.no_grad():
                zeroshot_weights = []
                for classname in tqdm(classnames):
                    texts = []
                    for t in imagenet_templates:
                        texts.append(t.format(classname))
                    texts = clip.tokenize(texts).to(device)  # tokenize
                    embeddings = self.model.encode_text(
                        texts)  # embed with text encoder
                    embeddings /= embeddings.norm(dim=-1, keepdim=True)

                    embeddings = embeddings.mean(dim=0, keepdim=True)
                    embeddings /= embeddings.norm()

                    zeroshot_weights.append(embeddings)

                zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
                zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

                zeroshot_weights *= logit_scale.exp()

                zeroshot_weights = zeroshot_weights.squeeze().float()
                zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)
            
            self.fc = ClassificationHead(normalize=True, weights=zeroshot_weights).to(device)
            for param in self.fc.parameters():
                param.requires_grad = True
        # self.model = self.model.visual
        self.model.train()
    
        self.text_cls = text_cls
        self.device = device
        
    def set_desc_loss_fn(self, target_layers, desc_emb, loss_fn=None):
        self.model.desc_emb = desc_emb
        self.model.loss_fn = loss_fn
        self.model.target_layers = target_layers
    
    def forward(self, x, text=None):
        if self.cl:
            if x is None and text is not None:
                text_features = self.model(x, text)
                return text_features
            elif text is None and x is not None:
                image_features, desc_loss, feat_loss = self.model(x, text)
                return image_features, desc_loss, feat_loss
            else:    
                logits_per_image, logits_per_text, desc_loss, feat_loss, cl_loss = self.model(x, text)
                return logits_per_image, logits_per_text, desc_loss, feat_loss, cl_loss
        else:
            out, desc_loss, feat_loss = self.model(x)
            if self.text_cls:
                return out, desc_loss, feat_loss
            else:
                return self.fc(out), desc_loss, feat_loss


class ProjectionHead(torch.nn.Linear):
    def __init__(self, weights, normalize=False):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size, bias=True)
        
        self.normalize = normalize
        self.weight = torch.nn.Parameter(weights.clone())
        self.bias.data.zero_()

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)


def pca(X, num_components):
    X = X - X.mean(dim=0, keepdim=True)  
    cov_matrix = X.T @ X / (X.shape[0] - 1) 
    eigvals, eigvecs = torch.linalg.eigh(cov_matrix) 
    principal_components = eigvecs[:, -num_components:]
    return principal_components


class CLIP_FT_desc(nn.Module):
    def __init__(self, args, device, class_descs, freeze_encoder=True):
        super().__init__()
        
        self.device = device
        self.model, self.preprocess, self.val_preprocess = load(args.arch, device=device)
        self.model.float()      # mixed precision -> underflow/overflow in optimizer.step()
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
        self.n_layers = self.model.visual.transformer.layers

        self.model.eval()
        self.logit_scale = self.model.logit_scale.exp()
        # with torch.no_grad():
        #     desc_embs = {cls_name: [] for cls_name in class_descs}
        #     for cls_name, descs in tqdm(class_descs.items()):
        #         for desc in descs:
        #             #texts = [t.format(desc) for t in imagenet_templates]
        #             texts = clip.tokenize(desc).to(device)
        #             embeddings = self.model.encode_text(texts)
        #             embeddings /= embeddings.norm(dim=-1, keepdim=True)
        #             desc_embs[cls_name].append(embeddings)
            
        #     desc_embs = {k: torch.stack(v, dim=0).to(device) for k,v in desc_embs.items()}
        #     desc_embs = {k: v.squeeze().float() for k,v in desc_embs.items()}
        # self.desc_embs = desc_embs

        # if args.pca_perCls:
        #     desc_embs_pca = {}
        #     for k, v in desc_embs.items():
        #         desc_embs_pca[k] = pca(v, args.n_pca)
        #     #TODO
        # else:
        #     all_embs = torch.cat([v for v in desc_embs.values()], dim=0)
        #     all_embs_pca = pca(all_embs, args.n_pca)
        #     all_embs_pca = all_embs_pca.detach()
        #     self.model.visual.desc_proj =  ProjectionHead(all_embs_pca @ all_embs_pca.T)
            
        self.model.train()
    
    # def set_pcaOrtho(self, loss_fn):
    #     self.model.visual.ortho_lossfn = loss_fn

    def set_lora_scaling(self, scaling: float):
        for name, module in self.named_modules():
            if isinstance(module, loralib.LoRAInjectedLinear):
                module.scaling = scaling
            elif isinstance(module, loralib.LoRAInjectedMultiheadAttention):
                for subname, submodule in module.named_modules():
                    if isinstance(submodule, loralib.LoRAInjectedLinear):
                        submodule.scaling = scaling


    def forward(self, x=None, text=None, return_dict=True):
        if x is not None and text is not None:
            image_features, text_features = self.model(x, text, return_dict=return_dict)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
  
            return image_features, text_features, self.logit_scale #, feat_loss

        if  x is None and text is not None:
            text_features = self.model(x, text)
            return text_features / text_features.norm(dim=-1, keepdim=True)
        
        if  x is not None and text is None:
            image_features = self.model(x, text)
            return image_features / image_features.norm(dim=-1, keepdim=True) #, feat_loss





class CLIP_FT_tmp(nn.Module):
    def __init__(self, model_arch, device, n_cls=2):
        super().__init__()
        
        self.model, self.preprocess, self.val_preprocess = load(model_arch, device=device)
        self.n_layers = self.model.transformer.layers
        #delattr(self.model, 'transformer')
        self.model.float()      # mixed precision -> underflow/overflow in optimizer.step()

        # for param in self.model.parameters():
        #     param.requires_grad = False
        
        self.fc = nn.Linear(self.model.output_dim, n_cls)
        self.fc.to(device)
        for param in self.fc.parameters():
            param.requires_grad = True
        
        self.device = device
        
    def set_desc_loss_fn(self, target_layers, desc_emb, loss_fn=None):
        self.model.desc_emb = desc_emb
        self.model.loss_fn = loss_fn
        self.model.target_layers = target_layers
    
    def forward(self, x):
        out, desc_loss, feat_loss = self.model(x)
        if isinstance(desc_loss, list):
            desc_loss = torch.stack(desc_loss).mean()
        if isinstance(feat_loss, list):
            feat_loss = torch.stack(feat_loss).mean()
        return self.fc(out), desc_loss, feat_loss


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None, shape=[512, 1000]):
        if weights is not None:
            output_size, input_size = weights.shape
            super().__init__(input_size, output_size)
        else:
            super().__init__(shape[0], shape[1])
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())

        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)


class CLIP_FT_new(nn.Module):
    def __init__(self, model_arch, device, classnames, freeze_encoder=True):
        super().__init__()
        
        self.model, self.preprocess, self.val_preprocess = load(model_arch, device=device)
        self.model.float()      # mixed precision -> underflow/overflow in optimizer.step()
        self.model.eval()
        
        logit_scale = self.model.logit_scale
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = []
                for t in imagenet_templates:
                    texts.append(t.format(classname))
                texts = clip.tokenize(texts).to(device)  # tokenize
                embeddings = self.model.encode_text(
                    texts)  # embed with text encoder
                embeddings /= embeddings.norm(dim=-1, keepdim=True)

                embeddings = embeddings.mean(dim=0, keepdim=True)
                embeddings /= embeddings.norm()

                zeroshot_weights.append(embeddings)

            zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
            zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

            zeroshot_weights *= logit_scale.exp()

            zeroshot_weights = zeroshot_weights.squeeze().float()
            zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)
        
        self.fc = ClassificationHead(normalize=True, weights=zeroshot_weights).to(device)
        for param in self.fc.parameters():
            param.requires_grad = True

        self.n_layers = self.model.transformer.layers
        delattr(self.model, 'transformer')
        self.model.train()
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.device = device
        
    def set_desc_loss_fn(self, target_layers, desc_emb, loss_fn=None):
        self.model.desc_emb = desc_emb
        self.model.loss_fn = loss_fn
        self.model.target_layers = target_layers
    
    def forward(self, x):
        out, desc_loss, feat_loss = self.model.visual(x)
        if isinstance(desc_loss, list):
            desc_loss = torch.stack(desc_loss).mean()
        if isinstance(feat_loss, list):
            feat_loss = torch.stack(feat_loss).mean()
        return self.fc(out), desc_loss, feat_loss