import torch
from torch import nn
import torchvision.models as models
import math
import numpy as np
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair

class Pool(nn.Module):
    def __init__(self, patch_size=32, embed_dim=768, embedding_key='cls', prompt_init='uniform',
                 pool_size=None, top_k=None, batchwise_prompt=False, candidate_key=None, dropout_value=0.0):
        super(Pool, self).__init__()
        patch_size_pair = _pair((patch_size, patch_size))
        self.embedding_key = embedding_key
        self.top_k = top_k
        self.top_k_idx = torch.zeros(top_k)
        self.batchwise_prompt = batchwise_prompt
        self.prompt = nn.Parameter(torch.zeros(pool_size, embed_dim) * 0.02)
        self.features_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.features_dropout = nn.Dropout(dropout_value)
        self.selected_prompt = None

        if candidate_key is not None:
            self.key = nn.Parameter(candidate_key.clone().detach()) 
        else:
            self.key = nn.Parameter(torch.randn(pool_size, embed_dim))
        
        if prompt_init == 'uniform':
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size_pair, 1) + embed_dim))
            nn.init.uniform_(self.prompt.data, -val, val)
            if candidate_key is None:
                nn.init.uniform_(self.key.data, -val, val)
            nn.init.uniform_(self.features_proj.weight, -1, 1)
        else:
            raise NotImplementedError("Not supported way of prompt initial!")
                
    def add_noise_to_prompt(self, epsilon):
        sensitivity = 1.0
        noise = torch.from_numpy(
                    np.random.laplace(loc=0.0, scale=sensitivity/epsilon, size=self.prompt.size())
                ).to(self.prompt.device).type(self.prompt.dtype)
        with torch.no_grad():
            self.prompt.data.add_(noise)

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
        
    def forward(self, x_embed, cls_features=None, used_prompts_frequency=None):
        current_pool_size = self.prompt.shape[0]
        if self.embedding_key == 'cls':
            if cls_features is None:
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            else:
                x_embed_mean = cls_features
        else:
            raise NotImplementedError("Not supported way of calculating embedding keys!")
            
        prompt_norm = self.l2_normalize(self.prompt, dim=1) # Pool_size, C
        key_norm = self.l2_normalize(self.key, dim=1)
        # x_embed_mean = self.features_proj(self.features_dropout(x_embed_mean))
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C
        similarity = torch.matmul(x_embed_norm, key_norm.t())
        _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
        # print(similarity)
        if self.batchwise_prompt:
            prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
            if prompt_id.shape[0] < current_pool_size:
                prompt_id = torch.cat([prompt_id, torch.full((current_pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                id_counts = torch.cat([id_counts, torch.full((current_pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
            _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
            major_prompt_id = prompt_id[major_idx] # top_k
            # expand to batch
            idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
        
        select_method = 'topk'  
        if select_method == 'topk':
            self.top_k_idx = idx
        elif select_method == 'random':
        #randomizing prompt index 
            # Random selection of top_k prompts per sample
            # print("Random selection of prompts from the pool")
            B = x_embed.shape[0]
            P = current_pool_size
            K = self.top_k

            # Random indices in [0, P)
            rand_idx = torch.randint(
                low=0, high=P,
                size=(B, K),
                device=idx.device
            )
            # print(idx, rand_idx)
            idx = rand_idx
            self.top_k_idx = idx
        # print(self.top_k_idx)
        
        batched_prompt = self.prompt[idx] # B, top_k, C
        self.selected_prompt = batched_prompt
        if used_prompts_frequency is not None and torch.sum(used_prompts_frequency)>0:
            used_prompts_frequency = 1 - (used_prompts_frequency/torch.max(used_prompts_frequency))
            weighted_prompt_norm = key_norm * used_prompts_frequency.unsqueeze(1).to(prompt_norm.device)
            batched_key_norm = weighted_prompt_norm[idx] # B, top_k, C
        else:
            batched_key_norm = key_norm[idx]
        
        # x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
        # sim = batched_key_norm * x_embed_norm # B, top_k, C
        # reduce_sim = torch.sum(sim) / x_embed.shape[0]
        
        # per query–key cosine similarity
        cos_sim = (batched_key_norm * x_embed_norm.unsqueeze(1) ).sum(dim=-1)   # [B, top_k]
        # print(cos_sim)
        # if paper uses cosine distance:
        cos_dist = 1.0 - cos_sim
        
        # sum over keys, mean over batch
        reduce_sim = cos_dist.sum(dim=1).mean()   # scalar
        return reduce_sim, batched_prompt
        
        
class L2P_ViT_B32(nn.Module):
    def __init__(self, prompt_method, batchwise_prompt, classification_adaptor=True,
                 pool_size=None, top_k=None, frozen_pretrian=True, num_classes=10, 
                 candidate_key=None):
        super(L2P_ViT_B32, self).__init__()
        self.num_classes = num_classes
        self.trainable_keys = list()

        # self.query_model_type = query_model_type
        self.candidate_key = candidate_key

        self.vit_b32 = models.vit_b_32(weights='IMAGENET1K_V1')
        if classification_adaptor:
            self.classification_head = nn.Sequential(
                nn.Linear(self.vit_b32.heads.head.out_features, 512),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:
            self.classification_head = nn.Linear(self.vit_b32.heads.head.out_features, num_classes)
        
        hidden_size_each_layers = self.record_hidden_size_each_layers(self.vit_b32)
        self.embed_dim = hidden_size_each_layers[0]

        self.pool = Pool(embed_dim=hidden_size_each_layers[0], pool_size=pool_size, top_k=top_k, batchwise_prompt=batchwise_prompt, candidate_key=self.candidate_key)
        self.trained_prompts_checklist = torch.zeros(self.pool.prompt.shape[0], dtype=torch.float32)
            
        if frozen_pretrian:
            self.vit_b32.requires_grad_(False)
        
    def record_hidden_size_each_layers(self, origin_model):
        num_encoder_layers = len(origin_model.encoder.layers)
        hidden_size_record = list()
        for i in range(num_encoder_layers):
            for n, p in origin_model.encoder.layers[i].named_parameters():
                if 'ln_1.weight' in n:
                    hidden_size_record.append(p.shape[0])
        return hidden_size_record
    
    def get_cls_keys(self, x):
        # if self.query_model_type == 'vit_cls':
        with torch.no_grad():
            x = self.vit_b32._process_input(x)
            n = x.shape[0]
            batch_class_token = self.vit_b32.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.vit_b32.encoder(x)
            x = x[:, 0]
        return x


    def forward_features(self, x, cls_features):
        x = self.vit_b32._process_input(x)
        n = x.shape[0]
        reduce_sim, batched_prompt = self.pool(x, cls_features=cls_features)
        batch_class_token = self.vit_b32.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x += self.vit_b32.encoder.pos_embedding
        x = self.vit_b32.encoder.dropout(x)
        x = torch.cat((
            x[:, :1, :],
            batched_prompt,
            x[:, 1:, :]
        ), dim=1)
        x = self.vit_b32.encoder.dropout(x)
        x = self.vit_b32.encoder.layers(x)
        features = self.vit_b32.encoder.ln(x)
        return reduce_sim, features
        
    def forward_head(self, features):
        x = features[:, 0]
        x = self.vit_b32.heads(x)
        x = self.classification_head(x)
        return x
        
    def build_trainable_keys(self):
        grad_keys = list()
        for n, p in self.named_parameters():
            if p.requires_grad:
                grad_keys.append(n)
        self.trainable_keys = grad_keys

    def checking_trained_prompt(self):
        if self.pool.top_k_idx.device != self.trained_prompts_checklist.device:
            self.trained_prompts_checklist = self.trained_prompts_checklist.to(self.pool.top_k_idx.device)
        # print(self.trained_prompts_checklist)
        self.trained_prompts_checklist[self.pool.top_k_idx.long()] += 1.0
        
    def reset_trained_pormpts_checklist(self):
        self.trained_prompts_checklist = torch.zeros(self.pool.prompt.shape[0], dtype=torch.torch.float32)
        
    def forward(self, x):
        cls_features = self.get_cls_keys(x)
        reduce_sim, features = self.forward_features(x, cls_features=cls_features)
        logits = self.forward_head(features)
        self.checking_trained_prompt()
        return reduce_sim, logits
        
class L2P_DeiT_B16(nn.Module):
    def __init__(self, prompt_method, batchwise_prompt, classification_adaptor=True,
                 pool_size=None, top_k=None, frozen_pretrian=True, num_classes=10,
                 candidate_key=None, pretrained=True):
        super(L2P_DeiT_B16, self).__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("L2P_DeiT_B16 requires 'timm'. Install with `pip install timm`.") from e

        self.num_classes = num_classes
        self.trainable_keys = []
        self.candidate_key = candidate_key

        # Backbone: DeiT-Base, patch16, 224x224
        self.deit = timm.create_model(
            'deit_base_patch16_224',
            pretrained=pretrained,
            num_classes=0,   # we add our own head
            global_pool=''   # keep sequence output
        )
        self.embed_dim = self.deit.embed_dim  # 768

        # Prompt pool
        self.pool = Pool(embed_dim=self.embed_dim, pool_size=pool_size, top_k=top_k,
                         batchwise_prompt=batchwise_prompt, candidate_key=self.candidate_key)

        # Classification head (same pattern as your ViT class)
        if classification_adaptor:
            self.classification_head = nn.Sequential(
                nn.Linear(self.embed_dim, 512),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:
            self.classification_head = nn.Linear(self.embed_dim, num_classes)

        if frozen_pretrian:
            self.deit.requires_grad_(False)

        self.trained_prompts_checklist = torch.zeros(self.pool.prompt.shape[0], dtype=torch.float32)

    # ---- CLS features used as the "key" for selection ----
    def get_cls_keys(self, x):
        with torch.no_grad():
            B = x.shape[0]
            x_tok = self.deit.patch_embed(x)                   # [B, N, C]
            cls = self.deit.cls_token.expand(B, -1, -1)        # [B, 1, C]
            x_seq = torch.cat((cls, x_tok), dim=1)             # [B, 1+N, C]
            x_seq = x_seq + self.deit.pos_embed                # pos for cls+patches
            x_seq = self.deit.pos_drop(x_seq)
            for blk in self.deit.blocks:
                x_seq = blk(x_seq)
            x_seq = self.deit.norm(x_seq)
            cls_feat = x_seq[:, 0]                             # [B, 768]
        return cls_feat

    # ---- Features with prompt injection ----
    def forward_features(self, x, cls_features):
        B = x.shape[0]
        # Pool() uses cls_features; token tensor is only a placeholder
        x_tok = self.deit.patch_embed(x)                       # [B, N, C]
        reduce_sim, batched_prompt = self.pool(x_tok, cls_features=cls_features)

        # Build input: [CLS] + prompts + patches
        cls = self.deit.cls_token.expand(B, -1, -1)            # [B, 1, C]
        x_seq = torch.cat((cls, x_tok), dim=1)                 # [B, 1+N, C]
        x_seq = x_seq + self.deit.pos_embed                    # (no pos on prompts)
        x_seq = self.deit.pos_drop(x_seq)

        x_seq = torch.cat((
            x_seq[:, :1, :],            # CLS
            batched_prompt,             # [B, top_k, C]
            x_seq[:, 1:, :]             # patches
        ), dim=1)

        for blk in self.deit.blocks:
            x_seq = blk(x_seq)
        features = self.deit.norm(x_seq)                       # [B, 1+top_k+N, C]
        return reduce_sim, features

    def forward_head(self, features):
        x = features[:, 0]                                     # CLS
        return self.classification_head(x)

    def build_trainable_keys(self):
        self.trainable_keys = [n for n, p in self.named_parameters() if p.requires_grad]

    def checking_trained_prompt(self):
        if self.pool.top_k_idx.device != self.trained_prompts_checklist.device:
            self.trained_prompts_checklist = self.trained_prompts_checklist.to(self.pool.top_k_idx.device)
        self.trained_prompts_checklist[self.pool.top_k_idx.long()] += 1.0

    def reset_trained_pormpts_checklist(self):
        self.trained_prompts_checklist = torch.zeros(self.pool.prompt.shape[0], dtype=torch.float32)

    def forward(self, x):
        cls_features = self.get_cls_keys(x)                    # [B, 768]
        reduce_sim, features = self.forward_features(x, cls_features=cls_features)
        logits = self.forward_head(features)                   # [B, num_classes]
        self.checking_trained_prompt()
        return reduce_sim, logits

class L2P_CaiT_S24(nn.Module):
    """
    L2P-style wrapper for timm's CaiT (e.g., 'cait_s24_224').
    """
    def __init__(self, prompt_method, batchwise_prompt, classification_adaptor=True,
                 pool_size=None, top_k=None, frozen_pretrian=True, num_classes=10,
                 candidate_key=None, model_name='cait_s24_224.fb_dist_in1k', pretrained=True):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("L2P_CaiT_S24 requires 'timm'. Install with `pip install timm`.") from e

        self.num_classes = num_classes
        self.candidate_key = candidate_key

        # Backbone: CaiT (flat transformer with CLS token)
        self.cait = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=''
        )
        self.embed_dim = self.cait.embed_dim  # 384/768 depending on variant

        # Prompt pool (operate at token dim)
        self.pool = Pool(embed_dim=self.embed_dim, pool_size=pool_size, top_k=top_k,
                         batchwise_prompt=batchwise_prompt, candidate_key=self.candidate_key)

        # Head (same pattern as yours)
        if classification_adaptor:
            self.classification_head = nn.Sequential(
                nn.Linear(self.embed_dim, 512),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:
            self.classification_head = nn.Linear(self.embed_dim, num_classes)

        if frozen_pretrian:
            self.cait.requires_grad_(False)

        self.trained_prompts_checklist = torch.zeros(self.pool.prompt.shape[0], dtype=torch.float32)
    
    @torch.no_grad()
    def get_cls_keys(self, x: torch.Tensor) -> torch.Tensor:
        """
        Robust CLS-like feature for CaiT:
        - Handles pos_embed that is patch-only [1, N, C] or includes CLS [1, 1+N, C]
        - Runs main transformer blocks and class-attention blocks (if present)
        - Returns [B, C] CLS embedding
        """
        B = x.size(0)

        # Patch tokens
        x_tok = self.cait.patch_embed(x)                 # [B, N, C]
        N = x_tok.shape[1]
        cls = self.cait.cls_token.expand(B, -1, -1)      # [B, 1, C]

        # Positional embeddings: robust to either style
        pe = getattr(self.cait, "pos_embed", None)       # [1, M, C] or None
        if pe is not None:
            if pe.shape[1] == N:                         # patch-only pe
                x_tok = x_tok + pe
                x_seq = torch.cat((cls, x_tok), dim=1)   # [B, 1+N, C]
            elif pe.shape[1] == N + 1:                   # pe includes CLS
                cls   = cls + pe[:, :1, :]
                x_tok = x_tok + pe[:, 1:, :]
                x_seq = torch.cat((cls, x_tok), dim=1)   # [B, 1+N, C]
            else:
                raise ValueError(f"Unexpected pos_embed length {pe.shape[1]} for N={N}")
        else:
            # No positional embedding provided
            x_seq = torch.cat((cls, x_tok), dim=1)       # [B, 1+N, C]

        # Positional dropout if present
        pos_drop = getattr(self.cait, "pos_drop", None)
        if isinstance(pos_drop, torch.nn.Module):
            x_seq = pos_drop(x_seq)

        # Run standard transformer blocks
        for blk in getattr(self.cait, "blocks", []):
            x_seq = blk(x_seq)

        # Run CaiT class-attention blocks (naming differs across timm versions)
        # Try common attribute names in order.
        # for attn_stack_name in ("attn_blocks", "blocks_token_only", "cls_blocks"):
        #     attn_stack = getattr(self.cait, attn_stack_name, None)
        #     if isinstance(attn_stack, (list, torch.nn.ModuleList)):
        #         for blk in attn_stack:
        #             x_seq = blk(x_seq)

        # Final norm
        x_seq = self.cait.norm(x_seq)  

        # Return CLS embedding
        return x_seq[:, 0]                                # [B, C]
                              # [B, C]

    def forward_features(self, x, cls_features):
        B = x.shape[0]
        x_tok = self.cait.patch_embed(x)                               # [B, N, C]
        reduce_sim, batched_prompt = self.pool(x_tok, cls_features=cls_features)

        # Prepare CLS and handle positional embeddings robustly
        cls = self.cait.cls_token.expand(B, -1, -1)                    # [B, 1, C]
        pe  = getattr(self.cait, "pos_embed", None)                    # [1, M, C] or None
        N   = x_tok.shape[1]

        if pe is not None:
            if pe.shape[1] == N:                                       # patch-only pe
                x_tok = x_tok + pe                                     # [B, N, C]
                x_seq = torch.cat((cls, x_tok), dim=1)                 # [B, 1+N, C]
            elif pe.shape[1] == N + 1:                                 # pe includes CLS
                cls   = cls + pe[:, :1, :]                             # [B, 1, C]
                x_tok = x_tok + pe[:, 1:, :]                           # [B, N, C]
                x_seq = torch.cat((cls, x_tok), dim=1)                 # [B, 1+N, C]
            else:
                raise ValueError(f"Unexpected pos_embed length {pe.shape[1]} for N={N}")
        else:
            x_seq = torch.cat((cls, x_tok), dim=1)                     # [B, 1+N, C]

        # Positional dropout (if present)
        pos_drop = getattr(self.cait, "pos_drop", None)
        if isinstance(pos_drop, torch.nn.Module):
            x_seq = pos_drop(x_seq)

        # Insert prompts right after CLS
        x_seq = torch.cat((x_seq[:, :1, :], batched_prompt, x_seq[:, 1:, :]), dim=1)  # [B, 1+top_k+N, C]
        # print(self.cait)
        # Main transformer blocks (single-tensor API)
        for blk in getattr(self.cait, "blocks", []):
            x_seq = blk(x_seq)

        # # Split for class-attention blocks (two-tensor API: (tokens, cls) -> (tokens, cls))
        x_cls = x_seq[:, :1, :]                                        # [B, 1, C]
        x_tok = x_seq[:, 1:, :]                                        # [B, top_k+N, C]

        for attn_stack_name in ("blocks_token_only", "attn_blocks" , "cls_blocks"):
            attn_stack = getattr(self.cait, attn_stack_name, None)
            if isinstance(attn_stack, (list, torch.nn.ModuleList)):
                for blk in attn_stack:
                    # print("hey")
                    x_seq = blk(x_tok, x_cls)

        # Reassemble and final norm
        # x_seq = torch.cat((x_cls, x_tok), dim=1)                       # [B, 1+top_k+N, C]
        features = self.cait.norm(x_seq)                               # [B, 1+top_k+N, C]
        return reduce_sim, features

    def forward_head(self, features):
        x = features[:, 0]                                 # CLS
        return self.classification_head(x)

    def build_trainable_keys(self):
        self.trainable_keys = [n for n, p in self.named_parameters() if p.requires_grad]

    def checking_trained_prompt(self):
        if self.pool.top_k_idx.device != self.trained_prompts_checklist.device:
            self.trained_prompts_checklist = self.trained_prompts_checklist.to(self.pool.top_k_idx.device)
        self.trained_prompts_checklist[self.pool.top_k_idx.long()] += 1.0

    def reset_trained_pormpts_checklist(self):
        self.trained_prompts_checklist = torch.zeros(self.pool.prompt.shape[0], dtype=torch.float32)

    def forward(self, x):
        cls_features = self.get_cls_keys(x)                  # [B, C]
        reduce_sim, features = self.forward_features(x, cls_features=cls_features)
        logits = self.forward_head(features)
        self.checking_trained_prompt()
        return reduce_sim, logits
    
class L2P_ConViT_Base(nn.Module):
    """
    Simple L2P wrapper for timm's ConViT-Base (e.g., 'convit_base.fb_in1k').
    """
    def __init__(self, prompt_method, batchwise_prompt, classification_adaptor=True,
                 pool_size=None, top_k=None, frozen_pretrian=True, num_classes=10,
                 candidate_key=None, model_name='convit_base.fb_in1k', pretrained=True):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("L2P_ConViT_Base requires 'timm'. Install with `pip install timm`.") from e

        self.num_classes = num_classes
        self.candidate_key = candidate_key

        # Backbone
        try:
            self.convit = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, global_pool=''
            )
        except Exception:
            # fallback if tagged variant isn't in your timm
            base_name = model_name.split('.')[0]
            self.convit = timm.create_model(
                base_name, pretrained=pretrained, num_classes=0, global_pool=''
            )

        self.embed_dim = getattr(self.convit, 'embed_dim', self.convit.num_features)

        # Prompt pool (token dim)
        self.pool = Pool(embed_dim=self.embed_dim, pool_size=pool_size, top_k=top_k,
                         batchwise_prompt=batchwise_prompt, candidate_key=self.candidate_key)

        # Head (same pattern as your DeiT)
        if classification_adaptor:
            self.classification_head = nn.Sequential(
                nn.Linear(self.embed_dim, 512),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:
            self.classification_head = nn.Linear(self.embed_dim, num_classes)

        if frozen_pretrian:
            self.convit.requires_grad_(False)

        self.trained_prompts_checklist = torch.zeros(self.pool.prompt.shape[0], dtype=torch.float32)

    # --- tiny helper to keep pos-embed handling simple & robust ---
    def _add_pos(self, x_seq, x_tok, cls):
        pe = getattr(self.convit, 'pos_embed', None)
        if pe is None:
            return x_seq
        # If PE length matches [CLS]+patches, add directly
        if pe.shape[1] == x_seq.shape[1]:
            return x_seq + pe
        # If PE is patch-only, add to patches then re-cat CLS
        if pe.shape[1] == x_tok.shape[1]:
            return torch.cat((cls, x_tok + pe), dim=1)
        # Otherwise, be explicit (keeps behavior clear & simple)
        raise ValueError(f"Unexpected pos_embed length {pe.shape[1]} for sequence/patch N={x_seq.shape[1]}/{x_tok.shape[1]}")

    # ---- CLS features used as the "key" for selection ----
    @torch.no_grad()
    def get_cls_keys(self, x: torch.Tensor) -> torch.Tensor:
        """Match timm ConViT forward: GPSA on patches only; add CLS at transition."""
        B = x.shape[0]
        x_tok = self.convit.patch_embed(x)                            # [B, N, C]
        if hasattr(self.convit, 'pos_embed'):
            x_tok = x_tok + self.convit.pos_embed                     # patch-only PE
        if hasattr(self.convit, 'pos_drop'):
            x_tok = self.convit.pos_drop(x_tok)

        cls = self.convit.cls_token.expand(B, -1, -1)                 # [B, 1, C]
        x_seq = x_tok
        for u, blk in enumerate(self.convit.blocks):
            if u == getattr(self.convit, 'local_up_to_layer', 0):
                x_seq = torch.cat((cls, x_seq), dim=1)                # add CLS at transition
            x_seq = blk(x_seq)
        x_seq = self.convit.norm(x_seq)
        return x_seq[:, 0]                                            # CLS [B, C]

    def forward_features(self, x: torch.Tensor, cls_features: torch.Tensor):
        """Select prompts from CLS, inject at the same transition point as CLS."""
        B = x.shape[0]
        x_tok = self.convit.patch_embed(x)                            # [B, N, C]
        reduce_sim, batched_prompt = self.pool(x_tok, cls_features=cls_features)

        if hasattr(self.convit, 'pos_embed'):
            x_tok = x_tok + self.convit.pos_embed                     # patch-only PE
        if hasattr(self.convit, 'pos_drop'):
            x_tok = self.convit.pos_drop(x_tok)

        cls = self.convit.cls_token.expand(B, -1, -1)
        x_seq = x_tok
        for u, blk in enumerate(self.convit.blocks):
            if u == getattr(self.convit, 'local_up_to_layer', 0):
                # insert CLS and prompts exactly once, at transition to MHSA
                x_seq = torch.cat((cls, batched_prompt, x_seq), dim=1)  # [B, 1+top_k+N, C]
            x_seq = blk(x_seq)

        features = self.convit.norm(x_seq)                            # [B, 1+top_k+N, C]
        return reduce_sim, features

    def forward_head(self, features):
        return self.classification_head(features[:, 0])      # CLS

    def build_trainable_keys(self):
        self.trainable_keys = [n for n, p in self.named_parameters() if p.requires_grad]

    def checking_trained_prompt(self):
        if self.pool.top_k_idx.device != self.trained_prompts_checklist.device:
            self.trained_prompts_checklist = self.trained_prompts_checklist.to(self.pool.top_k_idx.device)
        self.trained_prompts_checklist[self.pool.top_k_idx.long()] += 1.0

    def reset_trained_pormpts_checklist(self):
        self.trained_prompts_checklist = torch.zeros(self.pool.prompt.shape[0], dtype=torch.float32)

    def forward(self, x):
        cls_features = self.get_cls_keys(x)                  # [B, C]
        reduce_sim, features = self.forward_features(x, cls_features=cls_features)
        logits = self.forward_head(features)
        self.checking_trained_prompt()
        return reduce_sim, logits