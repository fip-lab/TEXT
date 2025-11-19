import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn, einsum
from torch.nn import functional as F
from einops import rearrange, repeat



def pair(t): 
    return t if isinstance(t, tuple) else (t, t) 


####################### Transformer #######################
####################### Transformer #######################
####################### Transformer #######################

class PreNormAttention(nn.Module):  
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)  
        self.norm_k = nn.LayerNorm(dim) 
        self.norm_v = nn.LayerNorm(dim)  
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        q = self.norm_q(q)  
        k = self.norm_k(k)  
        v = self.norm_v(v)  

        return self.fn(q, k, v)
    

class Attention(nn.Module):  
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):

        super().__init__()
        inner_dim = dim_head *  heads  
        project_out = not (heads == 1 and dim_head == dim) 
        self.heads = heads  
        self.scale = dim_head ** -0.5  

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential( 
            nn.Linear(inner_dim, dim), 
            nn.Dropout(dropout)  
        ) if project_out else nn.Identity()  
   

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        q = self.to_q(q)  
        k = self.to_k(k)    
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))  
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  

        attn = self.attend(dots)  

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)') 

        return self.to_out(out)


class PreNormForward(nn.Module):
    def __init__(self, dim, fn):

        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)
    

class TransformerEncoder(nn.Module):  
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):

        super().__init__()
        self.layers = nn.ModuleList([])  
        for _ in range(depth):          
            self.layers.append(nn.ModuleList([  
                PreNormAttention(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),  
                PreNormForward(dim, FeedForward(dim, mlp_dim, dropout = dropout))  
            ]))

    def forward(self, x, save_hidden=False):
        if save_hidden == True:
            hidden_list = []
            hidden_list.append(x)       
            for attn, ff in self.layers:  
                x = attn(x, x, x) + x  
                x = ff(x) + x  
                hidden_list.append(x)  
            return hidden_list
        else:
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
            return x
        


class Transformer(nn.Module):
    def __init__(self, *, num_frames, token_len, save_hidden, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):

        super().__init__()

        self.token_len = token_len  
        self.save_hidden = save_hidden  

        if token_len is not None:  
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + token_len, dim))
            self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim)) 
        else:
             self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim)) 
             self.extra_token = None 

        self.dropout = nn.Dropout(emb_dropout) 

        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout) 

        self.pool = pool
        self.to_latent = nn.Identity()


    def forward(self, x):
        b, n, _ = x.shape

        if self.token_len is not None:
            extra_token = repeat(self.extra_token, '1 n d -> b n d', b = b)  
            x = torch.cat((extra_token, x), dim=1) 
            x = x + self.pos_embedding[:, :n+self.token_len]  
        else:
            x = x + self.pos_embedding[:, :n]  

        x = self.dropout(x) 
        x = self.encoder(x, self.save_hidden) 

        return x
    


            # ***************** Cross Transformer *****************
            # ***************** Cross Transformer *****************
            
            # ***************** Cross Transformer *****************
            # ***************** Cross Transformer *****************

            # ***************** Cross Transformer *****************
            # ***************** Cross Transformer *****************
            
            # ***************** Cross Transformer *****************
            # ***************** Cross Transformer *****************

class CrossTransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAttention(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNormForward(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, source_x, target_x):
        for attn, ff in self.layers:
            target_x_tmp = attn(target_x, source_x, source_x)
            target_x = target_x_tmp + target_x
            target_x = ff(target_x) + target_x
        return target_x



class CrossTransformer(nn.Module):
    def __init__(self, *, source_num_frames, tgt_num_frames, token_len, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.token_len = token_len

        if token_len is not None:
            self.pos_embedding_s = nn.Parameter(torch.randn(1, source_num_frames + token_len, dim))
            self.pos_embedding_t = nn.Parameter(torch.randn(1, tgt_num_frames + token_len, dim))
            self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
        else:
            self.pos_embedding_s = nn.Parameter(torch.randn(1, source_num_frames, dim))
            self.pos_embedding_t = nn.Parameter(torch.randn(1, tgt_num_frames, dim))
            self.extra_token = None


        self.dropout = nn.Dropout(emb_dropout)
        self.CrossTransformerEncoder = CrossTransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, source_x, target_x):
        b, n_s, _ = source_x.shape  
        b, n_t, _ = target_x.shape 

        if self.token_len is not None:
            extra_token = repeat(self.extra_token, '1 n d -> b n d', b = b)

            source_x = torch.cat((extra_token, source_x), dim=1)
            source_x = source_x + self.pos_embedding_s[:, : n_s+self.token_len]

            target_x = torch.cat((extra_token, target_x), dim=1)
            target_x = target_x + self.pos_embedding_t[:, : n_t+self.token_len]

        else:
            source_x = source_x + self.pos_embedding_s[:, : n_s]
            target_x = target_x + self.pos_embedding_t[:, : n_t]

        source_x = self.dropout(source_x)
        target_x = self.dropout(target_x)

        x_s2t = self.CrossTransformerEncoder(source_x, target_x)

        return x_s2t
    
            # ***************** Cross Transformer *****************
            # ***************** Cross Transformer *****************


    
####################### Transformer #######################
####################### Transformer #######################
####################### Transformer #######################





####################### LanguageRouterMoeTransformer #######################
####################### LanguageRouterMoeTransformer #######################
####################### LanguageRouterMoeTransformer #######################
####################### LanguageRouterMoeTransformer #######################
####################### LanguageRouterMoeTransformer #######################
####################### LanguageRouterMoeTransformer #######################



class Expert(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    

class NoisyTopkRouter(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(dim, num_experts) 
        self.noise_linear = nn.Linear(dim, num_experts)

    def forward(self, router_input):
        router_input = router_input.mean(dim=1) 

        logits = self.topkroute_linear(router_input)
        noise_logits = self.noise_linear(router_input)
        
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices




class Sample_SparseMoE(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts, top_k, capacity_factor=1.0, dropout=0.):
        super().__init__()
        self.router = NoisyTopkRouter(dim, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(dim, hidden_dim, dropout) for _ in range(num_experts)])
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts

    def forward(self, router_input, x):
        batch_size, seq_len, _ = x.shape

        router_output, indices = self.router(router_input)
        final_output = torch.zeros_like(x)

        samples_per_batch = batch_size * self.top_k
        expert_capacity = int((samples_per_batch / self.num_experts) * self.capacity_factor)
        expert_capacity = max(expert_capacity, 1)  # Ensure at least one sample per expert

        # Flatten routing indices and outputs
        flat_indices = indices.view(-1)
        flat_router_output = router_output.gather(1, indices).view(-1)
        sample_indices = torch.arange(batch_size, device=x.device)[:, None].expand(-1, self.top_k).reshape(-1)

        # Sort elements based on expert assignments
        sorted_expert_indices, sort_idx = torch.sort(flat_indices)

        sorted_x = x[sample_indices[sort_idx]]
        sorted_router_weights = flat_router_output[sort_idx]
        sorted_sample_indices = sample_indices[sort_idx]

        # Create mask for capacity constraints
        mask = torch.zeros_like(sorted_expert_indices, dtype=torch.bool)
        expert_counts = torch.unique(sorted_expert_indices, return_counts=True)[1]
        ptr = 0
        for count in expert_counts:
            num_to_keep = min(count, expert_capacity)
            mask[ptr:ptr+num_to_keep] = True
            ptr += count

        # Apply capacity mask
        sorted_x = sorted_x[mask]
        sorted_router_weights = sorted_router_weights[mask]
        sorted_sample_indices = sorted_sample_indices[mask]

        # Process each expert's inputs
        unique_experts = torch.unique(sorted_expert_indices[mask])
        expert_inputs = sorted_x.split(torch.unique(sorted_expert_indices[mask], return_counts=True)[1].tolist(), dim=0)
        expert_outputs = []
        for i, expert_idx in enumerate(unique_experts):
            expert = self.experts[expert_idx.item()]
            expert_output = expert(expert_inputs[i])
            expert_outputs.append(expert_output)
        

        # Combine and re-weight expert outputs
        if expert_outputs:
            combined_output = torch.cat(expert_outputs, dim=0)
            weighted_output = combined_output * sorted_router_weights.view(-1, 1, 1)
            final_output.index_add_(0, sorted_sample_indices, weighted_output)

        return final_output


class PreNormSpareMoe(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, router_input, x):
        x = self.norm(x)
        return self.fn(router_input, x)



class LanguageRouterMoeTransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, num_experts, top_k, capacity_factor=1.0, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAttention(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNormSpareMoe(dim, Sample_SparseMoE(dim, hidden_dim=mlp_dim, num_experts=num_experts, top_k=top_k, capacity_factor=capacity_factor, dropout=dropout))
            ]))

    def forward(self, router_input, x):
        for attn, ff in self.layers:
            x = attn(x, x, x) + x
            x = ff(router_input, x) + x
        return x



class LanguageRouterMoeTransformer(nn.Module):
    def __init__(self, num_frames, token_len, dim, depth, heads, mlp_dim, dim_head=64, num_experts=5, top_k=3, capacity_factor=1.0, dropout=0., emb_dropout=0.):
        super().__init__()

        self.token_len = token_len

        if token_len is not None:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames+token_len, dim))
            self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
            self.extra_token = None

        self.dropout = nn.Dropout(emb_dropout)
        self.encoder = LanguageRouterMoeTransformerEncoder(dim, depth, heads, dim_head, mlp_dim, num_experts, top_k, capacity_factor, dropout)

    def forward(self, router_input, x):
        b, n, _ = x.shape

        if self.token_len is not None:
            extra_token = repeat(self.extra_token, '1 n d -> b n d', b=b)

            x = torch.cat((extra_token, x), dim=1)
            x = x + self.pos_embedding[:, :n+self.token_len]
        else:
            x = x + self.pos_embedding[:, :n]

        x = self.dropout(x)
        x = self.encoder(router_input, x)

        return x







####################### AV_Temporal_Attn #######################
####################### AV_Temporal_Attn #######################
####################### AV_Temporal_Attn #######################
####################### AV_Temporal_Attn #######################
####################### AV_Temporal_Attn #######################

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()

        self.use_mup = use_mup
        self.eps = eps

        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        if not self.use_mup:
            return output * self.weight
        else:
            return output




class AV_Temporal_Attn(nn.Module):
    def __init__(self, dim, inner_dim, d_conv):
        super().__init__()

        self.audio_RMSNorm = RMSNorm(dim, eps=1e-5, use_mup=False)
        self.visual_RMSNorm = RMSNorm(dim, eps=1e-5, use_mup=False)

        self.audio_in_proj = nn.Linear(dim, 2 * inner_dim , bias=False)
        self.visual_in_proj = nn.Linear(dim, 2 * inner_dim, bias=False)

        self.audio_conv1d = nn.Conv1d(in_channels=inner_dim, out_channels=inner_dim, 
                              kernel_size=d_conv, bias=True, 
                              groups=inner_dim,
                              padding=d_conv - 1)
        
        self.visual_conv1d = nn.Conv1d(in_channels=inner_dim, out_channels=inner_dim,
                              kernel_size=d_conv, bias=True, 
                              groups=inner_dim,
                              padding=d_conv - 1)
        
        self.audio_out_proj = nn.Linear(inner_dim, dim, bias=False)
        self.visual_out_proj = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, audio, visual):
        # audio: [B, L, D]
        # visual: [B, L, D]

        _, L, _ = audio.shape

        audio_norm = self.audio_RMSNorm(audio)
        visual_norm = self.visual_RMSNorm(visual)

        audio_x = self.audio_in_proj(audio_norm)
        visual_x = self.visual_in_proj(visual_norm)

        audio_x, audio_w = audio_x.chunk(2, dim=-1)
        visual_x, visual_w = visual_x.chunk(2, dim=-1)

        audio_x = audio_x.transpose(1, 2)
        audio_x = self.audio_conv1d(audio_x)[:, :, :L]
        audio_x = audio_x.transpose(1, 2)

        visual_x = visual_x.transpose(1, 2)
        visual_x = self.visual_conv1d(visual_x)[:, :, :L]
        visual_x = visual_x.transpose(1, 2)

        audio_w = F.silu(audio_w)
        visual_w = F.silu(visual_w)

        audio_x = audio_x * visual_w
        visual_x = visual_x * audio_w

        audio_x = self.audio_out_proj(audio_x)
        visual_x = self.visual_out_proj(visual_x)

        audio_x = audio + audio_x
        visual_x = visual + visual_x

        return torch.cat([audio_x, visual_x], dim=1)
        




####################################
####################################
####################################

class GatedFusion_mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(3*dim, 3),
            nn.Softmax(dim=1)
        )
        
        self.cls_head = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim//2, 1)
        )
        
    def forward(self, text, audio, visual):
        combined = torch.cat([text, audio, visual], dim=1)
        gates = self.gate(combined)

        fused = gates[:, 0:1]*text + gates[:, 1:2]*audio + gates[:, 2:3]*visual

        return self.cls_head(fused)



