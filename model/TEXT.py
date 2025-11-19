import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
import torch.nn.functional as F
from model.bert import BertTextEncoder
from model.module_layer import Transformer, CrossTransformer, LanguageRouterMoeTransformer, GatedFusion_mlp, AV_Temporal_Attn






class TEXT(nn.Module):

    def __init__(self, dataset, fusion_layer_depth=2, num_experts=5, top_k=3, capacity_factor=1.0, dropout=0.0, bert_pretrained='./pretrained_model/bert-base-uncased'):
        super(TEXT, self).__init__()

        print("model参数: ", "dataset: ", dataset, "fusion_layer_depth: ", fusion_layer_depth, "num_experts: ", num_experts, "top_k: ", top_k, "capacity_factor: ", capacity_factor, "dropout: ", dropout)

        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=bert_pretrained)

        # mosi
        if dataset == 'mosi':
            self.proj_t0 = nn.Linear(768, 128)
            self.proj_a0 = nn.Linear(5, 128)
            self.proj_v0 = nn.Linear(20, 128)
        elif dataset == 'mosei':
            self.proj_t0 = nn.Linear(768, 128)
            self.proj_a0 = nn.Linear(74, 128)
            self.proj_v0 = nn.Linear(35, 128)
        elif dataset == 'sims':
            self.proj_t0 = nn.Linear(768, 128)
            self.proj_a0 = nn.Linear(33, 128)
            self.proj_v0 = nn.Linear(709, 128)
        elif dataset == 'sims2':
            self.proj_t0 = nn.Linear(768, 128)
            self.proj_a0 = nn.Linear(25, 128)
            self.proj_v0 = nn.Linear(177, 128)
        else:
            assert False, "DatasetName must be mosi, mosei or sims sims2."


        self.proj_a_clue0 = nn.Linear(768, 128)
        self.proj_v_clue0 = nn.Linear(768, 128)

        self.proj_t = Transformer(num_frames=50, save_hidden=False, token_len=1, dim=128, depth=1, heads=8, mlp_dim=128, dropout=dropout, emb_dropout=dropout)

        self.proj_a_feat = Transformer(num_frames=50, save_hidden=False, token_len=1, dim=128, depth=1, heads=8, mlp_dim=128, dropout=dropout, emb_dropout=dropout)
        self.proj_a_clue = Transformer(num_frames=50, save_hidden=False, token_len=1, dim=128, depth=1, heads=8, mlp_dim=128, dropout=dropout, emb_dropout=dropout)

        self.proj_v_feat = Transformer(num_frames=50, save_hidden=False, token_len=1, dim=128, depth=1, heads=8, mlp_dim=128, dropout=dropout, emb_dropout=dropout)
        self.proj_v_clue = Transformer(num_frames=50, save_hidden=False, token_len=1, dim=128, depth=1, heads=8, mlp_dim=128, dropout=dropout, emb_dropout=dropout)

        self.corss_fusion_audio = CrossTransformer(source_num_frames=51, tgt_num_frames=51, token_len=None, dim=128, depth=1, heads=8, mlp_dim=128, dropout=dropout, emb_dropout=dropout)
        self.corss_fusion_visual = CrossTransformer(source_num_frames=51, tgt_num_frames=51, token_len=None, dim=128, depth=1, heads=8, mlp_dim=128, dropout=dropout, emb_dropout=dropout)

        self.cross_align_a_v = AV_Temporal_Attn(dim=128, inner_dim=256, d_conv=4)
        self.text_router_moe_trans = LanguageRouterMoeTransformer(num_frames=153, token_len=None, dim=128, depth=fusion_layer_depth, heads=8, mlp_dim=128, num_experts=num_experts, top_k=top_k, capacity_factor=capacity_factor, dropout=dropout, emb_dropout=dropout)
        
        self.cls_head = GatedFusion_mlp(dim=128)


    def forward(self, text, audio, visual, audio_clue, visual_clue):

        b = visual.size(0)  # batch size

        text = self.bertmodel(text)
        audio_clue = self.bertmodel(audio_clue)
        visual_clue = self.bertmodel(visual_clue)

        text = self.proj_t0(text)
        audio = self.proj_a0(audio)
        visual = self.proj_v0(visual)
        audio_clue = self.proj_a_clue0(audio_clue)
        visual_clue = self.proj_v_clue0(visual_clue)

        # transformer
        text = self.proj_t(text)

        audio = self.proj_a_feat(audio)
        audio_clue = self.proj_a_clue(audio_clue)
        
        visual = self.proj_v_feat(visual)
        visual_clue = self.proj_v_clue(visual_clue)

        # cross fusion
        audio_fusion = self.corss_fusion_audio(source_x=audio, target_x=audio_clue)
        visual_fusion = self.corss_fusion_visual(source_x=visual, target_x=visual_clue)

        # cross align
        temporal_attn_av = self.cross_align_a_v(audio_fusion, visual_fusion)

        cat_feat = torch.cat((text, temporal_attn_av), dim=1)
        moe_trans = self.text_router_moe_trans(text, cat_feat)

        text_extra = moe_trans[:, 0, :]
        audio_extra = moe_trans[:, 51, :]
        visual_extra = moe_trans[:, 102, :]

        output = self.cls_head(text_extra, audio_extra, visual_extra)
        return output
    






