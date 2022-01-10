# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import logging
import math
import torch
import numpy
import torch.nn.functional as F
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.layers import ClassifierLayer
from mmf.utils.build import build_image_encoder
from omegaconf import OmegaConf
from torch import nn
from transformers.modeling_bert import (
    BertConfig,
    BertLayerNorm
)
import json

logger = logging.getLogger(__name__)


# @registry.register_model("mma_sr")
class MMA_SR_Model(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.mmt_config = BertConfig(**self.config.mmt)
        self._datasets = registry.get("config").datasets.split(",")

    @classmethod
    def config_path(cls):
        return "configs/models/m4c_captioner/mma_sr_bsl.yaml"

    @classmethod
    def format_state_key(cls, key):
        key = key.replace("obj_faster_rcnn_fc7.module.lc", "obj_faster_rcnn_fc7.lc")
        key = key.replace("ocr_faster_rcnn_fc7.module.lc", "ocr_faster_rcnn_fc7.lc")
        return key

    def build(self):
        self.mmt_config.hidden_size = 1000
        # modules requiring custom learning rates (usually for finetuning)
        self.finetune_modules = []

        # split model building into several components
        self._build_obj_encoding()
        self._build_ocr_encoding()
        self._build_mma_sr()
        self._build_output()

    def _build_encoder_config(self):
        return OmegaConf.create(
            {
                "type": "finetune_faster_rcnn_fpn_fc7",
                "params": {
                    "in_dim": 2048,
                    "weights_file": "models/detectron.defaults/fc7_w.pkl",
                    "bias_file": "models/detectron.defaults/fc7_b.pkl",
                    "model_data_dir": self.config.model_data_dir,
                },
            }
        )

    def _build_obj_encoding(self):
        # object appearance feature: Faster R-CNN
        self.obj_faster_rcnn_fc7 = build_image_encoder(
            self._build_encoder_config(), direct_features=True
        )
        # apply smaller lr to pretrained Faster R-CNN fc7
        self.finetune_modules.append(
            {"module": self.obj_faster_rcnn_fc7, "lr_scale": self.config.lr_scale_frcn}
        )
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            self.config.obj.mmt_in_dim, self.mmt_config.hidden_size
        )
        
        self.obj_feat_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)

        self.obj_drop = nn.Dropout(self.config.obj.dropout_prob)

    def _build_ocr_encoding(self):

        self.remove_ocr_fasttext = getattr(self.config.ocr, "remove_ocr_fasttext", False)
        self.remove_ocr_phoc = getattr(self.config.ocr, "remove_ocr_phoc", False)
        self.remove_ocr_frcn = getattr(self.config.ocr, "remove_ocr_frcn", False)
        self.remove_ocr_semantics = getattr(
            self.config.ocr, "remove_ocr_semantics", False
        )
        self.remove_ocr_bbox = getattr(self.config.ocr, "remove_ocr_bbox", False)

        # OCR appearance feature: Faster R-CNN
        self.ocr_faster_rcnn_fc7 = build_image_encoder(
            self._build_encoder_config(), direct_features=True
        )
        self.finetune_modules.append(
            {"module": self.ocr_faster_rcnn_fc7, "lr_scale": self.config.lr_scale_frcn}
        )

        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            self.config.ocr.mmt_in_dim, self.mmt_config.hidden_size
        )
        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(4, self.mmt_config.hidden_size)

        self.ocr_feat_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.ocr_bbox_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)

        self.ocr_drop = nn.Dropout(self.config.ocr.dropout_prob)

    def _build_mma_sr(self):
        hidden_size = 1000
        self.mma_sr = MMA_SR(decoder_dim=hidden_size,
                             obj_dim=hidden_size,
                             ocr_dim=hidden_size,
                             emb_dim=hidden_size,
                             attention_dim=hidden_size,
                             mmt_config=self.mmt_config)

    def _build_output(self):
        self.answer_processor = registry.get(self._datasets[0] + "_answer_processor")

    def forward(self, sample_list):
        # fwd_results holds intermediate forward pass results
        # TODO possibly replace it with another sample list

        fwd_results = {}
        self._forward_obj_encoding(sample_list, fwd_results)
        self._forward_ocr_encoding(sample_list, fwd_results)
        self._foward_mma_sr(sample_list, fwd_results)
        results = {"scores": fwd_results["scores"]}
        
        return results

    def _forward_obj_encoding(self, sample_list, fwd_results):
        # object appearance feature: Faster R-CNN fc7
        obj_fc6 = sample_list.image_feature_0
        obj_fc7 = self.obj_faster_rcnn_fc7(obj_fc6)
        obj_fc7 = F.normalize(obj_fc7, dim=-1)

        obj_feat = obj_fc7
        obj_bbox = sample_list.obj_bbox_coordinates

        obj_in = self.obj_feat_layer_norm(self.linear_obj_feat_to_mmt_in(obj_feat))
        obj_in = self.obj_drop(obj_in)
        fwd_results["obj_in"] = obj_in

        # binary mask of valid object vs padding
        obj_nums = sample_list.image_info_0.max_features
        fwd_results["obj_mask"] = _get_mask(obj_nums, obj_in.size(1))

    def _forward_ocr_encoding(self, sample_list, fwd_results):
        # OCR FastText feature (300-dim)
        ocr_fasttext = sample_list.context_feature_0
        ocr_fasttext = F.normalize(ocr_fasttext, dim=-1)
        assert ocr_fasttext.size(-1) == 300

        # OCR PHOC feature (604-dim)
        ocr_phoc = sample_list.context_feature_1
        ocr_phoc = F.normalize(ocr_phoc, dim=-1)
        assert ocr_phoc.size(-1) == 604

        # OCR appearance feature: Faster R-CNN fc7
        ocr_fc6 = sample_list.image_feature_1[:, : ocr_fasttext.size(1), :]
        ocr_fc7 = self.ocr_faster_rcnn_fc7(ocr_fc6)
        ocr_fc7 = F.normalize(ocr_fc7, dim=-1)

        # OCR order vectors (legacy from LoRRA model; set to all zeros)
        # TODO remove OCR order vectors; they are not needed
        ocr_order_vectors = torch.zeros_like(sample_list.order_vectors)
        if self.remove_ocr_fasttext:
            ocr_fasttext = torch.zeros_like(ocr_fasttext)
        if self.remove_ocr_phoc:
            ocr_phoc = torch.zeros_like(ocr_phoc)
        if self.remove_ocr_frcn:
            ocr_fc7 = torch.zeros_like(ocr_fc7)
        ocr_feat = torch.cat(
            [ocr_fasttext, ocr_phoc, ocr_fc7, ocr_order_vectors], dim=-1
        )

        ocr_bbox = (sample_list.ocr_bbox_coordinates).float()

        if self.remove_ocr_semantics:
            ocr_feat = torch.zeros_like(ocr_feat)
        if self.remove_ocr_bbox:
            ocr_bbox = torch.zeros_like(ocr_bbox)
        ocr_in = self.ocr_feat_layer_norm(self.linear_ocr_feat_to_mmt_in(ocr_feat)) + self.ocr_bbox_layer_norm(self.linear_ocr_bbox_to_mmt_in(ocr_bbox))
        
        ocr_in = self.ocr_drop(ocr_in)
        fwd_results["ocr_in"] = ocr_in

        # binary mask of valid OCR vs padding
        ocr_nums = sample_list.context_info_0.max_features
        fwd_results["ocr_mask"] = _get_mask(ocr_nums, ocr_in.size(1))


    def _foward_mma_sr(self, sample_list, fwd_results):

        if self.training:

            fwd_results["prev_inds"] = sample_list.train_prev_inds.clone()
            target_caption = sample_list.train_prev_inds.clone()
            target_cap_len = sample_list["train_loss_mask"].sum(dim=-1).unsqueeze(1)
            
            fwd_results["scores"] = self.mma_sr(fwd_results["obj_in"], fwd_results["ocr_in"],
                                                      fwd_results["ocr_mask"], target_caption, target_cap_len)
        else:
     
            fwd_results["prev_inds"] = torch.zeros_like(sample_list.train_prev_inds)
            fwd_results["prev_inds"][:, 0] = 1  # self.answer_processor.BOS_IDX = 1
            fwd_results["scores"] = self.mma_sr(fwd_results["obj_in"], fwd_results["ocr_in"],
                                                      fwd_results["ocr_mask"], training=False,
                                                      )

    def get_optimizer_parameters(self, config):
        optimizer_param_groups = []

        base_lr = config.optimizer.params.lr
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append(
                {
                    "params": list(m["module"].parameters()),
                    "lr": base_lr * m["lr_scale"],
                }
            )
            finetune_params_set.update(list(m["module"].parameters()))
        # remaining_params are those parameters w/ default lr
        remaining_params = [
            p for p in self.parameters() if p not in finetune_params_set
        ]
        # put the default lr parameters at the beginning
        # so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})

        return optimizer_param_groups

    @classmethod
    def update_registry_for_pretrained(cls, config, checkpoint, full_output):
        from omegaconf import OmegaConf

        # Hack datasets using OmegaConf
        datasets = full_output["full_config"].datasets
        dataset = datasets.split(",")[0]
        config_mock = OmegaConf.create({"datasets": datasets})
        registry.register("config", config_mock)
        registry.register(
            f"{dataset}_num_final_outputs",
            # Need to add as it is subtracted
            checkpoint["classifier.module.weight"].size(0)
            + config.classifier.ocr_max_num,
        )
        # Fix this later, when processor pipeline is available
        answer_processor = OmegaConf.create({"BOS_IDX": 1})
        registry.register(f"{dataset}_answer_processor", answer_processor)


class OcrPtrNet(nn.Module):
    def __init__(self, hidden_size, query_key_size=None):
        super().__init__()

        if query_key_size is None:
            query_key_size = hidden_size
        self.hidden_size = hidden_size
        self.query_key_size = query_key_size

        self.query = nn.Linear(hidden_size, query_key_size)
        self.key = nn.Linear(hidden_size, query_key_size)


    def forward(self, query_inputs, key_inputs, attention_mask):
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        if extended_attention_mask.dim() == 2:
            extended_attention_mask = extended_attention_mask.unsqueeze(1)

        query_layer = self.query(query_inputs)
        if query_layer.dim() == 2:
            query_layer = query_layer.unsqueeze(1)
            squeeze_result = True
        else:
            squeeze_result = False
        key_layer = self.key(key_inputs)

        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scores = scores / math.sqrt(self.query_key_size)
        scores = scores + extended_attention_mask

        if squeeze_result:
            scores = scores.squeeze(1)

        return scores


class PrevPredEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        MAX_DEC_LENGTH = 100
        MAX_TYPE_NUM = 5
        hidden_size = config.hidden_size
        ln_eps = config.layer_norm_eps

        self.token_type_embeddings = nn.Embedding(MAX_TYPE_NUM, hidden_size)

        self.ans_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.ocr_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, ans_emb, ocr_emb, prev_inds):

        assert prev_inds.dim() == 2 and prev_inds.dtype == torch.long
        assert ans_emb.dim() == 2
        batch_size = prev_inds.size(0)
        ans_num = ans_emb.size(0)

        # apply layer normalization to both answer embedding and OCR embedding
        # before concatenation, so that they have the same scale
        assert ans_emb.size(-1) == ocr_emb.size(-1)

        # apply layer normalization to both answer embedding and OCR embedding
        # before concatenation, so that they have the same scale
        ans_emb = self.ans_layer_norm(ans_emb)
        ocr_emb = self.ocr_layer_norm(ocr_emb)

        ans_emb = ans_emb.unsqueeze(0).expand(batch_size, -1, -1)
        ans_ocr_emb_cat = torch.cat([ans_emb, ocr_emb], dim=1)
        raw_dec_emb = _batch_gather(ans_ocr_emb_cat, prev_inds)

        # Token type ids: 0 -- vocab; 1 -- OCR
        token_type_ids = prev_inds.ge(ans_num).long()
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_type_embeddings
        embeddings = self.emb_layer_norm(embeddings)
        embeddings = self.emb_dropout(embeddings)
        dec_emb = raw_dec_emb + embeddings

        return dec_emb


def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask


@functools.lru_cache(maxsize=32)
def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i + 1):
            mask[i, j] = 1.0
    return mask


def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.view(batch_size * length, dim)

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    if inds.dim() == 2:
        batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    inds_flat = batch_offsets + inds
    results = F.embedding(inds_flat, x_flat)
    return results


class AttentionC(nn.Module):

    def __init__(self, image_features_dim, decoder_dim, attention_dim):
        super(AttentionC, self).__init__()

        self.features_att = nn.Linear(image_features_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, attended_features, decoder_hidden, attention_mask=None):

        att1 = self.features_att(attended_features)  # (batch_size, attend_features_dim, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, decoder_features_dim, attention_dim)
        att = self.full_att(self.tanh(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, m, n)
        if attention_mask is not None:
            extended_attention_mask = (1.0 - attention_mask) * -10000.0
            alpha = self.softmax(att + extended_attention_mask)  # (batch_size, 36)
        else:
            alpha = self.softmax(att)
        context = (attended_features * alpha.unsqueeze(2)).sum(dim=1)  
        return context


class MMA_SR(nn.Module):

    def __init__(self,
                 decoder_dim=1000,
                 obj_dim=1000,
                 ocr_dim=1000,
                 emb_dim=1000,
                 attention_dim=1000,
                 mmt_config=None,
                 ss_prob=0.0):

        super(MMA_SR, self).__init__()
        self.dropout = nn.Dropout(0.3)
        self.decoder_dim = decoder_dim
        self.embed_config = mmt_config
        self.vocab_size = 6736
        self.ocr_size = 50
        self.voc_emb = nn.Embedding(self.vocab_size, emb_dim)
        self.embed = PrevPredEmbeddings(self.embed_config)
        self.ss_prob = 0.0

        self.obj_attention = AttentionC(obj_dim, decoder_dim, attention_dim)
        self.ocr_attention = AttentionC(ocr_dim, decoder_dim, attention_dim)

        self.fusion_lstm = nn.LSTMCell(decoder_dim + emb_dim + obj_dim, decoder_dim)
        self.obj_lstm = nn.LSTMCell(obj_dim + decoder_dim, decoder_dim)
        self.ocr_lstm = nn.LSTMCell(ocr_dim + decoder_dim, decoder_dim)
        self.ocr_prt = OcrPtrNet(hidden_size=attention_dim)

        self.fc = nn.Linear(decoder_dim, self.vocab_size)
        # self.fc2 = nn.Linear(self.vocab_size+50, self.vocab_size+50)

    def init_hidden_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.decoder_dim).to(device)  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size, self.decoder_dim).to(device)
        return h, c

    def forward(self, obj_features, ocr_features, ocr_mask, target_caption=None, target_cap_len=None, training=True,
                label=None):
        max_len = 30
        batch_size = obj_features.size(0)
        device = obj_features.device
        repeat_mask = torch.zeros([batch_size, self.vocab_size + self.ocr_size]).to(device)

        if training:
            caption_lengths, sort_ind = target_cap_len.squeeze(1).sort(dim=0, descending=True)
        else:
            target_caption = torch.zeros([batch_size, max_len], dtype=torch.long).to(device)
            target_caption[:, 0] = 1
            caption_lengths = torch.tensor([max_len for _ in range(batch_size)])


        h_obj, c_obj = self.init_hidden_state(batch_size, device)  # (batch_size, decoder_dim)
        h_ocr, c_ocr = self.init_hidden_state(batch_size, device)
        h_fu, c_fu = self.init_hidden_state(batch_size, device)
        decode_lengths = caption_lengths.tolist()

        predictions = torch.zeros(batch_size, max_len, self.vocab_size + self.ocr_size).to(device)

        ocr_num = ocr_mask.sum(dim=-1)
        ocr_nums = (ocr_num + (ocr_num == 0).long())
        ocr_mean = ocr_features.sum(dim=1) / ocr_nums.unsqueeze(1)
        obj_mean = obj_features.mean(1)
        dec_num = int(max(decode_lengths))
        if dec_num > max_len:
            dec_num = max_len
        for t in range(dec_num):

            if training and t >= 1 and self.ss_prob > 0.0:
                sample_prob = obj_mean.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = target_caption[:, t].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = target_caption[:, t].data.clone()
                    prob_prev = torch.exp(predictions[:, t - 1].detach())  # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = target_caption[:, t].clone()

            y = self.embed(self.voc_emb.weight, ocr_features, it)



            x_fu = torch.cat([h_obj + h_ocr, obj_mean + ocr_mean, y], dim=-1)
            h_fu, c_fu = self.fusion_lstm(x_fu, (h_fu, c_fu))

            v_obj_weighted = self.obj_attention(obj_features, h_fu)
            v_ocr_weighted = self.ocr_attention(ocr_features, h_fu, ocr_mask)
            # add the mask later

            h_obj, c_obj = self.obj_lstm(torch.cat([h_fu, v_obj_weighted], dim=-1), (h_obj, c_obj))
            h_ocr, c_ocr = self.ocr_lstm(torch.cat([h_fu, v_ocr_weighted], dim=-1), (h_ocr, c_ocr))

            s_v = self.fc( self.dropout(h_obj) )
            s_o = self.ocr_prt(self.dropout(h_ocr), ocr_features, ocr_mask)

            scores = torch.cat([s_v, s_o], dim=-1)

            if not training and t < dec_num - 1:
              
                scores[:, 3] = -1e10
                scores = scores + repeat_mask
                pre_idx = (scores.argmax(dim=-1)).long()
                target_caption[:, t + 1] = pre_idx
                for j in range(batch_size):
                    used_idx = pre_idx[j]
                    if used_idx >= self.vocab_size:
                        repeat_mask[j, used_idx] = -1e6

            predictions[:, t] = scores


        return predictions
   




