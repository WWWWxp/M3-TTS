# -*- coding: utf-8 -*-
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.model.scaling import ScheduledFloat
from src.model.solver import EulerSolver
from src.model.utils import (
    AttributeDict,
    make_pad_mask,
    to_int_tuple,
    condition_time_mask,
    condition_time_mask_reverse,
    masked_mean_time
)
from src.model.zipformer import TTSZipformer
from src.model.blocks import FMDecoder




# -----------------------
# Factory
# -----------------------
def get_fm_decoder_model(params: AttributeDict, distill: bool = False) -> nn.Module:
    nhead = params.fm_decoder_nhead
    block_dim = params.fm_decoder_block_dim
    joint_block_layers = params.fm_decoder_joint_block_layers
    single_block_layers = params.fm_decoder_single_block_layers
    feat_dim = params.feat_dim
    text_encoder_dim = params.text_encoder_dim
    return FMDecoder(feat_dim, block_dim, text_encoder_dim, nhead, joint_block_layers, single_block_layers)



def get_text_encoder_model(params: AttributeDict) -> nn.Module:
    encoder = TTSZipformer(
        in_dim=params.text_embed_dim,
        out_dim=params.text_embed_dim,
        downsampling_factor=to_int_tuple(params.text_encoder_downsampling_factor),
        num_encoder_layers=to_int_tuple(params.text_encoder_num_layers),
        cnn_module_kernel=to_int_tuple(params.text_encoder_cnn_module_kernel),
        encoder_dim=params.text_encoder_dim,
        feedforward_dim=params.text_encoder_feedforward_dim,
        num_heads=params.text_encoder_num_heads,
        query_head_dim=params.query_head_dim,
        pos_head_dim=params.pos_head_dim,
        value_head_dim=params.value_head_dim,
        pos_dim=params.pos_dim,
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        use_time_embed=False,
    )
    return encoder


def get_model(params: AttributeDict) -> nn.Module:
    """获取标准TTS模型"""
    fm_decoder = get_fm_decoder_model(params)
    text_encoder = get_text_encoder_model(params)

    model = TtsModel(
        fm_decoder=fm_decoder,
        text_encoder=text_encoder,
        text_embed_dim=params.text_embed_dim,
        condition_embed_dim=params.condition_embed_dim,
        feat_dim=params.feat_dim,
        vocab_size=params.vocab_size,
        pad_id=params.pad_id,
    )
    return model


# -----------------------
# TTS Model
# -----------------------
class TtsModel(nn.Module):
    """带有更新输入条件的标准TTS模型"""

    def __init__(
        self,
        fm_decoder: nn.Module,
        text_encoder: nn.Module,
        text_embed_dim: int,
        condition_embed_dim: int,
        feat_dim: int,
        vocab_size: int,
        pad_id: int = 0,
    ):
        """
        参数:
            fm_decoder: 流匹配编码器模型
            text_encoder: 文本编码器模型
            text_embed_dim: 文本嵌入维度
            condition_embed_dim: 条件嵌入维度
            feat_dim: 声学特征维度
            vocab_size: 词汇表大小
            pad_id: 填充ID
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.text_embed_dim = text_embed_dim
        self.condition_embed_dim = condition_embed_dim
        self.pad_id = pad_id
        
        self.fm_decoder = fm_decoder
        self.text_encoder = text_encoder
        
        # 嵌入层和投影层
        self.embed = nn.Embedding(vocab_size, text_embed_dim)
        
        self.distill = False

    def prepare_text_embed(
        self,
        token_ids: torch.Tensor,
        token_lens: torch.Tensor,
    ) -> torch.Tensor:

        # 将token_ids转换为嵌入
        token_embed = self.embed(token_ids)  # (B, T_token, D_text)
        
        # 创建token填充掩码
        tokens_padding_mask = make_pad_mask(token_lens, max_len=token_embed.size(1))
        
        # 通过文本编码器处理token嵌入
        token_embed = self.text_encoder(
            x=token_embed, 
            t=None, 
            padding_mask=tokens_padding_mask
        )  # (B, T_token, D_text)
        
        return token_embed

    def forward_fm_decoder(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        text_embed: torch.Tensor,
        speech_condition: torch.Tensor,
        guidance_scale: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert t.dim() in (0, 3)
        while t.dim() > 1 and t.size(-1) == 1:
            t = t.squeeze(-1)
        if guidance_scale is not None:
            while guidance_scale.dim() > 1 and guidance_scale.size(-1) == 1:
                guidance_scale = guidance_scale.squeeze(-1)
        if t.dim() == 0:
            t = t.repeat(xt.shape[0])
        if guidance_scale is not None and guidance_scale.dim() == 0:
            guidance_scale = guidance_scale.repeat(xt.shape[0])
        device = xt.device
        fm = self.fm_decoder
        if (fm.xt_rot is None or fm.text_rot is None or
            xt.shape[1] != fm.xt_rot[0].shape[0] or text_embed.shape[1] != fm.text_rot[0].shape[0]):
            fm.initialize_rotations(seq_len_xt=xt.shape[1], seq_len_text=text_embed.shape[1], device=device)
        xt_rot = fm.xt_rot
        text_rot = fm.text_rot
        t_emb = fm.encode_t(t.to(device))
        xt = fm.xt_input_proj(xt)
        text_embed = fm.text_proj(text_embed)

        speech_condition = fm.speech_proj(speech_condition)
        global_condition = fm.global_cond_mlp(t_emb)

        frame_condition = fm.frame_cond_mlp(speech_condition) + global_condition.unsqueeze(1)

        for block in fm.joint_blocks:
            xt, text_embed = block(xt, text_embed, global_condition,frame_condition, xt_rot=xt_rot, text_rot=text_rot)
        xt_out = xt
        for block in fm.single_blocks:
            xt_out = block(xt_out, frame_condition, rot=xt_rot)
        vt = fm.final_layer(xt_out, global_condition)
        return vt



    def forward(
        self,
        token_ids: torch.Tensor,
        token_lens: torch.Tensor,
        audio_mean: torch.Tensor,
        audio_lens: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        condition_drop_ratio: float = 0.0,
    ) -> torch.Tensor:

        target_length = audio_mean.size(1)
        
        # 准备条件输入
        text_embed = self.prepare_text_embed(
            token_ids=token_ids,
            token_lens=token_lens,
        )

        speech_condition_mask = condition_time_mask_reverse(
            features_lens=audio_lens,
            mask_percent=(0.7, 1.0),
            max_len=audio_mean.size(1),
        )

        speech_condition = torch.where(
            speech_condition_mask.unsqueeze(-1), 0, audio_mean
        )

        if condition_drop_ratio > 0.0:
            drop_mask = (torch.rand(speech_condition.size(0), 1, device=speech_condition.device) > condition_drop_ratio).float()
            text_embed = text_embed * drop_mask.unsqueeze(1)  # [B, N, D] * [B, 1, 1]
            # drop_mask_speech = (torch.rand(speech_condition.size(0), 1, device=speech_condition.device) > condition_drop_ratio).float()
            # speech_condition = speech_condition * drop_mask_speech.unsqueeze(1)
        
        # 创建填充掩码
        padding_mask = make_pad_mask(audio_lens, max_len=target_length)
        
        # 构建当前状态和目标速度
        xt = audio_mean * t + noise * (1 - t)
        ut = audio_mean - noise
        
        # 预测速度
        vt = self.forward_fm_decoder(
            t=t,
            xt=xt,
            text_embed=text_embed,
            speech_condition=speech_condition,
        )
        
        # 计算损失（只考虑非填充区域）
        loss_mask = speech_condition_mask & (~padding_mask)
        # fm_loss = torch.mean((vt[loss_mask] - ut[loss_mask]) ** 2)
        fm_loss = ((vt - ut).pow(2).mean(dim=-1) * loss_mask).sum() / loss_mask.sum().clamp(min=1)
        
        # aux_mask = (~speech_condition_mask) & (~padding_mask)
        # fm_loss_aux = ((vt - ut).pow(2).mean(dim=-1) * aux_mask).sum() / aux_mask.sum().clamp(min=1)
        # fm_loss = fm_loss + 0.1 * fm_loss_aux
        return fm_loss

    def sample(
        self,
        token_ids: torch.Tensor,
        token_lens: torch.Tensor,
        prompt_audio_mean: torch.Tensor,
        prompt_audio_token_id: torch.Tensor,
        prompt_audio_token_lens: torch.Tensor,
        audio_lens: torch.Tensor,
        t_shift: float = 1.0,
        num_step: int = 5,
        guidance_scale: float = 0.5,
    ) -> torch.Tensor:
      
        target_length = int(audio_lens.max())
        ref_length = int(prompt_audio_mean.size(1))
        # 准备完整条件
        token_ids = torch.cat([prompt_audio_token_id,token_ids],dim=1)
        token_lens = prompt_audio_token_lens + token_lens
        text_embed = self.prepare_text_embed(
            token_ids=token_ids,
            token_lens=token_lens,
        )

        speech_embed_pad = torch.zeros(prompt_audio_mean.size(0),target_length, prompt_audio_mean.size(2), device=prompt_audio_mean.device)
        speech_condition = torch.cat([prompt_audio_mean,speech_embed_pad],dim=1)
        # 创建填充掩码
        padding_mask = make_pad_mask(audio_lens, max_len=target_length)
        
        # 初始化噪声
        x0 = torch.randn(
            token_ids.size(0), ref_length+target_length, self.feat_dim, device=token_ids.device
        )
        
        # 使用求解器生成音频
        solver = EulerSolver(self, distill=self.distill, func_name="forward_fm_decoder")

        x1 = solver.sample(
            x=x0,
            text_embed = text_embed,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            num_step=num_step,
            guidance_scale=guidance_scale,
            t_shift=t_shift,
        )
        
        return x1[:,ref_length:,:]


if __name__ == '__main__':
    from params import params
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device('cuda')
    print(device)

    # 创建模拟输入数据
    batch_size = 4
    audio_length = 345
    token_length = 217
    prompt_len = 512

    audio_mean = torch.rand(batch_size, audio_length, params.feat_dim).to(device)
    token_ids = torch.randint(0, params.vocab_size, (batch_size, token_length)).to(device)
    token_lens = torch.randint(100, token_length, (batch_size,)).to(device)
    audio_lens = torch.randint(300, audio_length, (batch_size,)).to(device)
    noise = torch.randn(batch_size, audio_length, params.feat_dim).to(device)
    t = torch.rand(batch_size, 1, 1).to(device)
    
    # 创建模型
    model = get_model(params).to(device)

    # 测试前向传播
    loss = model(
        token_ids=token_ids,
        token_lens=token_lens,
        audio_mean=audio_mean,
        audio_lens=audio_lens,
        noise=noise,
        t=t
    )
    print("Loss:", loss.item())

    # Model Size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    print('param size: {:.3f}MB'.format(param_size/1024**2))
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    # print('buffer size: {:.3f}MB'.format(buffer_size/1024**2))
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Number of parameters: {num_params:.2f}M')

    require_gradient_num_params =  sum([param.nelement() for param in filter(lambda p: p.requires_grad, model.parameters())]) / 1e6
    print (f"Number_of_parameters_that_require_gradient: {require_gradient_num_params: .2f}M")