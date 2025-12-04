import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# --- MMAudio Joint/Single DiT相关结构 ---
from torch.nn import LayerNorm
from einops.layers.torch import Rearrange


class ChannelLastConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = super().forward(x)
        x = x.transpose(1, 2)
        return x

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class ConvMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = ChannelLastConv1d(in_dim, hidden_dim, kernel_size, padding=padding)
        self.act = nn.GELU()
        self.conv2 = ChannelLastConv1d(hidden_dim, in_dim, kernel_size, padding=padding)
    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))


def compute_rope_rotations(length: int, dim: int, theta: int = 10000, freq_scaling: float = 1.0, device='cpu'):
    assert dim % 2 == 0
    pos = torch.arange(length, dtype=torch.float32, device=device)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    freqs *= freq_scaling
    rot = torch.einsum('n,d->nd', pos, freqs)
    cos = torch.cos(rot)
    sin = torch.sin(rot)
    return cos, sin


def modulate(x, shift, scale):
    # x: (B, T, D), shift/scale: (B, D) or (B, 1, D)
    if shift.dim() == 2:
        shift = shift.unsqueeze(1)  # (B, 1, D)
    if scale.dim() == 2:
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    # training will crash without these contiguous calls and the CUDNN limitation
    # I believe this is related to https://github.com/pytorch/pytorch/issues/133974
    # unresolved at the time of writing
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = F.scaled_dot_product_attention(q, k, v)
    out = rearrange(out, 'b h n d -> b n (h d)').contiguous()
    return out

def apply_rope(x, cos, sin):
    # x: (B, nhead, N, head_dim)
    # cos, sin: (N, head_dim//2)
    # x最后一维必须是偶数
    head_dim = x.size(-1)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,N,head_dim//2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x_rope_even = x1 * cos - x2 * sin
    x_rope_odd = x1 * sin + x2 * cos
    x_rope = torch.empty_like(x)
    x_rope[..., ::2] = x_rope_even
    x_rope[..., 1::2] = x_rope_odd
    return x_rope

class SelfAttention(nn.Module):
    def __init__(self, dim, nhead):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.nhead = nhead
        self.dim = dim
        self.q_norm = nn.RMSNorm(dim // nhead)
        self.k_norm = nn.RMSNorm(dim // nhead)
        self.split_into_heads = Rearrange('b n (h d j) -> b h n d j',
                                          h=nhead,
                                          d=dim // nhead,
                                          j=3)

    def pre_attention(self, x, rot=None):
        # x: (B, N, D)
        qkv = self.qkv(x)
        q, k, v = self.split_into_heads(qkv).chunk(3, dim=-1)
        q = q.squeeze(-1)
        k = k.squeeze(-1)
        v = v.squeeze(-1)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if rot is not None:
            cos, sin = rot

            #只对tts文本做rope
            # num_rot_tokens = cos.shape[0]
            # q_rope = apply_rope(q[:, :, -num_rot_tokens:], cos, sin)
            # k_rope = apply_rope(k[:, :, -num_rot_tokens:], cos, sin)
            # q[:, :, -num_rot_tokens:] = q_rope
            # k[:, :, -num_rot_tokens:] = k_rope

            #全做rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        return q, k, v

class MMDitSingleBlock(nn.Module):
    def __init__(self, dim, nhead, mlp_ratio=4.0, pre_only=False, kernel_size=7, padding=3, condition_type="global"):
        super().__init__()
        self.norm1 = LayerNorm(dim, elementwise_affine=False)
        self.attn = SelfAttention(dim, nhead)
        self.pre_only = pre_only
        self.condition_type = condition_type
        if pre_only:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        else:
            if kernel_size == 1:
                self.linear1 = nn.Linear(dim, dim)
            else:
                self.linear1 = ChannelLastConv1d(dim, dim, kernel_size=kernel_size, padding=padding)
            self.norm2 = LayerNorm(dim, elementwise_affine=False)
            if kernel_size == 1:
                self.ffn = MLP(dim, int(dim * mlp_ratio))
            else:
                self.ffn = ConvMLP(dim, int(dim * mlp_ratio), kernel_size=kernel_size, padding=padding)
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
    def pre_attention(self, x, c, rot=None):
        # x: BS * N * D
        # c: BS * D (全局条件) 或 BS * N * D (帧级条件)
        modulation = self.adaLN_modulation(c)  # (B, 2*D) or (B, 6*D) 或 (B, N, 2*D) or (B, N, 6*D)
        
        if self.pre_only:
            if self.condition_type == "global":
                (shift_msa, scale_msa) = modulation.chunk(2, dim=-1)  # 每个都是 (B, D)
                # 广播到序列长度: (B, D) -> (B, N, D)
                B, N, D = x.shape
                shift_msa = shift_msa.unsqueeze(1).expand(B, N, D)
                scale_msa = scale_msa.unsqueeze(1).expand(B, N, D)
            else:  # frame
                (shift_msa, scale_msa) = modulation.chunk(2, dim=-1)  # 每个都是 (B, N, D)
            gate_msa = shift_mlp = scale_mlp = gate_mlp = None
        else:
            if self.condition_type == "global":
                (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = modulation.chunk(6, dim=-1)  # 每个都是 (B, D)
                # 广播到序列长度: (B, D) -> (B, N, D)
                B, N, D = x.shape
                shift_msa = shift_msa.unsqueeze(1).expand(B, N, D)
                scale_msa = scale_msa.unsqueeze(1).expand(B, N, D)
            else:  # frame
                (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = modulation.chunk(6, dim=-1)  # 每个都是 (B, N, D)
        
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        q, k, v = self.attn.pre_attention(x, rot)
        return (q, k, v), (gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attention(self, x, attn_out, c):
        if self.pre_only:
            return x
        (gate_msa, shift_mlp, scale_mlp, gate_mlp) = c
        if attn_out.dim() == 4:
            # attn_out: [B, nhead, N, D//nhead] -> [B, N, D]
            B, nhead, N, d_head = attn_out.shape
            attn_out = attn_out.transpose(1, 2).reshape(B, N, nhead * d_head)
        
        # 处理gate_msa
        if self.condition_type == "global":
            # 广播gate_msa到序列长度: (B, D) -> (B, N, D)
            B, N, D = x.shape
            gate_msa = gate_msa.unsqueeze(1).expand(B, N, D)
        # frame类型不需要广播，已经是(B, N, D)
        
        x = x + self.linear1(attn_out) * gate_msa
        
        # 处理MLP调制参数
        if self.condition_type == "global":
            # 广播MLP调制参数
            B, N, D = x.shape
            shift_mlp = shift_mlp.unsqueeze(1).expand(B, N, D)
            scale_mlp = scale_mlp.unsqueeze(1).expand(B, N, D)
            gate_mlp = gate_mlp.unsqueeze(1).expand(B, N, D)
        # frame类型不需要广播，已经是(B, N, D)
        
        r = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + self.ffn(r) * gate_mlp
        return x

    def forward(self, x, cond, rot=None):
        # x: BS * N * D
        # cond: BS * D (全局条件) 或 BS * N * D (帧级条件)
        x_qkv, x_conditions = self.pre_attention(x, cond, rot)
        attn_out = attention(*x_qkv)
        x = self.post_attention(x, attn_out, x_conditions)
        return x

class JointBlock(nn.Module):
    """
    Joint attention block for two modalities (e.g., xt, text),
    frame_condition通过adaLN_modulation分别调制xt和text。
    """
    def __init__(self, dim, nhead, mlp_ratio=4.0, pre_only=False):
        super().__init__()
        self.pre_only = pre_only
        # xt_block使用frame级别的condition (3维)
        self.xt_block = MMDitSingleBlock(dim, nhead, mlp_ratio, pre_only=False, kernel_size=3, padding=1, condition_type="frame")
        # text_block使用global级别的condition (2维)
        self.text_block = MMDitSingleBlock(dim, nhead, mlp_ratio, pre_only=pre_only, kernel_size=3, padding=1, condition_type="global")

    def forward(self, xt, text,global_condition, frame_condition=None, features=None, xt_rot=None, text_rot=None):
        # frame_condition: (B, N, D) - 帧级条件，用于调制xt
        # global_condition: (B, D) - 全局条件，用于调制text
        x_qkv, x_mod = self.xt_block.pre_attention(xt, frame_condition, xt_rot)
        t_qkv, t_mod = self.text_block.pre_attention(text, global_condition, text_rot)
        xt_len = xt.shape[1]
        text_len = text.shape[1]
        joint_qkv = [torch.cat([x_qkv[i], t_qkv[i]], dim=2) for i in range(3)]
        attn_out = attention(*joint_qkv)
        x_attn_out = attn_out[:, :xt_len, :]
        t_attn_out = attn_out[:, xt_len:, :]

        xt_out = self.xt_block.post_attention(xt, x_attn_out, x_mod)
        text_out = self.text_block.post_attention(text, t_attn_out, t_mod)
        return xt_out, text_out

class FinalBlock(nn.Module):
    def __init__(self, dim, out_dim, condition_type="global"):
        super().__init__()
        self.condition_type = condition_type  # "global" or "frame"
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.conv = ChannelLastConv1d(dim, out_dim, kernel_size=7, padding=3)

    def forward(self, latent, c):
        # c: (B, D) 或 (B, N, D)
        modulation = self.adaLN_modulation(c)
        if self.condition_type == "global":
            # 广播到序列长度: (B, D) -> (B, N, D)
            B, N, D = latent.shape
            modulation = modulation.unsqueeze(1).expand(B, N, 2 * D)
        
        shift, scale = modulation.chunk(2, dim=-1)
        latent = modulate(self.norm(latent), shift, scale)
        latent = self.conv(latent)
        return latent


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings (from zipformer)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[..., None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
    return embedding


class FMDecoder(nn.Module):
    def __init__(self, feat_dim, block_dim, text_encoder_dim, nhead, joint_block_layers, single_block_layers):
        super().__init__()
        self.block_dim = block_dim
        self.nhead = nhead
        self.feat_dim = feat_dim

        # self.audio_input_proj = nn.Sequential(
        #     nn.Linear(feat_dim, block_dim),
        #     nn.SiLU(),
        #     nn.Linear(block_dim, block_dim)
        # )

        self.xt_input_proj = nn.Sequential(
                ChannelLastConv1d(feat_dim, block_dim, kernel_size=7, padding=3),
                nn.SELU(),
                ConvMLP(block_dim, block_dim * 4, kernel_size=7, padding=3),
            )
        # self.xt_input_proj = nn.Sequential(
        #     nn.Linear(feat_dim, block_dim),
        #     nn.SiLU(),
        #     nn.Linear(block_dim, block_dim)
        # ) 

        self.joint_blocks = nn.ModuleList([
            JointBlock(block_dim, nhead, pre_only=(i == joint_block_layers - 1)) for i in range(joint_block_layers)
        ])
        self.single_blocks = nn.ModuleList([
            MMDitSingleBlock(block_dim, nhead, kernel_size=3, padding=1, condition_type="frame") for _ in range(single_block_layers)
        ])
        self.final_layer = FinalBlock(block_dim, feat_dim, condition_type="global")
        # 训练结构相关参数
        self.speech_proj = nn.Linear(feat_dim, block_dim)  
        # self.text_cond_proj = nn.Linear(block_dim, block_dim)
        
        # self.t_proj = nn.Sequential(
        #     nn.Linear(1, block_dim),
        #     nn.SiLU(),
        #     nn.Linear(block_dim, block_dim)
        # )
        self.text_proj = nn.Linear(text_encoder_dim, block_dim) #tts文本
        self.global_cond_mlp = nn.Sequential(
            nn.Linear(block_dim, block_dim * 4),
            nn.SiLU(),
            nn.Linear(block_dim * 4, block_dim)
        )
        self.frame_cond_mlp = nn.Sequential(
            nn.Linear(block_dim, block_dim * 4),
            nn.SiLU(),
            nn.Linear(block_dim * 4, block_dim)
        )
        self.rotary_theta = 10000
        self.seq_len_xt = None
        self.seq_len_text = None
        self.xt_rot = None
        self.text_rot = None
        self.t_embed_dim = block_dim
        self.t_embed = nn.Sequential(
            nn.Linear(self.t_embed_dim, self.t_embed_dim * 2),
            nn.GELU(),
            nn.Linear(self.t_embed_dim * 2, self.t_embed_dim),
        )
        self.initialize_weights() 

    def initialize_weights(self):
        """
        参考 DiT 论文 & 官方实现：
        1. Linear 层：Xavier Uniform
        2. Conv1d 层：Xavier Uniform
        3. AdaLN / 调制层最后一线性：全部置零（相当于一开始关闭调制）
        4. FinalBlock 的 conv & modulation：置零，确保初始输出≈0
        5. 其他特殊 buffer 置零
        """
        def _basic_init(m):
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 1) 递归给所有 Linear / Conv 做基本初始化
        self.apply(_basic_init)

        # 2) 时序嵌入 MLP 两层用 0.02 N(0,σ²)（BERT 同款）
        nn.init.normal_(self.t_embed[0].weight, std=0.02)
        nn.init.normal_(self.t_embed[2].weight, std=0.02)

        # 3) 将 joint_blocks 中所有 adaLN_modulation 的最终线性层置零
        for block in self.joint_blocks:                 # 逐个 JointBlock
            for submod in block.modules():              # 递归遍历其所有子模块
                if hasattr(submod, "adaLN_modulation"): # 只要模块内有该属性
                    ada = submod.adaLN_modulation[-1]   # 最后一层应为 nn.Linear
                    if isinstance(ada, nn.Linear):
                        nn.init.constant_(ada.weight, 0)
                        if ada.bias is not None:
                            nn.init.constant_(ada.bias, 0)

        # 4) 单独的 single_blocks
        for block in self.single_blocks:
            ada = block.adaLN_modulation[-1]
            nn.init.constant_(ada.weight, 0)
            nn.init.constant_(ada.bias,   0)

        # 5) FinalBlock：modulation 和 conv 全置零
        ada = self.final_layer.adaLN_modulation[-1]
        nn.init.constant_(ada.weight, 0)
        nn.init.constant_(ada.bias,   0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        if self.final_layer.conv.bias is not None:
            nn.init.constant_(self.final_layer.conv.bias, 0)

        # 6) 任何显式注册的 buffer（如空特征）直接清零
        for name, buf in self.named_buffers(recurse=False):
            if "rot_" not in name:          # RoPE 不清
                nn.init.constant_(buf, 0)

    def initialize_rotations(self, seq_len_xt, seq_len_text, device):
        base_freq = 1.0
        head_dim = self.block_dim // self.nhead
        self.seq_len_xt = seq_len_xt
        self.seq_len_text = seq_len_text

        self.xt_rot = compute_rope_rotations(
            length=seq_len_xt,
            dim=head_dim,
            theta=self.rotary_theta,
            freq_scaling=base_freq,
            device=device
        )

        self.text_rot = compute_rope_rotations(
            length=seq_len_text,
            dim=head_dim,
            theta=self.rotary_theta,
            freq_scaling=seq_len_xt * base_freq / seq_len_text,
            device=device
        )


    def encode_t(self, t):
        if t.dim() == 0:
            t = t.repeat(1)
        while t.dim() > 1:
            t = t.squeeze(-1)
        t_emb = timestep_embedding(t, self.t_embed_dim)
        t_emb = self.t_embed(t_emb)
        return t_emb

if __name__ == "__main__":
    # 测试样例
    batch = 2
    xt_len = 15
    text_len = 7
    cond_dim = 16
    nhead = 4

    xt = torch.randn(batch, xt_len, cond_dim)
    text = torch.randn(batch, text_len, cond_dim)
    global_condition = torch.randn(batch, cond_dim)  # (B, D)形状，全局条件
    frame_condition = torch.randn(batch, xt_len, cond_dim)  # (B, N, D)形状，帧级条件
    
    # 测试rotary embedding
    device = xt.device
    head_dim = cond_dim // nhead
    xt_rot = compute_rope_rotations(xt_len, head_dim, theta=10000, device=device)
    text_rot = compute_rope_rotations(text_len, head_dim, theta=10000, device=device)
    
    joint_block = JointBlock(cond_dim, nhead)
    single_block = MMDitSingleBlock(cond_dim, nhead, kernel_size=3, padding=1, condition_type="frame")
    
    print("xt shape:", xt.shape)
    print("text shape:", text.shape)
    print("global_condition shape:", global_condition.shape)  # (B, D)
    print("frame_condition shape:", frame_condition.shape)  # (B, N, D)
    print("xt_rot shape:", xt_rot[0].shape, xt_rot[1].shape)
    print("text_rot shape:", text_rot[0].shape, text_rot[1].shape)
    
    # 测试带rope的joint block
    xt_out, text_out = joint_block(xt, text, frame_condition, global_condition, xt_rot=xt_rot, text_rot=text_rot)
    print("xt_out shape:", xt_out.shape)
    print("text_out shape:", text_out.shape)
    
    # 测试带rope的single block (使用frame_condition)
    xt_final = single_block(xt_out, frame_condition, rot=xt_rot)
    print("xt_final shape:", xt_final.shape)
    
    # 测试不带rope的效果对比
    xt_out_no_rope, text_out_no_rope = joint_block(xt, text, frame_condition, global_condition)
    print("xt_out_no_rope shape:", xt_out_no_rope.shape)
    
    # 计算rope vs no_rope的差异
    rope_diff = torch.abs(xt_out - xt_out_no_rope).mean()
    print(f"RoPE vs No-RoPE difference: {rope_diff:.6f}")
    
    # 测试FinalBlock
    final_block = FinalBlock(cond_dim, cond_dim, condition_type="frame")
    final_out = final_block(xt_final, frame_condition)
    print("final_out shape:", final_out.shape)
    
    print("测试完成！")

    
    
