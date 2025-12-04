from typing import Optional, Union

import torch





class DiffusionModel_zipvoice(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        distill: bool = False,
        func_name: str = "forward_fm_decoder",
    ):
        super().__init__()
        self.model = model
        self.distill = distill
        self.func_name = func_name
        self.model_func = getattr(self.model, func_name)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        text_embed: torch.Tensor,
        speech_condition: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        guidance_scale: Union[float, torch.Tensor] = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        # guidance_scale follows t's dtype/device
        if not torch.is_tensor(guidance_scale):
            guidance_scale = torch.tensor(guidance_scale, dtype=t.dtype, device=t.device)

        # Distillation: direct pass-through
        if self.distill:
            return self.model_func(
                t=t,
                xt=x,
                text_embed=text_embed,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                guidance_scale=guidance_scale,
                **kwargs,
            )

        # No CFG
        if (guidance_scale == 0.0).all():
            return self.model_func(
                t=t,
                xt=x,
                text_embed=text_embed,
                speech_condition=speech_condition,
                padding_mask=padding_mask,
                **kwargs,
            )

        # ===== CFG path =====
        # Time handling: if batched, make it 2B; else keep scalar
        if torch.is_tensor(t) and t.dim() != 0:
            t = torch.cat([t, t], dim=0)  # 2B
            batch_size = x.size(0)
        else:
            batch_size = x.size(0)

        # Build concatenated batch
        x_cat = torch.cat([x, x], dim=0)  # [2B, ...]
        text_cat = torch.cat([torch.zeros_like(text_embed), text_embed], dim=0)

        pad_mask_cat = (
            torch.cat([padding_mask, padding_mask], dim=0) if padding_mask is not None else None
        )

        # Gate speech by t and double guidance_scale on uncond half where t <= 0.5
        if torch.is_tensor(t) and t.dim() != 0:
            larger_t_index = (t > 0.5).squeeze(1).squeeze(1)  # [2B]

            zero_speech_pair = torch.cat([torch.zeros_like(speech_condition), speech_condition], dim=0)
            speech_cat = torch.cat([speech_condition, speech_condition], dim=0)
            speech_cat[larger_t_index] = zero_speech_pair[larger_t_index]

            if guidance_scale.dim() == 0:
                guidance_scale = guidance_scale.expand(batch_size, 1, 1).clone()
            elif guidance_scale.dim() == 1:
                guidance_scale = guidance_scale.view(-1, 1, 1).clone()

            half = larger_t_index.size(0) // 2  # B
            idx_uncond_t_le = ~larger_t_index[:half]  # [B]
            guidance_scale[idx_uncond_t_le] = guidance_scale[idx_uncond_t_le] * 2
        else:
            if (t > 0.5) if torch.is_tensor(t) else (float(t) > 0.5):
                speech_cat = torch.cat([torch.zeros_like(speech_condition), speech_condition], dim=0)
            else:
                guidance_scale = guidance_scale * 2
                speech_cat = torch.cat([speech_condition, speech_condition], dim=0)

        y_uncond, y_cond = self.model_func(
            t=t,
            xt=x_cat,
            text_embed=text_cat,
            speech_condition=speech_cat,
            padding_mask=pad_mask_cat,
            **kwargs,
        ).chunk(2, dim=0)

        return (1 + guidance_scale) * y_cond - guidance_scale * y_uncond

class DiffusionModel(torch.nn.Module):
    """A wrapper of diffusion models for inference.
    Args:
        model: The diffusion model.
        distill: Whether it is a distillation model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        distill: bool = False,
        func_name: str = "forward_fm_decoder",
    ):
        super().__init__()
        self.model = model
        self.distill = distill
        self.func_name = func_name
        self.model_func = getattr(self.model, func_name)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        text_embed: torch.Tensor,
        speech_condition: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        guidance_scale: Union[float, torch.Tensor] = 0.0,
        **kwargs
    ) -> torch.Tensor:

        if not torch.is_tensor(guidance_scale):
            guidance_scale = torch.tensor(
                guidance_scale, dtype=t.dtype, device=t.device
            )
        
        # 蒸馏模型直接使用统一条件
        if self.distill:
            return self.model_func(
                t=t,
                xt=x,
                text_embed = text_embed,
                speech_condition = speech_condition,
                padding_mask=padding_mask,
                guidance_scale=guidance_scale,
                **kwargs
            )

        # 如果不需要引导，直接使用原始条件
        if (guidance_scale == 0.0).all():
            return self.model_func(
                t=t,
                xt=x,
                text_embed = text_embed,
                speech_condition = speech_condition,
                padding_mask=padding_mask,
                **kwargs
            )
        
        # 分类器无关引导
        else:
            # 扩展时间步
            if t.dim() != 0:
                t = torch.cat([t] * 2, dim=0)

            # 扩展输入和掩码
            x = torch.cat([x] * 2, dim=0)
            text_embed = torch.cat(
                [torch.zeros_like(text_embed), text_embed], dim=0
            )
            padding_mask = torch.cat([padding_mask] * 2, dim=0) if padding_mask is not None else None

            # 创建无条件输入（置零）
            zero_condition = torch.zeros_like(speech_condition)
            full_condition = torch.cat([zero_condition, speech_condition], dim=0)

            # 调用模型函数
            output = self.model_func(
                t=t,
                xt=x,
                text_embed=text_embed,
                speech_condition=full_condition,  # 包含有条件和无条件
                padding_mask=padding_mask,
                **kwargs
            )
            
            # 分离有条件和无条件输出
            data_uncond, data_cond = output.chunk(2, dim=0)
            
            # 应用引导比例
            res = (1 + guidance_scale) * data_cond - guidance_scale * data_uncond
            return res
            
class EulerSolver:
    def __init__(
        self,
        model: torch.nn.Module,
        distill: bool = False,
        func_name: str = "forward_fm_decoder",
    ):
        # self.model = DiffusionModel(model, distill=distill, func_name=func_name)
        self.model = DiffusionModel_zipvoice(model, distill=distill, func_name=func_name)

    def sample(
        self,
        x: torch.Tensor,
        text_embed: torch.Tensor,  # 修改为单一条件输入
        speech_condition: torch.Tensor,
        padding_mask: torch.Tensor,
        num_step: int = 10,
        guidance_scale: Union[float, torch.Tensor] = 0.0,
        t_start: Union[float, torch.Tensor] = 0.0,
        t_end: Union[float, torch.Tensor] = 1.0,
        t_shift: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        device = x.device

        if torch.is_tensor(t_start) and t_start.dim() > 0:
            timesteps = get_time_steps_batch(
                t_start=t_start,
                t_end=t_end,
                num_step=num_step,
                t_shift=t_shift,
                device=device,
            )
        else:
            timesteps = get_time_steps(
                t_start=t_start,
                t_end=t_end,
                num_step=num_step,
                t_shift=t_shift,
                device=device,
            )
        for step in range(num_step):
            v = self.model(
                t=timesteps[step],
                x=x,
                text_embed = text_embed,
                speech_condition = speech_condition,
                padding_mask=padding_mask,
                guidance_scale=guidance_scale,
                **kwargs
            )
            x = x + v * (timesteps[step + 1] - timesteps[step])
        return x

def get_time_steps(
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_step: int = 10,
    t_shift: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:

    timesteps = torch.linspace(t_start, t_end, num_step + 1).to(device)

    timesteps = t_shift * timesteps / (1 + (t_shift - 1) * timesteps)

    return timesteps


def get_time_steps_batch(
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    num_step: int = 10,
    t_shift: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
   
    while t_start.dim() > 1 and t_start.size(-1) == 1:
        t_start = t_start.squeeze(-1)
    while t_end.dim() > 1 and t_end.size(-1) == 1:
        t_end = t_end.squeeze(-1)
    assert t_start.dim() == t_end.dim() == 1

    timesteps_shape = (num_step + 1, t_start.size(0))
    timesteps = torch.zeros(timesteps_shape, device=device)

    for i in range(t_start.size(0)):
        timesteps[:, i] = torch.linspace(t_start[i], t_end[i], steps=num_step + 1)

    timesteps = t_shift * timesteps / (1 + (t_shift - 1) * timesteps)

    return timesteps.unsqueeze(-1).unsqueeze(-1)
