from pathlib import Path
from src.model.utils import str2bool  # 确保从utils模块导入str2bool转换函数

class AttributeDict(dict):
    """字典支持属性式访问 (e.g., params.key)"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

params = AttributeDict(
    # === 训练核心参数 ===
    # world_size=8,                      # DDP训练使用的GPU数量
    use_fp16=True,                     # 是否使用FP16混合精度训练
    dataset="libritts",                # 数据集名称 (emilia/libritts)
    token_type="phone",                 # 输入token类型 (phone/char/bpe)
    max_duration=200.0,                # 单批次最大音频时长（秒）<- 从add_arguments更新
    lr_hours=0,                        # 基于训练小时数的学习率调整
    lr_batches=7500,                   # 影响学习率衰减的步数
    lr_epochs=10,                      # 影响学习率衰减的周期数
    base_lr = 1e-4,
    weight_decay = 0.001,
    grad_clip = 5.0,
    min_lr = 1e-6,  # base_lr / 100
    token_file="/mmu-audio-ssd/tts/qiangchunyu/tts/zipvoice/ZipVoice/data/tokens_emilia.txt", # token映射文件路径
    vocab_size = 360, # 预留两个特殊token
    pad_id = 0,
    manifest_dir="data/fbank",         # 特征文件目录 <- 从add_arguments更新
    batch_size = 32,
    prefetch_factor = 4,
    num_epochs=1000,                     # 总训练周期数
    exp_dir="zipvoice/exp_zipvoice_libritts",  # 实验输出目录

    start_epoch=1,                     # 起"""  """始训练周期
    finetune=False,                    # 是否微调模式（固定学习率）

    # KL Loss
    kl_start  = 1e4,      # kl_loss梯度计算起始step
    kl_end    = 2e4,     # kl_loss梯度计算上限step
    kl_upper  = 1e-5,    # kl_loss权重
    kl_floor  = 500,       # kl_loss下限

    # === 数据加载参数 (从add_arguments添加) ===
    bucketing_sampler=True,            # 是否使用分桶采样器
    num_buckets=30,                    # 动态分桶采样器的桶数量
    on_the_fly_feats=False,            # 是否实时生成特征
    shuffle=True,                      # 是否打乱数据顺序
    drop_last=True,                    # 是否丢弃最后不完整批次
    return_cuts=True,                  # 是否返回批次对应的cut信息
    num_workers=8,                     # 数据加载工作进程数
    input_strategy="PrecomputedFeatures",  # 输入策略：AudioSamples/PrecomputedFeatures

    # === 模型结构参数 ===
    # 流匹配解码器配置
    fm_decoder_nhead = 10, #7
    fm_decoder_block_dim = 10 * 64, #7 * 64
    fm_decoder_joint_block_layers = 8, #12
    fm_decoder_single_block_layers = 8, #8

    # 文本编码器配置
    text_encoder_downsampling_factor="1",
    text_encoder_num_layers="4",
    text_encoder_feedforward_dim=512,  
    text_encoder_cnn_module_kernel="9",
    text_encoder_num_heads=4,
    text_encoder_dim=192,

    # 通用模型参数
    query_head_dim=32,
    value_head_dim=12,
    pos_head_dim=4,
    pos_dim=48,
    time_embed_dim=192,
    text_embed_dim=192,
    condition_embed_dim = 128,
    condition_drop_ratio=0.2,         # 训练时文本条件丢弃率

    # condition维度
    clip_dim = 1024,
    sync_dim = 768,
    metaclip_dim = 1280,

    # === 训练过程控制 ===
    ref_duration=50,                  # 参考批次时长（用于调整内部调度）
    seed=42,                           # 随机种子
    save_every_n=4000,                 # 每N批次保存检查点
    keep_last_k=30,                    # 保留的检查点数量
    average_period=200,                # 模型平均更新周期
    feat_scale=0.1,                    # Fbank特征缩放因子

    # === 运行时监控 ===
    tensorboard=True,                  # 启用TensorBoard日志
    print_diagnostics=False,            # 打印模型诊断信息
    inf_check=False,                    # 检查无限值/NaN

    # === 训练状态监控 ===
    best_train_loss=float("inf"),   # 当前最佳训练损失 (初始为无穷大)
    best_valid_loss=float("inf"),   # 当前最佳验证损失 (初始为无穷大)
    best_train_epoch=-1,            # 最佳训练损失对应的周期 (初始为-1)
    best_valid_epoch=-1,            # 最佳验证损失对应的周期 (初始为-1)
    batch_idx_train=0,              # 训练批次计数 (用于Tensoboard统计和周期性操作)
    log_interval=1,                 # 日志打印间隔 (每N批次打印一次训练损失)
    reset_interval=200,             # 统计重置间隔 (每N批次重置训练统计信息)
    valid_interval=4000,            # 验证运行间隔 (每N批次运行一次验证集评估)
    env_info={},                    # 环境信息字典 (包含PyTorch版本、Python版本、CUDA可用状态等)

    # === 音频处理参数 ===
    sampling_rate=44100,               # 音频采样率
    feat_dim=40,                      # 特征维度

    # 推理参数
    speed = 1.0,
    t_shift = 1.0,
    guidance_scale = 1.0,
    num_step = 32,

    # SecoustiVAE路径
    SecoustiVAE_dir = '/mmu-audio-ssd/tts/qiangchunyu/audio_vae/SecoustiVAE_BigVGAN_V2',
    SecoustiVAE_model_path = '/mmu-audio-ssd/tts/qiangchunyu/audio_vae/SecoustiVAE_BigVGAN_V2/pretrain_models/secoustivae_fusion_model.pt',
    Bigvgan_dir = '/mmu-audio-ssd/tts/qiangchunyu/audio_vae/SecoustiVAE_BigVGAN_V2/pretrain_models/BigVGAN_ft_secoustivae',
    Bigvgan_model_path = '/mmu-audio-ssd/tts/qiangchunyu/audio_vae/SecoustiVAE_BigVGAN_V2/pretrain_models/BigVGAN_ft_secoustivae/pretrain/bigvgan_v2_44khz_128band_512x',
    num_mels=128,
    embed_dim=40,
    hidden_dim=512,
    disc_start = 0,
    disc_loss = "hinge",
    r1_reg_weight = 3,
    disc_factor = 0.01,
    disc_only_step = 10000,
    kl_weight = 1.0e-06,
)

# 补充环境信息函数
def get_env_info():
    import torch, sys
    return {
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "cuda_available": torch.cuda.is_available()
    }

# 动态计算参数
params.manifest_dir = Path(params.manifest_dir)
params.exp_dir = Path(params.exp_dir)
params.frame_shift_ms = 256 / params.sampling_rate * 1000

# 填充环境信息
params.env_info = get_env_info()
