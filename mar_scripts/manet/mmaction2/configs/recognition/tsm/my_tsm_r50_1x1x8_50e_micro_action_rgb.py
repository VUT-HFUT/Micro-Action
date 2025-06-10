_base_ = [
    '../../_base_/models/my_tsm_r50.py', '../../_base_/schedules/sgd_tsm_50e.py',
    '../../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'VideoDataset'
data_root = './data/ma52/videos_train/'
data_root_val = './data/ma52/videos_val/'
data_root_test = './data/ma52/videos_test/'
ann_file_train = './data/ma52/train_list_videos.txt'
ann_file_val = './data/ma52/val_list_videos.txt'
ann_file_test = './data/ma52/test_list_videos.txt'
img_norm_cfg = dict(# 图像正则化参数设置
    mean=[123.675, 116.28, 103.53], # 图像正则化平均值
    std=[58.395, 57.12, 57.375], # 图像正则化方差
    to_bgr=False)# 是否将通道数从 RGB 转为 BGR

train_pipeline = [# 训练数据前处理流水线步骤组成的列表
    dict(type='DecordInit'),
    dict(type='SampleFrames', 
         clip_len=1,  #每个输出视频片段的帧
         frame_interval=1, # 所采相邻帧的时序间隔
         num_clips=8),# 所采帧片段的数量
    # TSN形式：将视频分为x个部分，每个部分随机取一帧。clip_len=1, num_clips=x，另外一个参数取值无所谓
    # 普通形式：在连续的帧中，间隔x帧提取帧，一共获取y帧。num_clips=1, clip_len=y, frames_interval=x
    dict(type='DecordDecode'),
    dict(type='Resize', # Resize 类的配置# 调整图片尺寸
         scale=(-1, 256)),# 调整比例
    dict(# MultiScaleCrop 类的配置
        type='MultiScaleCrop',# 多尺寸裁剪，随机从一系列给定尺寸中选择一个比例尺寸进行裁剪
        input_size=224,# 网络输入
        scales=(1, 0.875, 0.75, 0.66),# 长宽比例选择范围
        random_crop=False, # 是否进行随机裁剪
        max_wh_scale_gap=1,# 长宽最大比例间隔
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5), # 图片翻转
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label','emb'], meta_keys=[]),# Collect 类决定哪些键会被传递到行为识别器中
    dict(type='ToTensor', keys=['imgs', 'label','emb'])# ToTensor 类将其他类型转化为 Tensor 类型
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label','emb'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label','emb'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label','emb'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label','emb'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    lr=0.01/8,  # this lr is used for 8 gpus
)

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/my_tsm_r50_1x1x8_50e_micro_action_rgb/'
