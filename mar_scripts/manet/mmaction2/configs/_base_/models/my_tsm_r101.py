# model settings
model = dict(# 模型的配置
    type='Recognizer2D',# 动作识别器的类型
    backbone=dict(# Backbone 字典设置
        type='ResNetTSM',# Backbone 名
        pretrained='torchvision://resnet101', # 预训练模型的 url 或文件位置
        depth=101, # ResNet 模型深度
        norm_eval=False,# 训练时是否设置 BN 层为验证模式
        shift_div=8),
    cls_head=dict(# 分类器字典设置
        type='MyTSMHead',# 分类器名
        num_classes=59,# 分类类别数量
        in_channels=2048, # 分类器里输入通道数
        spatial_type='avg',# 空间维度的池化种类
        consensus=dict(type='AvgConsensus', dim=1),# consensus 模块设置
        dropout_ratio=0.5,# dropout 层概率
        init_std=0.001,# 线性层初始化 std 值
        is_shift=True),
    # model training and testing settings
    train_cfg=None,# 训练 TSM 的超参配置
    test_cfg=dict(average_clips='prob'))# 测试 TSM 的超参配置
