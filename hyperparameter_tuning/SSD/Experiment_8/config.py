batch = 16
lr = 0.01
optimizer = 'SGD'
epochs = 150
input_size = 300
model = dict(
    type='SingleStageDetector',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[1, 1, 1],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='SSDVGG',
        depth=16,
        with_last_pool=False,
        ceil_mode=True,
        out_indices=(3, 4),
        out_feature_indices=(22, 34),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://vgg16_caffe')),
    neck=dict(
        type='SSDNeck',
        in_channels=(512, 1024),
        out_channels=(512, 1024, 512, 256, 256, 256),
        level_strides=(2, 2, 1, 1),
        level_paddings=(1, 1, 0, 0),
        l2_norm_scale=20),
    bbox_head=dict(
        type='SSDHead',
        in_channels=(512, 1024, 512, 256, 256, 256),
        num_classes=13,
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=300,
            basesize_ratio_range=(0.15, 0.9),
            strides=[8, 16, 32, 64, 100, 300],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2])),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.0,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        sampler=dict(type='PseudoSampler'),
        smoothl1_beta=1.0,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200))
cudnn_benchmark = True
dataset_type = 'CocoDataset'
data_root = 'dataset'
classes = ('pad', 'scribe_line', 'contamination', 'corrosion', 'crack', 'chip',
           'discolor', 'particle', 'scratch', 'burr', 'fly_die', 'shift_sl',
           'flake')
class_weight = [
    0.20493364, 0.60980849, 1.69637635, 1.71051282, 1.71051282, 1.62647812,
    1.71051282, 1.71051282, 1.68247163, 1.70766671, 1.71051282, 1.71051282,
    1.69918492, 1.0
]
backend_args = None
metainfo = dict(
    classes=('pad', 'scribe_line', 'contamination', 'corrosion', 'crack',
             'chip', 'discolor', 'particle', 'scratch', 'burr', 'fly_die',
             'shift_sl', 'flake'),
    palette=[(222, 3, 75), (17, 46, 34), (220, 3, 164), (110, 176, 47),
             (132, 100, 151), (20, 230, 12), (90, 132, 236), (232, 174, 55),
             (2, 152, 197), (145, 206, 209), (31, 161, 86), (212, 220, 104),
             (101, 250, 98)])
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=[123.675, 116.28, 103.53],
        to_rgb=True,
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', scale=(300, 300), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(300, 300), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
val_evaluator = dict(
    type='CocoMetric',
    ann_file='dataset/val_coco.json',
    metric='bbox',
    backend_args=None)
test_evaluator = dict(
    type='CocoMetric',
    ann_file='dataset/test_coco.json',
    metric='bbox',
    backend_args=None)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=150, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005))
auto_scale_lr = dict(base_batch_size=64)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', save_best='auto', max_keep_ckpts=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook', draw=False, interval=1))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = '/home/computervision_zl/Documents/ssd/checkpoint/ssd300_coco_20210803_015428-d231a06e.pth'
resume = False
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        metainfo=dict(
            classes=('pad', 'scribe_line', 'contamination', 'corrosion',
                     'crack', 'chip', 'discolor', 'particle', 'scratch',
                     'burr', 'fly_die', 'shift_sl', 'flake')),
        data_root='dataset',
        ann_file='train_coco.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Expand',
                mean=[123.675, 116.28, 103.53],
                to_rgb=True,
                ratio_range=(1, 4)),
            dict(
                type='MinIoURandomCrop',
                min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                min_crop_size=0.3),
            dict(type='Resize', scale=(300, 300), keep_ratio=False),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(type='PackDetInputs')
        ],
        backend_args=None))
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        metainfo=dict(
            classes=('pad', 'scribe_line', 'contamination', 'corrosion',
                     'crack', 'chip', 'discolor', 'particle', 'scratch',
                     'burr', 'fly_die', 'shift_sl', 'flake')),
        data_root='dataset',
        ann_file='val_coco.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(300, 300), keep_ratio=False),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None))
test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        metainfo=dict(
            classes=('pad', 'scribe_line', 'contamination', 'corrosion',
                     'crack', 'chip', 'discolor', 'particle', 'scratch',
                     'burr', 'fly_die', 'shift_sl', 'flake')),
        data_root='dataset',
        ann_file='test_coco.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(300, 300), keep_ratio=False),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None))
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]
launcher = 'none'
work_dir = './work_dirs/ssd_8'
