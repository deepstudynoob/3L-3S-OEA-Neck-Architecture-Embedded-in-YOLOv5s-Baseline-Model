# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.
Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
"""

import argparse
import math
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = check_git_info()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
################################################ 1. 传入参数/基本配置 #############################################
def train(hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
        # opt传入的参数
    callbacks.run('on_pretrain_routine_start')

    # Directories
    w = save_dir / 'weights'  # weights dir
    # 新建文件夹 weights train evolve
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'
    # 保存训练结果的目录  如runs/train/exp*/weights/last.pt

    # Hyperparameters isinstance()是否是已知类型
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict 加载yaml文件
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    # 打印超参数 彩色字体
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings 如果不使用进化训练
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Config 画图
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True) # 随机种子
    with torch_distributed_zero_first(LOCAL_RANK): # 存在子进程-分布式训练
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val'] # 训练集和验证集的位路径
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes  # 设置类别 是否单类
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names # 类别对应的名称
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

    ################################################### 2. Model 模型加载/断点训练 ###########################################
    check_suffix(weights,'.pt')  # check weights 检查文件后缀是否是.pt
    # 加载预训练权重 yolov5提供了5个不同的预训练权重，大家可以根据自己的模型选择预训练权重
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK): #用于同步不同进程对数据读取的上下文管理器
            weights = attempt_download(weights)  # download if not found locally # 如果本地不存在就从网站上下载
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak  # 加载模型以及参数
        """
        两种加载模型的方式: opt.cfg / ckpt['model'].yaml
        使用resume-断点训练: 选择ckpt['model']yaml创建模型, 且不加载anchor
        使用断点训练时,保存的模型会保存anchor,所以不需要加载
        """
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect  # 筛选字典中的键值对  把exclude删除
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:  # 不适用预训练权重
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    amp = check_amp(model)  # check AMP

    ################################################ 3. Freeze/冻结训练  #########################################
    # 冻结训练的网络层
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False
            # 冻结训练的层梯度不更新

    # Image size 图片大小/batchsize设置
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple  # 检查图片的大小

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    ############################################ 4. Optimizer 优化器选择 / 分组优化设置 ###########################################
    """
        nbs = 64
        batchsize = 16
        accumulate = 64 / 16 = 4
        模型梯度累计accumulate次之后就更新一次模型 相当于使用更大batch_size
    """
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay # 权重衰减参数
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    ############################################ 5. Scheduler 学习率/ema/归一化/单机多卡 ##############################################
    if opt.cos_lr:  # 是否余弦学习率调整方式
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    # 使用EMA（指数移动平均）对模型的参数做平均, 一种给予近期数据更高权重的平均方法, 以求提高测试指标并增加模型鲁棒。
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # DP mode 单机多卡模式
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm 多卡归一化
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()') #打印信息

    ############################################## 6. Trainloader / 数据加载 / anchor调整 ######################################
    # 训练集数据加载
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True)
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class  # 标签编号最大值
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'
    # 判断编号是否正确

    # Process 0 # 验证集数据集加载
    if RANK in {-1, 0}:
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

        if not resume: # 没有使用断点训练
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # run AutoAnchor
            model.half().float()  # pre-reduce anchor precision

        callbacks.run('on_pretrain_routine_end', labels, names)

    #################################################### 7. 训练 ###############################################
    # DDP mode 多机多卡
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing # 标签平滑
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    # 从训练样本标签得到类别权重（和类别中的目标数即类别频率成反比）
    model.names = names

    # Start training 开始训练
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # 获取热身迭代的次数iterations： 3
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class  # 初始化maps(每个类别的map)和results
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    # 设置学习率衰减所进行到的轮次，即使打断训练，使用resume接着训练也能正常衔接之前的训练进行学习率衰减
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # 设置amp混合精度训练
    stopper, stop = EarlyStopping(patience=opt.patience), False # 早停止，不更新结束训练
    compute_loss = ComputeLoss(model)  # init loss class
    callbacks.run('on_train_start')
    # 打印信息
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):   # 开始走起训练 # epoch ------------------------------------------------------------------
        callbacks.run('on_train_epoch_start')
        model.train()

        # Update image weights (optional, single-GPU only)
        """
            如果设置进行图片采样策略，
            则根据前面初始化的图片采样权重model.class_weights以及maps配合每张图片包含的类别数
            通过random.choices生成图片索引indices从而进行采样
        """
        if opt.image_weights:
            # cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            cw = model.class_weights.to(device).numpy() * (1 - maps) ** 2 / nc  # class weights

            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
        if RANK in {-1, 0}: # 进度条显示
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        optimizer.zero_grad() # 梯度清零
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            """
            热身训练(前nw次迭代)
            在前nw次迭代中, 根据以下方式选取accumulate和学习率
            """
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    """
                    bias的学习率从0.1下降到基准学习率lr*lf(epoch)，
                    其他的参数学习率从0增加到lr*lf(epoch).
                    lf为上面设置的余弦退火的衰减函数
                    动量momentum也从0.9慢慢变到hyp['momentum'](default=0.937)
                    """
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            """
            Multi-scale  设置多尺度训练，从imgsz * 0.5, imgsz * 1.5 + gs随机选取尺寸
            """
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward /前相传播
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.
                # 计算损失，包括分类损失，objectness损失，框的回归损失
                # loss为总损失值，loss_items为一个元组，包含分类损失，objectness损失，框的回归损失和总损失

            # Backward
            torch.use_deterministic_algorithms(False)
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html 模型反向传播accumulate次之后再根据累积的梯度更新一次参数
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler 进行学习率衰减
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in {-1, 0}:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            # 将model中的属性赋值给ema
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            # 判断当前的epoch是否是最后一轮
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = validate.run(data_dict,
                                                batch_size=batch_size // WORLD_SIZE * 2,
                                                imgsz=imgsz,
                                                half=amp,
                                                model=ema.ema,
                                                single_cls=single_cls,
                                                dataloader=val_loader,
                                                save_dir=save_dir,
                                                plots=False,
                                                callbacks=callbacks,
                                                compute_loss=compute_loss)
                """
                测试使用的是ema（指数移动平均 对模型的参数做平均）的模型
                results: [1] Precision 所有类别的平均precision(最大f1时)
                         [1] Recall 所有类别的平均recall
                         [1] map@0.5 所有类别的平均mAP@0.5
                         [1] map@0.5:0.95 所有类别的平均mAP@0.5:0.95
                         [1] box_loss 验证集回归损失, obj_loss 验证集置信度损失, cls_loss 验证集分类损失
                maps: [80] 所有类别的mAP@0.5:0.95
                """

            # Update best mAP 这里的best mAP其实是[P, R, mAP@.5, mAP@.5-.95]的一个加权值
            # fi: [P, R, mAP@.5, mAP@.5-.95]的一个加权值 = 0.1*mAP@.5 + 0.9*mAP@.5-.95
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Save model
            """
            保存带checkpoint的模型用于inference或resuming training
            保存模型, 还保存了epoch, results, optimizer等信息
            optimizer将不会在最后一轮完成后保存
            model保存的是EMA的模型
            """
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,  # {remote, branch, commit} if a git repo
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    ############################################### 8. 打印训练信息 ##########################################
    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss)  # val best model with plots
                    # 模型训练完后, strip_optimizer函数将optimizer从ckpt中删除
                    # 并对模型进行model.half() 将Float32->Float16 这样可以减少模型大小, 提高inference速度
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, epoch, results) # 回调函数

    torch.cuda.empty_cache()  # 释放显存
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    """
    argparse 使用方法：
    parse = argparse.ArgumentParser()
    parse.add_argument('--s', type=int, default=2, help='flag_int')
    """
    parser.add_argument('--weights', type=str, default=ROOT / '/mnt/workspace/datasets/private-datasets/jsy/yolov5-7.0/yolov5-7.0/yolov5s.pt', help='initial weights path')
    # weights 权重的路径./weights/yolov5s.pt....
    # yolov5提供4个不同深度不同宽度的预训练权重 用户可以根据自己的需求选择下载
    parser.add_argument('--cfg', type=str, default='/mnt/workspace/datasets/private-datasets/jsy/yolov5-7.0/yolov5-7.0/models/0104_mymodel_ak+ak+dsc+dw+connect_1ceng+wusanwei.yaml', help='model.yaml path')
    # cfg 配置文件（网络结构） anchor/backbone/numclasses/head，训练自己的数据集需要自己生成
    # 生成方式——例如我的yolov5s_mchar.yaml 根据自己的需求选择复制./models/下面.yaml文件，5个文件的区别在于模型的深度和宽度依次递增
    parser.add_argument('--data', type=str, default=ROOT / '/mnt/workspace/datasets/private-datasets/jsy/basemodel/NEU-DET/MEU.yaml', help='dataset.yaml path')
    # data 数据集配置文件（路径） train/val/label/， 该文件需要自己生成
    # 生成方式——例如我的/data/mchar.yaml 训练集和验证集的路径 + 类别数 + 类别名称
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    # hpy超参数设置文件（lr/sgd/mixup）./data/hyps/下面有5个超参数设置文件，每个文件的超参数初始值有细微区别，用户可以根据自己的需求选择其中一个
    parser.add_argument('--epochs', type=int, default=400, help='total training epochs')
    # epochs 训练轮次， 默认轮次为100次l
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    # batchsize 训练批次， 默认bs=16
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    # imagesize 设置图片大小, 默认640*640
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    # rect 是否采用矩形训练，默认为False
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    # resume 是否接着上次的训练结果，继续训练
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    # nosave 不保存模型  默认False(保存)  在./runs/exp*/train/weights/保存两个模型 一个是最后一次的模型 一个是最好的模型
    # best.pt/ last.pt 不建议运行代码添加 --nosave
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    # noval 最后进行测试, 设置了之后就是训练结束都测试一下， 不设置每轮都计算mAP, 建议不设置
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    # noautoanchor 不自动调整anchor, 默认False, 自动调整anchor
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    # evolve参数进化， 遗传算法调参
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    # bucket谷歌优盘 / 一般用不到
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    # cache 是否提前缓存图片到内存，以加快训练速度，默认False
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    # mage-weights 使用图片采样策略，默认不使用
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # device 设备选择
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    # multi-scale 多测度训练
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    # single-cls 数据集是否多类/默认True
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='Adam', help='optimizer')
    # optimizer 优化器选择 / 提供了三种优化器
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    # sync-bn:是否使用跨卡同步BN,在DDP模式使用
    parser.add_argument('--workers', type=int, default=12, help='max dataloader workers (per RANK in DDP mode)')
    # workers/dataloader的最大worker数量
    parser.add_argument('--project', default=ROOT / 'runs/钢材', help='save to project/name')
    # 保存路径 / 默认保存路径
    parser.add_argument('--name', default='EXP', help='save to project/name')
    # 实验名称
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # 项目位置是否存在 / 默认是都不存在
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    # cos-lr 余弦学习率
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    # 标签平滑 / 默认不增强， 用户可以根据自己标签的实际情况设置这个参数，建议设置小一点 0.1 / 0.05
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    # 早停止忍耐次数 / 100次不更新就停止训练
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    # --freeze冻结训练 可以设置 default = [0] 数据量大的情况下，建议不设置这个参数
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    # --local_rank 进程编号 / 多卡使用
    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    # Weights & Biases arguments
    # 在线可视化工具，类似于tensorboard工具，想了解这款工具可以查看https://zhuanlan.zhihu.com/p/266337608
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    # upload_dataset: 是否上传dataset到wandb tabel(将数据集作为交互式 dsviz表 在浏览器中查看、查询、筛选和分析数据集) 默认False
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    # bbox_interval: 设置界框图像记录间隔 Set bounding-box image logging interval for W&B 默认-1   opt.epochs // 10
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')
    # 使用数据的版本

    return parser.parse_known_args()[0] if known else parser.parse_args()
    # 传入的基本配置中没有的参数也不会报错# parse_args()和parse_known_args()
    # parse = argparse.ArgumentParser()
    # parse.add_argument('--s', type=int, default=2, help='flag_int')
    # parser.parse_args() / parse_args()


def main(opt, callbacks=Callbacks()):
    ############################################### 1. Checks 打印关键词/安装环境 ##################################################
    if RANK in {-1, 0}:
        print_args(vars(opt))
        # 输出所有训练参数 / 参数以彩色的方式表现
        check_git_status()
        # 检查代码版本是否更新
        check_requirements()
        # 检查安装是否都安装了 requirements.txt， 缺少安装包安装。
        # 缺少安装包：建议使用 pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

    ############################################### 2. Resume 是否进行断点训练 ##################################################
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        # 初始化可视化工具wandb,wandb使用教程看https://zhuanlan.zhihu.com/p/266337608
        # 断点训练使用教程可以查看：https://blog.csdn.net/CharmsLUO/article/details/123410081
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        # isinstance()是否是已经知道的类型
        # 如果resume是True，则通过get_lastest_run()函数找到runs为文件夹中最近的权重文件last.pt
        opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
        #如果opt_yaml是文件就打开，相关的opt参数也要替换成last.pt中的opt参数 safe_load()yaml文件加载数据
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            # 不使用断点训练就在加载输入的参数
            d = torch.load(last, map_location='cpu')['opt']
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        # 保存相关信息

    ############################################## 3.DDP mode 是否分布式训练 ###############################################
    device = select_device(opt.device, batch_size=opt.batch_size)
    #选择设备cpu/gpu
    if LOCAL_RANK != -1:
        # 多卡训练GP
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        # 根据编号选择设备
        # 使用torch.cuda.set_device()可以更方便地将模型和数据加载到对应GPU上, 直接定义模型之前加入一行代码即可
        # torch.cuda.set_device(gpu_id) #单卡
        # torch.cuda.set_device('cuda:'+str(gpu_ids)) #可指定多卡
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
        # 初始化多进程

    ################################################ 4. Train 是否进化训练/遗传算法调参 #################################################
    if not opt.evolve:
        # 不设置evolve直接调用train训练
        train(opt.hyp, opt, device, callbacks)

    # Evolve hyperparameters (optional)
    # 遗传净化算法/一边训练一遍进化
    # 了解遗传算法可以查看我的博客：https://blog.csdn.net/CharmsLUO/article/details/123542598
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        # 超参数列表(突变范围 - 最小值 - 最大值)
        meta = {
            'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
            'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
            'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
            'box': (1, 0.02, 0.2),  # box loss gain
            'cls': (1, 0.2, 4.0),  # cls loss gain
            'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
            'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
            'iou_t': (0, 0.1, 0.7),  # IoU training threshold
            'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
            'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
            'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
            'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
            'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
            'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
            'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
            'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
            'mixup': (1, 0.0, 1.0),  # image mixup (probability)
            'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            # 加载yaml超参数
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        if opt.noautoanchor:
            del hyp['anchors'], meta['anchors']
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv' # 保存进化的超参数列表
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {evolve_csv}')  # download evolve.csv if exists

        """
           遗传算法调参：遵循适者生存、优胜劣汰的法则，即寻优过程中保留有用的，去除无用的。
           遗传算法需要提前设置4个参数: 群体大小/进化代数/交叉概率/变异概率

        """
        for _ in range(opt.evolve):  # generations to evolve 默认选择进化300代
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # 进化方式--single / --weight
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                # 加载evolve.txt文件
                n = min(5, len(x))  # number of previous results to consider
                # 选取进化结果代数
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                # 根据resluts计算hyp权重
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate 获取突变初始值
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)  设置突变
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate
                    # 将突变添加到base hyp上
                    # [i+7]是因为x中前7个数字为results的指标(P,R,mAP,F1,test_loss=(box,obj,cls)),之后才是超参数hyp

            # Constrain to limits
            # 限制超参再规定范围
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            # 训练 使用突变后的参超 测试其效果
            results = train(hyp.copy(), opt, device, callbacks)
            callbacks = Callbacks()
            # Write mutation results
            # 将结果写入results 并将对应的hyp写到evolve.txt evolve.txt中每一行为一次进化的结果
            # 每行前七个数字 (P, R, mAP, F1, test_losses(GIOU, obj, cls)) 之后为hyp
            # 保存hyp到yaml文件
            keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss',
                    'val/obj_loss', 'val/cls_loss')
            print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Usage example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # 执行这个脚本/ 调用train函数 / 开启训练
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items(): # setattr() 赋值属性，属性不存在则创建一个赋值
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

#能使用到的参数
'''
    python train.py --cfg yolov5l_mchar.yaml --weights ./weights/yolov5s.pt  --data ./data/mchar.yaml --epoch 200 --batch-size 8 --rect --noval --evolve 300 --image-weights --multi-scale --optimizer Adam --cos-lr --freeze 3 --bbox_interval 20
'''