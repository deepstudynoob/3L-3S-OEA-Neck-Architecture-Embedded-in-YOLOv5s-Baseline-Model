# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL 权重文件地址 默认 weights/可以是自己的路径
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)  自带电脑摄像头， 默认data/images/
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path 文件路径，包括类别/图片/标签等信息
        imgsz=(640, 640),  # inference size (height, width) 输入图片的大小 默认640*640
        conf_thres=0.25,  # confidence threshold 置信度阈值 默认0.25  用在nms中
        iou_thres=0.45,  # NMS IOU threshold 做nms的iou阈值 默认0.45   用在nms中
        max_det=1000,  # maximum detections per image 每张图片最多的目标数量  用在nms中
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu 设置代码执行的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results 是否展示预测之后的图片或视频 默认False
        save_txt=False,  # save results to *.txt 是否将预测的框坐标以txt文件形式保存, 默认False, 使用--save-txt 在路径runs/detect/exp*/labels/*.txt下生成每张图片预测的txt文件
        save_conf=False,  # save confidences in --save-txt labels 是否将置信度conf也保存到txt中, 默认False
        save_crop=False,  # save cropped prediction boxes 是否保存裁剪预测框图片, 默认为False, 使用--save-crop 在runs/detect/exp*/crop/剪切类别文件夹/ 路径下会保存每个接下来的目标
        nosave=False,  # do not save images/videos 不保存图片、视频, 要保存图片，不设置--nosave 在runs/detect/exp*/会出现预测的结果
        classes=None,  # filter by class: --class 0, or --class 0 2 3 设置只保留某一部分类别, 形如0或者0 2 3, 使用--classes = n, 则在路径runs/detect/exp*/下保存的图片为n所对应的类别, 此时需要设置data
        agnostic_nms=False,  # class-agnostic NMS 进行NMS去除不同类别之间的框, 默认False
        augment=False,  # augmented inference TTA测试时增强/多尺度预测，可以提分
        visualize=False,  # visualize features 是否可视化网络层输出特征
        update=False,  # update all models 如果为True,则对所有模型进行strip_optimizer操作,去除pt文件中的优化器等信息,默认为False
        project=ROOT / 'runs/detect',  # save results to project/name 保存测试日志的文件夹路径
        name='exp',  # save results to project/name 每次实验的名称
        exist_ok=False,  # existing project/name ok, do not increment 是否重新创建日志文件, False时重新创建文件
        line_thickness=3,  # bounding box thickness (pixels) 画框的线条粗细
        hide_labels=False,  # hide labels 可视化时隐藏预测类别
        hide_conf=False,  # hide confidences 可视化时隐藏置信度
        half=False,  # use FP16 half-precision inference 是否使用F16精度推理, 半进度提高检测速度
        dnn=False,  # use OpenCV DNN for ONNX inference 用OpenCV DNN预测
        vid_stride=1,  # video frame-rate stride
):
    ################################################# 1. 初始化配置 #####################################################
    source = str(source)
    # 输入的路径变为字符串
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # 是否保存图片和txt文件
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) # 提取文件后缀名是否符合要求的文件，例如：是否格式是jpg, png, asf, avi等
    # 判断文件是否是视频流
    # Path()提取文件名 例如：Path("./data/test_images/bus.jpg") Path.name->bus.jpg Path.parent->./data/test_images Path.suffix->.jpg
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # .lower()转化成小写 .upper()转化成大写 .title()首字符转化成大写，其余为小写, .startswith('http://')返回True or Flase
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download
        # 返回文件

    # Directories # 预测路径是否存在，不存在新建，按照实验文件以此递增新建
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model # 获取设备 CPU/CUDA
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    # 检测编译框架PYTORCH/TENSORFLOW/TENSORRT
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # 确保输入图片的尺寸imgsz能整除stride=32 如果不能则调整为能被整除并返回

    ################################################# 2. 加载数据 #####################################################
    # Dataloader 加载数据
    bs = 1  # batch_size
    if webcam: # 使用视频流或者页面
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        # 直接从source文件下读取图片
    vid_path, vid_writer = [None] * bs, [None] * bs
    # 保存的路径

    # Run inference 输入预测
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            # 转化到GPU上
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            # 是否使用半精度
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
                # 增加一个维度

        # Inference 可视化文件路径
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            """
            pred.shape=(1, num_boxes, 5+num_class)
            h,w为传入网络图片的长和宽,注意dataset在检测时使用了矩形推理,所以这里h不一定等于w
            num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
            pred[..., 0:4]为预测框坐标=预测框坐标为xywh(中心点+宽长)格式
            pred[..., 4]为objectness置信度
            pred[..., 5:-1]为分类结果
            """

        # NMS
            # 非极大值抑制
            """
            pred: 网络的输出结果
            conf_thres:置信度阈值
            ou_thres:iou阈值
            classes: 是否只保留特定的类别
            agnostic_nms: 进行nms是否也去除不同类别之间的框
            max-det: 保留的最大检测框数量
            ---NMS, 预测框格式: xywh(中心点+长宽)-->xyxy(左上角右下角)
            pred是一个列表list[torch.tensor], 长度为batch_size
            每一个torch.tensor的shape为(num_boxes, 6), 内容为box + conf + cls
            """
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        # 对每张图片做处理
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                # 如果输入源是webcam则batch_size>=1 取出dataset中的一张图片
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                # 但是大部分我们一般都是从LoadImages流读取本都文件中的照片或者视频 所以batch_size=1
                # p: 当前图片/视频的绝对路径 如 F:\yolo_v5\yolov5-U\data\images\bus.jpg
                # s: 输出信息 初始为 ''
                # im0: 原始图片 letterbox + pad 之前的图片
                # frame: 视频流
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # 当前路径yolov5/data/images/
            save_path = str(save_dir / p.name)  # im.jpg
            # 图片/视频的保存路径save_path 如 runs\\detect\\exp8\\bus.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            # 设置保存框坐标的txt文件路径，每张图片对应一个框坐标信息
            s += '%gx%g ' % im.shape[2:]  # print string
            # 设置打印图片的信息
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            # 保存截图
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # 将预测信息映射到原图

                # Print results
                # 打印检测到的类别数量
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # 保存结果： txt/图片画框/crop-image
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file  # 将每个图片的预测信息分别存入save_dir/labels下的xxx.txt中 每行: class_id + score + xywh
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        # 在原图上画框 + 将预测到的目标剪切出来 保存成图片 保存在save_dir/crops下 在原图像画图或者保存结果
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                        # 在原图上画框 + 将预测到的目标剪切出来 保存成图片 保存在save_dir/crops下

            # Stream results
            im0 = annotator.result()
            if view_img: # 显示图片
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img: # 保存图片
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / r'/mnt/workspace/datasets/private-datasets/jsy/yolov5-7.0/yolov5-7.0/runs/16397/base/yolov5s_86.6/weights/last.pt', help='model path or triton URL')
    # weights: 训练的权重路径,可以使用自己训练的权重,也可以使用官网提供的权重 yolov5的x\m\l等区别就在于网络的宽度和深度
    parser.add_argument('--source', type=str, default=r'/mnt/workspace/datasets/private-datasets/jsy/yolov5-7.0/yolov5-7.0/datasets/mix', help='file/dir/URL/glob/screen/0(webcam)')
    # source: 测试数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头),也可以是rtsp等视频流, 默认data/images
    parser.add_argument('--data', type=str, default=ROOT / '/mnt/workspace/datasets/private-datasets/jsy/yolov5-7.0/yolov5-7.0/datasets/fabric/VOC_1.yaml', help='(optional) dataset.yaml path')
    # data: 配置数据文件路径, 包括image/label/classes等信息, 训练自己的文件, 需要作相应更改, 可以不用管
    # 如果设置了只显示个别类别即使用了--classes = 0 或二者1, 2, 3等, 则需要设置该文件，数字和类别相对应才能只检测某一个类
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    # imgsz: 网络输入图片大小, 默认的大小是640
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    # conf-thres: 置信度阈值， 默认为0.25
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    # iou-thres:  做nms的iou阈值, 默认为0.45
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    # max-det: 保留的最大检测框数量, 每张图片中检测目标的个数最多为1000类
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # device: 设置设备CPU/CUDA, 可以不用设置
    parser.add_argument('--view-img', action='store_true', help='show results')
    # view-img: 是否展示预测之后的图片/视频, 默认False, --view-img 电脑界面出现图片或者视频检测结果
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # save-txt: 是否将预测的框坐标以txt文件形式保存, 默认False, 使用--save-txt 在路径runs/detect/exp*/labels/*.txt下生成每张图片预测的txt文件
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # save-conf: 是否将置信度conf也保存到txt中, 默认False
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    # save-crop: 是否保存裁剪预测框图片, 默认为False, 使用--save-crop 在runs/detect/exp*/crop/剪切类别文件夹/ 路径下会保存每个接下来的目标
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # nosave: 不保存图片、视频, 要保存图片，不设置--nosave 在runs/detect/exp*/会出现预测的结果
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    #  classes: 设置只保留某一部分类别, 形如0或者0 2 3, 使用--classes = n, 则在路径runs/detect/exp*/下保存的图片为n所对应的类别, 此时需要设置data
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # agnostic-nms: 进行NMS去除不同类别之间的框, 默认False
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # augment: TTA测试时增强/多尺度预测
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    # visualize: 是否可视化网络层输出特征
    parser.add_argument('--update', action='store_true', help='update all models')
    # update: 如果为True,则对所有模型进行strip_optimizer操作,去除pt文件中的优化器等信息,默认为False
    parser.add_argument('--project', default=ROOT / '/mnt/workspace/datasets/private-datasets/jsy/basemodel/ultralytics-main-11/ultralytics-main-11/runs/0813detect/11-mix2', help='save results to project/name')
    # project:保存测试日志的文件夹路径
    parser.add_argument('--name', default='class8', help='save results to project/name')
    # name:保存测试日志文件夹的名字, 所以最终是保存在project/name中
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # exist_ok: 是否重新创建日志文件, False时重新创建文件
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    # line-thickness: 画框的线条粗细
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    # hide-labels: 可视化时隐藏预测类别
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    # hide-conf: 可视化时隐藏置信度
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # half: 是否使用F16精度推理, 半进度提高检测速度
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # dnn: 用OpenCV DNN预测
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    # 扩充维度, 如果是一位就扩充一位
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # 输出所有参数
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    # 检查环境/打印参数,主要是requrement.txt的包是否安装，用彩色显示设置的参数
    run(**vars(opt))
    # 执行run()函数


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
