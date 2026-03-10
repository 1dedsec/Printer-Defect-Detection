import argparse
import cv2
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import os

sys.path.append("/home/ubuntu/yolo/")
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import time
import logging
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, \
    process_mask_native
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box


# from models.common import DetectMultiBackend
# from trackers.multi_tracker_zoo import create_tracker


def tlwh2center(bbox):
    if len(bbox) < 4:
        return
    center = [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)]
    return center


def iou(bbox1, bbox2, center=False):
    """Compute the iou of two boxes.
    Parameters
    ----------
    bbox1, bbox2: list.
        The bounding box coordinates: [xmin, ymin, xmax, ymax] or [xcenter, ycenter, w, h].
    center: str, default is 'False'.
        The format of coordinate.
        center=False: [xmin, ymin, xmax, ymax]
        center=True: [xcenter, ycenter, w, h]
    Returns
    -------
    iou: float.
        The iou of bbox1 and bbox2.
    """
    if center == False:
        xmin1, ymin1, xmax1, ymax1 = bbox1
        xmin2, ymin2, xmax2, ymax2 = bbox2
    else:
        xmin1, ymin1 = int(bbox1[0] - bbox1[2] / 2.0), int(bbox1[1] - bbox1[3] / 2.0)
        xmax1, ymax1 = int(bbox1[0] + bbox1[2] / 2.0), int(bbox1[1] + bbox1[3] / 2.0)
        xmin2, ymin2 = int(bbox2[0] - bbox2[2] / 2.0), int(bbox2[1] - bbox2[3] / 2.0)
        xmax2, ymax2 = int(bbox2[0] + bbox2[2] / 2.0), int(bbox2[1] + bbox2[3] / 2.0)

    # 获取矩形框交集对应的顶点坐标(intersection)
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    # 计算两个矩形框面积
    area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)

    # 计算交集面积 
    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    # 计算交并比
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)
    return iou


def iou_detect(output_det, output_sev, iou_thres):  # [bbox,conf,cls]
    det_numpy = []
    sev_numpy = []
    for det in output_det[0]:
        det_numpy.append(det.cpu().numpy())
    for sev in output_sev[0]:
        sev_numpy.append(sev.cpu().numpy())
    d = np.zeros((len(det_numpy), 2))
    output = det_numpy.copy()
    output = np.concatenate((output, d), axis=1)
    if output_sev is not None and len(sev_numpy):
        for i, sev in enumerate(sev_numpy):
            bbox_sev = sev[0:4]
            for j, det in enumerate(det_numpy):
                bbox_det = det[0:4]
                if iou(bbox_sev, bbox_det) > iou_thres:
                    output[j][6] = sev[4]
                    output[j][7] = sev[5]
    return output  # [:,8]


@torch.no_grad()
def run(
        source='0',
        yolo_weights_det=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        yolo_weights_sev=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=True,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs' / 'track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
):
    # 导入处理视频流或图像序列
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights_det, list):  # single yolo model
        exp_name = yolo_weights_det.stem
    elif type(yolo_weights_det) is list and len(yolo_weights_det) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights_det[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + yolo_weights_det.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    is_seg = '-seg' in str(yolo_weights_det)
    model_det = AutoBackend(yolo_weights_det, device=device, dnn=dnn, fp16=half)
    model_sev = AutoBackend(yolo_weights_sev, device=device, dnn=dnn, fp16=half)
    stride_det, names_det, pt_det = model_det.stride, model_det.names, model_det.pt
    stride_sev, names_sev, pt_sev = model_sev.stride, model_sev.names, model_sev.pt
    imgsz = check_imgsz(imgsz, stride=stride_det)  # check image size
    # batch-size = 1，一帧一帧处理
    # Dataloader
    bs = 1
    if webcam:  # 视频摄像头
        show_vid = check_imshow(warn=True)
        dataset = LoadStreams(
            source,
            imgsz=imgsz,
            # stride=stride_det,
            auto=pt_det,
            transforms=getattr(model_det.model, 'transforms', None),
            vid_stride=vid_stride
        )
        bs = len(dataset)
    else:
        dataset = LoadImages(
            source,
            imgsz=imgsz,
            # stride=stride_det,
            # auto=pt_det,
            # transforms=getattr(model_det.model, 'transforms', None),
            vid_stride=vid_stride
        )
    vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs
    model_det.warmup(imgsz=(1 if pt_det or model_det.triton else bs, 3, *imgsz))  # warmup

    # Run tracking
    # model_det.warmup(imgsz=(1 if pt_det else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    for frame_idx, batch in enumerate(dataset):
        # if (frame_idx + 1) % 50 == 0:
        # highthrow_detector.output_track()
        path, im, vid_cap, s = batch
        im0s = im[0].copy()
        im = cv2.resize(im[0], imgsz)
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        with torch.no_grad():
            with dt[0]:
                im = torch.from_numpy(im).to(device).unsqueeze(0).permute(0, 3, 1, 2)
                im = im.half() if half else im.float()  # uint8 to fp16/32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
            # print(im.size())
            # Inference
            with dt[1]:
                preds_det = model_det(im, augment=augment, visualize=visualize)
                preds_sev = model_sev(im, augment=augment, visualize=visualize)

                # Apply NMS
            with dt[2]:
                # if is_seg:
                #     masks = []
                #     p_det = non_max_suppression(preds[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det,
                #                             nm=32)
                #     proto = preds[1][-1]
                # else:
                p_det = non_max_suppression(preds_det, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                p_sev = non_max_suppression(preds_sev, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process detections
        output_det = []
        output_sev = []
        for i, det in enumerate(p_det):  # detections per image
            seen += 1
            if webcam:  # bs >= 1
                p, im0, _ = path[i], im0s, dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path[0], im0s, getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string

            if det is not None and len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()  # rescale boxes to im0 size

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names_det[int(c)]}{'s' * (n > 1)}, "  # add to string
                output_det.append(det)
                # # draw boxes for visualization
                # if len(det) > 0:
                #     # if is_seg:
                #     #     # Mask plotting
                #     #     annotator.masks(
                #     #         masks[i],
                #     #         colors=[colors(x, True) for x in det[:, 5]],
                #     #         im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(
                #     #             0).contiguous() /
                #     #                255 if retina_masks else im[i]
                #     #     )
                #     for j, (output) in enumerate(det[i]):

                #         bbox = output[0:4]
                #         id = output[4]
                #         cls = output[5]
                #         conf = output[6]
                #         if save_txt:
                #             # to MOT format
                #             # bbox_left = output[0]
                #             # bbox_top = output[1]
                #             # bbox_w = output[2] - output[0]
                #             # bbox_h = output[3] - output[1]
                #             # Write MOT compliant results to file
                #             # with open(txt_path + '.txt', 'a') as f:
                #             #     f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                #             #                                    bbox_top, bbox_w, bbox_h, -1, -1, -1, i))
                #             pass
                # if save_vid or save_crop or show_vid:  # Add bbox/seg to image
                #     c = int(cls)  # integer class
                #     id = int(id)  # integer id
                #     label = None if hide_labels else (f'{id} {names_det[c]}' if hide_conf else \
                #                                           (
                #                                               f'{id} {conf:.2f}' if hide_class else f'{id} {names_det[c]} {conf:.2f}'))
                #     color = colors(c, True)
                #     annotator.box_label(bbox, label, color=color)

                #             if save_crop:
                #                 txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                #                 save_one_box(np.array(bbox, dtype=np.int16), imc,
                #                              file=save_dir / 'crops' / txt_file_name / names[
                #                                  c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
            else:
                pass
        for i, det in enumerate(p_sev):  # detections per image
            if det is not None and len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()  # rescale boxes to im0 size

                # Print results
                # for c in det[:, 5].unique():
                #     n = (det[:, 5] == c).sum()  # detections per class
                #     s += f"{n} {names_det[int(c)]}{'s' * (n > 1)}, "  # add to string
                output_sev.append(det)
            else:
                pass
        out = []
        if output_det is not None and len(output_det):
            out = iou_detect(output_det, output_sev, iou_thres)
        annotator = Annotator(im0s, line_width=line_thickness, example=str(names_det))
        if out is not None and len(out):
            for i, output in enumerate(out):
                xyxy = output[0:4]
                conf_det = output[4]
                cls_det = output[5]
                conf_sev = output[6]
                cls_sev = output[7]

                if save_vid or save_crop or show_vid:  # Add bbox/seg to image
                    c_det = int(cls_det)  # integer class
                    c_sev = int(cls_sev)
                    if hide_labels:
                        label = None
                    elif conf_sev == 0:
                        label = (f'{names_det[c_det]} {conf_det:.2f} slight')
                    else:
                        label = (f'{names_det[c_det]} {conf_det:.2f} {names_sev[c_sev]} {conf_sev:.2f}')
                    color = colors(c_det, True)
                    annotator.box_label(xyxy, label, color=color)
                    # Stream results
        im0 = annotator.result()
        if show_vid:
            # if platform.system() == 'Linux' and p not in windows:
            windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            # cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                exit()

        # Save results (image with detections)
        if save_vid:
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

        # Print total time (preprocessing + inference + NMS + tracking)
        LOGGER.info(
            f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(
        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list((save_dir / 'tracks').glob('*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights-det', nargs='+', type=Path,
                        default='yolo/ultralytics/runs/detect/train4/weights/best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--yolo-weights-sev', nargs='+', type=Path,
                        default='yolo/ultralytics/runs/detect/train4/weights/best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='/home/ubuntu/wall_bulg.mp4',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.25, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', default=False, help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', default=False, help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    parser.add_argument('--save-vid', action='store_true', default=False, help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+',
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', default=True, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
