# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import os
import sys
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS
from utils.augmentations import letterbox
from utils.general import (check_img_size, non_max_suppression, non_max_suppression_obb, scale_polys)
from utils.plots import Annotator
from utils.torch_utils import select_device, time_sync
from utils.rboxs_utils import rbox2poly


@torch.no_grad()
class YOLOv5_OBB:
  def __init__(self,weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
          source=ROOT / 'data path',  # file/dir/URL/glob, 0 for webcam
          imgsz=(640, 640), # image shape
          #img_array=[], # image input
          conf_thres=0.25,  # confidence threshold
          iou_thres=0.45,  # NMS IOU threshold
          max_det=1000,  # maximum detections per image
          device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
          view_img=False,  # show results
          save_txt=False,  # save results to *.txt
          save_conf=False,  # save confidences in --save-txt labels
          save_crop=False,  # save cropped prediction boxes
          nosave=False,  # do not save images/videos
          classes=None,  # filter by class: --class 0, or --class 0 2 3
          agnostic_nms=False,  # class-agnostic NMS
          augment=False,  # augmented inference
          visualize=False,  # visualize features
          update=False,  # update all models
          project=ROOT / 'runs/detect',  # save results to project/name
          name='exp',  # save results to project/name
          exist_ok=False,  # existing project/name ok, do not increment
          line_thickness=3,  # bounding box thickness (pixels)
          hide_labels=False,  # hide labels
          hide_conf=False,  # hide confidences
          half=False,  # use FP16 half-precision inference
          dnn=False,  # use OpenCV DNN for ONNX inference
          ):

      self.weights = weights
      self.source = source
      self.imgsz = imgsz
      self.conf_thres = conf_thres
      self.iou_thres = iou_thres
      self.max_det = max_det
      self.device = device
      self.view_img = view_img
      self.save_txt = save_txt # Completamente necesario para obtener datos
      self.save_conf = save_conf
      self.save_crop = save_crop
      self.nosave = nosave
      self.classes = classes
      self.agnostic_nms = agnostic_nms
      self.augment = augment
      self.visualize = visualize
      self.update = update
      self.project = project
      self.name = name
      self.exist_ok = exist_ok
      self.line_thickness = line_thickness
      self.hide_labels = hide_labels
      self.hide_conf = hide_conf
      self.half = half
      self.dnn = dnn


  def run(self, img_array= []):
      self.img = img_array
      source = str(self.source)
      is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
      
      # Load model
      device = select_device(self.device)
      model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn)
      stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
      imgsz = check_img_size(self.imgsz, s=stride)  # check image size

      # Half
      self.half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
      if pt or jit:
          model.model.half() if self.half else model.model.float()

      # Run inference
      model.warmup(imgsz=(1, 3, *imgsz), half=self.half)  # warmup
      dt, seen = [0.0, 0.0, 0.0], 0
      all_data = []

      path = self.source
      im = letterbox(self.img)[0]
      im = im.transpose((2, 0, 1))[::-1]
      im = np.ascontiguousarray(im)
      vid_cap = None
      s = 'None'
      dataset_count = 0

      t1 = time_sync()
      im = torch.from_numpy(im).to(device)
      im = im.half() if self.half else im.float()  # uint8 to fp16/32
      im /= 255  # 0 - 255 to 0.0 - 1.0
      if len(im.shape) == 3:
          im = im[None]  # expand for batch dim
      t2 = time_sync()
      dt[0] += t2 - t1

      # Inference
      pred = model(im, augment=self.augment, visualize=False)
      t3 = time_sync()
      dt[1] += t3 - t2

      # NMS
      # pred: list*(n, [xylsθ, conf, cls]) θ ∈ [-pi/2, pi/2)
      pred = non_max_suppression_obb(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, multi_label=True, max_det=self.max_det)
      dt[2] += time_sync() - t3

      # Process predictions
      step_data = []
      for i, det in enumerate(pred):  # per image
          pred_poly = rbox2poly(det[:, :5]) # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
          seen += 1
          p, im0, frame = path, self.img.copy(), 0

          p = Path(p)  # to Path
          s += '%gx%g ' % im.shape[2:] 
          gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
          imc = im0.copy() if self.save_crop else im0  # for save_crop
          annotator = Annotator(im0, line_width=self.line_thickness, example=str(names))
          if len(det):
              # Rescale polys from img_size to im0 size
              pred_poly = scale_polys(im.shape[2:], pred_poly, im0.shape)
              det = torch.cat((pred_poly, det[:, -2:]), dim=1) # (n, [poly conf cls])

              # Print results
              for c in det[:, -1].unique():
                  n = (det[:, -1] == c).sum()  # detections per class
                  s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

              # Write results
              for *poly, conf, cls in reversed(det):
                  if self.save_txt:  # Write to file
                      line = (cls, *poly, conf) if self.save_conf else (cls, *poly)  # label format
                      var = ('%g ' * len(line)).rstrip() % line + " " + str(np.round(conf.item(),4))
                      res = [float(value) for value in var.split(' ')][1:]
                  step_data.append(res)
      all_data.append( np.reshape(step_data,(-1,9)) )
      
      if is_file == False:
        return np.array(all_data)
      else:
        return np.reshape(step_data,(-1,9)) 