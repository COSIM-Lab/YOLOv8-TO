# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops
import torch

def print_tensor_shapes(item, index_path=None):
    if index_path is None:
        index_path = []  # Initialize the index path for the top-level call

    if isinstance(item, torch.Tensor):
        # Print the index path and shape if the item is a tensor
        print(f"Tensor at Index Path {index_path}: Shape {item.shape}")
    elif isinstance(item, (list, tuple)):
        # Recursively call the function for nested lists and tuples
        for i, sub_item in enumerate(item):
            print_tensor_shapes(sub_item, index_path + [i])
    else:
        # Print the type of the item if it is not a tensor, list, or tuple
        print(f"Item at Index Path {index_path} is not a tensor, list, or tuple. It is a {type(item)}.")


class SegmentationPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.segment import SegmentationPredictor

        args = dict(model='yolov8n-seg.pt', source=ASSETS)
        predictor = SegmentationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """
    

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'segment'


    def postprocess(self, preds, img, orig_imgs):
        #print(preds[0].shape)
        regression_preds = preds[1][-1]
        p, final_reg = ops.non_max_suppression(prediction=preds[0],
                                               mask_coef = preds[1][1],
                                               proto = preds[1][-2],
                                               img_shape = img.shape[2:],
                                       conf_thres=self.args.conf,
                                       iou_thres=self.args.iou,
                                       agnostic=self.args.agnostic_nms,
                                       max_det=self.args.max_det,
                                       nc=len(self.model.names),
                                       regression_var=regression_preds,
                                       classes=self.args.classes)
        #print(p[0].shape)
        results = []
        is_list = isinstance(orig_imgs, list)  # input images are a list, not a torch.Tensor
        if len(preds[1])==3:
            proto = preds[1][-1]
        elif len(preds[1])==4:
            proto = preds[1][-2] 
        else:
            proto = preds[1]

        #print(regression_preds.shape)
        for i, pred in enumerate(p):
            orig_img = orig_imgs[i] if is_list else orig_imgs
            img_path = self.batch[0][i]

            
            if not len(pred):  # save empty boxes
                masks = None
            elif self.args.retina_masks:
                if is_list:
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                if is_list:
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            #print(masks.shape)
            #print(final_reg[i].shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks, regression_preds=final_reg[i]))
        return results
