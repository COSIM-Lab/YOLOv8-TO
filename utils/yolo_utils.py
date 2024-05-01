import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


class CustomTverskyLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.9, size_average=True):
        super(CustomTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.size_average = size_average

    def forward(self, inputs, targets, smooth=1):
        # If your model contains a sigmoid or equivalent activation layer, comment this line
        # inputs = F.sigmoid(inputs)

        # Check if the input tensors are of expected shape
        if inputs.shape != targets.shape:
            raise ValueError("Shape mismatch: inputs and targets must have the same shape")

        # Compute Tversky loss for each sample in the batch
        tversky_loss_values = []
        for input_sample, target_sample in zip(inputs, targets):
            # Flatten tensors for each sample
            input_sample = input_sample.view(-1)
            target_sample = target_sample.view(-1)

            # Calculate the true positives, false positives, and false negatives
            true_positives = (input_sample * target_sample).sum()
            false_positives = (input_sample * (1 - target_sample)).sum()
            false_negatives = ((1 - input_sample) * target_sample).sum()

            # Compute the Tversky index for each sample
            tversky_index = (true_positives + smooth) / (true_positives + self.alpha * false_positives + self.beta * false_negatives + smooth)

            tversky_loss_values.append(1 - tversky_index)

        # Convert list of Tversky loss values to a tensor
        tversky_loss_values = torch.stack(tversky_loss_values)

        # If you want the average loss over the batch to be returned
        if self.size_average:
            return tversky_loss_values.mean()
        else:
            # If you want individual losses for each sample in the batch
            return tversky_loss_values

class CustomDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CustomDiceLoss, self).__init__()
        self.size_average = size_average
    def forward(self, inputs, targets, smooth=1):
        
        # If your model contains a sigmoid or equivalent activation layer, comment this line
        #inputs = F.sigmoid(inputs)       
      
        # Check if the input tensors are of expected shape
        if inputs.shape != targets.shape:
            raise ValueError("Shape mismatch: inputs and targets must have the same shape")

        # Compute Dice loss for each sample in the batch
        dice_loss_values = []
        for input_sample, target_sample in zip(inputs, targets):
            
            # Flatten tensors for each sample
            input_sample = input_sample.view(-1)
            target_sample = target_sample.view(-1)

            intersection = (input_sample * target_sample).sum()
            dice = (2. * intersection + smooth) / (input_sample.sum() + target_sample.sum() + smooth)
            
            dice_loss_values.append(1 - dice)

        # Convert list of Dice loss values to a tensor
        dice_loss_values = torch.stack(dice_loss_values)

        # If you want the average loss over the batch to be returned
        if self.size_average:
            return dice_loss_values.mean()
        else:
            # If you want individual losses for each sample in the batch
            return dice_loss_values

def smooth_heaviside(phi, alpha, epsilon):
    # Scale and shift phi for the sigmoid function
    scaled_phi = (phi - alpha) / epsilon
    
    # Apply the sigmoid function
    H = torch.sigmoid(scaled_phi)

    return H
def calc_Phi(variable, LSgrid):
    device = variable.device  # Get the device of the variable

    x0 = variable[0]
    y0 = variable[1]
    L = variable[2]
    t = variable[3]  # Constant thickness
    angle = variable[4]

    # Rotation
    st = torch.sin(angle)
    ct = torch.cos(angle)
    x1 = ct * (LSgrid[0][:, None].to(device) - x0) + st * (LSgrid[1][:, None].to(device) - y0) 
    y1 = -st * (LSgrid[0][:, None].to(device) - x0) + ct * (LSgrid[1][:, None].to(device) - y0)

    # Regularized hyperellipse equation
    a = L / 2  # Semi-major axis
    b = t / 2  # Constant semi-minor axis
    small_constant = 1e-9  # To avoid division by zero
    temp = ((x1 / (a + small_constant))**6) + ((y1 / (b + small_constant))**6)

    # # Ensuring the hyperellipse shape
    allPhi = 1 - (temp + small_constant)**(1/6)
    
    # # Call Heaviside function with allPhi
    alpha = torch.tensor(0.0, device=device, dtype=torch.float32)
    epsilon = torch.tensor(0.001, device=device, dtype=torch.float32)
    H_phi = smooth_heaviside(allPhi, alpha, epsilon)
    return allPhi, H_phi



# utils.py

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from PIL import Image

def preprocess_image_pil(image, threshold_value=0.9, upscale=False, upscale_factor=2.0):
    # Ensure the image is in grayscale mode
    if image.mode != 'L':
        image = image.convert('L')

    # Apply threshold
    image = image.point(lambda x: 255 if x > threshold_value * 255 else 0, '1')
    
    # Upscale if requested
    if upscale:
        image = image.resize(
            (int(image.width * upscale_factor), int(image.height * upscale_factor)),
            resample=Image.BICUBIC
        )
    
    return image

def preprocess_image(image_path, threshold_value=0.9, upscale=False, upscale_factor=2.0):
    image = Image.open(image_path).convert('L')
    image = image.point(lambda x: 255 if x > threshold_value * 255 else 0, '1')
    
    if upscale:
        image = image.resize(
            (int(image.width * upscale_factor), int(image.height * upscale_factor)),
            resample=Image.BICUBIC
        )
    
    return image

def run_model(model, image, conf=0.05, iou=0.5, imgsz=640):
    results = model(image, conf=conf, iou=iou, imgsz=imgsz)
    return results


def process_results(results, input_image):
    diceloss = CustomDiceLoss()
    tverskyloss = CustomTverskyLoss()

    prediction_tensor = results[0].regression_preds.to('cpu').detach()
    input_image_array = np.array(input_image.convert('L'))
    input_image_array_tensor = torch.tensor(input_image_array) / 255.0
    input_image_array_tensor = 1.0 - input_image_array_tensor
    input_image_array_tensor = torch.flip(input_image_array_tensor, [0])
    
    for r in results:
        im_array = r.plot(boxes=True, labels=False, line_width=1)
        seg_result = Image.fromarray(im_array[..., ::-1])
    
    DH = input_image_array.shape[0] / min(input_image_array.shape[1], input_image_array.shape[0])
    DW = input_image_array.shape[1] / min(input_image_array.shape[1], input_image_array.shape[0])
    nelx = input_image_array.shape[1] - 1
    nely = input_image_array.shape[0] - 1
    
    x, y = torch.meshgrid(torch.linspace(0, DW, nelx+1), torch.linspace(0, DH, nely+1))
    LSgrid = torch.stack((x.flatten(), y.flatten()), dim=0)
    
    pred_bboxes = results[0].boxes.xyxyn.to('cpu').detach()
    constant_tensor_02 = torch.full((pred_bboxes.shape[0],), 0.2)
    constant_tensor_00 = torch.full((pred_bboxes.shape[0],), 0.001)
    
    xmax = torch.stack([pred_bboxes[:,2]*(DW*1.0), pred_bboxes[:,3]*(DH*1.0), pred_bboxes[:,2]*(DW*1.0), pred_bboxes[:,3]*(DH*1.0), constant_tensor_02], dim=1)
    xmin = torch.stack([pred_bboxes[:,0]*(DW*1.0), pred_bboxes[:,1]*(DH*1.0), pred_bboxes[:,0]*(DW*1.0), pred_bboxes[:,1]*(DH*1.0), constant_tensor_00], dim=1)
    
    unnormalized_preds = prediction_tensor * (xmax - xmin) + xmin
    
    x_center = (unnormalized_preds[:, 0] + unnormalized_preds[:, 2]) / 2
    y_center = (unnormalized_preds[:, 1] + unnormalized_preds[:, 3]) / 2
    
    L = torch.sqrt((unnormalized_preds[:, 0] - unnormalized_preds[:, 2])**2 + 
                (unnormalized_preds[:, 1] - unnormalized_preds[:, 3])**2)
    
    L = L + 1e-4
    t_1 = unnormalized_preds[:, 4]
    
    epsilon = 1e-10
    y_diff = unnormalized_preds[:, 3] - unnormalized_preds[:, 1] + epsilon
    x_diff = unnormalized_preds[:, 2] - unnormalized_preds[:, 0] + epsilon
    theta = torch.atan2(y_diff, x_diff)
    
    formatted_variables = torch.cat((x_center.unsqueeze(1), 
                        y_center.unsqueeze(1), 
                        L.unsqueeze(1), 
                        t_1.unsqueeze(1), 
                        theta.unsqueeze(1)), dim=1)
    
    pred_Phi, pred_H = calc_Phi(formatted_variables.T, LSgrid)
    
    sum_pred_H = torch.sum(pred_H.detach().cpu(), dim=1)
    sum_pred_H[sum_pred_H > 1] = 1
    
    final_H = np.flipud(sum_pred_H.detach().numpy().reshape((nely+1, nelx+1), order='F'))
    
    dice_loss = diceloss(torch.tensor(final_H.copy()), input_image_array_tensor)
    tversky_loss = tverskyloss(torch.tensor(final_H.copy()), input_image_array_tensor)
    
    return input_image_array_tensor, seg_result, pred_Phi, sum_pred_H, final_H, dice_loss, tversky_loss

def plot_results(input_image_array_tensor, seg_result, pred_Phi, sum_pred_H, final_H, dice_loss, tversky_loss, filename='combined_plots.png'):
    nelx = input_image_array_tensor.shape[1] - 1
    nely = input_image_array_tensor.shape[0] - 1
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    axes[0, 0].imshow(input_image_array_tensor.squeeze(), origin='lower', cmap='gray_r')
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('on')
    
    axes[0, 1].imshow(seg_result)
    axes[0, 1].set_title('Segmentation Result')
    axes[0, 1].axis('off')
    
    render_colors1 = ['yellow', 'g', 'r', 'c', 'm', 'y', 'black', 'orange', 'pink', 'cyan', 'slategrey', 'wheat', 'purple', 'mediumturquoise', 'darkviolet', 'orangered']
    for i, color in zip(range(0, pred_Phi.shape[1]), render_colors1*100):
        axes[1, 1].contourf(np.flipud(pred_Phi[:, i].numpy().reshape((nely+1, nelx+1), order='F')), [0, 1], colors=color)
    axes[1, 1].set_title('Prediction contours')
    axes[1, 1].set_aspect('equal')
    
    axes[1, 0].imshow(np.flipud(sum_pred_H.detach().numpy().reshape((nely+1, nelx+1), order='F')), origin='lower', cmap='gray_r')
    axes[1, 0].set_title('Prediction Projection')
    
    plt.subplots_adjust(hspace=0.3, wspace=0.01)
    
    plt.figtext(0.5, 0.05, f'Dice Loss: {dice_loss.item():.4f}', ha='center', fontsize=16)
    
    fig.savefig(filename, dpi=600)


import numpy as np
from PIL import Image
import io

def plot_results_gradio(input_image_array_tensor, seg_result, pred_Phi, sum_pred_H, final_H, dice_loss, tversky_loss):
    nelx = input_image_array_tensor.shape[1] - 1
    nely = input_image_array_tensor.shape[0] - 1
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    axes[0, 0].imshow(input_image_array_tensor.squeeze(), origin='lower', cmap='gray_r')
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('on')
    
    axes[0, 1].imshow(seg_result)
    axes[0, 1].set_title('Segmentation Result')
    axes[0, 1].axis('off')
    
    render_colors1 = ['yellow', 'g', 'r', 'c', 'm', 'y', 'black', 'orange', 'pink', 'cyan', 'slategrey', 'wheat', 'purple', 'mediumturquoise', 'darkviolet', 'orangered']
    for i, color in zip(range(0, pred_Phi.shape[1]), render_colors1*100):
        axes[1, 1].contourf(np.flipud(pred_Phi[:, i].numpy().reshape((nely+1, nelx+1), order='F')), [0, 1], colors=color)
    axes[1, 1].set_title('Prediction contours')
    axes[1, 1].set_aspect('equal')
    
    axes[1, 0].imshow(np.flipud(sum_pred_H.detach().numpy().reshape((nely+1, nelx+1), order='F')), origin='lower', cmap='gray_r')
    axes[1, 0].set_title('Prediction Projection')
    
    plt.subplots_adjust(hspace=0.3, wspace=0.01)
    plt.figtext(0.5, 0.05, f'Dice Loss: {dice_loss.item():.4f}', ha='center', fontsize=16)
    
    # Convert figure to a PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return img
