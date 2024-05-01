import gradio as gr
import sys
sys.path.append('./utils')

from yolo_utils import preprocess_image_pil, run_model, process_results, plot_results_gradio
import matplotlib.pyplot as plt
import io
try:
    from ultralytics import YOLO
except ImportError:
    import os
    os.system('pip install ./yolov8-to')
    from ultralytics import YOLO

def process_image(image,conf,iou):
    model = YOLO('./trained_models/nano.pt')
    # Preprocess the image
    preprocessed_image = preprocess_image_pil(image, threshold_value=0.9, upscale=False)

    # Run the model
    results = run_model(model, preprocessed_image, conf=conf, iou=iou, imgsz=640)

    # Process the results
    input_image_array_tensor, seg_result, pred_Phi, sum_pred_H, final_H, dice_loss, tversky_loss = process_results(results, preprocessed_image)

    # Plot the results
    fig = plot_results_gradio(input_image_array_tensor, seg_result, pred_Phi, sum_pred_H, final_H, dice_loss, tversky_loss)

    # Convert the plot to an image

    return fig

# Create the Gradio interface
title = "YOLOV8-TO Demo App"
description = "Upload an image and see the processed results. Adjust the confidence and IOU thresholds as needed."

iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type='pil'),
        gr.Slider(minimum=0, maximum=1, value=0.1, label="Confidence Threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.5, label="IOU Threshold")
    ],
    outputs="image",
    title=title,
    description=description
)

iface.launch()
