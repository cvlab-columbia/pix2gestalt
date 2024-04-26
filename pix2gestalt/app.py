import os
from typing import List

import cv2
from PIL import Image
import gradio as gr
import numpy as np
import fire
import imageio
from skimage.io import imsave
from ldm.util import create_carvekit_interface
from inference import SamPredictor, get_sam_predictor, run_inference, run_sam, load_model_from_config
from omegaconf import OmegaConf

_GPU_INDEX = 2
'''
conda activate pix2gestalt
cd pix2gestalt
python app.py
'''

def select_point(predictor: SamPredictor,
                 original_img: np.ndarray,
                 raw_image: np.ndarray,
                 sel_pix: list,
                 point_type: str, 
                 evt: gr.SelectData):
  """
  When the user clicks on the image, run Segment Anything, show points, and update the modal mask.

  Args:
      predictor (SamPredictor): Sam predictor.
      original_img (np.ndarray): Input image.
      sel_pix (list): List of selected points.
      point_type (str): Point type (positive prompt: on the visible (modal) region / negative prompt).
      evt (gr.SelectData): Event data.

  Returns:
      np.ndarray: Annotated image.
      np.ndarray: Image with mask.
      np.ndarray: Mask.
  """
  img = original_img.copy()
  h_original, w_original, _ = original_img.shape
  h_new, w_new = 256, 256

  scale_x = w_new / w_original
  scale_y = h_new / h_original

  if point_type == 'positive_prompt':
    sel_pix.append((evt.index, 1)) # positive_prompt
  elif point_type == 'negative_prompt':
    sel_pix.append((evt.index, 0)) # negative_prompt
  else:
    sel_pix.append((evt.index, 1)) # default positive_prompt

  # translate points from original dimensions to (256, 256)
  processed_sel_pix = []
  for point in sel_pix:
      (x, y), label = point
      new_x = int(x * scale_x)
      new_y = int(y * scale_y)
      processed_sel_pix.append(([new_x, new_y], label))

  visible_mask, overlay_mask = run_sam(predictor, processed_sel_pix)
  resized_overlay_mask = []
  mask = np.squeeze(overlay_mask[0][0]) # (256, 256)

  resized_mask = cv2.resize(mask.astype(np.uint8) * 255, (w_original, h_original), interpolation=cv2.INTER_AREA)
  resized_mask = resized_mask > 127

  resized_overlay_mask = [(resized_mask,'visible_mask')]

  COLORS = [(255, 0, 0), (0, 255, 0)]
  MARKERS = [1, 4]
  scaling_factor = min(h_original / 256, w_original / 256)
  marker_size = int(6 * scaling_factor)
  marker_thickness = int(2 * scaling_factor)
  for point, label in sel_pix:
    cv2.drawMarker(img, point, COLORS[label], markerType=MARKERS[label], markerSize=marker_size, thickness=marker_thickness)

  # visible_mask is for pix2gestalt input (256, 256)
  # (img, overlay_mask) is for SAM mask display (original size)
  return img, (raw_image, resized_overlay_mask), visible_mask 


# undo the selected point prompt
def undo_points(predictor, orig_img, sel_pix):
  temp = orig_img.copy()
  num_current_points = len(sel_pix)
  h_original, w_original, _ = orig_img.shape
  h_new, w_new = 256, 256


  COLORS = [(255, 0, 0), (0, 255, 0)]
  MARKERS = [0, 5]
  scaling_factor = min(h_original / 256, w_original / 256)
  marker_size = int(6 * scaling_factor)
  marker_thickness = int(2 * scaling_factor)
  if num_current_points > 1:
    sel_pix.pop()
    for point, label in sel_pix:
      cv2.drawMarker(temp, point, COLORS[label], markerType=MARKERS[label],
                     markerSize=marker_size, thickness=marker_thickness)

  else:
    if num_current_points == 1:
      sel_pix.pop()
    dummy_overlay_mask = [(np.zeros((1, h_original, w_original)),'visible_mask')]
    return orig_img, (orig_img, dummy_overlay_mask), [], []

  visible_mask, overlay_mask = run_sam(predictor, sel_pix)

  resized_overlay_mask = []
  mask = np.squeeze(overlay_mask[0][0]) # (256, 256)

  resized_mask = cv2.resize(mask.astype(np.uint8) * 255, (w_original, h_original), interpolation=cv2.INTER_AREA)
  resized_mask = resized_mask > 127

  resized_overlay_mask = [(resized_mask,'visible_mask')]

  return temp, (orig_img, resized_overlay_mask), visible_mask


def reset_image(predictor, img):
  
  preprocessed_image = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

  predictor.set_image(preprocessed_image)

  return (img, preprocessed_image, [], (img.copy(), []))


def write_mask(mask, file_path):
    cv2.imwrite(file_path, mask.astype(np.uint8))

def write_image(image, file_path):
    image.save(file_path)

def get_mask_from_pred(pred_image, thresholding = False):
    """
    Since pix2gestalt performs amodal completion and segmentation jointly, 
    the whole (amodal) object is synthesized on a white background.

    We can either perform traditional thresholding or utilize a background removal / 
    matting tool to extract the amodal mask (alpha channel) from pred_image. 

    For evaluation, we use direct thresholding. Below, we implement both.
    While we didn't empirically verify this, matting should slightly improve 
    the amodal segmentation performance. 
    """
    if thresholding:
      gray_image = cv2.cvtColor(pred_image, cv2.COLOR_BGR2GRAY)
      _, pred_mask = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY_INV)
    else:
      amodal_rgba = np.array(interface([pred_image])[0])
      alpha_channel = amodal_rgba[:, :, 3]
      visible_mask = (alpha_channel > 0).astype(np.uint8) * 255

      rgb_visible_mask = np.zeros((visible_mask.shape[0], visible_mask.shape[1], 3), dtype=np.uint8)
      rgb_visible_mask[:,:,0] = visible_mask
      rgb_visible_mask[:,:,1] = visible_mask
      rgb_visible_mask[:,:,2] = visible_mask # (256, 256, 3)
      pred_mask = rgb_visible_mask

    return pred_mask


def run_pix2gestalt(preprocessed_image, original_image, visible_mask, guidance_scale, n_samples, ddim_steps):

  pred_reconstructions = run_inference(preprocessed_image, visible_mask, model, guidance_scale, n_samples, ddim_steps)
  height, width = original_image.shape[:2]

  resized_images, resized_amodal_masks = list(), list()
  for image in pred_reconstructions:
      # Resize image to match the size of original_image using Lanczos interpolation
      resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
      resized_image = Image.fromarray(resized_image)
      resized_images.append(resized_image)

      pred_mask = get_mask_from_pred(resized_image)
      resized_amodal_masks.append(Image.fromarray(pred_mask))

  return resized_images, resized_amodal_masks

def update_button(selected_points, pix2gestalt_button):
  if len(selected_points) > 0:
    pix2gestalt_button.update(interactive=True)
    return pix2gestalt_button

def button_clickable(selected_points):
    if len(selected_points) > 0:
      return gr.Button.update(interactive=True)
    else:
      return gr.Button.update(interactive=False)

def run_demo(
  ckpt="./ckpt/epoch=000005.ckpt",
  config="./configs/sd-finetune-pix2gestalt-c_concat-256.yaml",
  device_idx =_GPU_INDEX):
  
  device = f"cuda:{device_idx}"
  config = OmegaConf.load(config)

  global model
  model = load_model_from_config(config, ckpt, device)
  global interface
  interface = create_carvekit_interface()

  _TITLE = "pix2gestalt: Amodal Segmentation by Synthesizing Wholes"
  _DESCRIPTION = '''
    This demo allows you to perform zero-shot amodal completion and segmentation with pix2gestalt. 
    Check out our [project webpage](https://gestalt.cs.columbia.edu/), [code](https://github.com/cvlab-columbia/pix2gestalt) and [paper](https://arxiv.org/abs/2401.14398) to learn more about the method!
  '''

  demo = gr.Blocks(title=_TITLE)
  with demo:
    gr.Markdown('# ' + _TITLE)
    gr.Markdown(_DESCRIPTION)

    predictor = gr.State(value = get_sam_predictor(device=device, model_type = 'vit_h'))
    selected_points = gr.State(value=[])
    original_image = gr.State(value=None)
    preprocessed_image = gr.State(value=None)
    visible_mask = gr.State(value=None)

    with gr.Row():
      with gr.Column():

        input_image = gr.Image(type="numpy", label='Input Occlusion Image', height=500)

        pix2gestalt_button = gr.Button(value = 'Run pix2gestalt', interactive = False)
        undo_button = gr.Button('Undo Prompt')
        
        fg_bg_radio = gr.Radio(
              ['positive_prompt', 'negative_prompt'],
              label='Point Prompt Type')
        
        gr.Markdown('''
        ### Instructions:
          - First, select the visible (modal) region of an occluded object with positive or negative point prompts. Default: `positive_prompt`.
          - Segment Anything (SAM) recommends using 2-9 point prompts for best performance.
          - Once your SAM generated modal mask is satisfactory, you can run pix2gestalt to perform amodal completion and segmentation!

        <details>
        <summary>Advanced Options</summary>

        **Number of Samples:** The model is probabilistic, which encodes the under-constrained nature of amodal completion. If the number of samples is selected to be bigger than 1 and results look different, that is expected behavior as the model tries to predict a diverse set of possibilities given the input image and modal mask.

        **Diffusion Guidance Scale:** Defines how much you want the model to respect the conditioning information (input image + modal mask). We recommend experimenting with 1 or 2.

        **DDIM Steps:** Controls the number of denoising iterations that are applied in order to generate each whole. We recommend experimenting in the range of 50-200. Higher values will lead to diminishing returns.

        </details>
        ''')
        guidance_scale = gr.Slider(0, 20, value=2, step=1, label="Diffusion Guidance Scale (CFG)")
        n_samples = gr.Slider(1, 8, value=4, step=1, label="Number of Samples")
        ddim_steps = gr.Slider(5, 500, value=200, step=5, label="DDIM Inference Steps")

      with gr.Column():
        output_mask = gr.AnnotatedImage(label='SAM Generated Visible (Modal) Mask', height=500)

        pred_reconstructions = gr.Gallery(label="Predicted Amodal Completion Samples", height=500)
        pred_masks = gr.Gallery(label="Corresponding Amodal Masks", height=500)

    input_image.upload(
        reset_image,
        [predictor, input_image],
        [original_image, preprocessed_image, selected_points, output_mask]
    )
    input_image.upload(
        button_clickable,
        [selected_points],
        [pix2gestalt_button]
    )
    undo_button.click(
        undo_points,
        [predictor, original_image, selected_points],
        [input_image, output_mask, visible_mask]
    )
    undo_button.click(
        button_clickable,
        [selected_points],
        [pix2gestalt_button]
    )
    input_image.select(
        select_point,
        [predictor, input_image, original_image, selected_points, fg_bg_radio],
        [input_image, output_mask, visible_mask]
    )
    input_image.select(
        button_clickable,
        [selected_points],
        [pix2gestalt_button]
    )
    pix2gestalt_button.click(
      run_pix2gestalt,
      [preprocessed_image, original_image, visible_mask, guidance_scale, n_samples, ddim_steps],
      [pred_reconstructions, pred_masks]
    )

  demo.queue(max_size=10)
  demo.launch(debug=True, share=True)


if __name__ == "__main__": 
    fire.Fire(run_demo)