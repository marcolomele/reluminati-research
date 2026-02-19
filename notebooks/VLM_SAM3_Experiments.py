# %% [markdown]
# <center> <h1> Testing VLMs + SAM3 </h1>
# </center>

# %% [markdown]
# # 1. Set up
# Set up APIs of VLMs and SAM 3.

# %%
# API KEYS

gemini_api = 'AIzaSyAr6BVb8gR3kGdjFj-LF8dCU-6FBmlLJb4'
ollama_api = '5e3528983b8e412d9f58687d49a41b11.3LUyjZTbzgbBdrUkqD47y8wh'
hf_api = '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'

# %% [markdown]
# ## 1.1 VLMs
# For each available VLM, set up API key and connect to model.

# %% [markdown]
# ### Gemini

# %%
!pip install decord

import google.generativeai as genai
from google.generativeai import types

genai.configure(api_key=gemini_api)

gemini_vlm = genai.GenerativeModel('gemini-2.0-flash')

# %% [markdown]
# ### Qwen

# %%
!pip install ollama

# %%
import os
from ollama import Client
import base64

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

client_ollama = Client(
    host='https://ollama.com',
    headers={'Authorization': 'Bearer ' + ollama_api}
)

# %% [markdown]
# ## 1.2 SAM 3
# Connect to SAM 3 via Huggingface API

# %%
HF_TOKEN = hf_api

from huggingface_hub import login
login(token=HF_TOKEN)
print("✓ Logged into Hugging Face")

# %%
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("facebook/sam3")
sam3_model = AutoModel.from_pretrained("facebook/sam3")

# %%
from transformers import Sam3Processor, Sam3Model
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

hf_sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(device)
hf_sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")

print("✓ Hugging Face Transformers SAM3 model and processor loaded.")

# %% [markdown]
# ## 1.3 Prompts and Utils

# %% [markdown]
# ### Prompts

# %%
vlm_prompt = (
    "First image = masked image with artificial red overlay. "
    "Second image = original image with true colors. "
    "Use first image only to know which object is selected. "
    "Use second image to determine the true color. "
    "Do NOT say red unless the object is actually red in the second image. "
    "Answer with: {color} {object_name}. No extra text."
)

# %% [markdown]
# ### Utils
# Get VLM captions, run SAM3 inference, compute metrics (IoU, BA, CA, LE).

# %% [markdown]
# Columns of table for data:
# 
# | take_id | scenario | experiment set-up | IoU | LE | CA | BA |
# 

# %%
from sklearn.metrics import balanced_accuracy_score
from skimage import segmentation
from PIL import Image
import numpy as np
import requests
import torch

def get_gemini_caption(src_with_mask_path=None, src_no_mask_path=None,
                       dest_path=None, vlm_prompt=None, src_prompt=None):

    if not src_with_mask_path and not src_no_mask_path:
        raise ValueError("At least one source image must be added.")

    contents = []

    if src_with_mask_path:
        with open(src_with_mask_path, 'rb') as f:
            src_with_mask_bytes = f.read()
        contents.append({'mime_type': 'image/jpeg', 'data': src_with_mask_bytes})
    if src_no_mask_path:
        with open(src_no_mask_path, 'rb') as f:
            src_no_mask_bytes = f.read()
        contents.append({'mime_type': 'image/jpeg', 'data': src_no_mask_bytes})
    if dest_path:
        with open(dest_path, 'rb') as f:
            dest_img_bytes = f.read()
        contents.append({'mime_type': 'image/jpeg', 'data': dest_img_bytes})
    if src_prompt:
        contents.append(src_prompt)
    if vlm_prompt:
        contents.append(vlm_prompt)

    response = gemini_vlm.generate_content(
        contents=contents)

    return response.text

def get_qwen_caption(src_with_mask_path=None, src_no_mask_path=None,
                       dest_path=None, vlm_prompt=None, src_prompt=None):

    if not src_with_mask_path and not src_no_mask_path:
        raise ValueError("At least one source image must be added.")

    qwen_messages = []
    image_list = []

    if src_with_mask_path:
        image_list.append(encode_image(src_with_mask_path))
    if src_no_mask_path:
        image_list.append(encode_image(src_no_mask_path))
    if dest_path:
        image_list.append(encode_image(dest_path))

    print(len(image_list))

    full_prompt_content = ""
    if src_prompt:
        full_prompt_content += src_prompt
    if vlm_prompt:
        if full_prompt_content:
            full_prompt_content += " "
        full_prompt_content += vlm_prompt

    message_item = {"role": "user"}
    if full_prompt_content:
        message_item["content"] = full_prompt_content
    if image_list:
        message_item["images"] = image_list
    if not full_prompt_content and not image_list:
        raise ValueError("No prompt or images provided for Qwen model.")

    qwen_messages.append(message_item)

    response_text = ""
    for part in client_ollama.chat('qwen3-vl:235b-cloud', messages=qwen_messages, stream=True):
        response_text += part.message.content

    return response_text

def get_sam3_masks(input_image_path, text_prompt):
    if isinstance(input_image_path, str):
        image = Image.open(input_image_path).convert("RGB")
    else:
        image = input_image_path

    inputs = hf_sam3_processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = hf_sam3_model(**inputs)

    results = hf_sam3_processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist())[0]

    return results

# %% [markdown]
# #3. Systematic Experiment on a Frame

# %% [markdown]
# ## 3.1 Understanding the data

# %%
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# **Create a Shortcut in your Drive**
# 1.   Go to the Google Drive web interface.
# 2.   Go to the "Shared with me" tab on the left.
# 3. Find your folder (the one containing output_data), it should be "The ReLUminati RA".
# 4. Right-click it and select "Add shortcut to Drive".
# 5. Pick "My Drive" as the destination.

# %%
import os

# This lists everything in your MyDrive.
# Look for the name of the folder you just added a shortcut for.
my_drive_path = "/content/drive/MyDrive"
folders = [f for f in os.listdir(my_drive_path) if os.path.isdir(os.path.join(my_drive_path, f))]

print("Folders found in My Drive:")
for f in folders:
    print(f"- {f}")

# %%
'''This could take a bit'''
import os

# 1. Path to the zip file in your Drive
ZIP_PATH = "/content/drive/MyDrive/EgoExo-ReLUminati/output_data.zip"
DEST_PATH = "/content/data_local"

if os.path.exists(ZIP_PATH):
    print("Found the ZIP file! Copying and unzipping...")
    !mkdir -p {DEST_PATH}
    # Unzip directly into data_local
    !unzip -o -q "{ZIP_PATH}" -d {DEST_PATH}
    print("✓ Done! Files are now at /content/data_local")
else:
    # If this fails, let's list the files in that folder to see the exact name
    print(f"File not found at {ZIP_PATH}. Checking folder contents...")
    parent_folder = os.path.dirname(ZIP_PATH)
    if os.path.exists(parent_folder):
        print(f"Contents of '{parent_folder}':", os.listdir(parent_folder))
    else:
        print("Could not even find the folder. Check the shortcut name again.")

# %%
import os

# Let's see how the folders are organized
root = "/content/data_local"
items = os.listdir(root)

# If 'output_data' is a folder inside the zip, we go deeper
data_folder = os.path.join(root, "output_data") if "output_data" in items else root

try:
    # Find the first 'Take' folder
    takes = sorted([t for t in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, t))])
    first_take = os.path.join(data_folder, takes[0])

    # Find the first 'Frame' folder inside that take
    frames = sorted([f for f in os.listdir(first_take) if os.path.isdir(os.path.join(first_take, f))])
    first_frame_path = os.path.join(first_take, frames[0])

    print(f"Structure Discovery:")
    print(f"- Data Folder: {data_folder}")
    print(f"- Sample Take: {takes[0]}")
    print(f"- Sample Frame: {frames[0]}")
    print(f"\nFiles found inside {frames[0]}:")
    print(os.listdir(first_frame_path))
except Exception as e:
    print(f"Discovery failed: {e}. Let's look at root: {items}")

# %%
import os
import glob

# Search for any PNG files in the entire unzipped folder
all_pngs = glob.glob(f"{DEST_PATH}/**/*.png", recursive=True)
all_json = glob.glob(f"{DEST_PATH}/**/*.json", recursive=True)

print(f"Total PNG files (potential masks): {len(all_pngs)}")
print(f"Total JSON files (potential metadata): {len(all_json)}")

if all_pngs:
    print(f"Sample mask path: {all_pngs[0]}")
else:
    print("WARNING: No PNG files found. Are the masks stored as JPGs or in a different zip?")

# Let's see if there is a 'masks' folder at the same level as 'output_data'
print("Root directory contents:", os.listdir(DEST_PATH))

# %%
import json
import glob
import os

# Find the 4 JSON files
json_paths = glob.glob("/content/data_local/**/*.json", recursive=True)
print(f"Found JSON files: {json_paths}")

for path in json_paths:
    print(f"\n--- Inspecting {os.path.basename(path)} ---")
    with open(path, 'r') as f:
        try:
            data = json.load(f)
            # If it's a list, show the first item; if it's a dict, show the keys
            if isinstance(data, dict):
                print("Keys found:", list(data.keys()))
                # Look for common mask keys
                for key in ['annotations', 'labels', 'objects', 'segments']:
                    if key in data:
                        print(f"Snippet of '{key}':", str(data[key])[:500])
            else:
                print("Data snippet:", str(data)[:500])
        except:
            print("Could not parse JSON.")

# %%
 import json

# Take one of the JSONs we found
json_path = "/content/data_local/output_data/64b60273-e935-4161-9300-2879b46f5785/annotation.json"

with open(json_path, 'r') as f:
    data = json.load(f)

# Let's look at the first frame's mask
frame_keys = list(data['masks'].keys())
print(f"Sample frame key: {frame_keys[0]}")
sample_mask_data = data['masks'][frame_keys[0]]

# Print the type and content to see if it's RLE or Polygons
print(f"Mask type: {type(sample_mask_data)}")
print("Mask Data (Snippet):", str(sample_mask_data)[:500])

# %%
!pip install pycocotools

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import json
import os
import torch
from sklearn.metrics import balanced_accuracy_score
from skimage import segmentation
import cv2 # Using cv2 for fast resizing

def decode_rle_mask(rle_data):
    """Converts the JSON 'counts' format into a numpy boolean mask."""
    # Ensure RLE is in the correct format for pycocotools
    if isinstance(rle_data, dict) and 'counts' in rle_data:
        return mask_utils.decode(rle_data).astype(bool)
    return None



def create_vlm_overlay(image_path, mask_np, alpha=0.5):
    """Creates the 'Source Masked' image with a red highlight, handling size mismatches."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # 1. Resize mask to match image size if they differ
    # We use NEAREST interpolation to keep the mask binary
    if mask_np.shape[0] != h or mask_np.shape[1] != w:
        mask_np = cv2.resize(mask_np.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    img_array = np.array(img)
    red_layer = np.zeros_like(img_array)
    red_layer[:, :] = [255, 0, 0]

    mask_indices = mask_np == 1
    img_array[mask_indices] = (img_array[mask_indices] * (1 - alpha) + red_layer[mask_indices] * alpha).astype(np.uint8)

    return Image.fromarray(img_array)

# %%
import pandas as pd
import cv2
import numpy as np

# 1. Define the missing prompts
vlm_prompt = (
    "First image = masked image with artificial red overlay. "
    "Second image = original image with true colors. "
    "Use first image only to know which object is selected. "
    "Use second image to determine the true color. "
    "Do NOT say red unless the object is actually red in the second image. "
    "Answer with: {color} {object_name}. No extra text."
)

# Added missing src_prompt
src_prompt = "Identify the object highlighted in red in the first image and describe its appearance using the second image."

# 2. Harmonized Metric Function (returns a dictionary for the runner)
def compute_metrics(gt_mask, sam_results):
    if len(sam_results['masks']) == 0:
        return {"iou": 0, "ba": 0, "ca": 0, "le": 1.0}

    best_iou = 0
    best_pred = None

    for pred_mask_tensor in sam_results['masks']:
        pred_mask = pred_mask_tensor.cpu().numpy().astype(bool)
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = intersection / union if union > 0 else 0
        if iou > best_iou:
            best_iou = iou
            best_pred = pred_mask

    if best_pred is None:
        return {"iou": 0, "ba": 0, "ca": 0, "le": 1.0}

    ba = balanced_accuracy_score(gt_mask.flatten(), best_pred.flatten())
    le = np.mean(best_pred != gt_mask)

    # Boundary IoU
    gt_b = segmentation.find_boundaries(gt_mask, mode='inner')
    pr_b = segmentation.find_boundaries(best_pred, mode='inner')
    ca = np.logical_and(gt_b, pr_b).sum() / np.logical_or(gt_b, pr_b).sum() if np.any(gt_b | pr_b) else 0.0

    return {"iou": best_iou, "ba": ba, "ca": ca, "le": le}

# %%
import time
import numpy as np
import pandas as pd
import cv2
import json
from PIL import Image
from google.api_core import exceptions as google_exceptions

def compute_spatial_covariates(mask_np):
    """
    Analyzes the ground truth mask to determine size and position.
    """
    H, W = mask_np.shape
    area = np.sum(mask_np)
    relative_area = area / (H * W)

    coords = np.argwhere(mask_np)
    if len(coords) == 0:
        return {"obj_rel_area": 0, "obj_dist_center": 1.0, "obj_size_cat": "none", "obj_is_peripheral": True}

    # Calculate Centroid
    centroid_y, centroid_x = coords.mean(axis=0)
    center_y, center_x = H / 2, W / 2

    # Distance to center normalized by max possible distance (corner)
    dist = np.sqrt((centroid_y - center_y)**2 + (centroid_x - center_x)**2)
    max_dist = np.sqrt((H/2)**2 + (W/2)**2)
    norm_dist = dist / max_dist

    # Heuristic categorization
    if relative_area < 0.005: # Less than 0.5% of image
        size_cat = "small"
    elif relative_area < 0.05: # 0.5% to 5%
        size_cat = "medium"
    else:
        size_cat = "large"

    return {
        "obj_rel_area": float(relative_area),
        "obj_dist_center": float(norm_dist),
        "obj_size_cat": size_cat,
        "obj_is_peripheral": norm_dist > 0.5
    }

def run_ablation_for_object(take_id, frame_folder, object_name, src_time='0', dest_time='300', vlm="gemini"):
    base_path = f"/content/data_local/output_data/{take_id}/{frame_folder}"
    json_path = f"/content/data_local/output_data/{take_id}/annotation.json"

    with open(json_path, 'r') as f:
        ann = json.load(f)

    # 1. Decode Masks and Image Paths
    src_mask_rle = ann['masks'][object_name][frame_folder][src_time]
    dest_mask_rle = ann['masks'][object_name][frame_folder][dest_time]
    src_mask_np = decode_rle_mask(src_mask_rle)
    dest_mask_gt = decode_rle_mask(dest_mask_rle)

    # --- NEW: Compute Spatial Covariates from Ground Truth ---
    spatial_info = compute_spatial_covariates(dest_mask_gt)

    src_raw = f"{base_path}/{src_time}.jpg"
    dest_raw = f"{base_path}/{dest_time}.jpg"

    # 2. Create and Save Source Overlay
    src_masked_img = create_vlm_overlay(src_raw, src_mask_np)
    src_masked_path = "temp_src_masked.jpg"
    src_masked_img.save(src_masked_path)

    results = []

    # 3. Define Ablation Scenarios
    scenarios = [
        {"id": "2.1", "args": {"src_with_mask_path": src_masked_path}},
        {"id": "2.2", "args": {"src_with_mask_path": src_masked_path, "src_no_mask_path": src_raw}},
        {"id": "2.3", "args": {"src_with_mask_path": src_masked_path, "src_no_mask_path": src_raw, "dest_path": dest_raw}},
        {"id": "2.4", "args": {"src_with_mask_path": src_masked_path, "dest_path": dest_raw}}
    ]

    for sc in scenarios:
        caption = "ERROR"
        max_retries = 3

        for attempt in range(max_retries):
            try:
                if vlm == "gemini":
                    caption = get_gemini_caption(**sc['args'], vlm_prompt=vlm_prompt)
                else:
                    caption = get_qwen_caption(**sc['args'], vlm_prompt=vlm_prompt)

                # Strip leading/trailing whitespace and newlines from VLM output
                caption = caption.strip()
                break
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait_time = 30 * (attempt + 1)
                    print(f"Quota reached for {vlm}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Unexpected error: {e}")
                    break

        # 4. Run SAM3 on Destination
        sam3_out = get_sam3_masks(dest_raw, caption)

        # 5. Compute Metrics (Handle resizing)
        h_sam, w_sam = sam3_out['masks'][0].shape[-2:] if len(sam3_out['masks']) > 0 else (dest_mask_gt.shape)
        if dest_mask_gt.shape[0] != h_sam:
            dest_mask_gt_resized = cv2.resize(dest_mask_gt.astype(np.uint8), (w_sam, h_sam), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            dest_mask_gt_resized = dest_mask_gt

        m = compute_metrics(dest_mask_gt_resized, sam3_out)

        # 6. Combine all data into row
        res_row = {
            "scenario": sc['id'],
            "vlm_model": vlm,
            "vlm_output": caption,
            "iou": m['iou'], "ba": m['ba'], "ca": m['ca'], "le": m['le']
        }
        res_row.update(spatial_info) # Add area, distance, size category

        results.append(res_row)

    return pd.DataFrame(results)

# %%
# Select a sample object
take = "64b60273-e935-4161-9300-2879b46f5785"
frame = "aria01_214-1"
obj = "hoop"

# Run the experiment
study_results = run_ablation_for_object(take, frame, obj, vlm="gemini")
display(study_results)

# %%
# Parameters from your discovery
take = "64b60273-e935-4161-9300-2879b46f5785"
frame = "aria01_214-1"
obj = "hoop"

# RUNNING WITH QWEN
print("Starting Ablation Study with Qwen...")
study_results_qwen = run_ablation_for_object(take, frame, obj, vlm="qwen")

display(study_results_qwen)

# %%
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

def visualize_ablation_results(take_id, frame_folder, object_name, ablation_df, src_time='0', dest_time='300'):
    """
    Enhanced visualization to fix mask alignment and display spatial metadata.
    """
    base_path = f"/content/data_local/output_data/{take_id}/{frame_folder}"
    json_path = f"/content/data_local/output_data/{take_id}/annotation.json"

    with open(json_path, 'r') as f:
        ann = json.load(f)

    # 1. Load Images
    src_raw_path = f"{base_path}/{src_time}.jpg"
    dest_raw_path = f"{base_path}/{dest_time}.jpg"
    src_img = Image.open(src_raw_path).convert("RGB")
    dest_img = Image.open(dest_raw_path).convert("RGB")
    dest_np = np.array(dest_img)
    H, W, _ = dest_np.shape

    # 2. Get masks and ensure they match image resolution
    src_mask_rle = ann['masks'][object_name][frame_folder][src_time]
    dest_mask_rle = ann['masks'][object_name][frame_folder][dest_time]
    src_mask_np = decode_rle_mask(src_mask_rle)
    dest_mask_gt = decode_rle_mask(dest_mask_rle)

    # Resize GT if there is a mismatch
    if dest_mask_gt.shape[:2] != (H, W):
        dest_mask_gt = cv2.resize(dest_mask_gt.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

    # 3. Create Source Overlay for VLM
    src_masked_img = create_vlm_overlay(src_raw_path, src_mask_np)

    # --- Setup Plot ---
    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(2, 4)

    # Row 0, Col 0: The Masked Source (VLM Input)
    ax_src_m = fig.add_subplot(gs[0, 0])
    ax_src_m.imshow(src_masked_img)
    ax_src_m.set_title(f"INPUT: Source (Masked)\nObject: {object_name}")
    ax_src_m.axis('off')

    # Row 0, Col 1: Clean Source
    ax_src_c = fig.add_subplot(gs[0, 1])
    ax_src_c.imshow(src_img)
    ax_src_c.set_title("CONTEXT: Source (Clean)")
    ax_src_c.axis('off')

    # Row 0, Col 2: Spatial Metadata Info
    ax_info = fig.add_subplot(gs[0, 2])
    info_row = ablation_df.iloc[0]
    info_text = (f"Spatial Covariates:\n\n"
                 f"Rel. Area: {info_row['obj_rel_area']:.4f}\n"
                 f"Size Cat: {info_row['obj_size_cat'].upper()}\n"
                 f"Dist to Center: {info_row['obj_dist_center']:.3f}\n"
                 f"Peripheral: {info_row['obj_is_peripheral']}")
    ax_info.text(0.1, 0.5, info_text, fontsize=14, family='monospace', va='center')
    ax_info.axis('off')

    # Row 0, Col 3: Ground Truth
    ax_gt = fig.add_subplot(gs[0, 3])
    ax_gt.imshow(dest_np)
    gt_overlay = np.zeros((H, W, 4))
    gt_overlay[dest_mask_gt > 0] = [0, 1, 0, 0.5] # Green Alpha
    ax_gt.imshow(gt_overlay)
    ax_gt.set_title("Ground Truth Mask")
    ax_gt.axis('off')

    # Row 1: The 4 Ablation Scenarios
    for i, scenario_id in enumerate(["2.1", "2.2", "2.3", "2.4"]):
        row = ablation_df[ablation_df['scenario'] == scenario_id].iloc[0]
        caption = row['vlm_output'].replace("\n", " ") # Clean for title
        iou = row['iou']

        ax = fig.add_subplot(gs[1, i])
        ax.imshow(dest_np)

        # Run/Fetch SAM3 result
        sam3_out = get_sam3_masks(dest_raw_path, row['vlm_output'])

        if len(sam3_out['masks']) > 0:
            pred_mask = sam3_out['masks'][0].cpu().numpy()
            # Resize mask to image size for perfect overlay
            if pred_mask.shape != (H, W):
                pred_mask = cv2.resize(pred_mask.astype(np.float32), (W, H))

            m_overlay = np.zeros((H, W, 4))
            m_overlay[pred_mask > 0.5] = [1, 0, 0, 0.5] # Red Alpha
            ax.imshow(m_overlay)

        ax.set_title(f"Scenario {scenario_id} (IoU: {iou:.3f})\nCaption: {caption[:40]}...", fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Run it
visualize_ablation_results(take, frame, obj, study_results_qwen)

# %% [markdown]
# ## Comment
# 
# This study evaluates how different visual inputs affect a Vision Language Model's (VLM) ability to identify a specific object and how that description subsequently guides **SAM3** to perform accurate correspondence and segmentation on a destination image.
# 
# ### 1. The Ablation Scenarios
# 
# We test four levels of information disclosure to the VLM to find the "optimal" prompt strategy.
# 
# | Scenario | Masked Source | Clean Source | Destination Image | Research Question |
# | --- | --- | --- | --- | --- |
# | **2.1** | ✅ | ❌ | ❌ | **Minimal Info:** Can the VLM identify an object based solely on shape/silhouette? |
# | **2.2** | ✅ | ✅ | ❌ | **Texture/Color:** Does "True Color" (e.g., *red* hoop) improve identification over silhouette? |
# | **2.3** | ✅ | ✅ | ✅ | **Full Context:** Does seeing the target view resolve perspective or semantic ambiguity? |
# | **2.4** | ✅ | ❌ | ✅ | **Efficiency:** Is the "Clean Source" redundant if the Destination image is provided? |
# 
# ### 2. Metric Definitions & Analysis
# 
# We use a suite of pixel-level and boundary-level metrics to evaluate the SAM3 output against the **Ground Truth (GT)** (EgoExo benchmark).
# 
# * **IoU (Intersection over Union):** * **Definition:** The area of overlap between the predicted mask and the ground truth divided by the area of their union.
# 
# 
# * **BA (Balanced Accuracy):** * **Definition:** The average of Recall obtained on each class (Object vs. Background). It prevents the background size from inflating the score.
# 
# 
# * **CA (Contour Accuracy):** * **Definition:** Measures how closely the **edges** of the predicted mask align with the GT boundaries (often using Boundary IoU).
# 
# * **LE (Local Error):** * **Definition:** The total pixel-wise error rate (Total Misclassified Pixels / Total Pixels).
# 
# 
# 

# %% [markdown]
# # 4. Automated Experiment on Takes
# 

# %%
import os
import json
import random
import pandas as pd

DATA_ROOT = "/content/data_local/output_data"

def get_dataset_inventory(data_root):
    """
    Scans the directory to find all available (take, frame, object) tuples.
    """
    inventory = []
    # Find all Take folders (UUIDs)
    takes = sorted([t for t in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, t))])

    for take_id in takes:
        json_path = os.path.join(data_root, take_id, "annotation.json")
        if not os.path.exists(json_path): continue

        with open(json_path, 'r') as f:
            ann = json.load(f)

        # Structure: ann['masks'][object_name][frame_folder][time_stamp]
        for obj_name, frames_dict in ann.get('masks', {}).items():
            for frame_folder, times_dict in frames_dict.items():
                # We need both timestamps to run the correspondence test
                if '0' in times_dict and '300' in times_dict:
                    inventory.append({
                        "take_id": take_id,
                        "frame_folder": frame_folder,
                        "object_name": obj_name
                    })
    return pd.DataFrame(inventory)

# Build the full list of candidates
full_inventory = get_dataset_inventory(DATA_ROOT)
print(f"Inventory built: {len(full_inventory)} candidate samples found across {full_inventory['take_id'].nunique()} takes.")

# %%
''' Here we just have 4 takes but could be useful later '''
def sample_takes(df, n_takes=4, seed=42):
    """
    Randomly selects N unique takes and 1 object from each.
    """
    unique_takes = df['take_id'].unique()
    sampled_take_ids = random.sample(list(unique_takes), min(n_takes, len(unique_takes)))

    final_sample = []
    for t_id in sampled_take_ids:
        take_candidates = df[df['take_id'] == t_id]
        # Pick one random object/frame from this take
        final_sample.append(take_candidates.sample(1, random_state=seed))

    return pd.concat(final_sample).reset_index(drop=True)

# Select the subset for the study
study_subset = sample_takes(full_inventory, n_takes=4)
print(f"Subset selected: {len(study_subset)} objects ready for ablation.")
display(study_subset)

# %%
def run_systematic_study(sample_df, vlm_type="qwen"):
    all_results = []

    for idx, row in sample_df.iterrows():
        print(f"Processing [{idx+1}/{len(sample_df)}]: Take {row['take_id'][:8]}... | Obj: {row['object_name']}")

        try:
            res_df = run_ablation_for_object(
                take_id=row['take_id'],
                frame_folder=row['frame_folder'],
                object_name=row['object_name'],
                vlm=vlm_type
            )

            # Save ALL metadata to the results
            res_df['take_id'] = row['take_id']
            res_df['object_name'] = row['object_name']
            res_df['frame_folder'] = row['frame_folder'] # <--- Added this line
            all_results.append(res_df)

        except Exception as e:
            print(f"Error on {row['take_id'][:8]}: {e}")

    return pd.concat(all_results).reset_index(drop=True)

# %%
'''This will take a bit'''
# Choose your model (qwen or gemini)
VLM_TO_TEST = "qwen"

print(f"Starting systematic study using {VLM_TO_TEST}...")
final_results_df = run_systematic_study(study_subset, vlm_type=VLM_TO_TEST)

# Save the raw data immediately so you don't lose it if the runtime crashes
final_results_df.to_csv("ablation_study_results.csv", index=False)
print("\n Study Complete! Data saved to 'ablation_study_results.csv'")
display(final_results_df.head())

# %%
# Load the full results file
import pandas as pd
df_results = pd.read_csv("ablation_study_results.csv")

print(f"Total experiments recorded: {len(df_results)}")
print(f"Unique objects processed: {df_results['object_name'].nunique()}")

# Show rows where IoU is greater than 0 to see successes
successes = df_results[df_results['iou'] > 0]
print(f"\nExperiments with successful segmentation: {len(successes)}")

if len(successes) > 0:
    display(successes[['scenario', 'object_name', 'vlm_output', 'iou', 'obj_size_cat']])
else:
    print("No successful segmentations found in this sample.")

# %%
# Average IoU per scenario across all 4 objects
performance_summary = df_results.groupby('scenario')[['iou', 'ba']].mean()
print("--- Mean Performance per Scenario ---")
print(performance_summary)

# Average IoU by size category
size_summary = df_results.groupby('obj_size_cat')[['iou']].mean()
print("\n--- Mean IoU by Object Size ---")
print(size_summary)

# %%
# Select the specific failure data
fail_take_id = "ba2086b2-f43f-4096-8b84-e9a3bfc0020b"
fail_obj_name = "napkin_0"
correct_folder = "cam01"

# Filter the results for this object
obj_results = final_results_df[
    (final_results_df['take_id'] == fail_take_id) &
    (final_results_df['object_name'] == fail_obj_name)
]

# Visualize
visualize_ablation_results(fail_take_id, correct_folder, fail_obj_name, obj_results)

# %%
# 1. Get the list of unique objects we actually tested in the study
# We use final_results_df (the 16-row dataframe)
tested_objects = final_results_df[['take_id', 'object_name']].drop_duplicates()

print(f"Generating visualizations for {len(tested_objects)} objects...")

for _, row in tested_objects.iterrows():
    t_id = row['take_id']
    o_name = row['object_name']

    # 2. Find the correct folder name for this take/object from the inventory
    # (Since different takes have different folder names like 'cam01' or 'aria01_214-1')
    match = full_inventory[(full_inventory['take_id'] == t_id) &
                           (full_inventory['object_name'] == o_name)]

    if len(match) > 0:
        f_folder = match['frame_folder'].iloc[0]

        # 3. Filter the results for just this specific object (all 4 scenarios)
        obj_df = final_results_df[(final_results_df['take_id'] == t_id) &
                                 (final_results_df['object_name'] == o_name)]

        print(f"\n--- Visualizing: {o_name} (Take: {t_id[:8]}...) ---")

        # 4. Call the visualization function
        visualize_ablation_results(t_id, f_folder, o_name, obj_df)
    else:
        print(f"Warning: Could not find folder for {o_name} in inventory.")

# %% [markdown]
# # Results & Analysis
# 
# In this section, we moved from single-image testing to a **systematic, take-level study**. We selected a random subset of 4 unique objects across our available data and evaluated each against the four defined ablation scenarios (16 total experiments).
# 
# ### 1. Methodology
# 
# To ensure scientific rigor, we analyzed the ground truth masks to extract **Spatial Covariates**:
# 
# * **Object Size:** Categorized as *Small* ( relative area) or *Medium*.
# * **Centricity:** Measuring distance from the image center to identify peripheral vs. central objects.
# * **Ablation Loop:** Each object was processed by the VLM (Qwen) and SAM3 through four stages of information disclosure (2.1–2.4).
# 
# ### 2. Previous Findings (another run)
# 
# The results reveal a contrast in performance based on object size and the type of prompt context provided. Moreover, we see a stark contrast in resulst across different runs.
# 
# #### **Performance by Scenario**
# 
# | Scenario | Mean IoU | Description |
# | --- | --- | --- |
# | **2.1** | 0.000 | Masked Source only |
# | **2.2** | 0.232 | Masked + Clean Source |
# | **2.3** | **0.456** | **Full Context (Source + Destination)** |
# | **2.4** | 0.232 | Masked Source + Destination |
# 
# #### **The "Small Object" Bottleneck**
# 
# | Size Category | Mean IoU |
# | --- | --- |
# | **Medium** | **0.460** |
# | **Small** | **0.000** |
# 
# **Observation:** There is a total failure (0.0 IoU) for objects classified as "Small." When objects are too tiny, the VLM essentially "loses" the red mask due to input resolution downsampling, leading to **hallucinations** (e.g., describing a person or a shirt instead of a napkin).
# 
# ### 3. Limitations
# 
# 
# 
# *   4 takes is not enough, we have to try on a larger subset of data randomized at take level
# *   Results differ on different runs, so we should make a larger and deeper analysis in order to understand
# 
# 


