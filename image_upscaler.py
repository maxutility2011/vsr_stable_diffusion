import requests
import sys
import time
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id)
pipeline = pipeline.to("cuda")

# let's download an image
#url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
url = sys.argv[1]
response = requests.get(url)
low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
low_res_img = low_res_img.resize((128, 128))

prompt = ""

upscaled_image = pipeline(prompt=prompt, image=low_res_img, num_inference_steps=75).images[0]
current_time = time.strftime("%b_%d_%Y_%H:%M:%S")
upscaled_image.save("output_" + current_time + ".png")