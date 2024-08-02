import requests
import sys
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id)
pipeline = pipeline.to("cuda")

# let's download the images from the base url given by the command line argument
url_base = sys.argv[1]
for i in range (1, 75):
    url = url_base + str(i) + ".png"

    #url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
    response = requests.get(url)
    low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
    low_res_img = low_res_img.resize((128, 128))

    prompt = ""

    upscaled_image = pipeline(prompt=prompt, image=low_res_img, num_inference_steps=75).images[0]
    output = "./new1/highres/new1_" + str(i) + "_highres.png"
    upscaled_image.save(output)