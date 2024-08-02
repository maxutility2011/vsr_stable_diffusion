# vsr_stable_diffusion
Video super resolution using **stable-diffusion-x4-upscaler** from stablility-AI.

## Convert the input low-resolution video to images.
Convert the input low-resolution video to a sequence of images at native frame rate.
```
mkdir images
cd images/
ffmpeg -i input.mov -vf fps=15 images/output_%4d.png
```
Before running the above command, please make sure your input video does not have more than 9999 frames since *%4d* is used for naming the images, or you can use more digits to represent frame numbers.

## Upscale low-resolution images
1. Get stable_diffusion source
```
git clone git@github.com:Stability-AI/stablediffusion.git
cd stablediffusion/
```
2. Create a Python virtual environment
```
conda env create -f environment.yaml
conda activate ldm
```
3. Install dependencies
```
pip install diffusers transformers accelerate scipy safetensors
```
4. Upgrade transformers if you see errors like “cannot import name 'CLIPImageProcessor' from 'transformers'”
```
pip install git+https://github.com/huggingface/transformers
```
5. Login to hugging_face
```
huggingface-cli login
```
Enter your hugging face token when prompted.
6. Run video_upscaler.py on your input low-resolution video (input file is to be hardcoded in the program)
```
python video_upscaler.py [base_url_prefix_of_your_input_images]
```
[base_url_prefix_of_your_input_images] is the prefix of the base url to your input images, e.g., *https://bzhang-test-bucket-public.s3.amazonaws.com/new1_*.

## Convert image sequence to video
```
ffmpeg -r 15  -s 512x512 -i output_%2d.png -vcodec libx264 -crf 20 ../output_highres.mp4
```