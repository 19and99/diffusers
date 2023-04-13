import sys

import torch
import imageio
from pipeline_pix2pix_controlnet import InstructPix2PixControlNetPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor
from diffusers import ControlNetModel, StableDiffusionInstructPix2PixPipeline, StableDiffusionControlNetPipeline

sys.path.insert(0, '/home/andranik/Documents/Projects/text2video/Text2Video-Zero')
from utils import prepare_video, pre_process_canny
sys.path.remove('/home/andranik/Documents/Projects/text2video/Text2Video-Zero')

device = 'cuda'
dtype = torch.float16

model_id = "timbrooks/instruct-pix2pix"
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=dtype)
pipe = InstructPix2PixControlNetPipeline.from_pretrained(model_id,
                                                         controlnet=controlnet,
                                                         safety_checker=None,
                                                         use_safetensors=False,
                                                         torch_dtype=dtype).to(device)
pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=3))
controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=3))

video_path = '/home/andranik/Documents/Projects/text2video/video_editing/Tune-A-Video/__assets__/pix2pix_video_2fps/ballet.mp4'
video, fps = prepare_video(video_path,
                           512,
                           device,
                           dtype,
                           False,
                           start_t=0, end_t=1, output_fps=4)

video_normalized = video / 127.5 - 1.0

# video = video[:1]
control = pre_process_canny(video).to(device).to(dtype)

f, c, h, w = video.size()

prompt = "make her a golden sculpture"
g = torch.Generator(device=device)



g.manual_seed(0)
result = pipe(
    image=video_normalized,
    control_image=control,
    prompt=[prompt] * f,
    num_inference_steps=20,
    image_guidance_scale=1.0,
    controlnet_conditioning_scale=1.0,
    generator=g,
).images



for i, r in enumerate(result):
    imageio.imwrite(f'{i}.png', r)
