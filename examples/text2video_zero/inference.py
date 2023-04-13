import os.path
import sys
import numpy as np
import torch
import imageio
from einops import rearrange

from pipeline_pix2pix_controlnet import InstructPix2PixControlNetPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor
from diffusers import ControlNetModel, StableDiffusionInstructPix2PixPipeline, StableDiffusionControlNetPipeline

sys.path.insert(0, '/home/andranik/Documents/Projects/text2video/Text2Video-Zero')
from utils import prepare_video, pre_process_canny, pre_process_pose, create_video
sys.path.remove('/home/andranik/Documents/Projects/text2video/Text2Video-Zero')


def inference_chunk(frame_ids, prompt_i, prompt_c, **kwargs):
    prompt_i = np.array(prompt_i)
    prompt_c = np.array(prompt_c)
    latents = None
    if 'latents' in kwargs:
        latents = kwargs.pop('latents')[frame_ids]
    if 'image' in kwargs:
        kwargs['image'] = kwargs['image'][frame_ids]
    if 'control_image' in kwargs:
        kwargs['control_image'] = kwargs['control_image'][frame_ids]
    return pipe(prompt_instruct=prompt_i[frame_ids].tolist(),
                prompt_control=prompt_c[frame_ids].tolist(),
                latents=latents,
                **kwargs)



device = 'cuda'
dtype = torch.float16

model_id = "timbrooks/instruct-pix2pix"
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=dtype)
pipe = InstructPix2PixControlNetPipeline.from_pretrained(model_id,
                                                         controlnet=controlnet,
                                                         safety_checker=None,
                                                         use_safetensors=False,
                                                         torch_dtype=dtype).to(device)
pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=3))
controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=3))
g = torch.Generator(device=device)

test_set_path = '/home/andranik/Desktop/video test set/test set'
video_prompts = {
    'full body/pexels-kindel-media-8164487-1080x1920-30fps.mp4': ['replace man with chimpanzee', 'make him santa claus', 'replace man with cleopatra', 'make him iron man'],
    'full body/pexels-mart-production-7331381-2160x3840-25fps.mp4': ['make him santa claus', 'replace man with cleopatra', 'replace man with chimpanzee', 'make him iron man'],
    'full body/pexels-mary-taylor-6002038-2160x3840-30fps.mp4': ['make him superman', 'replace man with kangaroo', 'make it cyberpunk style', 'make man look like terminator'],
    'full body/pexels-rodnae-productions-7334739-1080x1920-24fps.mp4': ['replace humans with aliens', 'replace people with marble sculptures', 'make it Van Gogh Starry Night style'],
    'full body/pexels-shvets-production-7197861-2160x3840-25fps.mp4': ['replace girl with bear', 'make girl a golden sculpture', 'what if it was iron man dancing', 'make her Disney Moana character'],
    'full body/pexels-tony-schnagl-5528734-2160x3840-25fps.mp4':  ['make him santa claus', 'replace man with cleopatra', 'replace man with chimpanzee', 'make him iron man', 'make him a Disney cartoon character'],
    'multiple/pexels-cottonbro-studio-2795172-3840x2160-25fps.mp4': ['replace girls with Disney carton characters', 'replace girls with wonder woman', 'make them look like golden sculptures'],
    'multiple/pexels-cottonbro-studio-4100357-4096x2160-50fps.mp4': ['replace men with chimpanzee', 'make them marble sculptures', 'make them golden sculptures', 'replace them with astronauts'],
    'multiple/pexels-fauxels-3253079-3840x2160-25fps.mp4': ['replace people with mortal kombat characters', 'make them marble sculptures', 'what if they were made of stone'],
    'people back:angle/pexels-diva-plavalaguna-6985525-3840x2160-50fps.mp4': ['make him santa claus', 'replace man with cleopatra', 'replace man with chimpanzee', 'make him iron man', 'make him a Disney cartoon character'],
    'portrait/pexels-koolshooters-8529808-3840x2160-25fps.mp4': ['make girl a golden sculpture', 'what if it was iron man dancing', 'make her Disney Moana character'],
    'portrait/pexels-shvets-production-8416580-1080x1920-25fps.mp4': ['make her a roman sculpture', 'make her an egyptian sculpture', 'replace her with terminator', 'replace her with orangutan'],
}

for video_name in video_prompts:
    video_path = os.path.join(test_set_path, video_name)
    video, fps = prepare_video(video_path,
                               512,
                               device,
                               dtype,
                               False,
                               start_t=0, end_t=5, output_fps=10)
    video_normalized = video / 127.5 - 1.0
    control = pre_process_pose(video, apply_pose_detect=True).to(device).to(dtype)
    pose_path = os.path.join('outputs', 'pix2pix_original',  os.path.splitext(video_name)[0], f'pose.mp4')
    os.makedirs(os.path.dirname(pose_path), exist_ok=True)
    create_video(rearrange(control, 'b c h w -> b h w c').cpu(), fps, path=pose_path, watermark=None)
    f, c, h, w = video.size()
    prompt_control = ''
    latents = torch.randn((1, 4, h // 8, w // 8), dtype=dtype, device=device).repeat(f, 1, 1, 1)

    chunk_size = 16
    seed = 0
    chunk_ids = np.arange(0, f, chunk_size - 1)

    for prompt_instruct in video_prompts[video_name]:
        result = []

        for i in range(len(chunk_ids)):
            ch_start = chunk_ids[i]
            ch_end = f if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
            frame_ids = [0] + list(range(ch_start, ch_end))
            g.manual_seed(seed)
            print(f'Processing chunk {i + 1} / {len(chunk_ids)}')
            result.append(inference_chunk(frame_ids=frame_ids,
                                          prompt_i=[prompt_instruct] * f,
                                          prompt_c=[prompt_control] * f,
                                          image=video_normalized,
                                          # control_image=control,
                                          num_inference_steps=20,
                                          image_guidance_scale=1.0,
                                          latents=latents,
                                          controlnet_conditioning_scale=1.0,
                                          generator=g,
                                          output_type='numpy'
                                          ).images[1:])
        result = np.concatenate(result)
        out_path = os.path.join('outputs', 'pix2pix_original', os.path.splitext(video_name)[0], f'{prompt_instruct}.mp4')
        create_video(result, fps, path=out_path, watermark=None)
