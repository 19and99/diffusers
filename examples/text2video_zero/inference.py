import os.path
import sys
import numpy as np
import torch
import imageio
from einops import rearrange

from pipeline_pix2pix_controlnet import InstructPix2PixControlNetPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor
from diffusers import ControlNetModel, StableDiffusionInstructPix2PixPipeline, StableDiffusionControlNetPipeline

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

sys.path.insert(0, '/home/andranik/Documents/Projects/text2video/Text2Video-Zero')
from utils1 import (prepare_video,
                    pre_process_canny,
                    pre_process_pose,
                    pre_process_depth,
                    pre_process_HED,
                    pre_process_normal,
                    create_video)
sys.path.remove('/home/andranik/Documents/Projects/text2video/Text2Video-Zero')


def inference_chunk(frame_ids, prompt_i, prompt_c, **kwargs):
    prompt_i = np.array(prompt_i)
    prompt_c = np.array(prompt_c)
    latents = None
    if 'latents' in kwargs:
        latents = kwargs.pop('latents')[frame_ids]
    if 'image' in kwargs:
        kwargs['image'] = kwargs['image'][frame_ids]
    if 'control_image' in kwargs and kwargs['control_image'] is not None:
        if isinstance(kwargs['control_image'], list):
            kwargs['control_image'] = [ci[frame_ids] for ci in kwargs['control_image']]
        else:
            kwargs['control_image'] = kwargs['control_image'][frame_ids]
    return pipe(prompt_instruct=prompt_i[frame_ids].tolist(),
                prompt_control=prompt_c[frame_ids].tolist(),
                latents=latents,
                **kwargs)



device = 'cuda'
dtype = torch.float16

model_id = "timbrooks/instruct-pix2pix"


test_set_path = '/home/andranik/Desktop/video test set/test set 20 crop'

prompts = [
    # "make it Starry Night style",
    "make it Anime style",
    "make it Golden sculpture",
    "make it Modigliani painting",
    # "make it Marble Sculpture",
    # "make it 1900's style",
    # "make it Claymation",
    # "make it Watercolor style",
    # "make it Paper origami",
    # "make it Pen and ink style",
    # "make it Charcoal sketch",
    # "make it Cloudscape"
]

video_prompts = {
    # 'pexels-chris-galkowski-1987421-1920x1080-30fps.mp4': prompts,
    # 'pexels-christopher-schultz-5147455-1080x1920-30fps.mp4': prompts,
    'pexels-cottonbro-studio-2795172-3840x2160-25fps.mp4': prompts,
    'pexels-cottonbro-studio-5700073-2160x4096-25fps.mp4': prompts,
    # 'pexels-diva-plavalaguna-6985525-3840x2160-50fps.mp4': prompts,
    'pexels-fauxels-3253079-3840x2160-25fps.mp4': prompts,
    'pexels-kindel-media-8164487-1080x1920-30fps.mp4': prompts,
    # 'pexels-koolshooters-8529808-3840x2160-25fps.mp4': prompts,
    # 'pexels-mart-production-7331381-2160x3840-25fps.mp4': prompts,
    # 'pexels-mary-taylor-6002038-2160x3840-30fps.mp4': prompts,
    # 'pexels-mikhail-nilov-6981411-1920x1080-25fps.mp4': prompts,
    # 'pexels-olia-danilevich-4753975-720x1280-25fps.mp4': prompts,
    # 'pexels-pixabay-854963-1920x1080-30fps.mp4': prompts,
    # 'pexels-rodnae-productions-7334739-1080x1920-24fps.mp4': prompts,
    # 'pexels-rodnae-productions-8624901-1920x1080-30fps.mp4': prompts,
    # 'pexels-shvets-production-7197861-2160x3840-25fps.mp4': prompts,
    # 'pexels-shvets-production-8416580-1080x1920-25fps.mp4': prompts,
    # 'pexels-taryn-elliott-9116112-3840x2160-25fps.mp4': prompts,
    # 'pexels-tony-schnagl-5528734-2160x3840-25fps.mp4': prompts,
    # 'pexels-zlatin-georgiev-7173031-3840x2160-25fps.mp4': prompts,
}

configurations = {
    # 'pix2pix_depth+HED+normal': [('depth', pre_process_depth), ('hed', pre_process_HED), ('normal', pre_process_normal)],
    # 'pix2pix_all': [('openpose', pre_process_pose), ('depth', pre_process_depth), ('hed', pre_process_HED), ('canny', pre_process_canny), ('normal', pre_process_normal)],
    # 'pix2pix_canny+depth+HED+normal': [('depth', pre_process_depth), ('hed', pre_process_HED), ('canny', pre_process_canny), ('normal', pre_process_normal)],
    # 'pix2pix_pose+canny':     [('openpose', pre_process_pose), ('canny', pre_process_canny)],
    # 'pix2pix_pose+depth':     [('openpose', pre_process_pose), ('depth', pre_process_depth)],
    # 'pix2pix_pose+HED':       [('openpose', pre_process_pose), ('hed', pre_process_HED)],
    # 'pix2pix_pose+normal':    [('openpose', pre_process_pose), ('normal', pre_process_normal)],
    # 'pix2pix_depth':          [('depth', pre_process_depth)],
    # 'pix2pix_depth+HED':      [('depth', pre_process_depth), ('hed', pre_process_HED)],
    'pix2pix_OF':    [('depth', pre_process_depth)],
}



for configuration_name in configurations:
    control_info = configurations[configuration_name]
    controlnets = []
    for (c_suffix, _) in control_info:
        controlnet = ControlNetModel.from_pretrained(f"lllyasviel/sd-controlnet-{c_suffix}", torch_dtype=dtype, use_safetensors=False)
        controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=3))
        controlnets.append(controlnet)

    if len(controlnets) == 0:
        dummy_controlnet = ControlNetModel.from_pretrained(f"lllyasviel/sd-controlnet-canny", torch_dtype=dtype, use_safetensors=False)
        controlnets.append(dummy_controlnet)

    pipe = InstructPix2PixControlNetPipeline.from_pretrained(model_id,
                                                             controlnet=controlnets,
                                                             safety_checker=None,
                                                             use_safetensors=False,
                                                             torch_dtype=dtype).to(device)
    pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=3))
    g = torch.Generator(device=device)

    for video_name in video_prompts:
        video_path = os.path.join(test_set_path, video_name)
        video, fps = prepare_video(video_path,
                                   512,
                                   device,
                                   dtype,
                                   False,
                                   start_t=0, end_t=5, output_fps=10)
        video_normalized = video / 127.5 - 1.0

        # save original video
        o_path = os.path.join('outputs', configuration_name, os.path.splitext(video_name)[0], f'original.mp4')
        os.makedirs(os.path.dirname(o_path), exist_ok=True)
        create_video(rearrange((video_normalized + 1.0) / 2.0, 'b c h w -> b h w c').cpu(), fps, path=o_path, watermark=None)

        controls = []
        for (c_suffix, pre_process) in control_info:
            control = pre_process(video).to(device).to(dtype)
            control_save_path = os.path.join('outputs', configuration_name, os.path.splitext(video_name)[0], f'{c_suffix}.mp4')
            os.makedirs(os.path.dirname(control_save_path), exist_ok=True)
            create_video(rearrange(control, 'b c h w -> b h w c').cpu(), fps, path=control_save_path, watermark=None)
            controls.append(control)

        if len(controls) == 0:
            controls = None

        f, c, h, w = video.size()

        seed = 0
        g.manual_seed(seed)
        latents = torch.randn((1, 4, h // 8, w // 8), dtype=dtype, device=device, generator=g).repeat(f, 1, 1, 1)

        num_inference_steps = 20
        chunk_size = 4
        chunk_ids = np.arange(0, f, chunk_size - 1)
        for prompt_instruct in video_prompts[video_name]:
            result = []
            prompt_control = prompt_instruct
            first_frame_latents = None
            for i in range(len(chunk_ids)):
                ch_start = chunk_ids[i]
                ch_end = f if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
                frame_ids = [0, max(0, ch_start-1)] + list(range(ch_start, ch_end))
                g.manual_seed(seed)
                print(f'Processing chunk {i + 1} / {len(chunk_ids)}')
                res, first_frame_latents = inference_chunk(frame_ids=frame_ids,
                                                           prompt_i=[prompt_instruct] * f,
                                                           prompt_c=[prompt_control] * f,
                                                           image=video_normalized,
                                                           control_image=controls,
                                                           num_inference_steps=num_inference_steps,
                                                           image_guidance_scale=1.0,
                                                           latents=latents,
                                                           controlnet_conditioning_scale=1.0,
                                                           generator=g,
                                                           output_type='numpy',
                                                           classifier_guidance_scale=list(np.linspace(200, 0, num_inference_steps)),
                                                           # classifier_guidance_scale=0.0,
                                                           first_frame_latents=first_frame_latents,
                                                           )
                result.append(res[2:])
            result = np.concatenate(result)
            out_path = os.path.join('outputs', configuration_name, os.path.splitext(video_name)[0], f'{prompt_instruct}.mp4')
            create_video(result, fps, path=out_path, watermark=None)