# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    PIL_INTERPOLATION,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from diffusers.pipelines import StableDiffusionControlNetPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import MultiControlNetModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from torch.nn import MSELoss, L1Loss


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess
def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


class InstructPix2PixControlNetPipeline(StableDiffusionControlNetPipeline):
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: ControlNetModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
        )

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)


    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt


            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds, negative_prompt_embeds])

        return prompt_embeds

    def prepare_image_latents(
        self, image, batch_size, num_images_per_prompt, dtype, device, do_classifier_free_guidance, generator=None
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            image_latents = [self.vae.encode(image[i : i + 1]).latent_dist.mode() for i in range(batch_size)]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = self.vae.encode(image).latent_dist.mode()

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # expand image_latents for batch_size
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)

        if do_classifier_free_guidance:
            uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents

    def prepare_control(
        self, image, width, height, batch_size, num_images_per_prompt, device, dtype, do_classifier_free_guidance
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 3)

        return image

    def latents_to_x0(self, model_output, timestep: int, sample):
        # 2. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )
        return pred_original_sample

    def get_guided_latents(self, noise_pred, t, latents, latents_prev, classifier_guidance_scale):
        t = int(t.item())
        latents.requires_grad = True
        pred_x0 = self.latents_to_x0(noise_pred, t, latents)
        f, c, h, w = pred_x0.size()
        grads = torch.zeros_like(latents)
        # coeffs = torch.linspace(0, 1, f, device=latents.device, dtype=latents.dtype)
        # coeffs = coeffs ** 0.5
        # coeffs /= coeffs.sum()
        for ind in range(1, f):
            L1Loss()(pred_x0[ind], pred_x0[ind-1]).backward(retain_graph=True)
            grads[ind] = latents.grad[ind]
        return latents_prev - classifier_guidance_scale * grads / (grads.norm() + 1e-7)

    # @torch.no_grad()
    def __call__(
        self,
        prompt_instruct: Union[str, List[str]] = None,
        prompt_control: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]] = None,
        control_image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        classifier_guidance_scale: Optional[Union[float, List[float]]] = 0.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        # prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: float = 1.0,
    ):
        # 0. Default height and width to unet
        height, width = self._default_height_width(height, width, image)

        # 1. Check inputs. Raise error if not correct
        if control_image is not None:
            self.check_inputs(
                prompt_instruct,
                control_image,
                height,
                width,
                callback_steps,
                negative_prompt,
                None,
                negative_prompt_embeds,
                controlnet_conditioning_scale,
            )

        # 2. Define call parameters
        if prompt_instruct is not None and isinstance(prompt_instruct, str):
            assert prompt_control is not None and not isinstance(prompt_control, list)
            batch_size = 1
        elif prompt_instruct is not None and isinstance(prompt_instruct, list):
            assert isinstance(prompt_control, list) and len(prompt_instruct) == len(prompt_control)
            batch_size = len(prompt_instruct)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(self.controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(self.controlnet.nets)

        # 3. Encode input prompt
        prompt_instruct_embeds = self._encode_prompt(
            prompt_instruct,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        prompt_control_embeds = self._encode_prompt(
            prompt_control,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare control image
        if control_image is not None:
            if isinstance(self.controlnet, ControlNetModel):
                control_image = self.prepare_control(
                    image=control_image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=self.controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                )
            elif isinstance(self.controlnet, MultiControlNetModel):
                control_images = []

                for control_image_ in control_image:
                    control_image_ = self.prepare_control(
                        image=control_image_,
                        width=width,
                        height=height,
                        batch_size=batch_size * num_images_per_prompt,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=self.controlnet.dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                    )

                    control_images.append(control_image_)

                control_image = control_images
            else:
                assert False

        # 5. Preprocess image
        image = preprocess(image)

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare Image latents
        with torch.no_grad():
            image_latents = self.prepare_image_latents(
                image,
                batch_size,
                num_images_per_prompt,
                prompt_instruct_embeds.dtype,
                device,
                do_classifier_free_guidance,
                generator,
            )

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_instruct_embeds.dtype,
            device,
            generator,
            latents,
        )

        if isinstance(classifier_guidance_scale, float) or isinstance(classifier_guidance_scale, int):
            classifier_guidance_scale = [classifier_guidance_scale] * len(timesteps)
        elif isinstance(classifier_guidance_scale, list):
            classifier_guidance_scale.extend([0]*(len(timesteps) - len(classifier_guidance_scale)))
        else:
            raise "Not supported classifier_guidance_scale type"

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                image_latent_model_input = torch.cat([latent_model_input, image_latents], dim=1)

                with torch.no_grad():
                    # controlnet(s) inference
                    if control_image is not None:
                        down_block_res_samples, mid_block_res_sample = self.controlnet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_control_embeds,
                            controlnet_cond=control_image,
                            conditioning_scale=controlnet_conditioning_scale,
                            return_dict=False,
                        )
                    else:
                        down_block_res_samples, mid_block_res_sample = None, None

                    # predict the noise residual
                    noise_pred = self.unet(
                        image_latent_model_input,
                        t,
                        encoder_hidden_states=prompt_instruct_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                    noise_pred = (
                            noise_pred_uncond
                            + guidance_scale * (noise_pred_text - noise_pred_image)
                            + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_prev = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                if abs(classifier_guidance_scale[i]) > 1e-5:
                    latents = self.get_guided_latents(noise_pred, t, latents, latents_prev, classifier_guidance_scale[i])
                else:
                    latents = latents_prev

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        latents = latents.detach()
        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        with torch.no_grad():
            if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
                self.unet.to("cpu")
                self.controlnet.to("cpu")
                torch.cuda.empty_cache()

            if output_type == "latent":
                image = latents
                has_nsfw_concept = None
            elif output_type == "pil":
                # 8. Post-processing
                image = self.decode_latents(latents)

                # 9. Run safety checker
                image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_instruct_embeds.dtype)

                # 10. Convert to PIL
                image = self.numpy_to_pil(image)
            else:
                # 8. Post-processing
                image = self.decode_latents(latents)

                # 9. Run safety checker
                image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_instruct_embeds.dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
