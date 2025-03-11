from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch
from diffusers import HunyuanVideoPipeline
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import HunyuanVideoPipelineOutput
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import MultiPipelineCallbacks
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import PipelineCallback
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import retrieve_timesteps
from PIL import Image


def resizecrop(image, th, tw):
    w, h = image.size
    if h / w > th / tw:
        new_w = int(w)
        new_h = int(new_w * th / tw)
    else:
        new_h = int(h)
        new_w = int(new_h * tw / th)
    left = (w - new_w) / 2
    top = (h - new_h) / 2
    right = (w + new_w) / 2
    bottom = (h + new_h) / 2
    image = image.crop((left, top, right, bottom))
    return image


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class SkyreelsVideoPipeline(HunyuanVideoPipeline):
    """
    support i2v and t2v
    support true_cfg
    """

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def do_classifier_free_guidance(self):
        # return self._guidance_scale > 1 and self.transformer.config.time_cond_proj_dim is None
        return self._guidance_scale > 1

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool,
        negative_prompt: str = "",
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 256,
    ):
        num_hidden_layers_to_skip = self.clip_skip if self.clip_skip is not None else 0
        print(f"num_hidden_layers_to_skip: {num_hidden_layers_to_skip}")
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_llama_prompt_embeds(
                prompt,
                prompt_template,
                num_videos_per_prompt,
                device=device,
                dtype=dtype,
                num_hidden_layers_to_skip=num_hidden_layers_to_skip,
                max_sequence_length=max_sequence_length,
            )
        if negative_prompt_embeds is None and do_classifier_free_guidance:
            negative_prompt_embeds, negative_attention_mask = self._get_llama_prompt_embeds(
                negative_prompt,
                prompt_template,
                num_videos_per_prompt,
                device=device,
                dtype=dtype,
                num_hidden_layers_to_skip=num_hidden_layers_to_skip,
                max_sequence_length=max_sequence_length,
            )
        if self.text_encoder_2 is not None and pooled_prompt_embeds is None:
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt,
                num_videos_per_prompt,
                device=device,
                dtype=dtype,
                max_sequence_length=77,
            )
            if negative_pooled_prompt_embeds is None and do_classifier_free_guidance:
                negative_pooled_prompt_embeds = self._get_clip_prompt_embeds(
                    negative_prompt,
                    num_videos_per_prompt,
                    device=device,
                    dtype=dtype,
                    max_sequence_length=77,
                )
        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_attention_mask,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def image_latents(
        self,
        initial_image,
        batch_size,
        height,
        width,
        device,
        dtype,
        num_channels_latents,
        video_length,
    ):
        initial_image = initial_image.unsqueeze(2)
        image_latents = self.vae.encode(initial_image).latent_dist.sample()
        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
            image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        else:
            image_latents = image_latents * self.vae.config.scaling_factor
        padding_shape = (
            batch_size,
            num_channels_latents,
            video_length - 1,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        latent_padding = torch.zeros(padding_shape, device=device, dtype=dtype)
        image_latents = torch.cat([image_latents, latent_padding], dim=2)
        return image_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion",
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        num_inference_steps: int = 50,
        sigmas: List[float] = None,
        guidance_scale: float = 1.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = 2,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        max_sequence_length: int = 256,
        embedded_guidance_scale: Optional[float] = 6.0,
        image: Optional[Union[torch.Tensor, Image.Image]] = None,
        cfg_for: bool = False,
    ):
        if hasattr(self, "text_encoder_to_gpu"):
            self.text_encoder_to_gpu()

        if image is not None and isinstance(image, Image.Image):
            image = resizecrop(image, height, width)

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            None,
            height,
            width,
            prompt_embeds,
            callback_on_step_end_tensor_inputs,
            prompt_template,
        )
        #  add negative prompt check
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_attention_mask,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_template=prompt_template,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_attention_mask=negative_attention_mask,
            device=device,
            max_sequence_length=max_sequence_length,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

        ## Embeddings are concatenated to form a batch.
        if self.do_classifier_free_guidance:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
            negative_attention_mask = negative_attention_mask.to(transformer_dtype)
            if negative_pooled_prompt_embeds is not None:
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(transformer_dtype)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_attention_mask is not None:
                prompt_attention_mask = torch.cat([negative_attention_mask, prompt_attention_mask])
            if pooled_prompt_embeds is not None:
                pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])

        # 4. Prepare timesteps
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        if image is not None:
            num_channels_latents = int(num_channels_latents / 2)
            image = self.video_processor.preprocess(image, height=height, width=width).to(
                device, dtype=prompt_embeds.dtype
            )
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )
        # add image latents
        if image is not None:
            image_latents = self.image_latents(
                image, batch_size, height, width, device, torch.float32, num_channels_latents, num_latent_frames
            )

            image_latents = image_latents.to(transformer_dtype)
        else:
            image_latents = None

        # 6. Prepare guidance condition
        if self.do_classifier_free_guidance:
            guidance = (
                torch.tensor([embedded_guidance_scale] * latents.shape[0] * 2, dtype=transformer_dtype, device=device)
                * 1000.0
            )
        else:
            guidance = (
                torch.tensor([embedded_guidance_scale] * latents.shape[0], dtype=transformer_dtype, device=device)
                * 1000.0
            )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if hasattr(self, "text_encoder_to_cpu"):
            self.text_encoder_to_cpu()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latents = latents.to(transformer_dtype)
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                # timestep = t.expand(latents.shape[0]).to(latents.dtype)
                if image_latents is not None:
                    latent_image_input = (
                        torch.cat([image_latents] * 2) if self.do_classifier_free_guidance else image_latents
                    )
                    latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=1)
                timestep = t.repeat(latent_model_input.shape[0]).to(torch.float32)
                if cfg_for and self.do_classifier_free_guidance:
                    noise_pred_list = []
                    for idx in range(latent_model_input.shape[0]):
                        noise_pred_uncond = self.transformer(
                            hidden_states=latent_model_input[idx].unsqueeze(0),
                            timestep=timestep[idx].unsqueeze(0),
                            encoder_hidden_states=prompt_embeds[idx].unsqueeze(0),
                            encoder_attention_mask=prompt_attention_mask[idx].unsqueeze(0),
                            pooled_projections=pooled_prompt_embeds[idx].unsqueeze(0),
                            guidance=guidance[idx].unsqueeze(0),
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        noise_pred_list.append(noise_pred_uncond)
                    noise_pred = torch.cat(noise_pred_list, dim=0)
                else:
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        pooled_projections=pooled_prompt_embeds,
                        guidance=guidance,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype) / self.vae.config.scaling_factor
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return HunyuanVideoPipelineOutput(frames=video)
