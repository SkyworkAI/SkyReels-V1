from typing import Any, Callable, Dict, List, Optional, Union

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
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

# 20250319 pftq: logging
import logging
logger = logging.getLogger("SkyreelsVideoPipeline")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 20250319 pftq: bad render detection - Start of detection function
def check_latent_transition(curr_latents, image_latents=None):
    """
    Compares consecutive frames within the current latent tensor and average adherence to input image.
    Args:
        latents: Current latent tensor (shape: [batch_size, channels, frames, height, width]).
        image_latents: Optional latent tensor of the input image (shape: [batch_size, channels, frames, height, width]).
    Returns:
        tuple: (float: max frame change (1 - similarity), int: max consecutive still frames, float: avg image adherence).
    """
    if curr_latents.shape[2] < 2:  # Need at least 2 frames
        return 0.0, 0
    
    max_frame_change = 0.0
    max_still_count_step = 0
    current_still_count = 0
    image_adherence_sum = 0.0
    image_adherence_count = 0

    # Reference input image latent (first frame)
    if image_latents is not None:
        input_frame = image_latents[:, :, 0].flatten()
        input_norm = input_frame / (torch.norm(input_frame) + 1e-8)
    
    # Compare consecutive frames within curr_latents within first 2 seconds of the video
    for t in range(1, min(curr_latents.shape[2], 48)):
        prev_frame = curr_latents[:, :, t - 1].flatten()
        curr_frame = curr_latents[:, :, t].flatten()
        # Normalize to avoid numerical issues
        prev_norm = prev_frame / (torch.norm(prev_frame) + 1e-8)
        curr_norm = curr_frame / (torch.norm(curr_frame) + 1e-8)
        similarity = torch.cosine_similarity(prev_norm, curr_norm, dim=0).item()
        # Clamp similarity to [-1, 1]
        similarity = max(min(similarity, 1.0), -1.0)
        frame_change = 1.0 - similarity  # Range: 0 to 2
        max_frame_change = max(max_frame_change, frame_change)
        
        # Stillness check per frame pair
        if similarity >= 0.999:  # Relaxed threshold for individual frames
            current_still_count += 1
            max_still_count_step = max(max_still_count_step, current_still_count)
        else:
            current_still_count = 0
        """    
        # Check against 24 frames back
        if t >= 24:
            one_second_frame = curr_latents[:, :, t - 24].flatten()
            one_second_norm = one_second_frame / (torch.norm(one_second_frame) + 1e-8)
            one_second_similarity = torch.cosine_similarity(one_second_norm, curr_norm, dim=0).item()
            one_second_similarity = max(min(one_second_similarity, 1.0), -1.0)
            one_second_frame_change = 1.0 - one_second_similarity
            max_frame_change = max(max_frame_change, one_second_frame_change)
        """
        # Image adherence check (accumulate for average)
        if image_latents is not None:
            adherence = torch.cosine_similarity(input_norm, curr_norm, dim=0).item()
            adherence = max(min(adherence, 1.0), -1.0)  # Clamp to [-1, 1]
            image_adherence_sum += adherence
            image_adherence_count += 1

    # Compute average image adherence
    image_adherence = image_adherence_sum / image_adherence_count if image_latents is not None else None

    return max_frame_change, max_still_count_step, image_adherence
def check_frame_transition(frames):
    """
    Compares consecutive decoded frames using inverted cosine similarity in pixel space.
    Args:
        frames: List of PIL Images (decoded video frames).
    Returns:
        tuple: (float: max frame change (1 - similarity), int: max consecutive still frames).
    """
    #logger.debug(f"Received {len(frames)} frames for transition check")
    if len(frames) < 2:
        #logger.warning(f"Fewer than 2 frames ({len(frames)}), returning 0.0, 0")
        return 0.0, 0
    
    # Convert PIL Images to numpy arrays (RGB, normalized to [0, 1])
    frame_arrays = [np.array(frame.convert('RGB'), dtype=np.float32).flatten() / 255.0 for frame in frames]
    #logger.debug(f"Processed {len(frame_arrays)} frames, shape of first frame array: {frame_arrays[0].shape}")
    
    # Check frame differences
    for i in range(1, min(3, len(frame_arrays))):
        diff = np.mean(np.abs(frame_arrays[i] - frame_arrays[i-1]))
        #logger.debug(f"Frame {i-1} to {i} mean abs diff: {diff:.4f}")
    
    max_frame_change = 0.0
    max_still_count = 0
    current_still_count = 0
    
    # Compare consecutive frames within first 2 seconds of the video
    for t in range(1, min(len(frame_arrays), 48)):
        prev_frame = torch.from_numpy(frame_arrays[t - 1])
        curr_frame = torch.from_numpy(frame_arrays[t])
        prev_norm = prev_frame / (torch.norm(prev_frame) + 1e-8)
        curr_norm = curr_frame / (torch.norm(curr_frame) + 1e-8)
        similarity = torch.cosine_similarity(prev_norm, curr_norm, dim=0).item()
        similarity = max(min(similarity, 1.0), -1.0)  # Clamp to [-1, 1]

        # Stillness check (relaxed threshold)
        if similarity >= 0.999:
            current_still_count += 1
            max_still_count = max(max_still_count, current_still_count)
        else:
            current_still_count = 0
        
        if t >= 24:
            one_second_frame = torch.from_numpy(frame_arrays[t - 24])
            one_second_norm = one_second_frame / (torch.norm(one_second_frame) + 1e-8)
            one_second_similarity = torch.cosine_similarity(one_second_norm, curr_norm, dim=0).item()
            one_second_similarity = max(min(one_second_similarity, 1.0), -1.0)  # Clamp to [-1, 1]
            similarity = min(similarity, one_second_similarity)
        frame_change = 1.0 - similarity  # Range: 0 to 2
        max_frame_change = max(max_frame_change, frame_change)

    return max_frame_change, max_still_count
# 20250319 pftq: bad render detection - End of detection function

# 20250319 pftq: bad render detection - Extend output to include pre- and post-decoding metrics
class HunyuanVideoPipelineOutputExtended(HunyuanVideoPipelineOutput):
    def __init__(self, frames, badRender: bool = False, maxFrameChange: float = 0.0, maxStillCount: int = 0, maxFrameChange_pre: float = 0.0, maxStillCount_pre: int = 0, badRenderAtStep: int = 0, initialMatch: Optional[float] = None):
        super().__init__(frames=frames)
        self.badRender = badRender
        self.maxFrameChange = maxFrameChange
        self.maxStillCount = maxStillCount
        self.maxFrameChange_pre = maxFrameChange_pre
        self.maxStillCount_pre = maxStillCount_pre
        self.badRenderAtStep = badRenderAtStep
        self.initialMatch = initialMatch
# 20250319 pftq: bad render detection - End of output extension

class SkyreelsVideoPipeline(HunyuanVideoPipeline):
    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def do_classifier_free_guidance(self):
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
        detect_bad_renders: Optional[bool] = False, # 20250320 pftq: auto-detect and skip/retry bad renders
        bad_render_threshold: Optional[float] = 0.02 # 20250320 pftq: optional setting to be more aggressive in cancelling renders, default 0.02 is most conservative. 0.04 and above is generally a good render
    ):
        if hasattr(self, "text_encoder_to_gpu"):
            self.text_encoder_to_gpu()

        if image is not None and isinstance(image, Image.Image):
            image = resizecrop(image, height, width)

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self.check_inputs(
            prompt,
            None,
            height,
            width,
            prompt_embeds,
            callback_on_step_end_tensor_inputs,
            prompt_template,
        )
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

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

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

        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )

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
            num_latent_frames,
            torch.float32,
            device,
            generator,
            latents,
        )
        if image is not None:
            image_latents = self.image_latents(
                image, batch_size, height, width, device, torch.float32, num_channels_latents, num_latent_frames
            )
            image_latents = image_latents.to(transformer_dtype)
        else:
            image_latents = None

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

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if hasattr(self, "text_encoder_to_cpu"):
            self.text_encoder_to_cpu()


        # 20250319 pftq: bad render detection - Start of detection initialization
        max_frame_change_pre = 0.0
        max_still_count_pre = 0
        current_still_count_pre = 0
        bad_render = False
        max_frame_change = 0.0
        max_still_count = 0
        bad_render_at = 0
        initial_image_adherence = 1  # Initialize to None
        image_adherence_pre_history = []
        # 20250319 pftq: bad render detection - End of detection initialization
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latents = latents.to(transformer_dtype)
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
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

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # 20250319 pftq: bad render detection - Start of latent analysis
                # Early check at 25% mark. initial image adherence <=0.02 generally means the render will be bad, .03 often does too if more aggressive.  Other things like max change between frames or still image detection don't really do much until the second-to-last step so you might as well not check them until post-decoding.
                # change i == to >= for debugging and analyzing how the numbers change with each step.
                if detect_bad_renders and i == num_inference_steps//4:  
                    frame_change_pre, still_count_step_pre, adherence = check_latent_transition(latents, image_latents)
                    max_frame_change_pre = max(max_frame_change_pre, frame_change_pre)
                    if adherence is not None:
                        image_adherence_pre_history.append(adherence)
                        if initial_image_adherence == 1:
                            initial_image_adherence = adherence  # Update with latest average
                    
                    # Update stillness count across steps
                    if still_count_step_pre > 0:
                        current_still_count_pre += still_count_step_pre
                        max_still_count_pre = max(max_still_count_pre, current_still_count_pre)
                    else:
                        current_still_count_pre = 0
                    
                    # Set bad_render and  interrupt
                    bad_render_pre = max_still_count_pre >= 24 or initial_image_adherence <=bad_render_threshold
                    if bad_render_pre:
                        # Optional: Uncomment to interrupt early
                        bad_render = True
                        bad_render_at = i
                        if num_inference_steps-i<=2:
                            logger.debug(f"Step {i} bad render detected, but almost done anyway: max_frame_change_pre={max_frame_change_pre:.4f}, max_still_count_pre={max_still_count_pre}, initial_image_adherence={initial_image_adherence} (<=0.02 generally means bad render, 0.03<= often does too if more aggressive, change via --bad_render_threshold)")
                            print('[%s]' % ', '.join(map(str, image_adherence_pre_history)))
                        else:
                            self._interrupt = True
                            logger.debug(f"Step {i} bad render detected early, aborting and retrying with new seed: max_frame_change_pre={max_frame_change_pre:.4f}, max_still_count_pre={max_still_count_pre}, initial_image_adherence={initial_image_adherence} (<=0.02 generally means bad render, 0.03<= often does too if more aggressive, change via --bad_render_threshold)")
                            print('[%s]' % ', '.join(map(str, image_adherence_pre_history)))
                            break
                    else:
                        logger.debug(f"Step {i}  bad render early check passed: max_frame_change_pre={max_frame_change_pre:.4f}, max_still_count_pre={max_still_count_pre}, initial_image_adherence={initial_image_adherence} (<=0.02 generally means bad render, 0.03<= often does too if more aggressive, change via --bad_render_threshold)")
                # 20250319 pftq: bad render detection - End of latent analysis

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # Decode latents to frames
        if not output_type == "latent":
            latents = latents.to(self.vae.dtype) / self.vae.config.scaling_factor
            video_tensor = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video_tensor, output_type=output_type)
        else:
            video = latents

        # 20250319 pftq: bad render detection - Start of post-decoding analysis 
        if detect_bad_renders and not output_type == "latent" and not self._interrupt and not bad_render:
            # Pass video[0] if nested, otherwise use video directly
            frames_to_check = video[0] if len(video) == 1 and isinstance(video[0], (list, tuple)) else video
            #logger.debug(f"Passing {len(frames_to_check)} frames to check_frame_transition (nested: {len(video) == 1 and isinstance(video[0], (list, tuple))})")
            max_frame_change, max_still_count = check_frame_transition(frames_to_check)
            bad_render = max_frame_change >= 0.19 or max_still_count >= 24
            logger.debug(f"Post-decoding bad render check: max_frame_change={max_frame_change:.4f}, max_still_count={max_still_count}, bad_render={bad_render}")
        # 20250319 pftq: bad render detection - End of post-decoding analysis

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        # 20250319 pftq: bad render detection - Start of output modification
        return HunyuanVideoPipelineOutputExtended(
            frames=video,
            badRender=bad_render,
            maxFrameChange=max_frame_change,
            maxStillCount=max_still_count,
            maxFrameChange_pre=max_frame_change_pre,
            maxStillCount_pre=max_still_count_pre,
            badRenderAtStep=bad_render_at,
            initialMatch=initial_image_adherence
        )
        # Original return (uncomment to revert):
        # return HunyuanVideoPipelineOutput(frames=video)
        # 20250319 pftq: bad render detection - End of output modification
