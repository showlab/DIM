import math
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, AnyStr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torchvision.utils import save_image

from transformers import AutoConfig, AutoProcessor, AutoModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    ModelOutput,
)
from transformers.cache_utils import (
    Cache,
    DynamicCache,
    SlidingWindowCache,
    StaticCache
)
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from .modeling_qwen2_5_vl import (
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLModel,
    Qwen2_5_VLConfig,
    Qwen2_5_VisionTransformerPretrainedModel,
    QWEN2_5_VL_INPUTS_DOCSTRING,
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLDecoderLayer,
    Qwen2RMSNorm,
    Qwen2_5_VLRotaryEmbedding, Qwen2_5_VLForConditionalGeneration,
)
from utils.vision_process import process_vision_info
from .utils import create_content_gpt
from PIL import Image
import time

import pyrallis
from safetensors.torch import load_file
from .sana_pipeline import guidance_type_select
from .utils import collate_dicts
from .sana_pipeline import SanaPipeline
from .diffusion.utils.config import SanaConfig
from .diffusion import DPMS, FlowEuler, Scheduler
from .diffusion.model.respace import compute_density_for_timestep_sampling
from .diffusion.model.utils import get_weight_dtype, prepare_prompt_ar, resize_and_crop_tensor
from .diffusion.model.nets.sana_blocks import PatchEmbedMS
from .diffusion.data.datasets.utils import (
    ASPECT_RATIO_512_TEST,
    ASPECT_RATIO_1024_TEST,
    ASPECT_RATIO_2048_TEST,
    ASPECT_RATIO_4096_TEST,
)
from utils.vision_process import process_vision_info
from torch.cuda.amp import autocast
from .multimodal.multimodal_projector.builder import build_projector

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Qwen2_5_VLConfig"


def guidance_type_select(default_guidance_type, pag_scale, attn_type):
    guidance_type = default_guidance_type
    if not (pag_scale > 1.0 and attn_type == "linear"):
        guidance_type = "classifier-free"
    elif pag_scale > 1.0 and attn_type == "linear":
        guidance_type = "classifier-free_PAG"
    return guidance_type


def classify_height_width_bin(height: int, width: int, ratios: dict) -> Tuple[int, int]:
    """Returns binned height and width."""
    ar = float(height / width)
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
    default_hw = ratios[closest_ratio]
    return int(default_hw[0]), int(default_hw[1])


class DIMDecoder(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.model_args = model_args
        self.config = pyrallis.load(SanaConfig, open(model_args.sana_config))
        self.image_size = self.resolution = self.config.model.image_size
        self.vis_sampler = self.config.scheduler.vis_sampler
        self.base_ratios = eval(f"ASPECT_RATIO_{self.image_size}_TEST")
        self.flow_shift = self.config.scheduler.flow_shift
        self.weight_dtype = get_weight_dtype(self.config.model.mixed_precision)
        self.progress_fn = lambda progress, desc: None

        # 0. init pretrained core components in SanaPipeline
        sana = SanaPipeline(model_args.sana_config)
        if model_args.model_name_or_path == '':
            sana.from_pretrained(model_args.sana_pretrained)
        else:
            print(f'Will load Sana later from {model_args.model_name_or_path}')

        # 1. build vae and freeze it
        self.vae = sana.vae
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae_dtype = sana.vae_dtype

        # 2. build Sana model
        self.model = sana.model

        # 3. build scheduler
        pred_sigma = getattr(self.config.scheduler, "pred_sigma", True)
        learn_sigma = getattr(self.config.scheduler, "learn_sigma", True) and pred_sigma
        self.train_diffusion = Scheduler(
            str(self.config.scheduler.train_sampling_steps),
            noise_schedule=self.config.scheduler.noise_schedule,
            predict_flow_v=self.config.scheduler.predict_flow_v,
            learn_sigma=learn_sigma,
            pred_sigma=pred_sigma,
            snr=self.config.train.snr_loss,
            flow_shift=self.config.scheduler.flow_shift,
        )

    def forward(
            self,
            mm_cond: Optional[torch.FloatTensor] = None,
            raw_images: Optional[List[torch.FloatTensor]] = None,
            target_images: Optional[List[torch.FloatTensor]] = None,
            raw_captions: Optional = None,
    ):
        # hard code: only support a single image here
        raw_images = raw_images[0].unsqueeze(0)

        raw_images = raw_images.to(self.vae_dtype)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.vae_dtype):
                z = self.vae_encode(
                    images=raw_images,
                    sample_posterior=self.config.vae.sample_posterior,
                    device=self.vae.device
                )

        if target_images is not None:
            # hard code: only support a single image here
            target_images = target_images[0].unsqueeze(0)

            target_images = target_images.to(self.vae_dtype)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=self.vae_dtype):
                    z_target = self.vae_encode(
                        images=target_images,
                        sample_posterior=self.config.vae.sample_posterior,
                        device=self.vae.device
                    )

        if target_images is not None:  # for X-PAD
            clean_images = z_target
            latents_condition = z
        else:  # for X-NPAD
            clean_images = z
            latents_condition = torch.zeros_like(z).to(z.device)

        if mm_cond is not None:
            # B x 1 x N x C
            y = mm_cond.unsqueeze(1)
            # B x 1 x 1 x N
            y_mask = torch.ones(y.shape[0], 1, 1, y.shape[2], device=y.device, dtype=torch.long)

            # truncation or padding
            max_length = self.model_args.max_condition_length
            B, _, N, C = y.shape

            if N < max_length:
                # need to pad to the right up to max_length
                pad_len = max_length - N

                # pad y with zeros (or any desired pad value)
                pad_y = torch.zeros(B, 1, pad_len, C, device=y.device, dtype=y.dtype)
                y = torch.cat([y, pad_y], dim=2)  # [B x 1 x max_length x C]

                # pad y_mask with zeros (mask=0 means “masked out”)
                pad_mask = torch.zeros(B, 1, 1, pad_len, device=y_mask.device, dtype=y_mask.dtype)
                y_mask = torch.cat([y_mask, pad_mask], dim=3)  # [B x 1 x 1 x max_length]
            else:
                # truncate to last max_length tokens
                y = y[:, :, -max_length:, :]  # [B x 1 x max_length x C]
                y_mask = y_mask[:, :, :, -max_length:]  # [B x 1 x 1 x max_length]
        else:
            raise RuntimeError('mm_cond cannot be None!')

        # Sample a random timestep for each image
        bs = clean_images.shape[0]
        timesteps = torch.randint(
            0, self.config.scheduler.train_sampling_steps, (bs,), device=clean_images.device
        ).long()
        if self.config.scheduler.weighting_scheme in ["logit_normal"]:
            # adapting from diffusers.training_utils
            u = compute_density_for_timestep_sampling(
                weighting_scheme=self.config.scheduler.weighting_scheme,
                batch_size=bs,
                logit_mean=self.config.scheduler.logit_mean,
                logit_std=self.config.scheduler.logit_std,
                mode_scale=None,  # not used
            )
            timesteps = (u * self.config.scheduler.train_sampling_steps).long().to(clean_images.device)

        # manully set data_info
        data_info = [{
            "img_hw": torch.tensor([self.resolution, self.resolution], dtype=torch.float32),
            "aspect_ratio": torch.tensor(1.0),
        } for _ in range(bs)]

        data_info = collate_dicts(data_info)

        loss_term = self.train_diffusion.training_losses(
            self.model, clean_images, timesteps,
            with_latents_condition=self.model_args.with_latents_condition,
            latents_condition=latents_condition,
            model_kwargs=dict(y=y, mask=y_mask, data_info=data_info)
        )

        diff_loss = loss_term["loss"].mean()

        return diff_loss

    @torch.inference_mode()
    def generate(
            self,
            mm_cond: Optional[torch.FloatTensor] = None,
            raw_images: Optional[List[torch.FloatTensor]] = None,
            raw_captions: Optional[List[AnyStr]] = None,
            task_type: Optional[str] = 'T2I-NPAD',
    ):
        if mm_cond is not None:
            # B x 1 x N x C
            y = mm_cond.unsqueeze(1)
            # B x 1 x 1 x N
            y_mask = torch.ones(y.shape[0], 1, 1, y.shape[2], device=y.device, dtype=torch.long)

            # truncation or padding
            max_length = self.model_args.max_condition_length
            B, _, N, C = y.shape

            if N < max_length:
                # need to pad to the right up to max_length
                pad_len = max_length - N

                # pad y with zeros (or any desired pad value)
                pad_y = torch.zeros(B, 1, pad_len, C, device=y.device, dtype=y.dtype)
                y = torch.cat([y, pad_y], dim=2)  # [B x 1 x max_length x C]

                # pad y_mask with zeros (mask=0 means “masked out”)
                pad_mask = torch.zeros(B, 1, 1, pad_len, device=y_mask.device, dtype=y_mask.dtype)
                y_mask = torch.cat([y_mask, pad_mask], dim=3)  # [B x 1 x 1 x max_length]
            else:
                # truncate to last max_length tokens
                y = y[:, :, -max_length:, :]  # [B x 1 x max_length x C]
                y_mask = y_mask[:, :, :, -max_length:]  # [B x 1 x 1 x max_length]
        else:
            raise RuntimeError('mm_cond cannot be None!')

        null_embeds = self.model.y_embedder.y_embedding
        null_embeds = null_embeds.unsqueeze(0)

        # hard code: only support a single image here
        raw_images = raw_images[0].unsqueeze(0)

        raw_images = raw_images.to(self.vae_dtype)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.vae_dtype):
                z = self.vae_encode(
                    images=raw_images,
                    sample_posterior=self.config.vae.sample_posterior,
                    device=self.vae.device
                )

        if task_type in ['T2I-NPAD', 'IR-NPAD', 'MM-NPAD']:
            latents_condition = torch.zeros_like(z).to(z.device)
        elif task_type in ['T2I-PAD', 'IR-PAD', 'MM-PAD']:
            latents_condition = z
        else:
            raise RuntimeError(f"unknown task type {task_type}")

        generator = torch.Generator(device=y.device).manual_seed(233)

        # generator = torch.Generator(device=y.device)
        # generator.seed()

        latents_denoised = self.denoise(
            prompt=raw_captions[0],
            height=self.image_size,
            width=self.image_size,
            caption_embs=y,
            emb_masks=y_mask.squeeze(1).squeeze(1),
            null_embeds=null_embeds,
            generator=generator,
            latents_condition=latents_condition,
        )

        # B x 32 x 32 x 32 for 1024 x 1024
        # print('denoised latents shape', latents_denoised[0].shape)
        # exit(0)

        image = self.latents2image(
            latents_denoised[0],
            height=self.image_size,
            width=self.image_size,
        )

        return image

    def expand_context(self, max_condition_length):
        null_embedding = self.model.y_embedder.y_embedding
        if null_embedding.shape[0] != max_condition_length:
            print(f'Expand context length from {null_embedding.shape[0]} to {max_condition_length}')

            in_channels = null_embedding.shape[1]
            token_num = max_condition_length

            # replace original null embedding
            self.model.y_embedder.y_embedding = nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5)
        else:
            print('No need to expand context length')

    def expand_channels(self, zero_init=False):
        conv_in = self.model.x_embedder.proj
        in_ch = conv_in.in_channels
        out_ch = conv_in.out_channels
        print(f'Expanding channels from {in_ch} to {in_ch * 2} for image editing...')

        new_conv_in = nn.Conv2d(
            in_channels=in_ch * 2,
            out_channels=out_ch,
            kernel_size=conv_in.kernel_size,
            stride=conv_in.stride,
            padding=conv_in.padding,
        )

        with torch.no_grad():
            new_conv_in.weight[:, :in_ch, :, :].copy_(conv_in.weight)
            if zero_init:
                new_conv_in.weight[:, in_ch:, :, :].zero_()
            new_conv_in.bias.copy_(conv_in.bias)

        self.model.x_embedder.proj = new_conv_in

    def vae_encode(self, images, sample_posterior, device):
        name = self.config.vae.vae_type
        if name == "sdxl" or name == "sd3":
            posterior = self.vae.encode(images.to(device)).latent_dist
            if sample_posterior:
                z = posterior.sample()
            else:
                z = posterior.mode()
            z = (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        elif "dc-ae" in name:
            ae = self.vae
            scaling_factor = ae.cfg.scaling_factor if ae.cfg.scaling_factor else 0.41407
            z = ae.encode(images.to(device))
            z = z * scaling_factor
        elif "AutoencoderDC" in name:
            ae = self.vae
            scaling_factor = ae.config.scaling_factor if ae.config.scaling_factor else 0.41407
            z = ae.encode(images.to(device))[0]
            z = z * scaling_factor
        else:
            print("error load vae")
            exit()
        return z

    def vae_decode(self, latent):
        name = self.config.vae.vae_type
        if name == "sdxl" or name == "sd3":
            latent = (latent.detach() / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            samples = self.vae.decode(latent).sample
        elif "dc-ae" in name:
            ae = self.vae
            vae_scale_factor = (
                2 ** (len(ae.config.encoder_block_out_channels) - 1)
                if hasattr(ae, "config") and ae.config is not None
                else 32
            )
            scaling_factor = ae.cfg.scaling_factor if ae.cfg.scaling_factor else 0.41407
            if latent.shape[-1] * vae_scale_factor > 4000 or latent.shape[-2] * vae_scale_factor > 4000:
                from patch_conv import convert_model

                ae = convert_model(ae, splits=4)
            samples = ae.decode(latent.detach() / scaling_factor)
        elif "AutoencoderDC" in name:
            ae = self.vae
            scaling_factor = ae.config.scaling_factor if ae.config.scaling_factor else 0.41407
            try:
                samples = ae.decode(latent / scaling_factor, return_dict=False)[0]
            except torch.cuda.OutOfMemoryError as e:
                print("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
                ae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024)
                samples = ae.decode(latent / scaling_factor, return_dict=False)[0]
        else:
            print("error load vae")
            exit()
        return samples

    @torch.inference_mode()
    def denoise(
            self,
            prompt=None,
            height=1024,
            width=1024,
            negative_prompt="",
            num_inference_steps=20,
            guidance_scale=4.5,
            pag_guidance_scale=1.0,
            num_images_per_prompt=1,
            generator=torch.Generator().manual_seed(42),
            latents=None,
            use_resolution_binning=True,
            caption_embs=None,
            emb_masks=None,
            null_embeds=None,
            latents_condition=None,
    ):
        if use_resolution_binning:
            height, width = classify_height_width_bin(height, width, ratios=self.base_ratios)

        latent_size_h, latent_size_w = (
            height // self.config.vae.vae_downsample_rate,
            width // self.config.vae.vae_downsample_rate,
        )

        guidance_type = guidance_type_select(
            'classifier-free', pag_guidance_scale, self.config.model.attn_type
        )

        if prompt is None:
            prompt = [""]
        prompts = prompt if isinstance(prompt, list) else [prompt]
        latents_denoised = []

        for prompt in prompts:
            # data prepare
            prompts, hw, ar = (
                [],
                torch.tensor([[self.image_size, self.image_size]], dtype=torch.float,
                             device=caption_embs.device).repeat(
                    num_images_per_prompt, 1
                ),
                torch.tensor([[1.0]], device=caption_embs.device).repeat(num_images_per_prompt, 1),
            )

            for _ in range(num_images_per_prompt):
                prompts.append(
                    prepare_prompt_ar(prompt, self.base_ratios, device=caption_embs.device, show=False)[0].strip())

            null_y = null_embeds.repeat(len(prompts), 1, 1)[:, None].to(self.weight_dtype)

            n = len(prompts)
            if latents is None:
                z = torch.randn(
                    n,
                    self.config.vae.vae_latent_dim,
                    latent_size_h,
                    latent_size_w,
                    generator=generator,
                    device=caption_embs.device,
                )
            else:
                z = latents.to(caption_embs.device)

            model_kwargs = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, mask=emb_masks)
            if self.vis_sampler == "flow_euler":
                flow_solver = FlowEuler(
                    self.model,
                    condition=caption_embs,
                    uncondition=null_y,
                    cfg_scale=guidance_scale,
                    model_kwargs=model_kwargs,
                )
                _latents_denoised = flow_solver.sample(
                    z,
                    steps=num_inference_steps,
                )
            elif self.vis_sampler == "flow_dpm-solver":
                scheduler = DPMS(
                    self.model,
                    condition=caption_embs,
                    uncondition=null_y,
                    guidance_type=guidance_type,
                    # cfg_scale=guidance_scale,
                    cfg_scale=7.5,
                    # cfg_scale=6.0,
                    # cfg_scale=4.5,
                    # cfg_scale=3.0,
                    pag_scale=pag_guidance_scale,
                    pag_applied_layers=self.config.model.pag_applied_layers,
                    model_type="flow",
                    model_kwargs=model_kwargs,
                    schedule="FLOW",
                    with_latents_condition=self.model_args.with_latents_condition,
                    latents_condition=latents_condition
                )
                scheduler.register_progress_bar(self.progress_fn)
                _latents_denoised = scheduler.sample(
                    z,
                    # steps=num_inference_steps,
                    steps=30,
                    order=2,
                    skip_type="time_uniform_flow",
                    method="multistep",
                    flow_shift=self.flow_shift,
                )
            else:
                raise NotImplementedError(f'vis_sampler "{self.vis_sampler}" not implemented')

            latents_denoised.append(_latents_denoised)

        return latents_denoised

    @torch.inference_mode()
    def latents2image(
            self,
            latents,
            height=1024,
            width=1024,
            use_resolution_binning=True
    ):
        latents = latents.to(self.vae_dtype)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.vae_dtype):
                sample = self.vae_decode(latents)

        if use_resolution_binning:
            sample = resize_and_crop_tensor(sample, height, width)

        return sample

    def from_pretrained(self, pretrained_model_name_or_path):
        state_dict = load_file(os.path.join(pretrained_model_name_or_path, "model.safetensors"))
        # remove "decoder." prefix (if any)
        state_dict = {k[8:]: v for k, v in state_dict.items() if k.startswith("decoder.")}
        # remove text encoder (if any)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("text_encoder.")}
        # load weights
        info = self.load_state_dict(state_dict)
        print(f'Load decoder pretrained weights from {pretrained_model_name_or_path}\n\n{info}\n\n')


@dataclass
class DIMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None


class DIM(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.model_args = model_args

        if model_args.condition_type == 'LMToken':
            # <-----mllm embedding-based model----->
            print(f'Loading MLLM pretrained weights from {model_args.pretrained_model_name_or_path}')

            self.mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.pretrained_model_name_or_path,
                torch_dtype="auto",
                attn_implementation='flash_attention_2',
            )
            self.projector = build_projector(
                proj_in=2048,
                proj_out=2304,
                projector_type='mlp2x_gelu'
            )

        else:
            raise NotImplementedError(f'encoder_type {model_args.encoder_type} not supported')

        self.decoder = DIMDecoder(model_args)

        # setup at inference time
        self.designer = None
        self.designer_name = None
        self.designer_processor = None

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            second_per_grid_ts: Optional[torch.Tensor] = None,
            raw_images: Optional[List[torch.FloatTensor]] = None,
            raw_captions: Optional[List[str]] = None,
            user_labels: Optional[torch.LongTensor] = None,
            target_images: Optional[List[torch.FloatTensor]] = None,
    ) -> Union[Tuple, DIMOutput]:
        outputs = self.mllm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
        )

        if self.model_args.condition_type == 'LMToken':
            # <-----mllm embedding-based model----->
            mm_cond_in = outputs.hidden_states[-1]
            # [B x L x D]
            mm_cond_out = self.projector(mm_cond_in)

            assert len(user_labels) == 1, 'currently only support batch size 1'

            mask = user_labels[0] != -100  # shape [1 x L]
            mm_cond_out = mm_cond_out[0, mask[0]]  # shape [L x D]
            mm_cond_out = mm_cond_out.unsqueeze(0)  # shape [1, L, D]

            if self.model_args.text_only_condition:
                labels_valid = user_labels[0][0, mask[0]]  # shape [L]

                # HARD CODE: Qwen2.5-VL image_start/image_end
                image_start_id, image_end_id = 151652, 151653

                image_start_idxs = (labels_valid == image_start_id).nonzero(as_tuple=True)[0]
                image_end_idxs = (labels_valid == image_end_id).nonzero(as_tuple=True)[0]

                if image_start_idxs.numel() != image_end_idxs.numel():
                    raise RuntimeError(
                        'start_idxs and end_idxs should have the same length, '
                        'got {} and {}'.format(image_start_idxs.numel(), image_end_idxs.numel())
                    )

                # text-only
                if image_start_idxs.numel() == 0:
                    text_cond, image_cond = mm_cond_out, None
                else:
                    s_idx = image_start_idxs[-1].item()
                    e_idx = image_end_idxs[-1].item()
                    # image-only
                    if e_idx - s_idx + 1 == mm_cond_out.shape[1]:
                        text_cond, image_cond = None, mm_cond_out
                    # image-text
                    else:
                        image_cond = mm_cond_out[0][s_idx: e_idx + 1].unsqueeze(0)
                        text_cond = mm_cond_out[0][e_idx + 1:].unsqueeze(0)

                assert text_cond is not None, 'text_only_condition is set to True, but no text input detected!'

                mm_cond_out = text_cond

        else:
            raise NotImplementedError(f'condition_type {self.model_args.condition_type} not supported')

        loss = self.decoder(
            mm_cond=mm_cond_out,
            raw_images=raw_images,
            target_images=target_images,
            raw_captions=raw_captions,
        )

        return DIMOutput(
            loss=loss,
        )

    def from_pretrained(self, pretrained_model_name_or_path):
        state_dict = load_file(os.path.join(pretrained_model_name_or_path, "model.safetensors"))
        # remove text encoder (if any)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("decoder.text_encoder.")}

        # load weights, need to check carefully
        info = self.load_state_dict(state_dict, strict=False)
        print(f'Load pretrained weights from {pretrained_model_name_or_path}\n\n{info}\n\n')

        assert not info.unexpected_keys, 'Only missing keys are allowed, got unexpected keys'

    @torch.inference_mode()
    def generate(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            second_per_grid_ts: Optional[torch.Tensor] = None,
            raw_images: Optional[List[torch.FloatTensor]] = None,
            raw_captions: Optional[List[str]] = None,
            user_labels: Optional[torch.LongTensor] = None,
            target_images: Optional[List[torch.FloatTensor]] = None,
            task_type: Optional[str] = 'T2I',
    ) -> torch.FloatTensor:
        outputs = self.mllm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
        )

        if self.model_args.condition_type == 'LMToken':
            # <-----mllm embedding-based model----->
            mm_cond_in = outputs.hidden_states[-1]
            # [B x L x D]
            mm_cond_out = self.projector(mm_cond_in)

            assert len(user_labels) == 1, 'currently only support batch size 1'
            mask = user_labels[0] != -100  # shape [1 x L]
            mm_cond_out = mm_cond_out[0, mask[0]]  # shape [L x D]
            mm_cond_out = mm_cond_out.unsqueeze(0)  # shape [1, L, D]

        else:
            raise NotImplementedError(f'condition_type {self.model_args.condition_type} not supported')

        images = self.decoder.generate(
            mm_cond=mm_cond_out,
            raw_images=raw_images,
            raw_captions=raw_captions,
            task_type=task_type,
        )

        return images

    def get_cot_from_designer(self, image_path, instruction):
        if self.designer_name == 'gpt-4o-2024-11-20':
            return self.get_cot_from_designer_gpt(image_path, instruction)
        elif self.designer_name in [
            'Qwen/Qwen2.5-VL-3B-Instruct',
            'Qwen/Qwen2.5-VL-7B-Instruct',
            'Qwen/Qwen2.5-VL-32B-Instruct',
            'Qwen/Qwen2.5-VL-72B-Instruct'
        ]:
            return self.get_cot_from_designer_qwen(image_path, instruction)
        else:
            raise NotImplementedError(f'designer_name {self.designer_name} not supported')

    def set_designer_gpt(self, api_key, version='gpt-4o-2024-11-20'):
        import openai
        self.designer = openai.AzureOpenAI(
            azure_endpoint="",
            api_version="2024-11-20",
            api_key=api_key
        )
        self.designer_name = version
        self.designer_processor = None

        print(f'Designer {self.designer_name} loaded')

    def get_cot_from_designer_gpt(self, image_path, instruction):
        image_obj = Image.open(image_path)

        prompt = (
            "You will receive two items: a source image, and an edit instruction. "
            "Your task is to outline your reasoning in four steps:\n"
            "1. Edit Instruction: [The edit instruction]\n"
            "2. Global Layout Perception: Identify and describe all key objects and their positions in the source image.\n"
            "3. Local Object Perception: Detail the appearance (shape, color, texture, state) of each object or background element in the source.\n"
            "4. Edit Area Localization: Specify which objects or regions will change, based on your refined instruction.\n"
            "5. Edited Image Imagination: Describe how the edited image will look, emphasizing the modified areas.\n\n"
            "Respond using exactly the format above, with no extra commentary.\n\n"
            f"Edit instruction: {instruction}"
        )

        content = create_content_gpt([image_obj, prompt])

        max_retries = 20
        for attempt in range(max_retries):
            try:
                resp = self.designer.chat.completions.create(
                    model=self.designer_name,
                    messages=[{"role": "user", "content": content}],
                    seed=233,
                    temperature=0,
                    timeout=60
                ).choices[0].message.content

                return resp.strip()
            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(3)

        raise RuntimeError("Failed to get completion after max retries")

    def set_designer_qwen(self, version='Qwen/Qwen2.5-VL-7B-Instruct'):
        # default: Load the model on the available device(s)
        self.designer = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            version,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.designer_name = version
        self.designer_processor = AutoProcessor.from_pretrained(version)

        print(f'Designer {self.designer_name} loaded')

    def get_cot_from_designer_qwen(self, image_path, instruction):
        prompt = (
            "You will receive two items: a source image, and an edit instruction. "
            "Your task is to outline your reasoning in four steps:\n"
            "1. Edit Instruction: [The edit instruction]\n"
            "2. Global Layout Perception: Identify and describe all key objects and their positions in the source image.\n"
            "3. Local Object Perception: Detail the appearance (shape, color, texture, state) of each object or background element in the source.\n"
            "4. Edit Area Localization: Specify which objects or regions will change, based on your refined instruction.\n"
            "5. Edited Image Imagination: Describe how the edited image will look, emphasizing the modified areas.\n\n"
            "Respond using exactly the format above, with no extra commentary.\n\n"
            f"Edit instruction: {instruction}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = self.designer_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.designer_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.designer.generate(**inputs, max_new_tokens=8192)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.designer_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0].strip()
