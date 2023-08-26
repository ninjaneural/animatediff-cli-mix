import os
import glob
import logging
import re
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from controlnet_aux import HEDdetector, LineartAnimeDetector, OpenposeDetector
from diffusers import AutoencoderKL, ControlNetModel, DiffusionPipeline, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionPipeline
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from animatediff import get_dir
from animatediff.models.clip import CLIPSkipTextModel
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines import AnimationPipeline, load_text_embeddings
from animatediff.schedulers import get_scheduler
from animatediff.settings import InferenceConfig, ModelConfig
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from animatediff.utils.model import ensure_motion_modules, get_checkpoint_weights
from animatediff.utils.util import get_resized_image, get_resized_images, save_video

logger = logging.getLogger(__name__)

data_dir = get_dir("data")
default_base_path = data_dir.joinpath("models/huggingface/stable-diffusion-v1-5")

re_clean_prompt = re.compile(r"[^\w\-, ]")

lineart_anime_processor = None
openpose_processor = None
softedge_processor = None


def create_controlnet_model(type_str):
    if type_str == "controlnet_tile":
        return ControlNetModel.from_pretrained("lllyasviel/control_v11f1e_sd15_tile")
    elif type_str == "controlnet_lineart_anime":
        return ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15s2_lineart_anime")
    elif type_str == "controlnet_ip2p":
        return ControlNetModel.from_pretrained("lllyasviel/control_v11e_sd15_ip2p")
    elif type_str == "controlnet_openpose":
        return ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose")
    elif type_str == "controlnet_softedge":
        return ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_softedge")
    else:
        raise ValueError(f"unknown controlnet type {type_str}")


def get_preprocessor(type_str):
    global lineart_anime_processor, openpose_processor, softedge_processor
    if type_str == "controlnet_lineart_anime":
        if not lineart_anime_processor:
            lineart_anime_processor = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
        return lineart_anime_processor
    elif type_str == "controlnet_openpose":
        if not openpose_processor:
            openpose_processor = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        return openpose_processor
    elif type_str == "controlnet_softedge":
        if not softedge_processor:
            softedge_processor = HEDdetector.from_pretrained("lllyasviel/Annotators")
        return softedge_processor
    else:
        raise ValueError(f"unknown controlnet type {type_str}")


def get_preprocessed_img(type_str, img):
    if type_str in ("controlnet_tile", "controlnet_ip2p"):
        return img
    elif type_str in ("controlnet_lineart_anime", "controlnet_openpose", "controlnet_softedge"):
        return get_preprocessor(type_str)(img)
    else:
        raise ValueError(f"unknown controlnet type {type_str}")


def create_pipeline(
    base_model: Union[str, PathLike] = default_base_path,
    model_config: ModelConfig = ...,
    infer_config: InferenceConfig = ...,
    use_xformers: bool = True,
) -> AnimationPipeline:
    """Create an AnimationPipeline from a pretrained model.
    Uses the base_model argument to load or download the pretrained reference pipeline model."""

    # make sure motion_module is a Path and exists
    logger.info("Checking motion module...")
    motion_module = data_dir.joinpath(model_config.motion_module)
    if not (motion_module.exists() and motion_module.is_file()):
        # check for safetensors version
        motion_module = motion_module.with_suffix(".safetensors")
        if not (motion_module.exists() and motion_module.is_file()):
            # download from HuggingFace Hub if not found
            ensure_motion_modules()
        if not (motion_module.exists() and motion_module.is_file()):
            # this should never happen, but just in case...
            raise FileNotFoundError(f"Motion module {motion_module} does not exist or is not a file!")

    logger.info("Loading tokenizer...")
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    logger.info("Loading text encoder...")
    text_encoder: CLIPSkipTextModel = CLIPSkipTextModel.from_pretrained(base_model, subfolder="text_encoder")
    logger.info("Loading VAE...")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    logger.info("Loading UNet...")
    unet: UNet3DConditionModel = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path=base_model,
        motion_module_path=motion_module,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(base_model, subfolder="feature_extractor")

    # set up scheduler
    sched_kwargs = infer_config.noise_scheduler_kwargs
    scheduler = get_scheduler(model_config.scheduler, sched_kwargs)
    logger.info(f'Using scheduler "{model_config.scheduler}" ({scheduler.__class__.__name__})')

    # Load the checkpoint weights into the pipeline
    if model_config.path is not None:
        model_path = data_dir.joinpath(model_config.path)
        logger.info(f"Loading weights from {model_path}")
        if model_path.is_file():
            logger.debug("Loading from single checkpoint file")
            unet_state_dict, tenc_state_dict, vae_state_dict = get_checkpoint_weights(model_path)
        elif model_path.is_dir():
            logger.debug("Loading from Diffusers model directory")
            temp_pipeline = StableDiffusionPipeline.from_pretrained(model_path)
            unet_state_dict, tenc_state_dict, vae_state_dict = (
                temp_pipeline.unet.state_dict(),
                temp_pipeline.text_encoder.state_dict(),
                temp_pipeline.vae.state_dict(),
            )
            del temp_pipeline
        else:
            raise FileNotFoundError(f"model_path {model_path} is not a file or directory")

        # Load into the unet, TE, and VAE
        logger.info("Merging weights into UNet...")
        _, unet_unex = unet.load_state_dict(unet_state_dict, strict=False)
        if len(unet_unex) > 0:
            raise ValueError(f"UNet has unexpected keys: {unet_unex}")
        tenc_missing, _ = text_encoder.load_state_dict(tenc_state_dict, strict=False)
        if len(tenc_missing) > 0:
            raise ValueError(f"TextEncoder has missing keys: {tenc_missing}")
        vae_missing, _ = vae.load_state_dict(vae_state_dict, strict=False)
        if len(vae_missing) > 0:
            raise ValueError(f"VAE has missing keys: {vae_missing}")
    else:
        logger.info("Using base model weights (no checkpoint/LoRA)")

    # enable xformers if available
    if use_xformers:
        logger.info("Enabling xformers memory-efficient attention")
        unet.enable_xformers_memory_efficient_attention()

    # I'll deal with LoRA later...

    # controlnet
    controlnet_map = {}
    if model_config.controlnet_map:
        c_image_dir = data_dir.joinpath(model_config.controlnet_map["input_image_dir"])

        for c in model_config.controlnet_map:
            item = model_config.controlnet_map[c]
            if type(item) is dict:
                if item["enable"] == True:
                    img_dir = c_image_dir.joinpath(c)
                    cond_imgs = sorted(glob.glob(os.path.join(img_dir, "[0-9]*.png"), recursive=False))
                    if len(cond_imgs) > 0:
                        controlnet_map[c] = create_controlnet_model(c)

    if not controlnet_map:
        controlnet_map = None

    logger.info("Creating AnimationPipeline...")
    pipeline = AnimationPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
        controlnet_map=controlnet_map,
    )

    # Load TI embeddings
    load_text_embeddings(pipeline)

    return pipeline


def run_inference(
    pipeline: AnimationPipeline,
    prompt: str = ...,
    n_prompt: str = ...,
    seed: int = -1,
    steps: int = 25,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    duration: int = 16,
    idx: int = 0,
    out_dir: PathLike = ...,
    context_frames: int = -1,
    context_stride: int = 3,
    context_overlap: int = 4,
    context_schedule: str = "uniform",
    clip_skip: int = 1,
    return_dict: bool = False,
    controlnet_map: Dict[str, Any] = None,
):
    out_dir = Path(out_dir)  # ensure out_dir is a Path

    controlnet_type_map = {}
    controlnet_image_map = {}
    # { 0 : { "type_str" : IMAGE, "type_str2" : IMAGE }  }

    if controlnet_map:
        c_image_dir = data_dir.joinpath(controlnet_map["input_image_dir"])

        for c in controlnet_map:
            item = controlnet_map[c]
            if type(item) is dict:
                if item["enable"] == True:
                    img_dir = c_image_dir.joinpath(c)
                    cond_imgs = sorted(glob.glob(os.path.join(img_dir, "[0-9]*.png"), recursive=False))
                    if len(cond_imgs) > 0:
                        controlnet_type_map[c] = {
                            "controlnet_conditioning_scale": item["controlnet_conditioning_scale"],
                            "control_guidance_start": item["control_guidance_start"],
                            "control_guidance_end": item["control_guidance_end"],
                            "control_scale_list": item["control_scale_list"],
                        }
                    for img_path in cond_imgs:
                        print(img_path)
                        frame_no = int(Path(img_path).stem)
                        if frame_no not in controlnet_image_map:
                            controlnet_image_map[frame_no] = {}
                        controlnet_image_map[frame_no][c] = get_preprocessed_img(c, get_resized_image(img_path, width, height))

    if not controlnet_type_map:
        controlnet_type_map = None
    if not controlnet_image_map:
        controlnet_image_map = None

    if seed != -1:
        torch.manual_seed(seed)
    else:
        seed = torch.seed()

    pipeline_output = pipeline(
        prompt=prompt,
        negative_prompt=n_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        video_length=duration,
        return_dict=return_dict,
        context_frames=context_frames,
        context_stride=context_stride + 1,
        context_overlap=context_overlap,
        context_schedule=context_schedule,
        clip_skip=clip_skip,
        controlnet_type_map=controlnet_type_map,
        controlnet_image_map=controlnet_image_map,
    )
    logger.info("Generation complete, saving...")

    # Trim and clean up the prompt for filename use
    prompt_tags = [re_clean_prompt.sub("", tag).strip().replace(" ", "-") for tag in prompt.split(",")]
    prompt_str = "_".join((prompt_tags[:6]))

    # generate the output filename and save the video
    out_str = f"{idx:02d}_{seed}_{prompt_str}"[:251]
    out_file = out_dir.joinpath(f"{out_str}.gif")
    if return_dict is True:
        save_video(pipeline_output["videos"], out_file)
    else:
        save_video(pipeline_output, out_file)

    logger.info(f"Saved sample to {out_file}")
    return pipeline_output
