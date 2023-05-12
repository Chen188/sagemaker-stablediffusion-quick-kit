# -*- coding: utf-8 -*-
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import json
import uuid
import io
import sys
import traceback
import base64
import random
import copy
from collections import defaultdict

from PIL import Image

import boto3
import sagemaker
import torch

from lpw import get_weighted_text_embeddings

from diffusers import (
    StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    AltDiffusionPipeline, AltDiffusionImg2ImgPipeline,
    ControlNetModel,
    EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler, LMSDiscreteScheduler, KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler, DDIMScheduler, PNDMScheduler, UniPCMultistepScheduler,
    DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler
)

from diffusers.utils import load_image
from safetensors.torch import load_file, save_file

import cv2
import numpy as np
import s3fs

# load utils and controle_net
from utils import quick_download_s3, get_bucket_and_key, untar

s3_client = boto3.client('s3')
fs = s3fs.S3FileSystem()

max_height = os.environ.get("max_height", 768)
max_width = os.environ.get("max_width", 768)
max_steps = os.environ.get("max_steps", 100)
max_count = os.environ.get("max_count", 4)
s3_bucket = os.environ.get("s3_bucket", "")
custom_region = os.environ.get("custom_region", None)
safety_checker_enable = json.loads(os.environ.get("safety_checker_enable", "false"))
control_net_enable = json.loads(os.environ.get("control_net_enable", "true"))
deepspeed_enable = json.loads(os.environ.get("deepspeed", 'false'))
base_models = json.loads(os.environ.get("base_models", '{}')) # "{'model_1':'s3://bkt/uncomporessed/', 'model_2':'hf/lora_model_id'}"
lora_models = json.loads(os.environ.get("lora_models", '{}')) # "{'model_1':'s3://bkt/obj.safetensors', 'model_2':'hf/lora_model_id'}"
save_image_to_local = json.loads(os.environ.get("save_image_to_local", 'false'))

DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"

if lora_models:
    if not isinstance(lora_models, dict):
        example_param = {'model_1':'s3://bkt/obj.safetensors', 'model_2':'hf/lora_model_id'}
        raise Exception(f"lora_models param error, {lora_models} is not supported. use format: {example_param}")

    for (model_alias, model_path) in lora_models.items():
        if model_path.endswith('.safetensors'):
            if model_path.startswith("s3://"):
                _dir_path = str(uuid.uuid4())
                local_lora_path = f"/tmp/{_dir_path}/"
                # print(f"LoRA {model_alias}: mkdir {local_lora_path}")
                os.makedirs(local_lora_path)
                fs.get(model_path, local_lora_path)
                local_lora_file = local_lora_path + os.path.basename(model_path)
                print(f"LoRA {model_alias}: downloaded to {local_lora_file}")
                lora_models[model_alias] = local_lora_file
                # state_dict = load_file(local_lora_file)
                # pipe = load_lora_weights(pipe, local_lora_file, 0.5, torch.float16)
        # else:
            # lora_models[model_alias]['type'] = 'hf' # hugging face model id
            #use huggingface lora
            # pipe.unet.load_attn_procs(lora_model)

if control_net_enable:
    print("control_net_enable: true")
    from controlnet_aux import OpenposeDetector, MLSDdetector, HEDdetector, HEDdetector
    from control_net import ControlNetDectecProcessor, init_control_net_pipeline, init_control_net_model

    processor = ControlNetDectecProcessor()
    init_control_net_model()

# control_net
control_net_prefix = "lllyasviel/sd-controlnet"
control_net_postfix = [
    "canny",
    "depth",
    "hed",
    "mlsd",
    "openpose",
    "scribble"
]

controle_net_cache = {}

def check_chontrole_net(model_list):
    model_list = model_list.split(",")
    valid_model = []
    for model in model_list:
        if model in control_net_postfix:
            valid_model.append(f"{model}")
    print(f"valid_control_net model: {valid_model} ")
    return valid_model


def canny_image_detector(image):
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    return canny_image


def get_default_bucket():
    try:
        sagemaker_session = sagemaker.Session() if custom_region is None else sagemaker.Session(
            boto3.Session(region_name=custom_region))
        bucket = sagemaker_session.default_bucket()
        return bucket
    except Exception:
        if s3_bucket != "":
            return s3_bucket
        else:
            return None


# need add more sampler
samplers = {
    "euler_a": EulerAncestralDiscreteScheduler,
    "eular": EulerDiscreteScheduler,
    "heun": HeunDiscreteScheduler,
    "lms": LMSDiscreteScheduler,
    "lms_k": LMSDiscreteScheduler,
    "dpm2": KDPM2DiscreteScheduler,   # dpm2
    "dpm2_a": KDPM2AncestralDiscreteScheduler,   # dpm2 a
    "dpmpp_2s_a": DPMSolverSinglestepScheduler,  # dpm++ 2s a
    "dpmpp_2m": DPMSolverMultistepScheduler,   #  dpm++ 2m
    "dpmpp_2m_k": DPMSolverMultistepScheduler, #  dpm ++ 2m karras
    "dpmpp_sde": DPMSolverSDEScheduler,
    "dpmpp_sde_k": DPMSolverSDEScheduler,
    "ddim": DDIMScheduler,  # ddim 
    "plms": PNDMScheduler,  # plms = pndm config.skip_prk_steps, ref: https://github.com/huggingface/diffusers/issues/960
    # "dpm_fast": "sample_dpm_fast",
    # "dpm_adaptive": "sample_dpm_adaptive",
}


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def load_lora_networks(pipeline, lora, device='cuda', dtype=torch.float16):
    for network, weight in lora.items():
        _load_lora_weights(pipeline, network, weight, False, device, dtype)

    return pipeline


def unload_lora_networks(pipeline, lora, device='cuda', dtype=torch.float16):
    for network, weight in lora.items():
        _load_lora_weights(pipeline, network, weight, True, device, dtype)

    return pipeline


def _load_lora_weights(pipeline, lora_network, multiplier, unload, device='cuda', dtype=torch.float16):
    checkpoint_path = lora_models.get(lora_network)

    if not lora_network:
        print(f'lora not downloaded: {lora_network}.')
        return pipeline

    if unload: # just unload weights
        print('unloading lora: ', lora_network, multiplier)
    else:
        print('loading lora: ', lora_network, multiplier)

    if not checkpoint_path.endswith('.safetensors'):
        print(f'not support lora format: {lora_network}. should be .safetensors')
        return pipeline
        # return pipeline.unet.load_attn_procs(checkpoint_path)

    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device=device)

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    # directly update weight in diffusers model
    for layer, elems in updates.items():
        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        elif "unet" in layer:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet
        else:
            raise Exception(f"error: f{lora_network} has invalid layer: {layer}")

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        backup_attr_name = f"origin_attr_{temp_name}"

        if unload:  # just unload weights
            if hasattr(curr_layer, backup_attr_name):
                # print(f'unload lora {lora_network}: layer attr {backup_attr_name}')
                del curr_layer.weight
                curr_layer.weight = getattr(curr_layer, backup_attr_name)
                delattr(curr_layer, backup_attr_name)
            # else:
                # print(f'unload lora {lora_network}: layer attr {temp_name} not found!!!')

            continue

        if not hasattr(curr_layer, backup_attr_name):
            # print(f'load lora {lora_network}: set layer attr {temp_name}')
            curr_layer.__setattr__(backup_attr_name, copy.deepcopy(curr_layer.weight.data))
        # else:
            # print(f'load lora {lora_network}: layer attr {temp_name} already found!!!')

        # get elements for this layer
        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)

        if 'alpha' in elems:
            alpha = elems['alpha'].item() / weight_up.shape[1]
        else:
            alpha = 1.0

        # update weight
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

    return pipeline


model_path = None


def init_pipeline(model_name: str, model_args=None):
    """
    help load model from s3
    """
    global model_path

    print(f"=================init_pipeline:{model_name}=================")

    if control_net_enable:
        model_name = DEFAULT_MODEL if "s3" in model_name else model_name
        controlnet_model = ControlNetModel.from_pretrained(f"{control_net_prefix}-canny", torch_dtype=torch.float16)
        # txt to image, with controlnet
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_name, controlnet=controlnet_model, torch_dtype=torch.float16
        )
        print(f"load {model_name} with controle net")
        return pipe
    else:
        print(f"load {model_name} without controle net")

    model_path = model_name
    base_name = os.path.basename(model_name)

    if model_name.startswith("s3://"):
        if base_name[-7:] == ".tar.gz":
            print('model in single tar file.')
            local_path = "/".join(model_name.split("/")[-2:-1])
            model_path = f"/tmp/{local_path}/"
            print(f"need copy {model_name} to {model_path}")
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            fs.get(model_name, model_path, recursive=True)
            untar(f"/tmp/{local_path}/{base_name}", model_path)
            os.remove(f"/tmp/{local_path}/{base_name}")
            print("download and untar completed")
        elif model_name[-1:] == '/':
            print('model in folder.')
            local_path = "/".join(model_name.split("/")[-2:])
            model_path = f"/tmp/{local_path}"
            print(f"need copy {model_name} to {model_path}")
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            fs.get(model_name, model_path, recursive=True)
            print("download completed")
        else:
            print(f'model file "{model_name}" is not support. if input is folder, pls append "/" at end of model_name.')

    print(f"pretrained model_path: {model_path}")
    if model_args is not None:
        return StableDiffusionPipeline.from_pretrained(
             model_path, torch_dtype=torch.float16, **model_args)
    return StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)


# model_name = os.environ.get("model_name", DEFAULT_MODEL)
# model_args = json.loads(os.environ['model_args']) if (
#         'model_args' in os.environ) else None
# #warm model load 
# init_pipeline(model_name,model_args)


def model_fn(model_name=None):
    """
    Load the model for inference,load model from os.environ['model_name'],diffult use runwayml/stable-diffusion-v1-5

    """
    print("=================model_fn=================")
    if model_name is None:
        model_name = os.environ.get("model_name", DEFAULT_MODEL)

    model_args = json.loads(os.environ.get('model_args', '{}'))

    print(f'model_name: {model_name},  model_args: {model_args}')

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    model = init_pipeline(model_name, model_args)

    if safety_checker_enable is False:
        # model.safety_checker = lambda images, clip_input: (images, False)
        model.safety_checker = None
    if deepspeed_enable:
        try:
            print("begin load deepspeed....")
            model=deepspeed.init_inference(
                model=getattr(model,"model", model),      # Transformers models
                mp_size=1,        # Number of GPU
                dtype=torch.float16, # dtype of the weights (fp16)
                replace_method="auto", # Lets DS autmatically identify the layer to replace
                replace_with_kernel_inject=False, # replace the model with the kernel injector
            )
            print("model accelarate with deepspeed!")
        except Exception as e:
            print("deepspeed accelarate excpetion!")
            print(e)

    model = model.to("cuda")
    model.enable_attention_slicing()
    model.enable_xformers_memory_efficient_attention()
    # model.enable_model_cpu_offload()

    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    print(f"=================input_fn=================\n{request_content_type}\n{request_body}")
    input_data = json.loads(request_body)
    return prepare_opt(input_data)


def clamp_input(input_data, minn, maxn):
    """
    clamp_input check input 
    """
    return max(min(maxn, input_data), minn)


def prepare_opt(input_data):
    """
    Prepare inference input parameter
    """
    opt = {}
    opt["prompt"] = input_data.get(
        "prompt", "a photo of an astronaut riding a horse on mars")
    opt["negative_prompt"] = input_data.get("negative_prompt", "")
    opt["steps"] = clamp_input(input_data.get(
        "steps", 20), minn=20, maxn=max_steps)
    opt["sampler"] = input_data.get("sampler")
    if opt["sampler"] not in samplers:
         raise Exception(f"sampler {opt['sampler']} is not supported. choose from {samplers.keys()}")

    opt["use_karras_sigmas"] = opt["sampler"].endswith('_k')
    opt["sampler"] = samplers[opt["sampler"]]

    opt["height"] = clamp_input(input_data.get(
        "height", 512), minn=64, maxn=max_height)
    opt["width"] = clamp_input(input_data.get(
        "width", 512), minn=64, maxn=max_width)
    opt["count"] = clamp_input(input_data.get(
        "count", 1), minn=1, maxn=max_count)
    opt["lora"] = input_data.get("lora", {})

    opt["seed"] = input_data.get("seed", -1)
    if opt["seed"] == -1:
        opt["seed"] = random.randrange(4294967294)

    # check task type, defaults to txt2img
    opt["task_type"] = input_data.get('task_type', 'txt2img')
    if opt["task_type"] not in TASK_TYPES_ALLOWED:
        raise Exception(f'task type "{opt["task_type"]}" is not supported. Choose from {TASK_TYPES_ALLOWED}')

    init_image = None
    if opt["task_type"] == 'img2img':
        opt["strength"] = clamp_input(input_data.get("strength", 0.8), minn=0, maxn=1)

        init_image = input_data.get('init_image')
        if not (init_image.startswith('http://') or init_image.startswith('https://')):
            print('base image seems not a URL, try base64 decode')
            init_image = base64.b64decode(init_image)
            init_image = Image.open(io.BytesIO(init_image))

        init_image = load_image(init_image)
        init_image.resize((input_data["width"], input_data["height"]))

    opt['init_image'] = init_image  # img2img init image

    opt["control_net_model"] = input_data.get("control_net_model", "")
    opt["control_net_detect"] = input_data.get("control_net_detect", "true")

    if opt["control_net_model"] not in control_net_postfix:
        opt["control_net_model"] = ""

    print(f"=================prepare_opt=================\n{opt}")
    return opt


bucket = get_default_bucket()

if bucket is None:
    raise Exception("Need setup default bucket")

default_output_s3uri = f's3://{bucket}/stablediffusion/asyncinvoke/images/'

TASK_TYPES_ALLOWED = [
    'txt2img',
    'img2img',
    # 'inpaint',
]


def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """
    print("=================predict_fn=================")
    print('input_data: ', input_data)
    prediction = []

    output_s3uri = input_data.get('output_s3uri', default_output_s3uri)

    # load different Pipeline for txt2img , img2img
    # referen doc: https://huggingface.co/docs/diffusers/api/diffusion_pipeline#diffusers.DiffusionPipeline.components
    #   text2img = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    #   img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
    #   inpaint = StableDiffusionInpaintPipeline(**text2img.components)

    if input_data['task_type'] == 'img2img':
        model = StableDiffusionImg2ImgPipeline(**model.components)

    if lora_models and input_data.get('lora'):
        load_lora_networks(model, input_data.get('lora'))

    generator = torch.Generator(device='cuda').manual_seed(input_data["seed"])

    with torch.inference_mode():
        prompt_embeds, negative_prompt_embeds = get_weighted_text_embeddings(model, input_data["prompt"], input_data["negative_prompt"])
        # model.scheduler = input_data["sampler"].from_config(model.scheduler.config, use_karras_sigmas=input_data["use_karras_sigmas"])
        model.scheduler = input_data["sampler"].from_pretrained(model_path, 'scheduler', use_karras_sigmas=input_data["use_karras_sigmas"])

        if isinstance(input_data["sampler"], PNDMScheduler):
            model.scheduler.skip_prk_steps = True

        if input_data['task_type'] == 'txt2img':
            images = model(height=input_data["height"], width=input_data["width"],
                           num_inference_steps=input_data["steps"], num_images_per_prompt=input_data["count"], 
                           prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                           generator=generator).images
        elif input_data['task_type'] == 'img2img':
            images = model(image=input_data['init_image'], strength=input_data["strength"], 
                           num_inference_steps=input_data["steps"], num_images_per_prompt=input_data["count"], 
                           prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                           generator=generator).images

        for image in images:
            bucket, key = get_bucket_and_key(output_s3uri)
            key = f'{key}{uuid.uuid4()}.jpg'
            buf = io.BytesIO()
            image.save(buf, format='JPEG')

            if save_image_to_local:
                local_name = f"/tmp/assets/model-output/{uuid.uuid4()}.jpg"
                with open(f"{local_name}", "wb") as f:
                    print(f"saved image to local: {local_name}")
                    f.write(buf.getbuffer())
            else:
                s3_client.put_object(
                    Body=buf.getvalue(),
                    Bucket=bucket,
                    Key=key,
                    ContentType='image/jpeg',
                    Metadata={
                        #s3 metadata only support ascii
                        "seed": str(input_data["seed"])
                    }
                )
                print('image: ', f's3://{bucket}/{key}')
                prediction.append(f's3://{bucket}/{key}')

    if lora_models and input_data.get('lora'):
        unload_lora_networks(model, input_data.get('lora'))

    print('prediction: ', prediction)
    return prediction


def output_fn(prediction, content_type):
    """
    Serialize and prepare the prediction output
    """
    print(content_type)
    return json.dumps(
        {
            'result': prediction
        }
    )
