import os
from dataclasses import asdict

import torch.cuda
from transformers import AutoProcessor, HfArgumentParser, TrainingArguments, logging
from torchvision.transforms import ToPILImage
from PIL import Image, ImageDraw

from models import ModelArguments
from models.modeling_dim import DIM
from utils.vision_process import process_vision_info
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

from data import DataArguments
from data.tos_dataset import TosDatasetBase, TosDatasetEdit
from tqdm import tqdm
import json

logger = logging.get_logger(__name__)

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    model_name = model_args.model_name_or_path.split('/')[2]

    model = DIM(model_args)

    model.decoder.expand_context(max_condition_length=model_args.max_condition_length)

    if model_args.with_latents_condition:
        model.decoder.expand_channels()

    model.from_pretrained(model_args.model_name_or_path)

    print(model)

    model.bfloat16().eval()
    model.cuda()

    processor = AutoProcessor.from_pretrained(model_args.pretrained_model_name_or_path, padding_side='right')

    # <----------gen CoT data begin---------->
    # GPT-4o as external designer
    # model.set_designer_gpt(api_key='')

    # Qwen2.5-VL as external designer
    # model.set_designer_qwen(version='Qwen/Qwen2.5-VL-3B-Instruct')
    # model.set_designer_qwen(version='Qwen/Qwen2.5-VL-7B-Instruct')

    # InternVL3.5 as external designer
    model.set_designer_internvl(version='OpenGVLab/InternVL3_5-8B-HF')

    # MiMo-VL as external designer
    # model.set_designer_mimo(version='XiaomiMimo/MiMo-VL-7B-RL-2508')

    # GLM-4.1V as external designer
    # model.set_designer_glm(version='THUDM/GLM-4.1V-9B-Thinking')

    with open(data_args.dataset_path, 'r') as f:
        records = [json.loads(line) for line in f]

    records_cot = []
    for record in tqdm(records, 'Generating CoT'):
        cot = model.get_cot_from_designer(
            image_path=record['image_path'],
            instruction=record['prompt']
        )

        record_cot = record.copy()
        record_cot['prompt'] = cot
        records_cot.append(record_cot)

    if not os.path.exists('./cache/inference/demo/DIM-4.6B-Edit'):
        os.makedirs('./cache/inference/demo/DIM-4.6B-Edit')

    with open('./cache/inference/demo/DIM-4.6B-Edit/tos_dataset_edit_cot_demo_gen.jsonl', 'w') as f:
        for record_cot in records_cot:
            f.write(json.dumps(record_cot) + '\n')

    data_args.dataset_path = './cache/inference/demo/DIM-4.6B-Edit/tos_dataset_edit_cot_demo_gen.jsonl'
    # <----------gen CoT data end---------->

    if data_args.dataset_type == 'TosDatasetBase':
        eval_dataset = TosDatasetBase(**asdict(data_args), processor=processor, output_meta_data=True)
    elif data_args.dataset_type == 'TosDatasetEdit':
        eval_dataset = TosDatasetEdit(**asdict(data_args), processor=processor, output_meta_data=True)
    else:
        raise ValueError(f'Unsupported dataset type: {data_args.dataset_type}')

    for idx, inputs in tqdm(enumerate(eval_dataset), desc='Demo-Edit'):
        meta_data = inputs['meta_data']
        inputs.pop('meta_data')

        save_dir = f'./cache/inference/demo/{model_name}'
        save_name = meta_data['id'] + '_edited.jpg'

        os.makedirs(save_dir, exist_ok=True)

        raw_image = inputs['raw_images'][0].unsqueeze(0)
        if inputs.get('target_images', None) is not None:
            target_image = inputs['target_images'][0].unsqueeze(0)
        else:
            target_image = None

        prompt = inputs['raw_captions'][0]

        print(f'Index {idx} Prompt:\n\n{prompt}\n\n')

        for k, v in inputs.items():
            if isinstance(v, torch.FloatTensor):
                inputs[k] = v.cuda()
            elif isinstance(v, torch.LongTensor):
                inputs[k] = v.cuda()
            else:
                inputs[k] = v

        # 1 x 3 x H x W
        gen_image = model.generate(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            task_type=data_args.task_type,
        )

        # save the edited image
        save_image(
            gen_image, os.path.join(save_dir, save_name),
            nrow=1, normalize=True, value_range=(-1, 1)
        )

        # save the source image
        save_name_source = meta_data['id'] + '_source.jpg'
        save_image(
            raw_image, os.path.join(save_dir, save_name_source),
            nrow=1, normalize=True, value_range=(-1, 1)
        )

        # save the target image
        if target_image is not None:
            save_name_target = meta_data['id'] + '_target.jpg'
            save_image(
                target_image, os.path.join(save_dir, save_name_target),
                nrow=1, normalize=True, value_range=(-1, 1)
            )
