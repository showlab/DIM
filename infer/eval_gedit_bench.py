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

    model.bfloat16().eval()

    print(model)

    model.cuda()

    processor = AutoProcessor.from_pretrained(model_args.pretrained_model_name_or_path, padding_side='right')

    if data_args.dataset_type == 'TosDatasetBase':
        eval_dataset = TosDatasetBase(**asdict(data_args), processor=processor, output_meta_data=True)
    elif data_args.dataset_type == 'TosDatasetEdit':
        eval_dataset = TosDatasetEdit(**asdict(data_args), processor=processor, output_meta_data=True)
    else:
        raise ValueError(f'Unsupported dataset type: {data_args.dataset_type}')

    id2key, id2task = {}, {}
    with open('./cache/GEdit-Bench/tos_dataset_edit_en.jsonl', 'r') as f:
        for line in f:
            item = json.loads(line)
            id2key[item['id']] = item['key']
            id2task[item['id']] = item['task_type']

    for idx, inputs in tqdm(enumerate(eval_dataset), desc='GEdit-Bench'):
        meta_data = inputs['meta_data']
        inputs.pop('meta_data')

        id = meta_data['id']
        save_id = id2key[id]
        edit_task = id2task[id]
        save_dir = f'./cache/inference/{model_name}/GEdit-Bench/fullset/{edit_task}/en'
        save_name = f'{save_id}.png'

        if os.path.exists(os.path.join(save_dir, save_name)):
            continue
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        raw_image = inputs['raw_images'][0].unsqueeze(0)

        prompt = inputs['raw_captions'][0]

        print(f'Index {id} Prompt:\n\n{prompt}\n\n')

        # 1 x 3 x H x W
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
