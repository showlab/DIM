import json
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

    if data_args.dataset_type == 'TosDatasetBase':
        eval_dataset = TosDatasetBase(**asdict(data_args), processor=processor, output_meta_data=True)
    elif data_args.dataset_type == 'TosDatasetEdit':
        eval_dataset = TosDatasetEdit(**asdict(data_args), processor=processor, output_meta_data=True)
    else:
        raise ValueError(f'Unsupported dataset type: {data_args.dataset_type}')

    base_idx = eval_dataset.chunk_start

    dset_name = data_args.dataset_path.split('/')[-1].replace('.jsonl', '')
    if dset_name.endswith('rewritten'):
        suffix = 'GenEval_rewritten'
    else:
        suffix = 'GenEval'

    for idx, inputs in tqdm(enumerate(eval_dataset), desc='GenEval'):
        meta_data = inputs['meta_data']
        inputs.pop('meta_data')

        save_dir = os.path.join(
            f'./cache/inference/{model_name}/{suffix}',
            f"{idx + base_idx:05d}"
        )

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "samples/"), exist_ok=True)

        meta_data_save_path = os.path.join(save_dir, 'metadata.jsonl')
        with open(meta_data_save_path, 'w') as f:
            if isinstance(meta_data['meta_data'], str):
                f.write(json.dumps(eval(meta_data['meta_data'])) + '\n')
            else:
                f.write(json.dumps(meta_data['meta_data']) + '\n')

        raw_image = inputs['raw_images'][0].unsqueeze(0)

        prompt = inputs['raw_captions'][0]

        for k, v in inputs.items():
            if isinstance(v, torch.FloatTensor):
                inputs[k] = v.cuda()
            elif isinstance(v, torch.LongTensor):
                inputs[k] = v.cuda()
            else:
                inputs[k] = v

        # 1 x 3 x H x W
        for _ in range(4):
            gen_image = model.generate(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
                task_type=data_args.task_type,
            )

            save_image(
                gen_image, os.path.join(save_dir, f"samples/{_:04d}.png"),
                nrow=1, normalize=True, value_range=(-1, 1)
            )
