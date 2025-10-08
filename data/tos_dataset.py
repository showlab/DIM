import sys

sys.path.append('..')

# must import mmcv first!!
import mmcv

from utils.vision_process import process_vision_info

import os, random
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import logging, AutoProcessor
from datasets import load_dataset, Features, Value
from io import BytesIO
from PIL import Image
import json
from tqdm import tqdm

try:
    from models.diffusion.data.transforms import get_transform
except:
    # debug on CPU machines
    import torchvision.transforms as T


    def get_transform(type, resolution):
        transform = default_train(resolution)
        transform = T.Compose(transform)
        transform.image_size = resolution
        return transform


    def default_train(n_px):
        transform = [
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize(n_px),  # Image.BICUBIC
            T.CenterCrop(n_px),
            # T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
        return transform

logger = logging.get_logger(__name__)

BASE_TASK_LIST = [
    'T2I-NPAD',  # Text-to-Image Generation without pad raw image
    'T2I-PAD',  # Text-to-Image Generation with pad raw image
    'IR-NPAD',  # Image Reconstruction without pad raw image
    'IR-PAD',  # Image Reconstruction with pad raw image
    'MM-NPAD',  # Mixed condition without pad raw image
    'MM-PAD',  # Mixed condition with pad raw image
]

EDIT_TASK_LIST = [
    'MM-NPAD',  # Mixed condition without pad raw image
    'MM-PAD',  # Mixed condition with pad raw image
]


class TosDatasetBase(Dataset):
    def __init__(
            self,
            *,
            dataset_path: str,
            sample_size: float = -1,
            gen_resolution: int = 512,
            force_gen_resolution: bool = False,
            processor: AutoProcessor,
            task_type: str = 'random',
            **kwargs
    ):
        super().__init__()

        assert os.path.isfile(dataset_path) and dataset_path.endswith('.jsonl'), \
            'The data_path must be a valid jsonl file.'

        try:
            print('Loading dataset with huggingface utils...')

            self.features = Features({
                "id": Value("string"),
                "image_path": Value("string"),
                "prompt": Value("string"),
                "image_path_target": Value("string"),
            })

            self.handles = load_dataset(
                "json",
                data_files=dataset_path,
                split="train",
                features=self.features,
            )

            if sample_size > 0:
                assert 0 < sample_size <= 1, 'sample size must be between 0 and 1 indicating the percentage'
                sample_idx = int(sample_size * len(self.handles))
                self.handles = self.handles.select(range(sample_idx))

            num_chunk = kwargs.get('num_chunk', None)
            chunk_idx = kwargs.get('chunk_idx', None)
            if num_chunk is not None and chunk_idx is not None:
                total = len(self.handles)
                per_chunk = total // num_chunk
                remainder = total % num_chunk

                start = chunk_idx * per_chunk + min(chunk_idx, remainder)
                end = start + per_chunk + (1 if chunk_idx < remainder else 0)

                self.handles = self.handles.select(range(start, end))

                print(f'num_chunk: {num_chunk}, chunk_idx: {chunk_idx}, start: {start}, end: {end}')

                self.chunk_start = start
                self.chunk_end = end

        except:
            print('Loading dataset with huggingface utils failed, switch to vanilla Python loading...')

            with open(dataset_path, 'r') as f:
                self.handles = [json.loads(line) for line in f.readlines()]

            if sample_size > 0:
                assert 0 < sample_size <= 1, 'sample size must be between 0 and 1 indicating the percentage'
                sample_idx = int(sample_size * len(self.handles))
                self.handles = self.handles[:sample_idx]

            num_chunk = kwargs.get('num_chunk', None)
            chunk_idx = kwargs.get('chunk_idx', None)
            if num_chunk is not None and chunk_idx is not None:
                total = len(self.handles)
                per_chunk = total // num_chunk
                remainder = total % num_chunk

                start = chunk_idx * per_chunk + min(chunk_idx, remainder)
                end = start + per_chunk + (1 if chunk_idx < remainder else 0)

                self.handles = self.handles[start:end]

                print(f'num_chunk: {num_chunk}, chunk_idx: {chunk_idx}, start: {start}, end: {end}')

                self.chunk_start = start
                self.chunk_end = end

        print('Number of samples: ', len(self.handles))

        self.output_meta_data = kwargs.get('output_meta_data', False)

        if 'Qwen2VL' in processor.__class__.__name__ or 'Qwen2_5_VL' in processor.__class__.__name__:
            self.im_start_id = processor.tokenizer('<|im_start|>').input_ids[0]
            self.im_end_id = processor.tokenizer('<|im_end|>').input_ids[0]
            self.system_id = processor.tokenizer('system').input_ids[0]
            self.user_id = processor.tokenizer('user').input_ids[0]
            self.assistant_id = processor.tokenizer('assistant').input_ids[0]
            self.newline_id = processor.tokenizer('\n').input_ids[0]
        else:
            raise NotImplementedError(f"Video preprocess not implemented for {processor.__class__.__name__}")
        self.processor = processor

        # gen-related configs
        self.gen_resolution = gen_resolution
        self.force_gen_resolution = force_gen_resolution
        self.task_type = task_type
        # diffusion decoder transform, not related to navit
        self.gen_transform = get_transform(type='default_train', resolution=self.gen_resolution)

        # for debug
        print('Task Type: ', self.task_type)

    def get_generation_conv(self, data, contain_image=False, contain_text=True):
        qwen_conv = []
        content = []

        if contain_image is True:
            img_msg = {'type': 'image', 'image': data['image_path']}

            if self.force_gen_resolution:
                img_msg['resized_height'] = self.gen_resolution
                img_msg['resized_width'] = self.gen_resolution

            content.append(img_msg)

        # process texts
        if contain_text is True:
            final_text = data['prompt']
        else:
            final_text = ''

        content.append({'type': 'text', 'text': final_text})

        # end by text, not image
        qwen_conv.append({"role": "user", "content": content})

        return qwen_conv

    def getitem(self, index):
        data = self.handles[index]

        is_gen_conv = True

        if is_gen_conv:
            if self.task_type == 'random':
                task = random.choice(BASE_TASK_LIST)
            elif self.task_type == 'random-NPAD':
                task = random.choice(BASE_TASK_LIST)
                task = task.replace('-PAD', '-NPAD')
            elif self.task_type == 'random-PAD':
                task = random.choice(BASE_TASK_LIST)
                task = task.replace('-NPAD', '-PAD')
            else:
                task = self.task_type

            if task in ['T2I-NPAD', 'T2I-PAD']:
                qwen_conv = self.get_generation_conv(data, contain_image=False, contain_text=True)
            elif task in ['IR-NPAD', 'IR-PAD']:
                qwen_conv = self.get_generation_conv(data, contain_image=True, contain_text=False)
            elif task in ['MM-NPAD', 'MM-PAD']:
                qwen_conv = self.get_generation_conv(data, contain_image=True, contain_text=True)
            else:
                raise ValueError(f'unsupported task: {task}, must be one of {BASE_TASK_LIST}')
        else:
            raise NotImplementedError('conv type not supported yet')

        image_inputs, video_inputs = process_vision_info(qwen_conv)
        texts = self.processor.apply_chat_template(
            qwen_conv, tokenize=False, add_generation_prompt=False, return_tensors='pt'
        )

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )

        # input_ids: [1 x L]
        input_ids = inputs.input_ids
        _, L = input_ids.shape

        # create user labels
        user_labels = torch.full_like(input_ids, -100, dtype=torch.long)
        user_begin, user_end = [], []
        for idx in range(3, L):
            # <|im_start|>user\n[CONTENT]<|im_end|>\n
            if (
                    input_ids[0, idx - 3] == self.im_start_id and
                    input_ids[0, idx - 2] == self.user_id and
                    input_ids[0, idx - 1] == self.newline_id
            ):
                user_begin.append(idx)
            if input_ids[0, idx] == self.im_end_id and len(user_begin) == len(user_end) + 1:
                user_end.append(idx)

        assert len(user_begin) == len(user_end), 'user_begin and user_end must have the same length'
        for i in range(len(user_begin)):
            user_labels[0, user_begin[i]:user_end[i]] = input_ids[0, user_begin[i]:user_end[i]]

        inputs['user_labels'] = [user_labels]

        # get raw images
        raw_images = []
        raw_img_path = data['image_path']

        if not os.path.exists(raw_img_path):
            raise FileNotFoundError(raw_img_path)
        else:
            raw_img_obj = Image.open(raw_img_path)

        raw_images.append(self.gen_transform(raw_img_obj))
        inputs['raw_images'] = raw_images

        # get raw captions
        raw_captions = []
        for conv in qwen_conv:
            caption_role = 'user' if is_gen_conv else 'assistant'
            if conv['role'] == caption_role:
                raw_captions.append(conv['content'][-1]['text'])
        inputs['raw_captions'] = raw_captions

        if task in ['T2I-NPAD', 'IR-NPAD', 'MM-NPAD']:
            inputs['target_images'] = None
        elif task in ['T2I-PAD', 'IR-PAD', 'MM-PAD']:
            inputs['target_images'] = raw_images
        else:
            raise ValueError(f'unsupported task: {task}, must be one of {BASE_TASK_LIST}')

        if self.output_meta_data is True:
            inputs['meta_data'] = data

        return inputs

    def __getitem__(self, index):
        # avoid IndexError be caught by other cases
        if index < 0 or index >= len(self.handles):
            raise IndexError

        max_tries = 10
        for _ in range(max_tries):
            try:
                return self.getitem(index)
            except Exception as e:
                logger.warning(f"Failed {_}-th try to get item {index}: {e}")
                index = random.randint(0, self.__len__() - 1)
                logger.warning(f"Retrying to get item {index}")
        raise Exception(f"Failed to get item after {max_tries} retries")

    def data_collator(self, batched_inputs, **kwargs):
        assert len(batched_inputs) == 1, f'currently only support batch size 1, got {len(batched_inputs)}'
        return batched_inputs[0]

    def __len__(self):
        return len(self.handles)


class TosDatasetEdit(TosDatasetBase):
    def getitem(self, index):
        data = self.handles[index]

        is_gen_conv = True

        if is_gen_conv:
            if self.task_type == 'random':
                task = random.choice(BASE_TASK_LIST)
            elif self.task_type == 'random-NPAD':
                task = random.choice(BASE_TASK_LIST)
                task = task.replace('-PAD', '-NPAD')
            elif self.task_type == 'random-PAD':
                task = random.choice(BASE_TASK_LIST)
                task = task.replace('-NPAD', '-PAD')
            else:
                task = self.task_type

            if task in ['T2I-NPAD', 'T2I-PAD']:
                qwen_conv = self.get_generation_conv(data, contain_image=False, contain_text=True)
            elif task in ['IR-NPAD', 'IR-PAD']:
                qwen_conv = self.get_generation_conv(data, contain_image=True, contain_text=False)
            elif task in ['MM-NPAD', 'MM-PAD']:
                qwen_conv = self.get_generation_conv(data, contain_image=True, contain_text=True)
            else:
                raise ValueError(f'unsupported task: {task}, must be one of {EDIT_TASK_LIST}')
        else:
            raise NotImplementedError('conv type not supported yet')

        image_inputs, video_inputs = process_vision_info(qwen_conv)
        texts = self.processor.apply_chat_template(
            qwen_conv, tokenize=False, add_generation_prompt=False, return_tensors='pt'
        )

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )

        # input_ids: [1 x L]
        input_ids = inputs.input_ids
        _, L = input_ids.shape

        # create user labels
        user_labels = torch.full_like(input_ids, -100, dtype=torch.long)
        user_begin, user_end = [], []
        for idx in range(3, L):
            # <|im_start|>user\n[CONTENT]<|im_end|>\n
            if (
                    input_ids[0, idx - 3] == self.im_start_id and
                    input_ids[0, idx - 2] == self.user_id and
                    input_ids[0, idx - 1] == self.newline_id
            ):
                user_begin.append(idx)
            if input_ids[0, idx] == self.im_end_id and len(user_begin) == len(user_end) + 1:
                user_end.append(idx)

        assert len(user_begin) == len(user_end), 'user_begin and user_end must have the same length'
        for i in range(len(user_begin)):
            user_labels[0, user_begin[i]:user_end[i]] = input_ids[0, user_begin[i]:user_end[i]]

        inputs['user_labels'] = [user_labels]

        # get raw images
        raw_images = []
        raw_img_path = data['image_path']

        if not os.path.exists(raw_img_path):
            raise FileNotFoundError(raw_img_path)
        else:
            raw_img_obj = Image.open(raw_img_path)

        raw_images.append(self.gen_transform(raw_img_obj))
        inputs['raw_images'] = raw_images

        # get raw captions
        raw_captions = []
        for conv in qwen_conv:
            caption_role = 'user' if is_gen_conv else 'assistant'
            if conv['role'] == caption_role:
                raw_captions.append(conv['content'][-1]['text'])
        inputs['raw_captions'] = raw_captions

        # get target images
        target_images = []
        target_img_path = data['image_path_target']

        if not os.path.exists(target_img_path):
            raise FileNotFoundError(target_img_path)
        else:
            target_img_obj = Image.open(target_img_path)

        target_images.append(self.gen_transform(target_img_obj))

        if task in ['T2I-NPAD', 'IR-NPAD', 'MM-NPAD']:
            inputs['target_images'] = None
        elif task in ['T2I-PAD', 'IR-PAD', 'MM-PAD']:
            inputs['target_images'] = target_images
        else:
            raise ValueError(f'unsupported task: {task}, must be one of {EDIT_TASK_LIST}')

        if self.output_meta_data is True:
            inputs['meta_data'] = data

        return inputs


if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained(
        './cache/Qwen2.5-VL-3B-Instruct', padding_side='right')

    dataset = TosDatasetEdit(
        dataset_path='./cache/tos_dataset_edit_cot_demo.jsonl',
        sample_size=1.0,
        gen_resolution=1024,
        force_gen_resolution=False,
        processor=processor,
        task_type='MM-PAD',
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=64, collate_fn=dataset.data_collator)

    for idx, batch in tqdm(enumerate(dataloader)):
        print(idx, batch['raw_captions'][0])
        if idx >= 5:
            break

    print('Finished!')
