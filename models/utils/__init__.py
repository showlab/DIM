import torch
import base64
from PIL import Image
from io import BytesIO

def collate_dicts(dict_list):
    if not dict_list:
        return {}

    collated = {}
    for key in dict_list[0].keys():
        values = [d[key] for d in dict_list]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = values
    return collated

def create_content_gpt(user_inputs):
    content = []
    for user_input in user_inputs:
        if isinstance(user_input, str):
            content.append({'type': 'text', 'text': user_input})
        elif isinstance(user_input, Image.Image):
            if user_input.mode != 'RGB':
                user_input = user_input.convert('RGB')
            quality = 100
            while quality > 5:
                buffered = BytesIO()
                user_input.save(buffered, format="JPEG", quality=quality)
                if len(buffered.getvalue()) < 18 * 1024 * 1024:
                    break
                quality -= 5
            else:
                raise RuntimeError("Image too large even at low quality")
            content.append({
                'type': 'image_url',
                'image_url': {
                    'url': f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
                }
            })
        else:
            raise NotImplementedError(f"Unsupported user input type: {type(user_input)}")
    return content