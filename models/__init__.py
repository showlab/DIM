from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str = ''
    model_name_or_path: str = ''
    condition_type: str = 'LMToken'
    max_condition_length: int = 16384
    sana_config: str = ''
    sana_pretrained: str = ''
    with_latents_condition: bool = False
    text_only_condition: bool = False
    expand_channel: bool = False
    expand_before_from_pretrained: bool = False
    freeze_modules: list[str] = field(default_factory=lambda: [])
    unfreeze_modules: list[str] = field(default_factory=lambda: [])
