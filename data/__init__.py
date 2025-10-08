from dataclasses import dataclass, field


@dataclass
class DataArguments:
    dataset_type: str = 'TosDatasetBase'
    dataset_path: str = ''
    sample_size: float = -1
    gen_resolution: int = 512
    force_gen_resolution: bool = False
    task_type: str = 'random'
    num_chunk: int = None
    chunk_idx: int = None
