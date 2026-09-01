import os
import torch
from transformers import AutoModel
from huggingface_hub import snapshot_download
import logging

from FlagEmbedding.finetune.embedder.encoder_only.m3.runner import EncoderOnlyEmbedderM3Runner 

logger = logging.getLogger(__name__)

# Viết lại hàm tĩnh (static method)
@staticmethod
def patched_get_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    colbert_dim: int = -1,
    cache_dir: str | None = None,
    torch_dtype = None,
):
    cache_folder = os.getenv('HF_HUB_CACHE', None) if cache_dir is None else cache_dir
    if not os.path.exists(model_name_or_path):
        model_name_or_path = snapshot_download(
            repo_id=model_name_or_path,
            cache_dir=cache_folder,
            ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5']
        )

    # ĐÃ SỬA LỖI Ở ĐÂY: dtype -> torch_dtype
    model = AutoModel.from_pretrained(
        model_name_or_path,
        cache_dir=cache_folder,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype, 
    )
    
    colbert_linear = torch.nn.Linear(
        in_features=model.config.hidden_size,
        out_features=model.config.hidden_size if colbert_dim <= 0 else colbert_dim,
        dtype=torch_dtype,
    )
    sparse_linear = torch.nn.Linear(
        in_features=model.config.hidden_size,
        out_features=1,
        dtype=torch_dtype,
    )

    colbert_model_path = os.path.join(model_name_or_path, 'colbert_linear.pt')
    sparse_model_path = os.path.join(model_name_or_path, 'sparse_linear.pt')
    if os.path.exists(colbert_model_path) and os.path.exists(sparse_model_path):
        logger.info('loading existing colbert_linear and sparse_linear---------')
        colbert_state_dict = torch.load(colbert_model_path, map_location='cpu', weights_only=True)
        sparse_state_dict = torch.load(sparse_model_path, map_location='cpu', weights_only=True)
        colbert_linear.load_state_dict(colbert_state_dict)
        sparse_linear.load_state_dict(sparse_state_dict)
    else:
        logger.info('The parameters of colbert_linear and sparse linear is new initialize.')

    return {
        'model': model,
        'colbert_linear': colbert_linear,
        'sparse_linear': sparse_linear
    }

# Ghi đè class của thư viện
def fix_bug_FlagEmbedding():
    EncoderOnlyEmbedderM3Runner.get_model = patched_get_model