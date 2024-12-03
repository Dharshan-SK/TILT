import sys
sys.path.append("/mnt/c/Javis/TiLT-Implementation/src")

from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
import datasets
import torch
import torch.nn as nn

# from torchvision import transforms
from tqdm.auto import tqdm

## Custom imports
from visual_backbone import Unet_encoder, RoIPool
from t5 import T5ForConditionalGeneration, T5Stack
from transformers import AutoModel, T5Tokenizer
from model import TiLTTransformer, CustomAutoProcessor
from tokenizer_ernie import ErnieLayoutTokenizerFast


device = "cuda" if torch.cuda.is_available() else "cpu"

hf_ds = load_dataset("nielsr/funsd-layoutlmv3")
model_name = "t5-base"
## Visual Embedding extractor's parameters
in_channels = 3
num_pool_layers = 3
channels = 16
sampling_ratio = 2
spatial_scale = 112 / 1000 # bboxes [0-1000], img_inp [48], conv_features_width [112]
output_size = (3,3)
load_weights = True

## Tokenizer's parameter
model_max_length = 512

t5_config = AutoConfig.from_pretrained(model_name)
## Adding new parameters
t5_config.update(dict(in_channels = in_channels, num_pool_layers = num_pool_layers,  channels = channels, model_max_length = model_max_length,
                      output_size = output_size, spatial_scale = spatial_scale, sampling_ratio = sampling_ratio, use_cache = False, load_weights = load_weights,
                      lr =  2e-4, load_vision_weights = True))

## Tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True, model_max_length = model_max_length)
tokenizer = ErnieLayoutTokenizerFast.from_pretrained(pretrained_model_name_or_path="t5-base", )
tokenizer.padding_side = "right"
tokenizer.pad_token="<pad>"
tokenizer.bos_token = " "
tokenizer.unk_token = "<unk>"
tokenizer.sep_token = " "

from transformers.models.layoutlmv3 import LayoutLMv3ImageProcessor
image_processor = LayoutLMv3ImageProcessor(apply_ocr=False, size= {"height": 448, "width": 448})
processor = CustomAutoProcessor(tokenizer, image_processor)
from PIL import Image
image = Image.open(f'/mnt/c/mnt/data/pharma_new_short_test_20231201_20240114_triplehead/images/20240671415039_0.png').convert('RGB')
text = ["▁T", "▁", "▁E", "▁P", "▁IT", "T", "▁"]*90
inputs = processor(text = [text]*2,images = [image]*2, boxes = [[[j for j in range(i,i+4, 1)] for i in range(len(text))]]*2, word_labels = [[1,2,0]*210]*2, 
                   stride=128,   max_length=512, truncation=True, padding="max_length",
                            return_overflowing_tokens=True, return_offsets_mapping=True, is_pretokenized=True, add_special_tokens=False)

inputs["distances"] = None
# inputs["bboxes"] = torch.rand(1, len(inputs["input_ids"][0]), 4, dtype=torch.float32)*1000
# inputs['labels'] = torch.ones(1,len(inputs["input_ids"][0]), dtype=torch.long)

data = datasets.load_from_disk("/mnt/c/mnt/data/pharma_new_short_test_20231201_20240114_triplehead/val").select(range(4))
# inputs["bboxes"] = torch.tensor([data[3]["bboxes"][:len(inputs["input_ids"][0])]], dtype=torch.float32)

tilt_model = TiLTTransformer(t5_config, 3).to(device)

output = tilt_model(inputs)

output.logits