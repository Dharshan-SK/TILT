from javis_t5 import T5ForTokenClassification
from ernie_visual_backbone import ResNetCustomized
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
import torch
import torch.nn as nn

from dataset import FUNSDDs
from torchvision import transforms
from tqdm.auto import tqdm

## Custom imports
from visual_backbone import Unet_encoder, RoIPool

class VisualEmbedding(nn.Module):
  def __init__(self, config):
    super().__init__()
    # self.unet_encoder = Unet_encoder(in_channels = config.in_channels, channels = config.channels, num_pool_layers = config.num_pool_layers)
    self.unet_encoder = ResNetCustomized(101, 3)
    self.roi_pool = RoIPool(output_size = config.output_size, spatial_scale = config.spatial_scale)
    self.proj = nn.Linear(in_features = 256 * 3 * 3, out_features = config.d_model)
    self.config = config

  def forward(self, pixel_values, bboxes):
    image_embedding = self.unet_encoder(pixel_values)
    print(image_embedding.shape)
    feature_maps_bboxes = self.roi_pool(image_embedding, bboxes).flatten(2)
    print(feature_maps_bboxes.shape, self.config.d_model, bboxes.shape)
    projection = self.proj(feature_maps_bboxes)
    return projection

class TiLTTransformer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.visual_embedding_extractor = VisualEmbedding(config)
    self.t5_model = T5ForTokenClassification(config, 3)
    if self.config.load_vision_weights:
      checkpoint = torch.load("src/unet_encoder_weights.pth")
      self.visual_embedding_extractor.unet_encoder.load_state_dict(checkpoint)
      print("Loaded Custom weights for vision module")


#   def generate(self, batch):
#     total_embedding = self.common_step(batch)
#     return self.t5_model.generate(input_embeds = total_embedding)

  def compute_horizontal_vertical_distances(self, bboxes):
    x_min, y_min, x_max, y_max = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    midx = ((x_min + x_max) / 2).unsqueeze(-1)
    midy = ((y_min + y_max) / 2).unsqueeze(-1)
    h_distances = (midx - midx.transpose(1, 2)).to(torch.long)  # (batch, num_tokens, num_tokens)
    v_distances = (midy - midy.transpose(1, 2)).to(torch.long)

    return (h_distances, v_distances)
  

  def common_step(self, batch):
    ## Visual embedding
    visual_embedding = self.visual_embedding_extractor(pixel_values = batch['pixel_values'], bboxes = batch['bboxes'])

    ## Semantic embedding from t5_model's embedding layer
    semantic_embedding = self.t5_model.shared(batch['input_ids'])

    ## Net embedding is addition of both the embeddings
    total_embedding = (visual_embedding,semantic_embedding)

    return total_embedding

  def forward(self, batch):

    total_embedding = self.common_step(batch)
    if batch["distances"]==None:
      batch["distances"] = self.compute_horizontal_vertical_distances(batch["bboxes"])

    ## This is then fed to t5_model
    final_output = self.t5_model(attention_mask = batch['attention_mask'], inputs_embeds = total_embedding,
                            labels = batch['labels'], distances = batch["distances"])

    return final_output
  

from transformers import ProcessorMixin
import torch

class CustomAutoProcessor(ProcessorMixin):
    def __init__(self, tokenizer, image_processor):
        # super().__init__(tokenizer, image_processor)
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __call__(self, text=None, images=None, return_tensors="pt", **kwargs):
        inputs = {}

        # Process text if provided
        if text is not None:
            inputs.update(self.tokenizer(text, return_tensors=return_tensors, **kwargs))

        # Process images if provided
        if images is not None:
            pixel_values = self.image_processor(images, return_tensors=return_tensors)
            # print(pixel_values)
            inputs["pixel_values"] = pixel_values['pixel_values']

        return inputs