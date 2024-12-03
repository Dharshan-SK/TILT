from javis_t5 import T5ForTokenClassification, TripleHeadTILTForTokenClassification
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
    num_imgs = pixel_values.shape[0]
    image_embeddings = self.unet_encoder(pixel_values)
    print(image_embeddings.shape)
    feature_maps_bboxes = []
    for i in range(num_imgs):
        image_embedding = image_embeddings[i:i+1]
        feature_maps_bboxes.append(self.roi_pool(image_embedding, bboxes[i:i+1]).flatten(2))
    feature_maps_bboxes=torch.concat(feature_maps_bboxes)
    print(feature_maps_bboxes.shape, self.config.d_model, bboxes.shape)
    projection = self.proj(feature_maps_bboxes)
    return projection

class TiLTTransformer(nn.Module):
  def __init__(self, config, num_labels_0, num_labels_1, num_labels_2):
    super().__init__()
    self.config = config
    self.visual_embedding_extractor = VisualEmbedding(config)
    self.t5_model = TripleHeadTILTForTokenClassification(config, num_labels_0, num_labels_1, num_labels_2)
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
    batch["bboxes"]=batch["bbox"]
    total_embedding = self.common_step(batch)
    if batch["distances"]==None:
      batch["distances"] = self.compute_horizontal_vertical_distances(batch["bboxes"])

    ## This is then fed to t5_model
    final_output = self.t5_model(attention_mask = batch['attention_mask'], inputs_embeds = total_embedding,
                            labels_1 = batch['labels_1'], labels_2 = batch['labels_2'], labels_3 = batch['labels_3'], distances = batch["distances"])

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
            text_inputs = self.tokenizer(text, **kwargs)
            overflow_mapping = text_inputs["overflow_to_sample_mapping"]
            labels = kwargs["word_labels"]
            if labels!=None:
                assert len(kwargs["boxes"])==len(labels)
                boxes = kwargs.get("boxes", None)
                updated_labels = []
                for i, overflow_idx in enumerate(overflow_mapping):
                    original_boxes = boxes[overflow_idx]
                    original_labels = labels[overflow_idx]

                    # Create bbox-to-label mapping for the current original sample
                    bbox_label_map = {str(bbox): label for bbox, label in zip(original_boxes, original_labels)}

                    # Update labels for the current tokenized sequence
                    example_labels = []
                    for label, bbox in zip(text_inputs["labels"][i], text_inputs["bbox"][i]):
                        if label == -100 and bbox != [0, 0, 0, 0]:
                            example_labels.append(bbox_label_map.get(str(bbox), label))
                        else:
                            example_labels.append(label)
                    updated_labels.append(example_labels)

                # Update the labels in the tokenizer outputs
                text_inputs["labels"] = updated_labels
            inputs.update(text_inputs)

        # Process images if provided
        if images is not None:
            pixel_values = self.image_processor(images, return_tensors=return_tensors)
            # print(pixel_values)
            inputs["pixel_values"] = [pixel_values['pixel_values'][i] for i in inputs["overflow_to_sample_mapping"]]

        inputs["input_ids"] = torch.tensor(inputs["input_ids"], dtype=torch.long)
        inputs["attention_mask"] = torch.tensor(inputs["attention_mask"], dtype=torch.long)
        inputs["bbox"] = torch.tensor(inputs["bbox"], dtype=torch.float32)
        inputs["pixel_values"] = torch.stack(inputs["pixel_values"])
        inputs["labels"] = torch.tensor(inputs["labels"], dtype=torch.long)
        return inputs

