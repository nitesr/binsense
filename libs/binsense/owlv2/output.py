from dataclasses import dataclass, is_dataclass, fields
from typing import Tuple, Optional, Any
from collections import OrderedDict

import torch

# Placeholder for transformers.utils.ModelOutput
class LocalModelOutput(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        is_modeloutput_subclass = self.__class__ != LocalModelOutput
        if is_modeloutput_subclass and not is_dataclass(self):
            raise TypeError(
                f"{self.__module__}.{self.__class__.__name__} is not a dataclasss."
                " This is a subclass of ModelOutput and so must use the @dataclass decorator."
            )

    def __post_init__(self):
        for f in fields(self):
            super().__setitem__(f.name, self.__getattribute__(f.name))
    
    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]
    
    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())

# Copied from transformers.models.owlv2.modeling_owlv2.BaseModelOutput
# and tailored for local needs
@dataclass
class BaseModelOutput(LocalModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

# Copied from transformers.models.owlv2.modeling_owlv2.BaseModelOutput
# and tailored for local needs
@dataclass
class BaseModelOutputWithPooling(LocalModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

# Copied from transformers.models.owlvit.modeling_owlv2.Owlv2ImageGuidedObjectDetectionOutput
# and tailored for local needs
@dataclass
class Owlv2ForObjectDetectionOutput(LocalModelOutput):
    """
    Output type of [`Owlv2ForObjectDetection.image_guided_detection`].

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, num_patches, num_queries)`):
            Classification logits (including no-object) for all queries.
        target_pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual target image in the batch
            (disregarding possible padding). You can use [`~Owlv2ImageProcessor.post_process_object_detection`] to
            retrieve the unnormalized bounding boxes.
        query_pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual query image in the batch
            (disregarding possible padding). You can use [`~Owlv2ImageProcessor.post_process_object_detection`] to
            retrieve the unnormalized bounding boxes.
        image_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, output_dim`):
            Pooled output of [`Owlv2VisionModel`]. OWLv2 represents images as a set of image patches and computes
            image embeddings for each patch.
        query_image_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, output_dim`):
            Pooled output of [`Owlv2VisionModel`]. OWLv2 represents images as a set of image patches and computes
            image embeddings for each patch.
        class_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
            Class embeddings of all image patches. OWLv2 represents images as a set of image patches where the total
            number of patches is (image_size / patch_size)**2.
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`Owlv2VisionModel`].
    """

    logits: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    query_image_embeds: torch.FloatTensor = None
    target_pred_boxes: torch.FloatTensor = None
    query_pred_boxes: torch.FloatTensor = None
    class_embeds: torch.FloatTensor = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )