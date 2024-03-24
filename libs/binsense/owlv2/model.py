from .config import Owlv2Config
from .transformer import Owlv2VisionTransformer, Owlv2Attention, Owlv2MLP
from .embedding import Owlv2VisionEmbeddings
from .output import Owlv2ForObjectDetectionOutput
from .utils import center_to_corners_format_torch
from .utils import box_iou, generalized_box_iou

from torch import nn
from typing import Optional, Tuple

import torch

# Copied from transformers.modeling_utils
_init_weights = True

# TODO: analyze and merge with OwlV2PretainedModel
# Copied from transformers.modeling_utils.PreTrainedModel
# tailed to local needs
class LocalPretrainedModel(nn.Module):
    supports_gradient_checkpointing = False
    
    def __init__(self, config: Owlv2Config, *inputs, **kwargs):
        super().__init__()
        self.config = config
    
    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        self.init_weights()
        # TODO: analyze and remove it
        # self._backward_compatibility_gradient_checkpointing()
    
    def init_weights(self):
        """
        If needed prunes and maybe initializes weights. If using a custom `LocalPretrainedModel`, you need to implement any
        initialization logic in `_init_weights`.
        """
        # change: commented as its not required for us
        # Prune heads if needed
        # if self.config.pruned_heads:
        #     self.prune_heads(self.config.pruned_heads)

        if _init_weights:
            # Initialize weights
            self.apply(self._initialize_weights)

            # Tie weights should be skipped when not initializing all weights
            # since from_pretrained(...) calls tie weights anyways
            # TODO: analyze and remove it
            # self.tie_weights()

    def _initialize_weights(self, module):
        """
        Initialize the weights if they are not already initialized.
        """
        if getattr(module, "_is_hf_initialized", False):
            return
        self._init_weights(module)
        module._is_hf_initialized = True

    
    def _init_weights(self, module):
        """
        Initialize the weights. This method should be overridden by derived class and is
        the only initialization method that will be called when loading a checkpoint
        using `from_pretrained`. Any attempt to initialize outside of this function
        will be useless as the torch.nn.init function are all replaced with skip.
        """
        pass
    
    # def _backward_compatibility_gradient_checkpointing(self):
    #     if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
    #         self.gradient_checkpointing_enable()
    #         # Remove the attribute now that is has been consumed, so it's no saved in the config.
    #         delattr(self.config, "gradient_checkpointing")


# Copied from transformers.models.owlv2.modeling_owlv2.Owlv2PreTrainedModel
# and tailored for local needs
class Owlv2PreTrainedModel(LocalPretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Owlv2Config
    base_model_prefix = "owlv2"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Owlv2EncoderLayer"]

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        # if isinstance(module, Owlv2TextEmbeddings):
        #     module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        #     module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        if isinstance(module, Owlv2VisionEmbeddings):
            factor = self.config.initializer_factor
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        elif isinstance(module, Owlv2Attention):
            factor = self.config.initializer_factor
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, Owlv2MLP):
            factor = self.config.initializer_factor
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
class Owlv2ClassPredictionHead(nn.Module):
    def __init__(self, config: Owlv2Config):
        super().__init__()

        #TODO: check on this hidden_size, we don't have text_config
        #out_dim = config.text_config.hidden_size
        out_dim = 512
        self.query_dim = config.vision_config.hidden_size

        self.dense0 = nn.Linear(self.query_dim, out_dim)
        self.logit_shift = nn.Linear(self.query_dim, 1)
        self.logit_scale = nn.Linear(self.query_dim, 1)
        self.elu = nn.ELU()

    def forward(
        self,
        image_embeds: torch.FloatTensor,
        query_embeds: Optional[torch.FloatTensor],
        query_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.FloatTensor]:
        image_class_embeds = self.dense0(image_embeds)
        if query_embeds is None:
            device = image_class_embeds.device
            batch_size, num_patches = image_class_embeds.shape[:2]
            pred_logits = torch.zeros((batch_size, num_patches, self.query_dim)).to(device)
            return (pred_logits, image_class_embeds)

        # Normalize image and text features
        image_class_embeds = image_class_embeds / (torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6)
        query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)

        # Get class predictions
        pred_logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)

        # Apply a learnable shift and scale to logits
        logit_shift = self.logit_shift(image_embeds)
        logit_scale = self.logit_scale(image_embeds)
        logit_scale = self.elu(logit_scale) + 1
        pred_logits = (pred_logits + logit_shift) * logit_scale

        # if query_mask is not None:
        #     if query_mask.ndim > 1:
        #         query_mask = torch.unsqueeze(query_mask, dim=-2)

        #     pred_logits = pred_logits.to(torch.float64)
        #     pred_logits = torch.where(query_mask == 0, -1e6, pred_logits)
        #     pred_logits = pred_logits.to(torch.float32)

        return (pred_logits, image_class_embeds)

# Copied from transformers.models.owlv2.modeling_owlv2.Owlv2BoxPredictionHead
class Owlv2BoxPredictionHead(nn.Module):
    def __init__(self, config: Owlv2Config, out_dim: int = 4):
        super().__init__()

        width = config.vision_config.hidden_size
        self.dense0 = nn.Linear(width, width)
        self.dense1 = nn.Linear(width, width)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(width, out_dim)

    def forward(self, image_features: torch.Tensor) -> torch.FloatTensor:
        output = self.dense0(image_features)
        output = self.gelu(output)
        output = self.dense1(output)
        output = self.gelu(output)
        output = self.dense2(output)
        return output
    
# Copied from transformers.models.owlv2.modeling_owlv2.Owlv2ForObjectDetection
# and tailored for local needs
class Owlv2ForObjectDetection(Owlv2PreTrainedModel):
    config_class = Owlv2Config

    def __init__(self, config: Owlv2Config):
        super().__init__(config)
        
        self.vision_model = Owlv2VisionTransformer(config.vision_config)
        # self.owlv2 = Owlv2Model(config)
        self.class_head = Owlv2ClassPredictionHead(config)
        self.box_head = Owlv2BoxPredictionHead(config)
        self.objectness_head = Owlv2BoxPredictionHead(config, out_dim=1)

        self.layer_norm = nn.LayerNorm(config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps)
        self.sigmoid = nn.Sigmoid()

        self.sqrt_num_patches = config.vision_config.image_size // config.vision_config.patch_size
        self.box_bias = self._compute_box_bias(self.sqrt_num_patches)
        
        self.post_init()
    
    def _normalize_grid_corner_coordinates(self, num_patches: int) -> torch.Tensor:
        # Create grid coordinates using torch
        x_coordinates = torch.arange(1, num_patches + 1, dtype=torch.float32)
        y_coordinates = torch.arange(1, num_patches + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x_coordinates, y_coordinates, indexing="xy")

        # Stack the coordinates and divide by num_patches
        box_coordinates = torch.stack((xx, yy), dim=-1)
        box_coordinates /= num_patches

        # Flatten (h, w, 2) -> (h*w, 2)
        box_coordinates = box_coordinates.view(-1, 2)

        return box_coordinates
    
    def _compute_box_bias(self, num_patches: int, feature_map: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        if feature_map is not None:
            raise ValueError("feature_map has been deprecated as an input. Please pass in num_patches instead")
        # The box center is biased to its position on the feature grid
        box_coordinates = self._normalize_grid_corner_coordinates(num_patches)
        box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)

        # Unnormalize xy
        box_coord_bias = torch.log(box_coordinates + 1e-4) - torch.log1p(-box_coordinates + 1e-4)

        # The box size is biased to the patch size
        box_size = torch.full_like(box_coord_bias, 1.0 / num_patches)
        box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)

        # Compute box bias
        box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)
        return box_bias
    
    def class_predictor(
        self,
        image_feats: torch.FloatTensor,
        query_embeds: Optional[torch.FloatTensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            image_feats:
                Features extracted from the `image_text_embedder`.
            query_embeds:
                Text query embeddings.
            query_mask:
                Must be provided with query_embeddings. A mask indicating which query embeddings are valid.
        """
        (pred_logits, image_class_embeds) = self.class_head(image_feats, query_embeds, query_mask)

        return (pred_logits, image_class_embeds)
    
    def objectness_predictor(self, image_features: torch.FloatTensor) -> torch.FloatTensor:
        """Predicts the probability that each image feature token is an object.

        Args:
            image_features (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_dim)`)):
                Features extracted from the image.
        Returns:
            Objectness scores.
        """
        image_features = image_features.detach()
        objectness_logits = self.objectness_head(image_features)
        objectness_logits = objectness_logits[..., 0]
        return objectness_logits
    
    def box_predictor(
        self,
        image_feats: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Args:
            image_feats:
                Features extracted from the image, returned by the `image_text_embedder` method.
            feature_map:
                A spatial re-arrangement of image_features, also returned by the `image_text_embedder` method.
        Returns:
            pred_boxes:
                List of predicted boxes (cxcywh normalized to 0, 1) nested within a dictionary.
        """
        # Bounding box detection head [batch_size, num_boxes, 4].
        pred_boxes = self.box_head(image_feats)

        # Compute the location of each token on the grid and use it to compute a bias for the bbox prediction
        box_bias = self.box_bias.to(image_feats.device)
        pred_boxes += box_bias
        pred_boxes = self.sigmoid(pred_boxes)
        return pred_boxes
    
    def image_embedder(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor]:
        # Get Owlv2Model vision embeddings (same as CLIP)
        # ignoring output_attentions and output_hidden_states to vision_model pass
        vision_outputs = self.vision_model(pixel_values=pixel_values, return_dict=True)

        # Apply post_layernorm to last_hidden_state, return non-projected output
        last_hidden_state = vision_outputs[0]
        image_embeds = self.vision_model.post_layernorm(last_hidden_state)

        # Resize class token
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], image_embeds[:, :-1].shape)

        # Merge image embedding with class tokens
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)

        # Resize to [batch_size, num_patches, num_patches, hidden_size]
        # new_size = (
        #     image_embeds.shape[0],
        #     self.sqrt_num_patches,
        #     self.sqrt_num_patches,
        #     image_embeds.shape[-1],
        # )
        # image_embeds = image_embeds.reshape(new_size)

        return (image_embeds, vision_outputs)
    
    def embed_image_query(
        self, 
        query_image_features: torch.FloatTensor
    ) -> torch.FloatTensor:
        _, class_embeds = self.class_predictor(query_image_features)
        pred_boxes = self.box_predictor(query_image_features)
        pred_boxes_as_corners = center_to_corners_format_torch(pred_boxes)

        # Loop over query images
        best_class_embeds = []
        best_box_indices = []
        pred_boxes_device = pred_boxes_as_corners.device

        for i in range(query_image_features.shape[0]):
            each_query_box = torch.tensor([[0, 0, 1, 1]], device=pred_boxes_device)
            each_query_pred_boxes = pred_boxes_as_corners[i]
            ious, _ = box_iou(each_query_box, each_query_pred_boxes)

            # If there are no overlapping boxes, fall back to generalized IoU
            if torch.all(ious[0] == 0.0):
                ious = generalized_box_iou(each_query_box, each_query_pred_boxes)

            # Use an adaptive threshold to include all boxes within 80% of the best IoU
            iou_threshold = torch.max(ious) * 0.8

            selected_inds = (ious[0] >= iou_threshold).nonzero()
            if selected_inds.numel():
                selected_embeddings = class_embeds[i][selected_inds.squeeze(1)]
                mean_embeds = torch.mean(class_embeds[i], axis=0)
                mean_sim = torch.einsum("d,id->i", mean_embeds, selected_embeddings)
                best_box_ind = selected_inds[torch.argmin(mean_sim)]
                best_class_embeds.append(class_embeds[i][best_box_ind])
                best_box_indices.append(best_box_ind)

        if best_class_embeds:
            query_embeds = torch.stack(best_class_embeds)
            box_indices = torch.stack(best_box_indices)
        else:
            query_embeds, box_indices = None, None

        return query_embeds, box_indices, pred_boxes
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        query_pixel_values: Optional[torch.FloatTensor] = None, 
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Owlv2ForObjectDetectionOutput:
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.
            query_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values of query image(s) to be detected. Pass in one query image per target image.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            [Owlv2ForObjectDetectionOutput] or `tuple(torch.FloatTensor)`: A [Owlv2ForObjectDetectionOutput] or a tuple of
            `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
            elements depending on the configuration ([`Owlv2Config`]) and inputs.

        Examples:
        ```python
        >>> import requests
        >>> from PIL import Image
        >>> import torch
        >>> import numpy as np
        >>> from transformers import AutoProcessor, Owlv2ForObjectDetection
        >>> from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

        >>> processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
        >>> model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> query_url = "http://images.cocodataset.org/val2017/000000001675.jpg"
        >>> query_image = Image.open(requests.get(query_url, stream=True).raw)
        >>> inputs = processor(images=image, query_images=query_image, return_tensors="pt")

        >>> # forward pass
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # Note: boxes need to be visualized on the padded, unnormalized image
        >>> # hence we'll set the target image sizes (height, width) based on that

        >>> def get_preprocessed_image(pixel_values):
        ...     pixel_values = pixel_values.squeeze().numpy()
        ...     unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
        ...     unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        ...     unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        ...     unnormalized_image = Image.fromarray(unnormalized_image)
        ...     return unnormalized_image

        >>> unnormalized_image = get_preprocessed_image(inputs.pixel_values)

        >>> target_sizes = torch.Tensor([unnormalized_image.size[::-1]])

        >>> # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> results = processor.post_process_image_guided_detection(
        ...     outputs=outputs, threshold=0.9, nms_threshold=0.3, target_sizes=target_sizes
        ... )
        >>> i = 0  # Retrieve predictions for the first image
        >>> boxes, scores = results[i]["boxes"], results[i]["scores"]
        >>> for box, score in zip(boxes, scores):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(f"Detected similar object with confidence {round(score.item(), 3)} at location {box}")
        Detected similar object with confidence 0.938 at location [490.96, 109.89, 821.09, 536.11]
        Detected similar object with confidence 0.959 at location [8.67, 721.29, 928.68, 732.78]
        Detected similar object with confidence 0.902 at location [4.27, 720.02, 941.45, 761.59]
        Detected similar object with confidence 0.985 at location [265.46, -58.9, 1009.04, 365.66]
        Detected similar object with confidence 1.0 at location [9.79, 28.69, 937.31, 941.64]
        Detected similar object with confidence 0.998 at location [869.97, 58.28, 923.23, 978.1]
        Detected similar object with confidence 0.985 at location [309.23, 21.07, 371.61, 932.02]
        Detected similar object with confidence 0.947 at location [27.93, 859.45, 969.75, 915.44]
        Detected similar object with confidence 0.996 at location [785.82, 41.38, 880.26, 966.37]
        Detected similar object with confidence 0.998 at location [5.08, 721.17, 925.93, 998.41]
        Detected similar object with confidence 0.969 at location [6.7, 898.1, 921.75, 949.51]
        Detected similar object with confidence 0.966 at location [47.16, 927.29, 981.99, 942.14]
        Detected similar object with confidence 0.924 at location [46.4, 936.13, 953.02, 950.78]
        ```"""
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # Compute feature maps for the input and query images
        query_image_feats = self.image_embedder(pixel_values=query_pixel_values)[0]
        image_feats, vision_outputs = self.image_embedder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # Get top class embedding and best box index for each query image in batch
        query_embeds, best_box_indices, query_pred_boxes = self.embed_image_query(query_image_feats)

        # Predict object classes [batch_size, num_patches, num_queries+1]
        (pred_logits, class_embeds) = self.class_predictor(image_feats=image_feats, query_embeds=query_embeds)
        
        # Predict objectness
        objectness_logits = self.objectness_predictor(image_feats)
        
        # Predict object boxes
        target_pred_boxes = self.box_predictor(image_feats)
        
        if not return_dict:
            output = (
                image_feats,
                query_image_feats,
                target_pred_boxes,
                query_pred_boxes,
                pred_logits,
                class_embeds,
                vision_outputs.to_tuple(),
            )
            output = tuple(x for x in output if x is not None)
            return output

        return Owlv2ForObjectDetectionOutput(
            image_embeds=image_feats,
            query_image_embeds=query_image_feats,
            target_pred_boxes=target_pred_boxes,
            query_pred_boxes=query_pred_boxes,
            logits=pred_logits,
            class_embeds=class_embeds,
            text_model_output=None,
            vision_model_output=vision_outputs,
        )