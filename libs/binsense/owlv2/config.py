from typing import Dict

import logging

logger = logging.getLogger(__name__)

class LocalPretainedConfig:
    def __init__(self, **kwargs):
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)
        
        self.torchscript = kwargs.pop("torchscript", False)
        self.return_dict = kwargs.pop("return_dict", True)
    
    @property
    def use_return_dict(self) -> bool:
        """
        `bool`: Whether or not return [`~ModelOutput`] instead of tuples.
        """
        # If torchscript is set, force `return_dict=False` to avoid jit errors
        return self.return_dict and not self.torchscript
        
# Copied from transformers.models.owlv2.configuration_owlv2.Owlv2VisionConfig
# and tailored for local needs
class Owlv2VisionConfig(LocalPretainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`Owlv2VisionModel`]. It is used to instantiate
    an OWLv2 image encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the OWLv2
    [google/owlv2-base-patch16](https://huggingface.co/google/owlv2-base-patch16) architecture.

    # TODO: analyze it and remove it
    # Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    # documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        image_size (`int`, *optional*, defaults to 768):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import Owlv2VisionConfig, Owlv2VisionModel

    >>> # Initializing a Owlv2VisionModel with google/owlv2-base-patch16 style configuration
    >>> configuration = Owlv2VisionConfig()

    >>> # Initializing a Owlv2VisionModel model from the google/owlv2-base-patch16 style configuration
    >>> model = Owlv2VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "owlv2_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=768,
        patch_size=16,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        class_embed_size=512,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.class_embed_size = class_embed_size

    # TODO: analyze it and remove it
    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
    #     cls._set_token_in_kwargs(kwargs)

    #     config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

    #     # get the vision config dict if we are loading from Owlv2Config
    #     if config_dict.get("model_type") == "owlv2":
    #         config_dict = config_dict["vision_config"]

    #     if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
    #         logger.warning(
    #             f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
    #             f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
    #         )

    #     return cls.from_dict(config_dict, **kwargs)

# Copied from transformers.models.owlv2.configuration_owlv2.Owlv2Config
# tailored for local needs
class Owlv2Config(LocalPretainedConfig):
    r"""
    [`Owlv2Config`] is the configuration class to store the configuration of an [`Owlv2Model`]. It is used to
    instantiate an OWLv2 model according to the specified arguments, defining the text model and vision model
    configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the OWLv2
    [google/owlv2-base-patch16](https://huggingface.co/google/owlv2-base-patch16) architecture.

    # TODO: analyze it and remove it
    # Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    # documentation from [`PretrainedConfig`] for more information.

    Args:
        # TODO: analyze it and remove it
        # text_config (`dict`, *optional*):
        #     Dictionary of configuration options used to initialize [`Owlv2TextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Owlv2VisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* parameter. Default is used as per the original OWLv2
            implementation.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return a dictionary. If `False`, returns a tuple.
        kwargs (*optional*):
            Dictionary of keyword arguments.
    """

    model_type = "owlv2"

    def __init__(
        self,
        # text_config=None,
        vision_config=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        return_dict=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # TODO: analyze it and remove it
        # if text_config is None:
        #     text_config = {}
        #     logger.info("text_config is None. Initializing the Owlv2TextConfig with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the Owlv2VisionConfig with default values.")

        # TODO: analyze it and remove it
        # self.text_config = Owlv2TextConfig(**text_config)
        self.vision_config = Owlv2VisionConfig(**vision_config)
        self.use_no_object_mask = False

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.return_dict = return_dict
        self.initializer_factor = 1.0

    # TODO: analyze it and remove it
    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
    #     cls._set_token_in_kwargs(kwargs)

    #     config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

    #     if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
    #         logger.warning(
    #             f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
    #             f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
    #         )

    #     return cls.from_dict(config_dict, **kwargs)

    # TODO: analyze it and remove it
    # @classmethod
    # def from_text_vision_configs(cls, text_config: Dict, vision_config: Dict, **kwargs):
    #     r"""
    #     Instantiate a [`Owlv2Config`] (or a derived class) from owlv2 text model configuration and owlv2 vision
    #     model configuration.

    #     Returns:
    #         [`Owlv2Config`]: An instance of a configuration object
    #     """
    #     config_dict = {}
    #     config_dict["text_config"] = text_config
    #     config_dict["vision_config"] = vision_config

    #     return cls.from_dict(config_dict, **kwargs)
    
    @classmethod
    def from_vision_config(cls, vision_config: Dict, **kwargs):
        r"""
        Instantiate a [`Owlv2Config`] (or a derived class) from owlv2 vision
        model configuration.

        Returns:
            [`Owlv2Config`]: An instance of a configuration object
        """
        config_dict = {}
        config_dict["vision_config"] = vision_config

        return cls.from_dict(config_dict, **kwargs)
    