from .constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from .utils import ChannelDimension, PaddingMode
from .utils import to_channel_dimension_format
from .utils import infer_channel_dimension_format
from .utils import get_channel_dimension_axis
from .utils import get_size_dict, get_image_size, to_pil_image
from .utils import make_list_of_images, valid_images, is_scaled_image, is_valid_image
from .utils import center_to_corners_format_torch
from .utils import box_iou

from typing import Union, Optional, List, Dict, Iterable, Tuple
from scipy import ndimage as ndi
from PIL.Image import Resampling as PILImageResampling

import numpy as np

import warnings, PIL, torch, logging

logger = logging.getLogger(__name__)

class BaseImageProcessor:
    def __init__(self, **kwargs) -> None:
        pass
    
    def rescale(
        self,
        image: np.ndarray,
        scale: float,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        dtype: np.dtype = np.float32,
        **kwargs,
    ) -> np.ndarray:
        """
        Rescale an image by a scale factor. image = image * scale.

        Args:
            image (`np.ndarray`):
                Image to rescale.
            scale (`float`):
                The scaling factor to rescale pixel values by.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

        Returns:
            `np.ndarray`: The rescaled image.
        """
        rescaled_image = image * scale
        if data_format is not None:
            rescaled_image = to_channel_dimension_format(rescaled_image, data_format, input_data_format)
        
        rescaled_image = rescaled_image.astype(dtype)
        return rescaled_image

    def normalize(
        self,
        image: np.ndarray,
        mean: Union[float, Iterable[float]],
        std: Union[float, Iterable[float]],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Normalize an image. image = (image - image_mean) / image_std.

        Args:
            image (`np.ndarray`):
                Image to normalize.
            mean (`float` or `Iterable[float]`):
                Image mean to use for normalization.
            std (`float` or `Iterable[float]`):
                Image standard deviation to use for normalization.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

        Returns:
            `np.ndarray`: The normalized image.
        """
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        channel_axis = get_channel_dimension_axis(image, input_data_format=input_data_format)
        num_channels = image.shape[channel_axis]

        # We cast to float32 to avoid errors that can occur when subtracting uint8 values.
        # We preserve the original dtype if it is a float type to prevent upcasting float16.
        if not np.issubdtype(image.dtype, np.floating):
            image = image.astype(np.float32)

        if isinstance(mean, Iterable):
            if len(mean) != num_channels:
                raise ValueError(f"mean must have {num_channels} elements if it is an iterable, got {len(mean)}")
        else:
            mean = [mean] * num_channels
        mean = np.array(mean, dtype=image.dtype)

        if isinstance(std, Iterable):
            if len(std) != num_channels:
                raise ValueError(f"std must have {num_channels} elements if it is an iterable, got {len(std)}")
        else:
            std = [std] * num_channels
        std = np.array(std, dtype=image.dtype)

        if input_data_format == ChannelDimension.LAST:
            image = (image - mean) / std
        else:
            image = ((image.T - mean) / std).T

        image = to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
        return image

    def center_crop(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        return_numpy: Optional[bool] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Center crop an image to `(size["height"], size["width"])`. If the input size is smaller than `crop_size` along
        any edge, the image is padded with 0's and then center cropped.

        Args:
            image (`np.ndarray`):
                Image to center crop.
            size (`Dict[str, int]`):
                Size of the output image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        """
        size = get_size_dict(size)
        if "height" not in size or "width" not in size:
            raise ValueError(f"The size dictionary must have keys 'height' and 'width'. Got {size.keys()}")
        
        if not isinstance(size, Iterable) or len(size) != 2:
            raise ValueError("size must have 2 elements representing the height and width of the output image")

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        output_data_format = data_format if data_format is not None else input_data_format

        # We perform the crop in (C, H, W) format and then convert to the output format
        image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)

        orig_height, orig_width = get_image_size(image, ChannelDimension.FIRST)
        crop_height, crop_width = size
        crop_height, crop_width = int(crop_height), int(crop_width)

        # In case size is odd, (image_shape[0] + size[0]) // 2 won't give the proper result.
        top = (orig_height - crop_height) // 2
        bottom = top + crop_height
        # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.
        left = (orig_width - crop_width) // 2
        right = left + crop_width

        # Check if cropped area is within image boundaries
        if top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width:
            image = image[..., top:bottom, left:right]
            image = to_channel_dimension_format(image, output_data_format, ChannelDimension.FIRST)
            return image

        # Otherwise, we may need to pad if the image is too small. Oh joy...
        new_height = max(crop_height, orig_height)
        new_width = max(crop_width, orig_width)
        new_shape = image.shape[:-2] + (new_height, new_width)
        new_image = np.zeros_like(image, shape=new_shape)

        # If the image is too small, pad it with zeros
        top_pad = (new_height - orig_height) // 2
        bottom_pad = top_pad + orig_height
        left_pad = (new_width - orig_width) // 2
        right_pad = left_pad + orig_width
        new_image[..., top_pad:bottom_pad, left_pad:right_pad] = image

        top += top_pad
        bottom += top_pad
        left += left_pad
        right += left_pad

        new_image = new_image[..., max(0, top) : min(new_height, bottom), max(0, left) : min(new_width, right)]
        new_image = to_channel_dimension_format(new_image, output_data_format, ChannelDimension.FIRST)

        if not return_numpy:
            new_image = to_pil_image(new_image)

        return new_image

# Copied from transformers.models.owlv2.image_processing_owlv2.Owlv2ImageProcessor
# tailored for local needs
class Owlv2ImageProcessor(BaseImageProcessor):
    r"""
    Constructs an OWLv2 image processor.

    Args:
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overriden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overriden by `rescale_factor` in the `preprocess`
            method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image to a square with gray pixels on the bottom and the right. Can be overriden by
            `do_pad` in the `preprocess` method.
        do_resize (`bool`, *optional*, defaults to `True`):
            Controls whether to resize the image's (height, width) dimensions to the specified `size`. Can be overriden
            by `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 960, "width": 960}`):
            Size to resize the image to. Can be overriden by `size` in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling method to use if resizing the image. Can be overriden by `resample` in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `OPENAI_CLIP_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `OPENAI_CLIP_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
    """
    model_input_names = ["pixel_values"]
    
    def __init__(
        self,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_pad: bool = True,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_pad = do_pad
        self.do_resize = do_resize
        self.size = size if size is not None else {"height": 960, "width": 960}
        self.resample = resample
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self._valid_processor_keys = [
            "images",
            "do_pad",
            "do_resize",
            "size",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]
    
    def pad(
        self,
        image: np.array,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        mode: PaddingMode = PaddingMode.CONSTANT
    ):
        """
        Pad an image to a square with gray pixels on the bottom and the right, as per the original OWLv2
        implementation.

        Args:
            image (`np.ndarray`):
                Image to pad.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred from the input
                image.
            mode (`PaddingMode`):
                The padding mode to use. Can be one of:
                    - `"constant"`: pads with a constant value.
                    - `"reflect"`: pads with the reflection of the vector mirrored on the first and last values of the
                    vector along each axis.
                    - `"replicate"`: pads with the replication of the last value on the edge of the array along each axis.
                    - `"symmetric"`: pads with the reflection of the vector mirrored along the edge of the array.
                
        """
        height, width = get_image_size(image)
        size = max(height, width)
        
        #padding (`int` or `Tuple[int, int]` or `Iterable[Tuple[int, int]]`):
        # Padding to apply to the edges of the height, width axes. Can be one of three formats:
        # - `((before_height, after_height), (before_width, after_width))` unique pad widths for each axis.
        # - `((before, after),)` yields same before and after pad for height and width.
        # - `(pad,)` or int is a shortcut for before = after = pad width for all axes.
        padding=((0, size - height), (0, size - width))
        
        # The value to use for the padding if `mode` is `"constant"`.
        constant_values=0.5
        
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        def _expand_for_data_format(values):
            """
            Convert values to be in the format expected by np.pad based on the data format.
            """
            if isinstance(values, (int, float)):
                values = ((values, values), (values, values))
            elif isinstance(values, tuple) and len(values) == 1:
                values = ((values[0], values[0]), (values[0], values[0]))
            elif isinstance(values, tuple) and len(values) == 2 and isinstance(values[0], int):
                values = (values, values)
            elif isinstance(values, tuple) and len(values) == 2 and isinstance(values[0], tuple):
                values = values
            else:
                raise ValueError(f"Unsupported format: {values}")

            # add 0 for channel dimension
            values = ((0, 0), *values) if input_data_format == ChannelDimension.FIRST else (*values, (0, 0))

            # Add additional padding if there's a batch dimension
            values = (0, *values) if image.ndim == 4 else values
            return values

        padding = _expand_for_data_format(padding)

        if mode == PaddingMode.CONSTANT:
            constant_values = _expand_for_data_format(constant_values)
            image = np.pad(image, padding, mode="constant", constant_values=constant_values)
        elif mode == PaddingMode.REFLECT:
            image = np.pad(image, padding, mode="reflect")
        elif mode == PaddingMode.REPLICATE:
            image = np.pad(image, padding, mode="edge")
        elif mode == PaddingMode.SYMMETRIC:
            image = np.pad(image, padding, mode="symmetric")
        else:
            raise ValueError(f"Invalid padding mode: {mode}")

        image = to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
        return image
    
    def _preprocess_resize_output_shape(self, image, output_shape):
        """Validate resize output shape according to input image.

        Args:
            image (`np.ndarray`):
            Image to be resized.
            output_shape (`iterable`):
                Size of the generated output image `(rows, cols[, ...][, dim])`. If `dim` is not provided, the number of
                channels is preserved.

        Returns
            image (`np.ndarray):
                The input image, but with additional singleton dimensions appended in the case where `len(output_shape) >
                input.ndim`.
            output_shape (`Tuple`):
                The output shape converted to tuple.

        Raises ------ ValueError:
            If output_shape length is smaller than the image number of dimensions.

        Notes ----- The input image is reshaped if its number of dimensions is not equal to output_shape_length.

        """
        output_shape = tuple(output_shape)
        output_ndim = len(output_shape)
        input_shape = image.shape
        if output_ndim > image.ndim:
            # append dimensions to input_shape
            input_shape += (1,) * (output_ndim - image.ndim)
            image = np.reshape(image, input_shape)
        elif output_ndim == image.ndim - 1:
            # multichannel case: append shape of last axis
            output_shape = output_shape + (image.shape[-1],)
        elif output_ndim < image.ndim:
            raise ValueError("output_shape length cannot be smaller than the " "image number of dimensions")

        return image, output_shape
    
    # https://github.com/scikit-image/scikit-image/blob/b4b521d6f0a105aabeaa31699949f78453ca3511/skimage/transform/_warps.py#L640.
    def _clip_warp_output(self, input_image, output_image):
        """Clip output image to range of values of input image.

        Note that this function modifies the values of *output_image* in-place.
        Args:
            input_image : ndarray
                Input image.
            output_image : ndarray
                Output image, which is modified in-place.
        """
        min_val = np.min(input_image)
        if np.isnan(min_val):
            # NaNs detected, use NaN-safe min/max
            min_func = np.nanmin
            max_func = np.nanmax
            min_val = min_func(input_image)
        else:
            min_func = np.min
            max_func = np.max
        max_val = max_func(input_image)

        output_image = np.clip(output_image, min_val, max_val)

        return output_image

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        anti_aliasing: bool = True,
        anti_aliasing_sigma=None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image as per the original implementation.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary containing the height and width to resize the image to.
            anti_aliasing (`bool`, *optional*, defaults to `True`):
                Whether to apply anti-aliasing when downsampling the image.
            anti_aliasing_sigma (`float`, *optional*, defaults to `None`):
                Standard deviation for Gaussian kernel when downsampling the image. If `None`, it will be calculated
                automatically.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred from the input
                image.
        """

        output_shape = (size["height"], size["width"])
        image = to_channel_dimension_format(image, ChannelDimension.LAST)
        image, output_shape = self._preprocess_resize_output_shape(image, output_shape)
        input_shape = image.shape
        factors = np.divide(input_shape, output_shape)

        # Translate modes used by np.pad to those used by scipy.ndimage
        ndi_mode = "mirror"
        cval = 0
        order = 1
        if anti_aliasing:
            if anti_aliasing_sigma is None:
                anti_aliasing_sigma = np.maximum(0, (factors - 1) / 2)
            else:
                anti_aliasing_sigma = np.atleast_1d(anti_aliasing_sigma) * np.ones_like(factors)
                if np.any(anti_aliasing_sigma < 0):
                    raise ValueError("Anti-aliasing standard deviation must be " "greater than or equal to zero")
                elif np.any((anti_aliasing_sigma > 0) & (factors <= 1)):
                    warnings.warn(
                        "Anti-aliasing standard deviation greater than zero but " "not down-sampling along all axes"
                    )
            filtered = ndi.gaussian_filter(image, anti_aliasing_sigma, cval=cval, mode=ndi_mode)
        else:
            filtered = image

        zoom_factors = [1 / f for f in factors]
        out = ndi.zoom(filtered, zoom_factors, order=order, mode=ndi_mode, cval=cval, grid_mode=True)

        image = self._clip_warp_output(image, out)

        image = to_channel_dimension_format(image, input_data_format, ChannelDimension.LAST)
        image = (
            to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
        )
        return image

    def _validate_kwargs(self, captured_kwargs: List[str]):
        unused_keys = set(captured_kwargs).difference(set(self._valid_processor_keys))
        if unused_keys:
            unused_key_str = ", ".join(unused_keys)
            # TODO raise a warning here instead of simply logging?
            logger.warning(f"Unused or unrecognized kwargs: {unused_key_str}.")
        
    def preprocess(
        self,
        images: Union[PIL.Image.Image, np.ndarray, torch.Tensor, List[PIL.Image.Image], List[np.ndarray], List[torch.Tensor]],
        do_pad: bool = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether to pad the image to a square with gray pixels on the bottom and the right.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size to resize the image to.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        Returns:
            dict({"pixel_values", torch.Tensor})
        """
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_pad = do_pad if do_pad is not None else self.do_pad
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        size = size if size is not None else self.size

        images = make_list_of_images(images)

        self._validate_kwargs(captured_kwargs=kwargs.keys())

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
            
        if do_rescale and rescale_factor is None:
            raise ValueError("rescale_factor must be specified if do_rescale is True.")

        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError("image_mean and image_std must both be specified if do_normalize is True.")

        # All transformations expect numpy arrays.
        def _to_numpy_array(img) -> np.ndarray:
            if not is_valid_image(img):
                raise ValueError(f"Invalid image type: {type(img)}")

            if isinstance(img, PIL.Image.Image):
                return np.array(img)
            
            if isinstance(img, torch.Tensor):
                return img.detach().cpu().numpy()
            
            raise ValueError(f"Invalid image type: {type(img)}")
        
        images = [_to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        if do_rescale:
            images = [
                self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                for image in images
            ]

        if do_pad:
            images = [self.pad(image=image, input_data_format=input_data_format) for image in images]

        if do_resize:
            images = [
                self.resize(
                    image=image,
                    size=size,
                    input_data_format=input_data_format,
                )
                for image in images
            ]

        if do_normalize:
            images = [
                self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                for image in images
            ]

        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]
        images = np.array(images)
        data = {"pixel_values": torch.tensor(images)}
        return data
    
    
    def unnormalize_pixels(
        self, pixels: Union[torch.Tensor, np.ndarray]
    ):
        """
        this is to get the pre-processed image for visualization
            resized and padded before normalizing
        Args:
            pixles: Union[torch.Tensor, np.ndarray]
                returned by the image embedder (`Owlv2VisionTransformer`)
                expects shape B x C x W x H
        Returns:
            unnormalized_pixels: Union[torch.Tensor, np.ndarray]
                returned pixels are in format B x W x H x C
        """
        if isinstance(pixels, torch.Tensor):
            pixels = pixels.detach().numpy()
        
        if len(pixels.shape) < 4:
            raise ValueError("pixels shape should be B x C x W x H")
        
        mean = np.array(self.image_mean)
        std = np.array(self.image_std)
        mean = np.expand_dims(self.image_mean, axis=(0, 2, -1)) # make it B x C x W x H
        std = np.expand_dims(self.image_std, axis=(0, 2, -1)) # make it B x C x W x H
        
        unnormalized_pixels = pixels * std + mean
        unnormalized_pixels = (unnormalized_pixels * 255).astype(np.uint8)
        return np.moveaxis(unnormalized_pixels, 1, -1)
    
    def post_process_bounding_boxes(
        self, boxes, target_sizes: Union[torch.Tensor, List[Tuple]] = None
    ):
        """
        Converts the bounding box raw output of [`Owlv2ForObjectDetectionOutput`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format.

        Args:
            boxes ([`Owlv2ForObjectDetectionOutput.pred_boxes`])::
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
        Returns:
            scaled bounding boxes to target_sizes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format.
        """
        if target_sizes is not None:
            if len(boxes) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )
        
        # Convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format_torch(boxes)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]
        return boxes


    def post_process_object_detection(
        self, outputs, threshold: float = 0.1, target_sizes: Union[torch.Tensor, List[Tuple]] = None
    ):
        """
        Converts the raw output of [`Owlv2ForObjectDetectionOutput`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format.

        Args:
            outputs ([`Owlv2ForObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        logits, boxes = outputs.logits, outputs.pred_boxes

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        probs = torch.max(logits, dim=-1)
        scores = torch.sigmoid(probs.values)
        labels = probs.indices

        # Convert to [x0, y0, x1, y1] format & scale
        boxes = self.post_process_bounding_boxes(boxes, target_sizes)

        results = []
        for s, l, b in zip(scores, labels, boxes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({"scores": score, "labels": label, "boxes": box})

        return results

    def post_process_image_guided_detection(self, outputs, threshold=0.0, nms_threshold=0.3, target_sizes=None):
        """
        Converts the output of [`Owlv2ForObjectDetection.image_guided_detection`] into the format expected by the COCO
        api.

        Args:
            outputs ([`Owlv2ForObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*, defaults to 0.0):
                Minimum confidence threshold to use to filter out predicted boxes.
            nms_threshold (`float`, *optional*, defaults to 0.3):
                IoU threshold for non-maximum suppression of overlapping boxes.
            target_sizes (`torch.Tensor`, *optional*):
                Tensor of shape (batch_size, 2) where each entry is the (height, width) of the corresponding image in
                the batch. If set, predicted normalized bounding boxes are rescaled to the target sizes. If left to
                None, predictions will not be unnormalized.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model. All labels are set to None as
            `Owlv2ForObjectDetection.image_guided_detection` perform one-shot object detection.
        """
        logits, target_boxes = outputs.pred_logits, outputs.pred_boxes

        if len(logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        if target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        probs = torch.max(logits, dim=-1)
        scores = torch.sigmoid(probs.values)

        # Convert to [x0, y0, x1, y1] format
        target_boxes = center_to_corners_format_torch(target_boxes)

        # Apply non-maximum suppression (NMS)
        if nms_threshold < 1.0:
            for idx in range(target_boxes.shape[0]):
                for i in torch.argsort(-scores[idx]):
                    if not scores[idx][i]:
                        continue

                    ious = box_iou(target_boxes[idx][i, :].unsqueeze(0), target_boxes[idx])[0][0]
                    ious[i] = -1.0  # Mask self-IoU.
                    scores[idx][ious > nms_threshold] = 0.0

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(target_boxes.device)
        target_boxes = target_boxes * scale_fct[:, None, :]

        # Compute box display alphas based on prediction scores
        results = []
        alphas = torch.zeros_like(scores)

        for idx in range(target_boxes.shape[0]):
            # Select scores for boxes matching the current query:
            query_scores = scores[idx]
            if not query_scores.nonzero().numel():
                continue

            # Apply threshold on scores before scaling
            query_scores[query_scores < threshold] = 0.0

            # Scale box alpha such that the best box for each query has alpha 1.0 and the worst box has alpha 0.1.
            # All other boxes will either belong to a different query, or will not be shown.
            max_score = torch.max(query_scores) + 1e-6
            query_alphas = (query_scores - (max_score * 0.1)) / (max_score * 0.9)
            query_alphas = torch.clip(query_alphas, 0.0, 1.0)
            alphas[idx] = query_alphas

            mask = alphas[idx] > 0
            box_scores = scores[idx][mask]
            boxes = target_boxes[idx][mask]
            results.append({"scores": box_scores, "labels": None, "boxes": boxes})

        return results