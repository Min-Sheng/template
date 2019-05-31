import torch
import random
import importlib
import numpy as np


def compose(transforms):
    """Compose several transformers together.
    Args:
        transforms (Box): The preprocessing and augmentation techniques applied to the data.

    Returns:
        transforms (list of BaseTransformer): The list of transformers.
    """
    if transforms is None:
        return None

    _transforms = []
    for transform in transforms:
        if transform.do:
            cls_name = ''.join([str_.capitalize() for str_ in transform.name.split('_')])
            cls = getattr(importlib.import_module('src.data.transformers'), cls_name)
            _transforms.append(cls(**transform.kwargs))

    # Append the default transformer ToTensor.
    _transforms.append(ToTensor())

    transforms = Compose(_transforms)
    return transforms


class BaseTransformer:
    """The base class for all transformers.
    """
    def __call__(self, *imgs, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__


class Compose(BaseTransformer):
    """Compose several transforms together.
    Args:
         transforms (Box): The preprocessing and augmentation techniques applied to the data.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *imgs, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be transformed.

        Returns:
            imgs (tuple of torch.Tensor): The transformed images.
        """
        for transform in self.transforms:
            imgs = transform(*imgs, **kwargs)
        return imgs

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(BaseTransformer):
    """Convert a tuple of numpy.ndarray to a tuple of torch.Tensor.
    """
    def __call__(self, *imgs, dtypes=None, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be converted to tensor.
            dtypes (sequence of torch.dtype): The corresponding dtype of the images (default: None, transform all the images' dtype to torch.float).

        Returns:
            imgs (tuple of torch.Tensor): The converted images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if dtypes:
            if not all(isinstance(dtype, torch.dtype) for dtype in dtypes):
                raise TypeError('All of the dtypes should be torch.dtype.')
            if len(dtypes) != len(imgs):
                raise ValueError('The number of the dtypes should be the same as the images.')
            imgs = tuple(img.to(dtype) for img, dtype in zip(map(torch.from_numpy, imgs), dtypes))
        else:
            imgs = tuple(img.float() for img in map(torch.from_numpy, imgs))
        return imgs


class Normalize(BaseTransformer):
    """Normalize a tuple of images with mean and standard deviation.
    Args:
        means (int or list): A sequence of means for each channel.
        stds (int or list): A sequence of standard deviations for each channel.
    """
    def __init__(self, means, stds):
        if means is None and stds is None:
            pass
        elif means is not None and stds is not None:
            if len(means) != len(stds):
                raise ValueError('The number of the means should be the same as the standard deviations.')
            means = tuple(means)
            stds = tuple(stds)
        else:
            raise ValueError('Both the means and the standard deviations should have values or None.')

        self.means = means
        self.stds = stds

    def __call__(self, *imgs, normalize_tags=None, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be normalized.
            normalize_tags (sequence of bool): The corresponding tags of the images (default: None, normalize all the images).

        Returns:
            imgs (tuple of numpy.ndarray): The normalized images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if normalize_tags:
            if len(normalize_tags) != len(imgs):
                raise ValueError('The number of the tags should be the same as the images.')
            if not all(normalize_tag in [True, False] for normalize_tag in normalize_tags):
                raise ValueError("All of the tags should be either True or False.")
        else:
            normalize_tags = [None] * len(imgs)

        _imgs = []
        for img, normalize_tag in zip(imgs, normalize_tags):
            if normalize_tag is None or normalize_tag is True:
                if self.means is None and self.stds is None: # Apply image-level normalization.
                    axis = tuple(range(img.ndim - 1))
                    means = img.mean(axis=axis)
                    stds = img.std(axis=axis)
                    img = self._normalize(img, means, stds)
                else:
                    img = self._normalize(img, self.means, self.stds)
            elif normalize_tag is False:
                pass
            _imgs.append(img)
        imgs = tuple(_imgs)
        return imgs

    @staticmethod
    def _normalize(img, means, stds):
        img = img.copy()
        for c, mean, std in zip(range(img.shape[-1]), means, stds):
            img[..., c] = (img[..., c] - mean) / std
        return img


class RandomCrop(BaseTransformer):
    """Crop a tuple of images at the same random location.
    Args:
        size (list): The desired output size of the crop.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, *imgs, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be croped.

        Returns:
            imgs (tuple of numpy.ndarray): The croped images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        ndim = imgs[0].ndim
        if ndim - 1 != len(self.size):
            raise ValueError(f'The dimensions of the crop size should be the same as the image ({ndim - 1}). Got {len(self.size)}')

        if ndim == 3:
            h0, hn, w0, wn = self._get_coordinates(imgs[0], self.size)
            imgs = tuple([img[h0: hn, w0: wn] for img in imgs])
        elif ndim == 4:
            h0, hn, w0, wn, d0, dn = self._get_coordinates(imgs[0], self.size)
            imgs = tuple([img[h0: hn, w0: wn, d0: dn] for img in imgs])
        return imgs

    @staticmethod
    def _get_coordinates(img, size):
        if any(i - j < 0 for i, j in zip(img.shape, size)):
            raise ValueError(f'The image ({img.shape}) is smaller than the crop size ({size}). Please use a smaller crop size.')

        if img.ndim == 3:
            h, w = img.shape[:-1]
            ht, wt = size
            h0, w0 = random.randint(0, h - ht), random.randint(0, w - wt)
            return h0, h0 + ht, w0, w0 + wt
        elif img.ndim == 4:
            h, w, d = img.shape[:-1]
            ht, wt, dt = size
            h0, w0, d0 = random.randint(0, h - ht), random.randint(0, w - wt), random.randint(0, d - dt)
            return h0, h0 + ht, w0, w0 + wt, d0, d0 + dt


class RandomElasticDeformation(BaseTransformer):
    """Do the random elastic deformation as used in U-Net and V-Net by using the bspline transform.
    Args:
        do_z_deformation (bool): Whether to apply the deformation along the z dimension (default: False).
        num_ctrl_points (int): The number of the control points to form the control point grid (default: 4).
        sigma (int or float): The number to determine the extent of deformation (default: 15).
        prob (float): The probability of applying the deformation (default: 0.5).
    """
    def __init__(self, do_z_deformation=False, num_ctrl_points=4, sigma=15, prob=0.5):
        self.do_z_deformation = do_z_deformation
        self.num_ctrl_points = max(num_ctrl_points, 2)
        self.sigma = max(sigma, 1)
        self.prob = max(0, min(prob, 1))
        self.bspline_transform = None

    def __call__(self, *imgs, elastic_deformation_orders=None, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be deformed.
            elastic_deformation_orders (sequence of int): The corresponding interpolation order of the images (default: None, the interpolation order would be 3 for all the images).

        Returns:
            imgs (tuple of numpy.ndarray): The deformed images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        if random.random() < self.prob:
            self._init_bspline_transform(imgs[0].shape)
            if elastic_deformation_orders:
                _imgs = []
                for img, order in zip(imgs, elastic_deformation_orders):
                    img = self._apply_bspline_transform(img, order)
                    _imgs.append(img)
                imgs = tuple(_imgs)
            else:
                imgs = map(self._apply_bspline_transform, imgs)
        return imgs

    def _init_bspline_transform(self, shape):
        """Initialize the bspline transform.
        Args:
            shape (tuple): The size of the control point grid.
        """
        # Remove the channel dimension.
        shape = shape[:-1]

        # Initialize the control point grid.
        img = sitk.GetImageFromArray(np.zeros(shape))
        mesh_size = [self.num_ctrl_points] * img.GetDimension()
        self.bspline_transform = sitk.BSplineTransformInitializer(img, mesh_size)

        # Set the parameters of the bspline transform randomly.
        params = self.bspline_transform.GetParameters()
        params = np.asarray(params, dtype=np.float64)
        params = params + np.random.randn(params.shape[0]) * self.sigma
        if len(shape) == 3 and not self.do_z_deformation:
            params[0: len(params) // 3] = 0
        params = tuple(params)
        self.bspline_transform.SetParameters(params)

    def _apply_bspline_transform(img, order=3):
        """Apply the bspline transform.
        Args:
            img (np.ndarray): The image to be deformed.
            order (int): The interpolation order (default: 3, should be 0, 1 or 3).

        Returns:
            img (np.ndarray): The deformed image.
        """
        # Create the resampler.
        resampler = sitk.ResampleImageFilter()
        if order == 0:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        elif order == 1:
            resampler.SetInterpolator(sitk.sitkLinear)
        elif order == 3:
            resampler.SetInterpolator(sitk.sitkBSpline)
        else:
            raise ValueError(f'The interpolation order should be 0, 1 or 3. Got {order}.')

        # Apply the bspline transform.
        shape = img.shape
        img = sitk.GetImageFromArray(np.squeeze(img))
        resampler.SetReferenceImage(img)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(self.bspline_transform)
        img = resampler.Execute(img)
        img = sitk.GetArrayFromImage(img).reshape(shape)
        return img
