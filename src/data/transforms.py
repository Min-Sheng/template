import torch
import random
import math
import numbers
import functools
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize, rescale, rotate

import src.data.transforms


def compose(transforms=None):
    """Compose several transforms together.
    Args:
        transforms (Box): The preprocessing and augmentation techniques applied to the data (default: None, only contain the default transform ToTensor).

    Returns:
        transforms (Compose): The list of BaseTransform.
    """
    if transforms is None:
        return Compose([ToTensor()])

    _transforms = []
    for transform in transforms:
        #print(transform)
        cls = getattr(src.data.transforms, transform.name)
        kwargs = transform.get('kwargs')
        _transforms.append(cls(**kwargs) if kwargs else cls())

    transforms = Compose(_transforms)
    return transforms


class BaseTransform:
    """The base class for all transforms.
    """
    def __call__(self, *imgs, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__


class Compose(BaseTransform):
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

        # Returns the torch.Tensor instead of a tuple of torch.Tensor if there is only one image.
        if len(imgs) == 1:
            imgs = imgs[0]
        return imgs

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(BaseTransform):
    """Convert a tuple of numpy.ndarray to a tuple of torch.Tensor.
    """
    def __call__(self, *imgs, dtypes=None, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be converted to tensor.
            dtypes (sequence of torch.dtype, optional): The corresponding dtype of the images (default: None, transform all the images' dtype to torch.float).

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


class Normalize(BaseTransform):
    """Normalize a tuple of images with the means and the standard deviations.
    Args:
        means (list, optional): A sequence of means for each channel (default: None).
        stds (list, optional): A sequence of standard deviations for each channel (default: None).
    """
    def __init__(self, means=None, stds=None):
        if means is None and stds is None:
            pass
        elif means is not None and stds is not None:
            if len(means) != len(stds):
                raise ValueError('The number of the means should be the same as the standard deviations.')
        else:
            raise ValueError('Both the means and the standard deviations should have values or None.')

        self.means = means
        self.stds = stds

    def __call__(self, *imgs, normalize_tags=None, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be normalized.
            normalize_tags (sequence of bool, optional): The corresponding tags of the images (default: None, normalize all the images).

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
                    self.means = img.mean(axis=axis)
                    self.stds = img.std(axis=axis)
                    img = self._normalize(img, self.means, self.stds)
                else:
                    img = self._normalize(img, self.means, self.stds)
            elif normalize_tag is False:
                pass
            _imgs.append(img)
        imgs = tuple(_imgs)
        return imgs

    @staticmethod
    def _normalize(img, means, stds):
        """Normalize the image with the means and the standard deviations.
        Args:
            img (numpy.ndarray): The image to be normalized.
            means (list): A sequence of means for each channel.
            stds (list): A sequence of standard deviations for each channel.

        Returns:
            img (numpy.ndarray): The normalized image.
        """
        img = img.copy()
        for c, mean, std in zip(range(img.shape[-1]), means, stds):
            img[..., c] = (img[..., c] - mean) / (std + 1e-10)
        return img


class Resize(BaseTransform):
    """Resize a tuple of images to the same size.
    Args:
        size (list): The desired output size of the resized images.
    """
    def __init__(self, size):
        self.size = size
        self._resize = functools.partial(resize, mode='constant', preserve_range=True)

    def __call__(self, *imgs, interpolation_orders=None, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be resized.
            interpolation_orders (sequence of int, optional): The corresponding interpolation order of the images (default: None, the interpolation order would be 1 for all the images).

        Returns:
            imgs (tuple of numpy.ndarray): The resized images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')
        
        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        ndim = imgs[0].ndim
        if ndim - 1 != len(self.size):
            raise ValueError(f'The dimensions of the resized size should be the same as the image ({ndim - 1}). Got {len(self.size)}')

        if interpolation_orders:
            imgs = tuple(self._resize(img, self.size, order).astype(img.dtype) for img, order in zip(imgs, interpolation_orders))
        else:
            imgs = tuple(self._resize(img, self.size) for img in imgs)
        return imgs

class RandomResize(BaseTransform):
    """Resize a tuple of images to a random size and aspect ratio.
    Args:
        scale (list, optional): The range of size of the origin size (default: 0.75 to 1.33).
        ratio (list, optional): The range of aspect ratio of the origin aspect ratio (default: 3/4, 4/3)
        prob  (float, optional): The probability of applying the resize (default: 0.5).
    """
    def __init__(self, scale=[0.75, 1.33], ratio=[3. / 4., 4. / 3.], prob=0.5):
        if len(scale) != 2:
            raise ValueError("Scale must be a sequence of len 2.")
        if len(ratio) != 2:
            raise ValueError("ratio must be a sequence of len 2.")
        self.scale = scale
        self.ratio = ratio
        self._rescale = functools.partial(rescale, mode='constant', preserve_range=True)
        self._resize = functools.partial(resize, mode='constant', preserve_range=True)
        self.prob = max(0, min(prob, 1))

    def __call__(self, *imgs, interpolation_orders=None, label_type=None, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be resized.
            interpolation_orders (sequence of int, optional): The corresponding interpolation order of the images (default: None, the interpolation order would be 1 for all the images).

        Returns:
            imgs (tuple of numpy.ndarray): The resized images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')
        
        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        if random.random() < self.prob:
            h, w = imgs[0].shape[:-1]
            area = h* w
            random_scale = random.uniform(*self.scale)
            target_area = random_scale * area
            
            if self.ratio[0] != 1.0 and self.ratio[1] != 1.0:
                resize_flag=True
                log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
                random_ratio = math.exp(random.uniform(*log_ratio))
                h = int(round(math.sqrt(target_area * random_ratio)))
                w = int(round(math.sqrt(target_area * random_ratio)))
            else:
                resize_flag=False
            if label_type!='watershed_label':
                if interpolation_orders:
                    imgs = tuple(self._rescale(img, random_scale, order, multichannel=True).astype(img.dtype) for img, order in zip(imgs, interpolation_orders))
                    if resize_flag:
                        imgs = tuple(self._resize(img, (h, w), order).astype(img.dtype) for img, order in zip(imgs, interpolation_orders))
                else:
                    imgs = tuple(self._rescale(img, random_scale, multichannel=True) for img in imgs)
                    if resize_flag:
                        imgs = tuple(self._resize(img, (h, w)) for img in imgs)
            else: 
                if interpolation_orders:
                    new_imgs = []
                    for i, (img, order) in enumerate(zip(imgs, interpolation_orders)):
                        if i == 2 or i == 3:
                            img_1 = self._rescale(img[...,0:2], random_scale, order=1, multichannel=True).astype(img.dtype)
                            img_2 = self._rescale(img[...,2], random_scale, order=order, multichannel=False).astype(img.dtype)
                            img = np.dstack((img_1, img_2[...,None]))
                            if resize_flag:
                                img_1 = self._resize(img[...,0:2], (h,w), order=1).astype(img.dtype)
                                img_2 = self._resize(img[...,2], (h,w), order=order).astype(img.dtype)
                                img = np.dstack((img_1, img_2[...,None]))
                            new_imgs.append(img)
                        else:
                            img = self._rescale(img, random_scale, order, multichannel=True).astype(img.dtype) 
                            if resize_flag:
                                img = self._resize(img, (h, w), order).astype(img.dtype) 
                            new_imgs.append(img)
                    imgs = tuple(new_imgs)
                else:
                    imgs = tuple(self._rescale(img, random_scale, multichannel=True) for img in imgs)
                    if resize_flag:
                        imgs = tuple(self._resize(img, (h, w)) for img in imgs)
        return imgs

class RandomRotation(BaseTransform):
    """Rotate a tuple of images to a random degree.
    Args:
        degrees (list, optional): The range of degrees to select from (default: -30 to 30).
            If degrees is a nmber instead of a list like [min, max], the range of degrees will be [-degree, degree]
        center (list, optional): Optional center of rotation (default is the center of the image).
            Origin is the upper left corner.
        prob  (float, optional): The probability of applying the resize (default: 0.5).
    """
    def __init__(self, degrees=[-30, 30], center=None, prob=0.5):
        
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = [-degrees, degrees]
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.center = center
        self._rotate = functools.partial(rotate, mode='constant', preserve_range=True)
        self.prob = prob
    def __call__(self, *imgs, interpolation_orders=None, label_type=None,  **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be resized.
            interpolation_orders (sequence of int, optional): The corresponding interpolation order of the images (default: None, the interpolation order would be 1 for all the images).

        Returns:
            imgs (tuple of numpy.ndarray): The rotated images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')
        
        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")
        
        if random.random() < self.prob:
            
            angle = random.uniform(*self.degrees)
            if label_type != 'watershed_label':
                if interpolation_orders:
                    imgs = tuple(self._rotate(img, angle, order=order, resize=False).astype(img.dtype) for img, order in zip(imgs, interpolation_orders))
                else:
                    imgs = tuple(self._rotate(img, angle, resize=False) for img in imgs)
            else:
                if interpolation_orders:
                    new_imgs = []
                    for i, (img, order) in enumerate(zip(imgs, interpolation_orders)):
                        if i == 2 or i == 3:
                            img_1 = self._rotate(img[...,0:2], angle, order=1, resize=False).astype(img.dtype)
                            img_1[...,0] = img_1[...,0] * np.cos(angle * np.pi / 180) - img_1[...,1] * np.sin(angle * np.pi / 180)
                            img_1[...,1] = img_1[...,0] * np.sin(angle * np.pi / 180) + img_1[...,1] * np.cos(angle * np.pi / 180)
                            img_2 = self._rotate(img[...,2], angle, order=order, resize=False).astype(img.dtype)
                            img = np.dstack((img_1, img_2[...,None]))
                            new_imgs.append(img)
                        else:
                            img = self._rotate(img, angle, order=order, resize=False).astype(img.dtype)
                            new_imgs.append(img)
                    imgs=tuple(new_imgs)
                else:
                    new_imgs = []
                    for i, img in enumerate(imgs):
                        if i == 2 or i == 3:
                            img_1 = self._rotate(img[...,0:2], angle, resize=False).astype(img.dtype)
                            img_1[...,0] = img_1[...,0] * np.cos(angle * np.pi / 180) - img_1[...,1] * np.sin(angle * np.pi / 180)
                            img_1[...,1] = img_1[...,0] * np.sin(angle * np.pi / 180) + img_1[...,1] * np.cos(angle * np.pi / 180)
                            img_2 = self._rotate(img[...,2], angle, resize=False).astype(img.dtype)
                            img = np.dstack((img_1, img_2[...,None]))
                            new_imgs.append(img)
                        else:
                            img = self._rotate(img, angle, resize=False).astype(img.dtype)
                            new_imgs.append(img)
                    imgs=tuple(new_imgs)
        return imgs

class RandomHorizontalFlip(BaseTransform):
    """Do the random flip horizontally.
    Args:
        prob (float, optional): The probability of applying the flip (default: 0.5).
    """
    def __init__(self, prob=0.5):
        self.prob = max(0, min(prob, 1))

    def __call__(self, *imgs, label_type=None, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be flipped.
        Returns:
            imgs (tuple of numpy.ndarray): The flipped images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        if random.random() < self.prob:
            if label_type == 'watershed_label':
                new_imgs=[]
                for i, img in enumerate(imgs):
                    img = np.flip(img , 1).copy()
                    if i == 2 or i == 3:
                        img[...,1] = -img[...,1]
                    new_imgs.append(img)
                imgs=tuple(new_imgs)
            else:
                imgs = tuple([np.flip(img, 1).copy() for img in imgs])
        return imgs
    
    
class RandomVerticalFlip(BaseTransform):
    """Do the random flip vertically.
    Args:
        prob (float, optional): The probability of applying the flip (default: 0.5).
    """
    def __init__(self, prob=0.5):
        self.prob = max(0, min(prob, 1))

    def __call__(self, *imgs, label_type=None,  **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be flipped.
        Returns:
            imgs (tuple of numpy.ndarray): The flipped images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        if 1:#random.random() < self.prob:
            if label_type == 'watershed_label':
                new_imgs=[]
                for i, img in enumerate(imgs):
                    img = np.flip(img , 0).copy()
                    if i == 2 or i == 3:
                        img[...,0] = -img[...,0]
                    new_imgs.append(img)
                imgs=tuple(new_imgs)
            else:
                imgs = tuple([np.flip(img, 0).copy() for img in imgs])
        return imgs

class RandomCrop(BaseTransform):
    """Crop a tuple of images at the same random location.
    Args:
        size (list): The desired output size of the cropped images.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, *imgs, **kwargs):
        """
        Args:
            imgs (tuple of numpy.ndarray): The images to be cropped.

        Returns:
            imgs (tuple of numpy.ndarray): The cropped images.
        """
        if not all(isinstance(img, np.ndarray) for img in imgs):
            raise TypeError('All of the images should be numpy.ndarray.')

        if not all(img.ndim == 3 for img in imgs) and not all(img.ndim == 4 for img in imgs):
            raise ValueError("All of the images' dimensions should be 3 (2D images) or 4 (3D images).")

        ndim = imgs[0].ndim
        if ndim - 1 != len(self.size):
            raise ValueError(f'The dimensions of the cropped size should be the same as the image ({ndim - 1}). Got {len(self.size)}')
        
        if ndim == 3:
            new_img_h0, new_img_hn, new_img_w0, new_img_wn, img_h0, img_hn, img_w0, img_wn = self._get_coordinates(imgs[0], self.size)
            new_imgs=[]
            for img in imgs:
                new_img = np.zeros((self.size[0], self.size[1], img.shape[-1]), img.dtype)
                new_img[new_img_h0: new_img_hn, new_img_w0: new_img_wn] = img[img_h0: img_hn, img_w0: img_wn]
                new_imgs.append(new_img)
            imgs = tuple(new_imgs)
        elif ndim == 4:
            new_img_h0, new_img_hn, new_img_w0, new_img_wn, new_img_d0, new_img_dn, img_h0, img_hn, img_w0, img_wn, img_d0, img_dn = self._get_coordinates(imgs[0], self.size)
            new_imgs=[]
            for img in imgs:
                new_img = np.zeros((self.size[0], self.size[1], self.size[2], img.shape[-1]), img.dtype)
                new_img[new_img_h0: new_img_hn, new_img_w0: new_img_wn, new_img_d0: new_img_dn] = img[img_h0: img_hn, img_w0: img_wn, img_d0: img_dn]
                new_imgs.append(new_img)
            imgs = tuple(new_imgs)
        return imgs

    @staticmethod
    def _get_coordinates(img, size):
        """Compute the coordinates of the cropped image.
        Args:
            img (numpy.ndarray): The image to be cropped.
            size (list): The desired output size of the cropped image.

        Returns:
            coordinates (tuple): The coordinates of the cropped image.
        """
        #if any(i - j < 0 for i, j in zip(img.shape, size)):
        #    raise ValueError(f'The image ({img.shape}) is smaller than the cropped size ({size}). Please use a smaller cropped size.')

        if img.ndim == 3:
            h, w = img.shape[:-1]
            ht, wt = min(size[0], h), min(size[1], w)
            h_space, w_space = h - size[0], w -size[1]
            
            if h_space > 0:
                new_img_h0 = 0
                img_h0 = random.randrange(h_space + 1)
            else:
                new_img_h0 = random.randrange(-h_space + 1)
                img_h0 = 0
            if w_space > 0:
                new_img_w0 = 0
                img_w0 = random.randrange(w_space + 1)
            else:
                new_img_w0 = random.randrange(-w_space + 1)
                img_w0 = 0

            return new_img_h0, new_img_h0 + ht, new_img_w0, new_img_w0 + wt, img_h0, img_h0 + ht, img_w0, img_w0 + wt

        elif img.ndim == 4:
            h, w, d = img.shape[:-1]
            ht, wt, dt = min(size[0], h), min(size[1], w), min(size[2], d)
            h_space, w_space, d_space = h - size[0], w -size[1], d- size[2] 
            
            if h_space > 0:
                new_img_h0 = 0
                img_h0 = random.randrange(h_space + 1)
            else:
                new_img_h0 = random.randrange(-h_space + 1)
                img_h0 = 0
            if w_space > 0:
                new_img_w0 = 0
                img_w0 = random.randrange(w_space + 1)
            else:
                new_img_w0 = random.randrange(-w_space + 1)
                img_w0 = 0
            if d_space > 0:
                new_img_d0 = 0
                img_d0 = random.randrange(d_space + 1)
            else:
                new_img_d0 = random.randrange(-d_space + 1)
                img_d0 = 0

            return new_img_h0, new_img_h0 + ht, new_img_w0, new_img_w0 + wt, new_img_d0, new_img_do + dt, img_h0, img_h0 + ht, img_w0, img_w0 + wt, img_d0, img_d0 + dt


class RandomElasticDeformation(BaseTransform):
    """Do the random elastic deformation as used in U-Net and V-Net by using the bspline transform.
    Args:
        do_z_deformation (bool, optional): Whether to apply the deformation along the z dimension (default: False).
        num_ctrl_points (int, optional): The number of the control points to form the control point grid (default: 4).
        sigma (int or float, optional): The number to determine the extent of deformation (default: 15).
        prob (float, optional): The probability of applying the deformation (default: 0.5).
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
            elastic_deformation_orders (sequence of int, optional): The corresponding interpolation order of the images (default: None, the interpolation order would be 3 for all the images).

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
                imgs = tuple(self._apply_bspline_transform(img, order) for img, order in zip(imgs, elastic_deformation_orders))
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

    def _apply_bspline_transform(self, img, order=3):
        """Apply the bspline transform.
        Args:
            img (np.ndarray): The image to be deformed.
            order (int, optional): The interpolation order (default: 3, should be 0, 1 or 3).

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
