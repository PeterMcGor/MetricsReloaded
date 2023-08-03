import numpy as np
from monai.transforms.utils import rescale_array

def create_test_image_3d(
    height: int,
    width: int,
    depth: int,
    num_objs: int = 12,
    rad_max: int = 30,
    rad_min: int = 5,
    noise_max: float = 0.0,
    num_seg_classes: int = 5,
    channel_dim: int | None = None,
    random_state: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a noisy 3D image and segmentation.
    Args:
        height: height of the image. The value should be larger than `2 * rad_max`.
        width: width of the image. The value should be larger than `2 * rad_max`.
        depth: depth of the image. The value should be larger than `2 * rad_max`.
        num_objs: number of circles to generate. Defaults to `12`.
        rad_max: maximum circle radius. Defaults to `30`.
        rad_min: minimum circle radius. Defaults to `5`.
        noise_max: if greater than 0 then noise will be added to the image taken from
            the uniform distribution on range `[0,noise_max)`. Defaults to `0`.
        num_seg_classes: number of classes for segmentations. Defaults to `5`.
        channel_dim: if None, create an image without channel dimension, otherwise create
            an image with channel dimension as first dim or last dim. Defaults to `None`.
        random_state: the random generator to use. Defaults to `np.random`.
    Returns:
        Randomised Numpy array with shape (`height`, `width`, `depth`)
    See also:
        :py:meth:`~create_test_image_2d`
    """

    if rad_max <= rad_min:
        raise ValueError(f"`rad_min` {rad_min} should be less than `rad_max` {rad_max}.")
    if rad_min < 1:
        raise ValueError("f`rad_min` {rad_min} should be no less than 1.")
    min_size = min(height, width, depth)
    if min_size <= 2 * rad_max:
        raise ValueError(f"the minimal size {min_size} of the image should be larger than `2 * rad_max` 2x{rad_max}.")

    image = np.zeros((height, width, depth))
    rs: np.random.RandomState = np.random.random.__self__ if random_state is None else random_state  # type: ignore

    for _ in range(num_objs):
        x = rs.randint(rad_max, height - rad_max)
        y = rs.randint(rad_max, width - rad_max)
        z = rs.randint(rad_max, depth - rad_max)
        rad = rs.randint(rad_min, rad_max)
        spy, spx, spz = np.ogrid[-x : height - x, -y : width - y, -z : depth - z]
        circle = (spx * spx + spy * spy + spz * spz) <= rad * rad

        if num_seg_classes > 1:
            image[circle] = np.ceil(rs.random() * num_seg_classes)
        else:
            image[circle] = rs.random() * 0.5 + 0.5

    labels = np.ceil(image).astype(np.int32, copy=False)

    norm = rs.uniform(0, num_seg_classes * noise_max, size=image.shape)
    noisyimage: np.ndarray = rescale_array(np.maximum(image, norm))  # type: ignore

    if channel_dim is not None:
        if not (isinstance(channel_dim, int) and channel_dim in (-1, 0, 3)):
            raise AssertionError("invalid channel dim.")
        noisyimage, labels = (
            (noisyimage[None], labels[None]) if channel_dim == 0 else (noisyimage[..., None], labels[..., None])
        )

    return noisyimage, labels