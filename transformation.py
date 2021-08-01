from PIL import Image
import numpy as np
import torch


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, mask):

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        mask = mask[top: top + new_h, left: left + new_w]
        return image, mask


class RandomBrightness(object):
    """Adding noises randomly to the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, std):
        self.std = std

    def __call__(self, image, mask):
        noise_val = np.random.uniform(0, self.std, size=image.shape)

        image = image + torch.Tensor(noise_val.astype(np.float32))
        image = (image - image.min()) / (image.max() - image.min())

        return image, mask


class Translate_and_Rotate(object):
    """Translate and rotate images randomly.

        Args:
            p_translate (tuple or int): Desired output size. If int, square crop
                is made.
            degree
        """
    """Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_xtranslation, max_ytranslation, max_rotation, flip):
        self.max_xtranslation = max_xtranslation
        self.max_ytranslation = max_ytranslation
        self.max_rotation = max_rotation
        self.flip = flip * 100

    def __call__(self, old, old_mask):
        xtranslation = np.random.randint(-self.max_xtranslation,
                                         self.max_xtranslation,
                                         size=1)
        ytranslation = np.random.randint(-self.max_ytranslation,
                                         self.max_ytranslation,
                                         size=1)
        degree = np.random.randint(-self.max_rotation,
                                   self.max_rotation,
                                   size=1)

        old_image = Image.fromarray(255 * old.reshape((256, 256)))
        old_mask = Image.fromarray(255 * old_mask.reshape((256, 256)))
        xsize, ysize = old_image.size

        new_image = Image.new("L", (xsize, ysize))
        new_mask = Image.new("L", (xsize, ysize))

        new_image.paste(old_image, box=None)
        new_mask.paste(old_mask, box=None)

        new_image = new_image.rotate(
            degree, translate=(xtranslation, ytranslation))
        new_mask = new_mask.rotate(
            degree, translate=(xtranslation, ytranslation))
        chance = np.random.randint(0, 100, size=1)
        if chance < self.flip:
            new_image = new_image.transpose(Image.FLIP_LEFT_RIGHT)
            new_mask = new_mask.transpose(Image.FLIP_LEFT_RIGHT)
        new_image = np.array(new_image).astype(np.float32)
        new_mask = np.array(new_mask).astype(np.float32)

        pixel_coor = np.where(new_image == 0)
        new_image[pixel_coor] = 255 * old[10, 10]

        return torch.Tensor(new_image.reshape((256, 256, 1)) / 255), torch.Tensor(new_mask.reshape((256, 256, 1)) / 255)


class RandomTranslateWithReflect:
    """Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image
