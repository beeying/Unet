from PIL import Image
import cv2
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

        # top = np.random.randint(0, h - new_h)
        # left = np.random.randint(0, w - new_w)
        top = 200
        left = 200

        image = image[top: top + new_h,
                      left: left + new_w]

        mask = mask[top: top + new_h, left: left + new_w]
        return torch.Tensor(image), torch.Tensor(mask)


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
        new_image = new_image.reshape((256, 256, 1))/ 255
        # new_image = np.concatenate((new_image,new_image,new_image), axis=2)

        return torch.Tensor(new_image), torch.Tensor(new_mask.reshape((256, 256, 1)) / 255)


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


class RandomRotateb4Crop:
    def __init__(self, max_rotation):
        self.max_rotation = max_rotation
        self.img_rows = 512
        self.img_cols = 512

    def __call__(self, image, mask):
        cen_rows = image.shape[0] // 2
        cen_cols = image.shape[1] // 2
        width = 256
        height = 256
        while True:
            center = (np.random.randint(cen_rows - 10, cen_rows + 10), np.random.randint(cen_cols - 10, cen_cols + 10))
            angle = np.random.randint(low=-self.max_rotation, high=self.max_rotation)
            rect = (center, (width, height), angle)
            if self._inside_rect(rect=rect):
                break
        rotated_im = self._crop_rotated_rectangle(image = image, rect = rect)
        rotated_mask = self._crop_rotated_rectangle(image = mask, rect = rect)
        new_image = rotated_im.reshape((256, 256, 1))
        # new_image = np.concatenate((new_image, new_image, new_image), axis=2)
        return torch.Tensor(new_image), torch.Tensor(rotated_mask.reshape((256, 256, 1)))

    def _inside_rect(self, rect):
        '''
            Determine if the four corners of the rectangle are inside the rectangle with width and height
            rect tuple
            center (x,y), (width, height), angle of rotation (to the row)
            center  The rectangle mass center.
            center tuple (x, y): x is regarding to the width (number of columns) of the image, y is regarding to the height (number of rows) of the image.
            size    Width and height of the rectangle.
            angle   The rotation angle in a clockwise direction. When the angle is 0, 90, 180, 270 etc., the rectangle becomes an up-right rectangle.
            Return:
            True: if the rotated sub rectangle is side the up-right rectange
            False: else
        '''

        rect_center = rect[0]
        rect_center_x = rect_center[0]
        rect_center_y = rect_center[1]

        if (rect_center_x < 0) or (rect_center_x > self.img_cols):
            return False
        if (rect_center_y < 0) or (rect_center_y > self.img_rows):
            return False

        # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
        box = cv2.boxPoints(rect)

        x_max = int(np.max(box[:, 0]))
        x_min = int(np.min(box[:, 0]))
        y_max = int(np.max(box[:, 1]))
        y_min = int(np.min(box[:, 1]))

        if (x_max <= self.img_cols) and (x_min >= 0) and (y_max <= self.img_rows) and (y_min >= 0):
            return True
        else:
            return False
#
#
    def _rect_bbx(self, rect):
        '''
        Rectangle bounding box for rotated rectangle
        Example:
        rotated rectangle: height 4, width 4, center (10, 10), angle 45 degree
        bounding box for this rotated rectangle, height 4*sqrt(2), width 4*sqrt(2), center (10, 10), angle 0 degree
        '''

        box = cv2.boxPoints(rect)

        x_max = int(np.max(box[:, 0]))
        x_min = int(np.min(box[:, 0]))
        y_max = int(np.max(box[:, 1]))
        y_min = int(np.min(box[:, 1]))

        center = (int((x_min + x_max) // 2), int((y_min + y_max) // 2))
        width = int(x_max - x_min)
        height = int(y_max - y_min)
        angle = 0

        return (center, (width, height), angle)


    def _image_rotate_without_crop(self, mat, angle):
        # https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
        # angle in degrees

        height, width = mat.shape[:2]
        image_center = (width / 2, height / 2)

        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

        return rotated_mat


    def _crop_rectangle(self, image, rect):
        # rect has to be upright

        num_rows = image.shape[0]
        num_cols = image.shape[1]

        if not self._inside_rect(rect=rect):
            print("Proposed rectangle is not fully in the image.")
            return None

        rect_center = rect[0]
        rect_center_x = rect_center[0]
        rect_center_y = rect_center[1]
        rect_width = rect[1][0]
        rect_height = rect[1][1]

        return image[rect_center_y - rect_height // 2:rect_center_y + rect_height - rect_height // 2,
               rect_center_x - rect_width // 2:rect_center_x + rect_width - rect_width // 2]
#
#
    def _crop_rotated_rectangle(self, image, rect):
        # Crop a rotated rectangle from a image

        if not self._inside_rect(rect=rect):
            print("Proposed rectangle is not fully in the image.")
            return None

        rotated_angle = rect[2]

        rect_bbx_upright = self._rect_bbx(rect=rect)
        rect_bbx_upright_image = self._crop_rectangle(image=image, rect=rect_bbx_upright)

        rotated_rect_bbx_upright_image = self._image_rotate_without_crop(mat=rect_bbx_upright_image, angle=rotated_angle)

        rect_width = rect[1][0]
        rect_height = rect[1][1]

        crop_center = (rotated_rect_bbx_upright_image.shape[1] // 2, rotated_rect_bbx_upright_image.shape[0] // 2)

        return rotated_rect_bbx_upright_image[
               crop_center[1] - rect_height // 2: crop_center[1] + (rect_height - rect_height // 2),
               crop_center[0] - rect_width // 2: crop_center[0] + (rect_width - rect_width // 2)]
#
#