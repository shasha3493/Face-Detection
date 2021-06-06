import cv2
import numpy as np
import random
from utils.box_utils import matrix_iof
import sys

def _crop(image, boxes, labels, img_dim):
    '''
    @params
    image: (height, width,3)
    boxes: 2d numpy array where each row represents a bounding box in an image containing face
    labels: 1d array of all 1s
    img_dim = 1024

    Returns:
    image_t: cropped image (width, height, 3)
    boxes_t: 2d numpy array of shape (num_faces (less thanequal to original num_faces), 4), converting xmin,ymin,xmiz,ymax of boxes_t relative to the the roi
    labels_t: 1d numpy array of all 1s (less than equal to num_faces)
    pad_image_flag: False


    1. Scale of the square crop is chosen randomly
    2. Location of the square crop is chosen randomly
    3. bboxes coordinates are scaled as per roi
    4. min(width, height) of bbox scaled to 1024 is 16 pixels (min face dimensions)

    Randomly square crops the image, but only those boxes are returned such that roi contains the bbox.

    So loops runs for max 250 times to meet the above criterias and returns as soon as it is met. If the above criterias are not met
    in 250 times it returns original image and bbox coordinates.


    '''

    height, width, _ = image.shape # actual width, height of an image,however,img_dim = 1024
    pad_image_flag = True

    for _ in range(250):

        # randomly picking up a value of scale
        if random.uniform(0, 1) <= 0.2:
            scale = 1
        else:
            scale = random.uniform(0.3, 1.)
        short_side = min(width, height) # shorted side
        w = int(scale * short_side) # scaling the shorter side by 'scale'
        h = w # square image. This section of the image will be cropped out. 

        # Now the location from which the above h*w will be cropped out is again decided randomly.
        # A caveat is that if w == width, entire width of the image has to be kept. Similarly,
        # if h == height, entire height of the image has to be kept.
        if width == w: # possible when short side = height and scale = width/height
            l = 0
        else:
            l = random.randrange(width - w) # random integer 0 <= random integer < width - w
        
        if height == h: # possible when short side = width and scale = height/width
            t = 0
        else:
            t = random.randrange(height - h) # random integer 0 <= random integer < height - h
        roi = np.array((l, t, l + w, t + h)) # region of interest (xmin, ymin, xmax, ymax) to be crooped
        value = matrix_iof(boxes, roi[np.newaxis]) # roi[np.newaxis] adds a new axis at index 0, so shape becomes (1,4)# value is intersection area between bbox and roi/area of bbox, (number of faces, 1)
        flag = (value >= 1)
        if not flag.any(): # no roi conatains the bbox completely.
            continue

        # if roi contains the bbox completely 
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2 # centre of the bbox, (number of faces, 2)

        
        # If the roi contains atleast a single bbox completely
        # Out of all bbox, we select only those whose whoe centre of bbox is included in the roi and then we crop the image
        # and scale the bbox coords as per the roi
        
        # roi[:2] < centers and centers < roi[2:] are of shape (number of faces, 2). Basically it looks if the
        # roi includes centre of bbox i.e np.logical_and does element wise and to satify both the condition.
        # .all(axis = 1) is an 1d array of shape (number of faces, ) which is true only if both the conditions are 
        # satisfied for a particular bbox.

        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)

        # Picking only those bounding boxes and labels
        boxes_t = boxes[mask_a].copy() # 2d array
        labels_t = labels[mask_a].copy() # 1d array of all 1's
        if boxes_t.shape[0] == 0: # if no boxes satify we continue
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]] # cropping the image

        # converting xmin,ymin,xmiz,ymax of boxes_t relative to the the roi i.e. if xmin of bbox is same as 
        # xmin of roi, then the new xmin of bbox will be 0 since image has been cropped to roi
        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

	    # make sure that the cropped image contains at least one face > 16 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim # bbox width relative to 1024
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim # bbox height relative to 1024
        mask_b = np.minimum(b_w_t, b_h_t) > 16.0
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False

        return image_t, boxes_t, labels_t, pad_image_flag
    return image, boxes, labels, pad_image_flag


def _distort(image):

    def _convert(image, alpha=1, beta=0): # linear tranformation of an image
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def _expand(image, boxes, fill, p):
    if random.randrange(2):
        return image, boxes

    height, width, depth = image.shape

    scale = random.uniform(1, p)
    w = int(scale * width)
    h = int(scale * height)

    left = random.randint(0, w - width)
    top = random.randint(0, h - height)

    boxes_t = boxes.copy()
    boxes_t[:, :2] += (left, top)
    boxes_t[:, 2:] += (left, top)
    expand_image = np.empty(
        (h, w, depth),
        dtype=image.dtype)
    expand_image[:, :] = fill
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    return image, boxes_t


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1] # inverting the image upside down
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2] # corresponding bbox
    return image, boxes


def _pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def _resize_subtract_mean(image, insize, rgb_mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= rgb_mean
    return image.transpose(2, 0, 1)


class preproc(object):

    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim # 1024
        self.rgb_means = rgb_means

    def __call__(self, image, targets):
        '''
        @params:
        image: of shape (height, width,3)
        targets: 2d numpy array of shape (num_faces, 5) where each row represents (xmin, ymin, xmax, ymax, label) label = 1 always for a face
        
        Returns:

        
        '''

        assert targets.shape[0] > 0, "this image does not have gt" # if there are no labels (i.e no bbox for the image)
        # Making copies so that changes doesn't reflect to the original array
        boxes = targets[:, :-1].copy() # 2d numpy array where each row represents a bounding box in an image containing face
        labels = targets[:, -1].copy() # 1d array of all 1's representing labels for face

        #image_t = _distort(image)
        #image_t, boxes_t = _expand(image_t, boxes, self.cfg['rgb_mean'], self.cfg['max_expand_ratio'])
        #image_t, boxes_t, labels_t = _crop(image_t, boxes, labels, self.img_dim, self.rgb_means)

        # image_t: random;y cropped image (width, height, 3)
        # boxes_t: 2d numpy array of shape (num_faces (less than or equal to original num_faces), 4), converting xmin,ymin,xmiz,ymax of boxes_t relative to the the roi but not scaled to 0-1
        # labels_t: 1d numpy array of all 1s (less than equal to num_faces)
        # pad_image_flag: true
        
        image_t, boxes_t, labels_t, pad_image_flag = _crop(image, boxes, labels, self.img_dim)
        image_t = _distort(image_t) # passes the cropped image, brighness, contrast, hue, saturation distortion
        image_t = _pad_to_square(image_t,self.rgb_means, pad_image_flag) # does nothing as pad_image_flag = False since it's already an sqaure crop
        image_t, boxes_t = _mirror(image_t, boxes_t) # randomly inverst the image (left-right) and corresponding bbox
        height, width, _ = image_t.shape
        image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means) # resizes the image to 1024*1024, subtract the measns and transposes it, shape is (3,1024,1024)
        boxes_t[:, 0::2] /= width # previously bbox coordinates was with respec to the crop size of the image (height, width), rescaling it to 0-1
        boxes_t[:, 1::2] /= height
        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, labels_t))
        return image_t, targets_t
        # image_t is of shape (3,1024,1024) and targets_t is of shape (number of labels, 5) where 5 is xmin, ymin, xmax, ymax(all in 0-1 scale), 1
