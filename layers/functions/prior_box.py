import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        #self.aspect_ratios = cfg['aspect_ratios']
        # Aspect ratio of anchors is 1:1  because the face box is approximately square
        self.min_sizes = cfg['min_sizes'] # [[32, 64, 128], [256], [512]] scale of 5 default anchor boxes
        self.steps = cfg['steps'] # [32, 64, 128] interval of anchor boxes
        self.clip = cfg['clip'] # False
        self.image_size = image_size # (1024, 1024) During testing it's the resized height and widht of the image
        # The input image of size (1024, 1024) is divided into tiles of shape [32, 32], [64, 64], [128,128] respectivey. Each tile will have some anchors attached to it. In other words, model will focus on these tiles of different scale.
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps] #[[32, 32], [16, 16], [8,8]], these represent the number of tiles. 


    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k] # size of the anchors
            
            for i, j in product(range(f[0]), range(f[1])):
                
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1] # width of the bbox (normalized to 0-1 scale)
                    s_ky = min_size / self.image_size[0] # height of the bbox (normalized to 0-1 scale)
                    
                    '''
                    For anchors of scale 32, we have to densify 4 times => each tile hould have now 4*4 = 16 anchors of scale 32.
                    For every tile of size 32*32, there are 4 values of dense_cx (normalized top left x coordinate) and 4 values of dense_cy
                    (normalized top left y coordinate) and s_kx, s_ky are the normalized width and height repectively. So using this
                    we can have in total 16 anchor boxes when we take the cartesian product of dense_cx and dense_cy. 
                    '''
                    # Even if the last anchor box extends beyond the right and bottom edge, if the object lies at corners, last anchor box would scale to fit the actual bbox
                    if min_size == 32: # densify 4 times => 16 anchors per 32*32 tile
                        dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.25, j+0.5, j+0.75]] # (top left x coordinate normalized to 0-1 scale)
                        dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.25, i+0.5, i+0.75]] # (top left y coordinate normalized to 0-1 scale)
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky] # anchors attributes (top left x, top left y, width, height)

                    elif min_size == 64: # densify 2 times => 4 anchors per 32*32 tile
                        dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.5]]
                        dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]

                    else: # densify 1 times => 1 anchors per 32*32, 64*64, 128*128 tile
                        cx = (j + 0.5) * self.steps[k] #/ self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] #/ self.image_size[0]
                        anchors += [cx, cy, s_kx, s_ky]
         
        
        '''
        First Scale:
        Each tile of size 32*32 will have 16 anchors of 32, 4 anchors of scale 64 and 1 anchor of scale 128 (total 21 anchors).
        There are 32*32 total numer of such tiles. So total anchors would be 32*32*21.

        Second Scale:
        Each tile of size 64*64 would have 1 anchor of scale 256 and there are 16*16 such tiles. So tota number of anchors would 
        be 16*16*1.

        Third Scale:
        Each tile of size 128*128 would have 1 anchor of scale 512 and there are 8*8 such tiles. So total number of anchors would 
        be 8*8*1.

        output shape = (32*32*21 + 16*16 + 8*8, 4) = (21824, 4)

        Note: Top left coordinate and width and height are normalized to 0-1 scale.

        As the size of tile increases, scale of the anchor boxes also increase
        '''
        
        # back to torch land
        # shape (num of anchor boxes at all scales(21824), 4) in the form (top left x, top left y, width, height), all attributes are normalized to 0-1 scale
        output = torch.Tensor(anchors).view(-1, 4) 

        # This doesn't execute as self.execute = False

        if self.clip:
            output.clamp_(max=1, min=0)
        return output
