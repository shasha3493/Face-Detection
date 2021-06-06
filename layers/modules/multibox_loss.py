import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp
from data import cfg
GPU = cfg['gpu_train']

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) ' orboxes' that have jaccard index (IoU) > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes # 2
        self.threshold = overlap_thresh # 0.35
        self.background_label = bkg_label # 0
        self.encode_target = encode_target # False
        self.use_prior_for_matching = prior_for_matching # True
        self.do_neg_mining = neg_mining # True
        self.negpos_ratio = neg_pos # 7
        self.neg_overlap = neg_overlap # 0.35
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data = predictions # (batch_size, num_priors, 4), (batch_size, num_priors, num_classes), # these predictions represent the encoded predictions which will be compared against the expected predictions loc_t and conf_t for each anchor box for an image
        priors = priors
        num = loc_data.size(0) # batch_size
        num_priors = (priors.size(0)) # num_priors = 21824

        # match priors (default boxes) and ground truth boxes
        # these are to store ground truth labels for all anchor boxes for all image
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data # 2D tensor of shape (number of faces, 4) for an image
            labels = targets[idx][:, -1].data # 1D tensor of shape (1)
            defaults = priors.data # (21824, 4)
            
            # Inside match function, for an image, loc_t is of shape (num_priors, 4) where columns represent gt for each prior (i.e the prior with which the jaccard overlap with the gt bbox was max will have the coords as the gt coords. For others, priors coords are same as the prefixed ones. It wouldn't matter as the labels for those would be 0)
            # conf_t for an image is of shape (num_priors,) which has the value 1 for the prior haveing the max jaccard overlap with the gt box else 0

            # Basically we pair ground truths and default boxes, and marking the remaining default boxes as background,

            # This happens for all the images in a batch one by one. Therefore, eventually, loc_t is of shape (batch_size, num_priors, 4) which represents the expected coords for all anchor box for every image.
            # Similarly, conf_t is of shape (batch_size, num_priors) which represents expected label for all the anchor boxes of all the images
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        pos = conf_t > 0 # locations of priors which had the max jaccard overlap with gt boxes, (batch_size, num_priors)


        # https://towardsdatascience.com/learning-note-single-shot-multibox-detector-with-pytorch-part-3-f0711caa65ad

        # Localization Loss (Smooth L1): The localization loss is calculated only on positive boxes (ones with a matched ground truth). For other boxes, it's trained to give a very low confodence so during testuing others will be filtered out on NMS
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data) # shape: (batch_size,21824,4) # all coumns are same, value of all columns are T if the particular anchor box has the max jaccard  overlap any of the gt box
        loc_p = loc_data[pos_idx].view(-1, 4) # filtering out encoded bbox predictions where conf = 1 (i.e. prescence of a face)
        loc_t = loc_t[pos_idx].view(-1, 4) # filtering out true encoded bbox where conf = 1 (i.e. prescence of a face)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum') # L1 loss

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes) # (batch_size*num_priors, 2)
        
        # Lconf = -log(exp(logits)/sum(exp(logits)))  => log(sum(exp of logits)) - logit
        # batch_conf.gather(1, conf_t.view(-1, 1)) picks up the logit for correct label i.e. if the true label is 0 it gathers the logit at index 0 and at index 1 if the true logit  = 1
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1)) # it is of shape (batch_size*num_anchors, 1). It calculates the loss per anchor per image individually
        

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1) # (batch_size, num_anchors)

        # For every image, sort the loss of -ve bbox anchors in descending order
        # loss_idx, idx_rank, neg are of shape (batch_size, num_anchors)
        '''
        Firstly get the sorted index, then get the sorted index of sorted index as the rank.
        This gives a rank of 0 for the -ve bbox whose loss is max

        Eg: support batch size = 1, loss_c = [[5,2,7,9,1]]
        loss_idx = [[3,2,0,1,4]]
        idx_rank = [[2,3,1,0,4]]

        idx_rank = 0 has the max correponding loss.
        '''
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1) 
        num_pos = pos.long().sum(1, keepdim=True) # (batch_size, 1) number of positive anchor boxes for each image
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank) # choosing the  top num_neg -ve bbox with max loss
        # **********************************************************************************************************************************
        # This is one of the reasons why facebox can give FP i.e. detecting background as a face. We don't penalize al the -ve bbox 
        # to force their probability to 0. So since we are not covering all the possible cases for -ve bbox to force their probability to 0
        # during training, so in teting their could be some bbox enclosing the background with Probability more than threhold. 
        # **********************************************************************************************************************************

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data) # (batch_size, num_anchors, 2) all coumns are same, value of all columns are T if it corresponds to +ve bbox else F
        neg_idx = neg.unsqueeze(2).expand_as(conf_data) # (batch_size, num_anchors, 2) all coumns are same, value of all columns are T if it its rank < num_neg else F
        
        # (pos_idx+neg_idx).gt(0) will be satisfies if pos_idx is [T, T] and neg_idx is [F,F] or vice versa. Both T is not possible
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes) # collecting predictions for pos bbox and top -ve bbx
        targets_weighted = conf_t[(pos+neg).gt(0)] # # collecting expected labels for pos bbox and top -ve bbx
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum') # Calcualting the CE loss

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        
        # Averaging the loss
        N = max(num_pos.data.sum().float(), 1) # to deal with the case where there are no faces 
        loss_l /= N
        loss_c /= N # as per the paper this should have been divided by num_pos.data.sum() + num_neg.data.sum() as for thr confidence loss both positive and top -ve bboxes are used

        return loss_l, loss_c
