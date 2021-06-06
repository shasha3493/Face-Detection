import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):
  '''
  Inception Network as described in paper

  '''

  def __init__(self):
    super(Inception, self).__init__()
    self.branch1x1 = BasicConv2d(128, 32, kernel_size=1, padding=0)
    self.branch1x1_2 = BasicConv2d(128, 32, kernel_size=1, padding=0)
    self.branch3x3_reduce = BasicConv2d(128, 24, kernel_size=1, padding=0)
    self.branch3x3 = BasicConv2d(24, 32, kernel_size=3, padding=1)
    self.branch3x3_reduce_2 = BasicConv2d(128, 24, kernel_size=1, padding=0)
    self.branch3x3_2 = BasicConv2d(24, 32, kernel_size=3, padding=1)
    self.branch3x3_3 = BasicConv2d(32, 32, kernel_size=3, padding=1)

  def forward(self, x):
    branch1x1 = self.branch1x1(x)

    branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch1x1_2 = self.branch1x1_2(branch1x1_pool)

    branch3x3_reduce = self.branch3x3_reduce(x)
    branch3x3 = self.branch3x3(branch3x3_reduce)

    branch3x3_reduce_2 = self.branch3x3_reduce_2(x)
    branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2)
    branch3x3_3 = self.branch3x3_3(branch3x3_2)

    outputs = [branch1x1, branch1x1_2, branch3x3, branch3x3_3]
    return torch.cat(outputs, 1)


class CRelu(nn.Module):

  """
  CRelu is motivated from the observation in CNN taht the filters in the lower layers form pairs(i.e. filters with opposite phase).
  From this observation, CRelu cab double the number of the output channels by simply concatenating negated outputs before 
  applying ReLU. In this way, computation cost for doing convolution for oppoite phased filters can be reduced, increasing the 
  computational efficiency
  
  """

  def __init__(self, in_channels, out_channels, **kwargs):
    super(CRelu, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
    self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

  def forward(self, x):
    x = self.conv(x) # Convolution
    x = self.bn(x) # Batch Normalization
    x = torch.cat([x, -x], 1) # Concatenation the output with -output
    x = F.relu(x, inplace=True) # Applying RelU # inplace=True means that it will modify the input directly, without allocating any additional output. 
                                # It can sometimes slightly decrease the memory usage, but may not always be a valid operation (because the original input is destroyed).
                                # However, if you don’t see an error, it means that your use case is valid. In both the cases, it return a tensor
    return x


class FaceBoxes(nn.Module):

  def __init__(self, phase, size, num_classes):
    super(FaceBoxes, self).__init__()
    self.phase = phase # (train/test)
    self.num_classes = num_classes # 2
    self.size = size # 1024  during training, input size during training (1024*1024)

    # CRelu
    self.conv1 = CRelu(3, 24, kernel_size=7, stride=4, padding=3) # in the paper, kernel size was 5.
    self.conv2 = CRelu(48, 64, kernel_size=5, stride=2, padding=2) # in the paper, kernel size was 3

    # Inception Network
    self.inception1 = Inception()
    self.inception2 = Inception()
    self.inception3 = Inception()

    self.conv3_1 = BasicConv2d(128, 128, kernel_size=1, stride=1, padding=0)
    self.conv3_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1) # output channels were 128 in the paper

    self.conv4_1 = BasicConv2d(256, 128, kernel_size=1, stride=1, padding=0)
    self.conv4_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1) # output channels were 128 in the paper

    self.loc, self.conf = self.multibox(self.num_classes)

    if self.phase == 'test':
        self.softmax = nn.Softmax(dim=-1)

    if self.phase == 'train':
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

  def multibox(self, num_classes):
    loc_layers = []
    conf_layers = []
    loc_layers += [nn.Conv2d(128, 21 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(128, 21 * num_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
    return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

  def forward(self, x):
    detection_sources = list()
    loc = list()
    conf = list()

    # Rapidly Digested Convolution Layers (RDCL)
    ############################################

    # input(x): (batch_size, 3, 1024, 1024)

    x = self.conv1(x) # (batch_size, 48, 256, 256) # (batch_size, 48, 256, 256) No. of channels is 2*24 as negated output is concatenated
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1) # (batch_size, 48, 128, 128)
    x = self.conv2(x) # (batch_size, 128, 64, 64) No. of channels is 2*64 as negated output is concatenated
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1) # (batch_size, 128, 32, 32)


    # Multiple Scale Convolutional Layers
    ####################################
    x = self.inception1(x) # (batch_size, 128, 32, 32)
    x = self.inception2(x) # (batch_size, 128, 32, 32)
    x = self.inception3(x) # (batch_size, 128, 32, 32)
    detection_sources.append(x)

    x = self.conv3_1(x)
    x = self.conv3_2(x) # (batch_size, 256, 16, 16)
    detection_sources.append(x)

    x = self.conv4_1(x)
    x = self.conv4_2(x) # (batch_size, 256, 8, 8)
    detection_sources.append(x)

    # detection_sources is a list of feature map at different scale on which detection will happen.

    # Unlike paper, FPN is not done and instead convolution is done to get the feature map in the required size
    
    for (x, l, c) in zip(detection_sources, self.loc, self.conf):
        loc.append(l(x).permute(0, 2, 3, 1).contiguous())
        conf.append(c(x).permute(0, 2, 3, 1).contiguous())
    """
    loc[0].shape = (batch_size, 32, 32, 21(num anchors)*4)
    loc[1].shape = (batch_size, 16, 16, 1(num anchors)*4)
    loc[2].shape = (batch_size, 8, 8, 1(num anchors)*4)

    conf[0].shape = (batch_size, 32, 32, 21(num anchors)*2)
    conf[1].shape = (batch_size, 16, 16, 1(num anchors)*2)
    conf[2].shape = (batch_size, 8, 8, 1(num anchors)*2)
    """
    # print(loc[0].shape, loc[1].shape, loc[2].shape)
    # print(conf[0].shape, conf[1].shape, conf[2].shape)

    loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
    conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
  
    # During training, logits are directly returned and in testing probability is returned
    if self.phase == "test":
      output = (loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)))
    else:
      output = (loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes))

    return output # output is a 2d tuple with 1st element of shape (batch_size, 21824, 4) and 2nd element of shape (batch_size, 21824, 2)
