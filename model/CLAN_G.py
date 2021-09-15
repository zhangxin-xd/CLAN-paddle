import numpy as np
import paddle


affine_par = True
weight_attr_conv = paddle.nn.initializer.Normal(mean=0.0,std=0.01)
weight_attr_bn = paddle.nn.initializer.Constant(value=1.0)
bais_attr_bn = paddle.nn.initializer.Constant(value=0.0)

class Bottleneck(paddle.nn.Layer):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = paddle.nn.Conv2D(inplanes, planes, kernel_size=1, stride=stride, bias_attr=False, weight_attr=paddle.ParamAttr(initializer=weight_attr_conv))  # change
        self.bn1 = paddle.nn.BatchNorm2D(planes ,weight_attr=paddle.ParamAttr(initializer=weight_attr_bn), bias_attr=paddle.ParamAttr(initializer=bais_attr_bn))
        for i in self.bn1.parameters():
            i.stop_gradient = False

        padding = dilation
        self.conv2 = paddle.nn.Conv2D(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, dilation=dilation, bias_attr=False, weight_attr=paddle.ParamAttr(initializer=weight_attr_conv))
        self.bn2 = paddle.nn.BatchNorm2D(planes,weight_attr=paddle.ParamAttr(initializer=weight_attr_bn), bias_attr=paddle.ParamAttr(initializer=bais_attr_bn))
        for i in self.bn2.parameters():
            i.stop_gradient = False
        self.conv3 = paddle.nn.Conv2D(planes, planes * 4, kernel_size=1, bias_attr=False, weight_attr=paddle.ParamAttr(initializer=weight_attr_conv))
        self.bn3 = paddle.nn.BatchNorm2D(planes * 4, weight_attr=paddle.ParamAttr(initializer=weight_attr_bn), bias_attr=paddle.ParamAttr(initializer=bais_attr_bn))
        for i in self.bn3.parameters():
            i.stop_gradient = False
        self.relu = paddle.nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(paddle.nn.Layer):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = paddle.nn.LayerList()
        for dilation, padding in zip(dilation_series, padding_series):
            weight_attr = paddle.nn.initializer.Normal(mean=0.0,std=0.01)
            conv = paddle.nn.Conv2D(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias_attr=True,weight_attr=paddle.ParamAttr(initializer=weight_attr_conv))
            self.conv2d_list.append(conv)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class ResNet(paddle.nn.Layer):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = paddle.nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3 , bias_attr=False, weight_attr=paddle.ParamAttr(initializer=weight_attr_conv))
        self.bn1 = paddle.nn.BatchNorm2D(64, weight_attr=paddle.ParamAttr(initializer=weight_attr_bn), bias_attr=paddle.ParamAttr(initializer=bais_attr_bn))
        for i in self.bn1.parameters():
            i.stop_gradient = True
        self.relu = paddle.nn.ReLU()
        self.maxpool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = paddle.nn.Sequential(
                paddle.nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, weight_attr=paddle.ParamAttr(initializer=weight_attr_conv)),
                paddle.nn.BatchNorm2D(num_features = (planes * block.expansion), weight_attr=paddle.ParamAttr(initializer=weight_attr_bn), bias_attr=paddle.ParamAttr(initializer=bais_attr_bn)))
        
        for i in downsample['1'].parameters():
            i.stop_gradient = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return paddle.nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1 = self.layer5(x)
        x2 = self.layer6(x)
       
        return x1, x2

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if not k.stop_gradient:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())
        #b.append(self.layer7.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]


def Res_Deeplab(num_classes=21):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model


