import paddle


class FCDiscriminator(paddle.nn.Layer):

    def __init__(self, num_classes, ndf = 64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = paddle.nn.Conv2D(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = paddle.nn.Conv2D(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = paddle.nn.Conv2D(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = paddle.nn.Conv2D(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = paddle.nn.Conv2D(ndf*8, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = paddle.nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)

        return x


class FCDiscriminator_Local(paddle.nn.Layer):

    def __init__(self, num_classes, ndf = 64):
        super(FCDiscriminator_Local, self).__init__()

        self.conv1 = paddle.nn.Conv2D(num_classes + 2048, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = paddle.nn.Conv2D(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = paddle.nn.Conv2D(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.classifier = paddle.nn.Conv2D(ndf*4, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = paddle.nn.LeakyReLU(negative_slope=0.2)
        self.up_sample = paddle.nn.Upsample(scale_factor=32, mode='bilinear')
        #self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        x = self.up_sample(x)
        #x = self.sigmoid(x) 

        return x