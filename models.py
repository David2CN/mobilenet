from torch.nn import Module, Conv2d, BatchNorm2d, ModuleList
from torch.nn import ReLU, AvgPool2d, Linear, Flatten


class DSConvLayer(Module):
    """
    Module consists of a depthwise convolution layer
    and a pointwise convolution layer, both followed by 
    one batchnormalization and relu layer.

    Together this presents depthwise separable convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.depth_conv = Conv2d(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=kernel_size,
                                groups=in_channels, 
                                stride=stride, 
                                padding=1)
                                
        self.point_conv = Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1)

        self.bn1 = BatchNorm2d(in_channels)
        self.bn2 = BatchNorm2d(out_channels)
        self.relu = ReLU()
    
    def forward(self, x) -> None:
        x = self.depth_conv(x)
        x = self.relu(self.bn1(x))
        x = self.point_conv(x)
        x = self.relu(self.bn2(x))
        return x


class DSConvModuleBlock(Module):
    """
    nx DSConvModule
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, n: int=5) -> None:
        super().__init__()

        block = [DSConvLayer(in_channels=in_channels,
                                 out_channels=out_channels, 
                                 kernel_size=kernel_size, 
                                 stride=stride) 
                    for _ in range(n)]

        self.block = ModuleList(block)

    def forward(self, x):
        
        for conv in self.block:
            x = conv(x)

        return x


class MobileNet(Module):
    """
    mobilenet
    """
    def __init__(self, input_dim: int=3, classes: int=10):
        super().__init__()
        self.input_dim = input_dim
        self.classes = classes
        self.conv1 = Conv2d(self.input_dim, 32, kernel_size=3, stride=2, padding=1)
        self.dsconv1 = DSConvLayer(32, 64, 3, stride=1)
        self.dsconv2 = DSConvLayer(64, 128, 3, stride=2)
        self.dsconv3 = DSConvLayer(128, 128, 3, stride=1)
        self.dsconv4 = DSConvLayer(128, 256, 3, stride=2)
        self.dsconv5 = DSConvLayer(256, 256, 3, stride=1)
        self.dsconv6 = DSConvLayer(256, 512, 3, stride=2)
        self.dsconv7 = DSConvModuleBlock(512, 512, 3, stride=1)
        self.dsconv8 = DSConvLayer(512, 1024, 3, stride=2)
        self.dsconv9 = DSConvLayer(1024, 1024, 3, stride=1)
        self.pool = AvgPool2d(7, stride=1)
        self.flatten = Flatten()
        self.fc = Linear(1024, 1000)
        self.classifier = Linear(1000, self.classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.dsconv3(x)
        x = self.dsconv4(x)
        x = self.dsconv5(x)
        x = self.dsconv6(x)
        x = self.dsconv7(x)
        x = self.dsconv8(x)
        x = self.dsconv9(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.classifier(x)
        return x



class MobileNetMini(Module):
    """
    mobilenet
    """
    def __init__(self, input_dim: int=3, classes: int=10):
        super().__init__()
        self.input_dim = input_dim
        self.classes = classes
        self.conv1 = Conv2d(self.input_dim, 64, kernel_size=3, stride=2, padding=1)
        self.dsconv1 = DSConvLayer(64, 128, 3, stride=1)
        self.dsconv2 = DSConvLayer(128, 256, 3, stride=2)
        self.dsconv3 = DSConvLayer(256, 512, 3, stride=1)
        self.dsconv4 = DSConvLayer(512, 512, 3, stride=2)
        self.pool = AvgPool2d(4, stride=1)
        self.flatten = Flatten()
        self.fc = Linear(512, 256)
        self.classifier = Linear(256, self.classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.dsconv3(x)
        x = self.dsconv4(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.classifier(x)
        return x
