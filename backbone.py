import timm
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class basic_cnn_layer(torch.nn.Module):
    """
    initiate basic cnn layer class
    conv -> batch normalization -> relu

    Parameters:
        input_channel (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding of the convolution.
        dilation (int): Dilation of the convolution.
    """
    def __init__(self, 
                 input_channel:int,
                 out_channels:int,
                 kernel_size:int,
                 stride:int=1,
                 padding:int=0,
                 dilation:int=1):
        super(basic_cnn_layer, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels= input_channel,
                                    out_channels = out_channels, 
                                    kernel_size = kernel_size,
                                    stride = stride,
                                    padding = padding ,dilation=dilation)   
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.LeakyReLU(0.1)
    
    def forward(self, 
                x:torch.Tensor)->torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SPPBlock(nn.Module):
    """
    Make Sapatial Pyramid pooling, it has 4 parallel CNN network
    1. original
    2. kernel size = 5*5, padding = 2
    3. kernel size = 9*9, padding = 4
    4. kernel size = 13*13, padding = 6
    """
    def __init__(self):
        super(SPPBlock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

    def forward(self, x):
        p1 = self.pool1(x)
        p2 = self.pool2(x)
        p3 = self.pool3(x)
        return torch.cat([x, p1, p2, p3], dim=1)

class backbone(nn.Module):
    """
    Backbone Design based on Pretrained CSPDarknet53 and FPN layer 
    """
    def __init__(self):
        super().__init__()
        self.layer_sequence = timm.create_model('cspdarknet53.ra_in1k', pretrained=True, features_only=True)
        self.bt_neck1 = nn.Sequential(
            basic_cnn_layer(1024,512,1,1,0,1),
            basic_cnn_layer(512,1024,3,1,1,1),
            basic_cnn_layer(1024,512,1,1,0,1)
        )
        self.SPP = SPPBlock()
        self.bt_neck2 = nn.Sequential(
            basic_cnn_layer(2048,512,1,1,0,1),
            basic_cnn_layer(512,1024,3,1,1,1),
            basic_cnn_layer(1024,512,1,1,0,1)
        )
        self.conv1 = basic_cnn_layer(512,256,1,1,0,1)
        self.conv1_1 = basic_cnn_layer(512,256,1,1,0,1) 

        self.bt_neck3 = nn.Sequential(
            basic_cnn_layer(512,256,1,1,0,1),
            basic_cnn_layer(256,512,3,1,1,1),
            basic_cnn_layer(512,256,1,1,0,1),
            basic_cnn_layer(256,512,3,1,1,1),
            basic_cnn_layer(512,256,1,1,0,1),
        )
        self.conv2 = basic_cnn_layer(256,128,1,1,0,1)
        self.conv2_1 = basic_cnn_layer(256,128,1,1,0,1)

        self.bt_neck4 = nn.Sequential(
            basic_cnn_layer(256,128,1,1,0,1),
            basic_cnn_layer(128,256,3,1,1,1),
            basic_cnn_layer(256,128,1,1,0,1),
            basic_cnn_layer(128,256,3,1,1,1),
            basic_cnn_layer(256,128,1,1,0,1),
        )
        self.conv3 = basic_cnn_layer(256,128,1,1,0,1)

    def forward(self,x):
        x = self.layer_sequence(x)

        route_1 = x[3]
        route_2 = x[4]
        route_3 = x[5]
        
        route_3 = self.bt_neck1(route_3)
        route_3 = self.SPP(route_3)
        route_3 = self.bt_neck2(route_3)
        P_large = route_3

        #--------------------------------------------+
        route_3 = self.conv1(route_3)
        route_3 = F.interpolate(route_3,scale_factor=2,mode ='nearest')
        route_2 = self.conv1_1(route_2)

        route_2 = torch.cat([route_2,route_3],dim = 1)
        route_2 = self.bt_neck3(route_2)
        P_mid = route_2

        #--------------------------------------------
        route_2 = self.conv2(route_2)
        route_2 = F.interpolate(route_2,scale_factor=2,mode ='nearest')
        route_1 = self.conv2_1(route_1)

        route_1 = torch.cat([route_1,route_2],dim = 1)
        P_small = self.conv3(route_1)

        return (P_large,P_mid,P_small)
    
class neck(nn.Module):
    """
    Neck is designed based on PANet.\\
    it ruturn the layer for object detetion
    """
    def __init__(self):
        super().__init__()
        self.conv1 = basic_cnn_layer(128,256,3,2,1,1)
        self.bt_neck1 = nn.Sequential(
            basic_cnn_layer(512,256,1,1,0,1),
            basic_cnn_layer(256,512,3,1,1,1),
            basic_cnn_layer(512,256,1,1,0,1),
            basic_cnn_layer(256,512,3,1,1,1),
            basic_cnn_layer(512,256,1,1,0,1),
        )

        self.conv2 = basic_cnn_layer(256,512,3,2,1,1)
        self.bt_neck2 = nn.Sequential(
            basic_cnn_layer(1024,512,1,1,0,1),
            basic_cnn_layer(512,1024,3,1,1,1),
            basic_cnn_layer(1024,512,1,1,0,1),
            basic_cnn_layer(512,1024,3,1,1,1),
            basic_cnn_layer(1024,512,1,1,0,1),
        )

    def forward(self,x):
        x_in = x[2]
        x_in = self.conv1(x_in)
        x_in = torch.cat([x[1],x_in],dim = 1)
        x_in = self.bt_neck1(x_in)
        Neck_mid = x_in

        x_in = self.conv2(x_in)
        x_in = torch.cat([x[0],x_in],dim = 1)
        Neck_large = self.bt_neck2(x_in)

        return (Neck_large,Neck_mid,x[2])
    

class head(nn.Module):
    """
    head similar to YOLOv3-based heads.\\
    it has small,mid,large scale result.\\
    each scale block has 3 anchors
    """
    def __init__(self,num_class):
        super().__init__()

        out_channel_length = (5+num_class)*3
        self.conv1 = basic_cnn_layer(128,256,3,1,1,1)
        self.conv1_1 = torch.nn.Conv2d(in_channels= 256,
                                    out_channels = out_channel_length, 
                                    kernel_size = 1,
                                    stride = 1,
                                    padding = 0 ,dilation=1)
                
        self.conv2 = basic_cnn_layer(256,512,3,1,1,1)
        self.conv2_1 = torch.nn.Conv2d(in_channels= 512,
                                    out_channels = out_channel_length, 
                                    kernel_size = 1,
                                    stride = 1,
                                    padding = 0 ,dilation=1)
        
        self.conv3 = basic_cnn_layer(512,1024,3,1,1,1)
        self.conv3_1 = torch.nn.Conv2d(in_channels= 1024,
                                    out_channels = out_channel_length, 
                                    kernel_size = 1,
                                    stride = 1,
                                    padding = 0 ,dilation=1)


    def forward(self,x):
        small = self.conv1(x[2])
        small = self.conv1_1(small)

        mid = self.conv2(x[1])
        mid = self.conv2_1(mid)

        large = self.conv3(x[0])
        large = self.conv3_1(large)

        return (large,mid,small)   
    
class YoloV4Model(nn.Module):
    """
    Combine Backbone, Neck, and Head to create a full object detection model (YOLOv4-like).
    
    prameters:
        num_classes (int): Number of classes for object detection.
    """
    def __init__(self, 
                 num_classes:int,
                 to_vector:bool = False,):
        super(YoloV4Model, self).__init__()
        self.backbone = backbone()
        self.neck = neck()
        self.head = head(num_classes)
        self.to_veoctor = to_vector
        self.num_classes = num_classes
        
    def forward(self, x):
        P_large, P_mid, P_small = self.backbone(x)
        Neck_large, Neck_mid, _ = self.neck((P_large, P_mid, P_small))
        large, mid, small = self.head((Neck_large, Neck_mid, P_small))
        result = []

        if self.to_veoctor:
            for detect_box in [large, mid, small]:
                grid_box = detect_box.permute(0, 2, 3, 1).contiguous()
                grid_box = grid_box.view(detect_box.size(0), -1, 5+self.num_classes)
                # [N, anchor_num * (5+C),H,W] (Start)
                # -> [N,H,W, anchor_num * (5+C)] (after permuate)
                # -> [N,H*W*anchor_num,(5+C)] (afeter view method)
                result.append(grid_box)
            return result[0], result[1], result[2]
        
        else:
            return large, mid, small

        
            

        