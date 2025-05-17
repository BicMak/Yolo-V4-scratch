from dataclasses import dataclass

# Data class variables impossible to use dict, list, set
@dataclass
class dataset_config:
    image_dir:str = "dataset/image/"
    label_dir:str = "dataset/label/"
    image_size:int = 448
    bboxes_format:str = "yolo"
    classes:tuple = ('NG','OK')

 # i5-14400F has 6 Performance Cores and 4 Efficient Cores
 # If you have a different CPU, you may need to adjust the number of workers.
@dataclass
class dataloader_config:
    batch_size:int = 32
    num_workers:int = 6
    shuffle:bool = True
    pin_memory:bool = True
    drop_last:bool = False

@dataclass
class augmentation_config:
    transform:bool = True
    alpha:float = 0.7
    lamda:float = 0.5
    prob:float = 0.5


#Warmup is on the training phase
@dataclass
class OptimizerConfig:
    lr0:float = 0.01  # initial learning rate
    lrf:float = 0.1 # final learning rate (lr0 * lrf)
    momentum:float= 0.937  # SGD momentum/Adam beta1
    weight_decay:float = 0.0005  # optimizer weight decay 5e-4  # warmup epochs (fractions ok)
    warmup_momentum:float = 0.8  # warmup initial momentum
    warmup_bias_lr:float = 0.1  # warmup initial bias lr
    iou_normalizer:float = 0.05  # iou normalizer
    obj_normalizer:float = 4.0
    cls_normalizer:float = 0.5

@dataclass
class training_config:
    warmup_epochs:int = 3
    epochs:int = 100
    save_period:int = 1
    save_dir:str = "weights/"
    device:str = "cuda"
    lr_sheduler:bool = True

