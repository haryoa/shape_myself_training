from torchvision.transforms import Compose, Lambda, ToPILImage, RandomHorizontalFlip, ToTensor, Resize, CenterCrop
import numpy as np


reverse_transform = Compose([
     Lambda(lambda t: (t + 1) / 2),
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),
     ToPILImage(),
])


def produce_transform_fn(image_size: int = 128):
    return Compose([
        Resize(image_size),
        CenterCrop(image_size),
        RandomHorizontalFlip(),
        ToTensor(),
        Lambda(lambda t: (t * 2) - 1)
    ])
