import torch
import os
import glob
import torchvision
import torchcodec
import functools
import pandas
import random

# backbone = torchvision.models.mobilenet_v2(
#     weights='MobileNet_V2_Weights.IMAGENET1K_V1'
# )
# model = torch.nn.Sequential(
#     *list(backbone.features.children())[:-1], 
#     torch.nn.AdaptiveAvgPool2d((1,1))
# )
# model.eval()

def getCollation(queue: list, device: str) -> tuple:
    collection = []
    iteration = queue
    for item in iteration:
        path = item
        video = torchcodec.decoders.VideoDecoder(path)
        domain = len(video)
        current = random.randint(0, int(domain*0.8))
        while(True):
            ceiling = min(current+1+1500, domain)
            positive = random.randint(current+1, ceiling)
            negative = random.randint(current+1, ceiling)
            if(positive!=negative): break
            _, _ = positive, negative
            continue
        size = (128, 128)
        getTransform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize(size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
        moment = getTransform(video[current])
        future = getTransform(video[positive])
        forgery = getTransform(video[negative])
        timestep = torch.tensor([positive - current])
        collection += [(moment, future, forgery, timestep)]
        continue
    _ = iteration
    moment = torch.stack(
        list(map(lambda item: item[0], collection)), dim=0
    )
    future = torch.stack(
        list(map(lambda item: item[1], collection)), dim=0
    )
    forgery = torch.stack(
        list(map(lambda item: item[2], collection)), dim=0
    )
    timestep = torch.cat(
        list(map(lambda item: item[3], collection))
    )
    collation = (
        moment.to(device), 
        future.to(device), 
        forgery.to(device),
        timestep.to(device)
    )
    return(collation)

class Library(torch.utils.data.Dataset):

    def __init__(self, queue: list, randomness: bool) -> None:
        self.queue = queue
        self.randomness = randomness
        return
    
    def getLength(self) -> int:
        length = len(self.queue)
        return(length)

    def getItem(self, index: int) -> tuple:
        item = self.queue[index]
        return(item)

    def getEmployment(
        self, batch: int, device: str
    ) -> torch.utils.data.DataLoader:
        employment = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch,
            shuffle=self.randomness,
            collate_fn=functools.partial(getCollation, device=device),
            drop_last=self.randomness
        )
        return(employment)

    __len__ = getLength
    __getitem__ = getItem
    pass
        
class Hub:

    def __init__(self, folder: str) -> None:
        self.folder = folder
        return
    
    def initiateLibrary(self) -> bool:
        tag = 'data'
        if(tag):
            with open(os.path.join(self.folder, f'{tag}.txt'), "r") as paper:
                source = [os.path.join(self.folder, line.strip()) for line in paper.readlines()]
                pass
            _ = paper
            data = Library(source, randomness=True)
            pass
        tag = 'validation'
        if(tag):
            with open(os.path.join(self.folder, f'{tag}.txt'), "r") as paper:
                source = [os.path.join(self.folder, line.strip()) for line in paper.readlines()]
                pass
            _ = paper
            validation = Library(source, randomness=True)
            pass
        tag = 'test'
        if(tag):
            with open(os.path.join(self.folder, f'{tag}.txt'), "r") as paper:
                source = [os.path.join(self.folder, line.strip()) for line in paper.readlines()]
                pass
            _ = paper
            test = Library(source, randomness=False)
            pass
        self.library = {
            'data': data,
            'validation': validation,
            'test': test
        }
        return(True)

    def getData(
        self, batch: int, device: str
    ) -> torch.utils.data.DataLoader:
        data = self.library['data'].getEmployment(batch, device)
        return(data)

    def getValidation(
        self, batch: int, device: str
    ) -> torch.utils.data.DataLoader:
        validation = self.library['validation'].getEmployment(batch, device)
        return(validation)
    
    def getTest(
        self, batch: int, device: str
    ) -> torch.utils.data.DataLoader:
        test = self.library['test'].getEmployment(batch, device)
        return(test)

    pass
