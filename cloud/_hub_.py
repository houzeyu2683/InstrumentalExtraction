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
#     torch.nn.AdaptiveAvgPool2d((1,1)),
#     torch.nn.Flatten(1, -1)
# )
# model.eval()

def getCollation(queue: list, device: str) -> tuple:
    collection = []
    iteration = queue
    for item in iteration:
        path = item
        video = torchcodec.decoders.VideoDecoder(path)
        domain = len(video)
        floor = random.randint(0, int(domain*0.8))
        while(True):
            ceiling = min(floor+1+1500, domain-1)
            destination = random.randint(floor+1, ceiling)
            deviation = random.randint(floor+1, ceiling)
            if(destination!=deviation): break
            _, _ = destination, deviation
            continue
        outset = floor
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
        anchor = getTransform(video[outset])
        positive = getTransform(video[destination])
        negative = getTransform(video[deviation])
        timestep = torch.tensor([destination - outset])
        collection += [(anchor, positive, negative, timestep)]
        continue
    _ = iteration
    anchor = torch.stack(
        list(map(lambda item: item[0], collection)), dim=0
    ).to(device)
    if('positive'):
        image = torch.stack(
            list(map(lambda item: item[1], collection)), dim=0
        )
        # embedding = model(image)
        # assert isinstance(embedding, torch.Tensor)
        # positive = (image.to(device), embedding.to(device))
        positive = image.to(device)
        pass
    if('negative'):
        image = torch.stack(
            list(map(lambda item: item[2], collection)), dim=0
        )
        # embedding = model(image)
        # assert isinstance(embedding, torch.Tensor)
        # negative = (image.to(device), embedding.to(device))
        negative = image.to(device)
        pass
    timestep = torch.cat(
        list(map(lambda item: item[3], collection))
    ).to(device)
    collation = (
        anchor, 
        positive, 
        negative,
        timestep
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
