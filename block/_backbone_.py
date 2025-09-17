import torch
import torchvision

class Backbone(torch.nn.Module):

    def __init__(self, weight: str, device: str) -> None:
        super().__init__()
        self.weight = weight
        self.device = device
        return
    
    def initiateLayer(self) -> bool:
        assert self.weight=='mobilenet'
        net = torchvision.models.mobilenet_v2(
            weights='MobileNet_V2_Weights.IMAGENET1K_V1'
        )
        layer = torch.nn.Sequential(
            *list(net.features.children())[:-1], 
            torch.nn.AdaptiveAvgPool2d((1,1)),
            torch.nn.Flatten(1, -1)
        )
        self.layer = layer.to(self.device)
        return(True)

    def getFeedback(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layer(x)
        return(y)

    forward = getFeedback
    pass


