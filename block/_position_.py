import torch
import math

class Position(torch.nn.Module):

    def __init__(self, length: int, dimension: int, trainable: bool, device: str) -> None:
        super().__init__()
        self.length = length
        self.dimension = dimension
        self.trainable = trainable
        self.device = device
        return
    
    def initiateLayer(self) -> bool:
        layer = {}
        code = torch.zeros(self.length, self.dimension)
        sequence = torch.arange(
            0, self.length, dtype=torch.float
        ).unsqueeze(1)  # shape: [max_len, 1]
        denominator = (-math.log(10000.0) / self.dimension)
        term = torch.exp(
            torch.arange(0, self.dimension, 2).float() * denominator
        )
        code[:, 0::2] = torch.sin(sequence * term)
        code[:, 1::2] = torch.cos(sequence * term)
        code = code.to(self.device)#.unsqueeze(0).to(self.device)
        if(self.trainable):
            layer['(1) code'] = torch.nn.Parameter(code, requires_grad=True)
            self.layer = torch.nn.ParameterDict(layer)
            return(True)
        layer['(1) code'] = torch.nn.Parameter(code, requires_grad=False)
        self.layer = torch.nn.ParameterDict(layer) # shape: [max_len, d_model]
        return(True)
    
    def getFeedback(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, l, d)
        _, l, _ = x.shape
        y = x + self.layer['(1) code'][None, :l, :]
        return(y)

    forward = getFeedback
    pass

# x = torch.randn(4, 20, 128)
# position = Position(length=20, dimension=128, trainable=False, device='cpu')
# position.initiateLayer()
# y = position(x)


