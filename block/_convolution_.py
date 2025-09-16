import torch

class Convolution(torch.nn.Module):

    def __init__(
        self, tunnel: tuple, group: int, inverse: bool, device: str
    ) -> None:
        super().__init__()
        self.tunnel = tunnel
        self.group = group
        self.inverse = inverse
        self.device = device
        return

    def initiateLayer(
        self, 
    ) -> bool:
        if(not self.inverse):
            layer = {
                '1': torch.nn.Sequential(
                    torch.nn.Conv2d(
                        *self.tunnel, 
                        kernel_size=3, 
                        stride=2, 
                        padding=1
                    ),
                    torch.nn.GroupNorm(
                        num_groups=self.group, 
                        num_channels=self.tunnel[-1]
                    ),
                    torch.nn.ReLU()
                ),
                '2': torch.nn.Sequential(
                    torch.nn.Conv2d(
                        self.tunnel[-1], 
                        self.tunnel[-1], 
                        kernel_size=3, 
                        stride=1, 
                        padding=1
                    ),
                    torch.nn.GroupNorm(
                        num_groups=self.group, 
                        num_channels=self.tunnel[-1]
                    ),
                    torch.nn.ReLU()
                )
            }
            self.layer = torch.nn.ModuleDict(layer).to(self.device)
            return(True)
        layer = {
            '1': torch.nn.Sequential(
                torch.nn.ConvTranspose2d(
                    *self.tunnel, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1,
                    output_padding=1
                ),
                torch.nn.GroupNorm(
                    num_groups=self.group, 
                    num_channels=self.tunnel[-1]
                ),
                torch.nn.ReLU()
            ),
            '2': torch.nn.Sequential(
                torch.nn.ConvTranspose2d(
                    self.tunnel[-1], 
                    self.tunnel[-1], 
                    kernel_size=3, 
                    stride=1, 
                    padding=1
                ),
                torch.nn.GroupNorm(
                    num_groups=self.group, 
                    num_channels=self.tunnel[-1]
                ),
                torch.nn.ReLU()
            )
        }
        self.layer = torch.nn.ModuleDict(layer).to(self.device)
        return(True)
    
    def getFeedback(self, x: torch.Tensor) -> torch.Tensor:
        o = self.layer['1'](x)
        y = o + self.layer['2'](o)
        return(y)

    forward = getFeedback
    pass
