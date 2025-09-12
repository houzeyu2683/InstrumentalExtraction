import torch
import einops
import flash_attn.flash_attn_interface

class Edge(torch.nn.Module):
    
    def __init__(self, device: str) -> None:
        super().__init__()
        self.device = device
        return
    
    def initiateLayer(self) -> bool:
        layer = {
            "(1) transposition": torch.nn.Sequential(
                torch.nn.Linear(320, 256),
                torch.nn.LayerNorm(256),
                torch.nn.LeakyReLU(0.2)
            ),
            "(2) transposition": torch.nn.Sequential(
                torch.nn.ConvTranspose2d(1, 256, 4, 2, 1),
                torch.nn.GroupNorm(32, 256),
                torch.nn.LeakyReLU(0.2)
            ),
            "(3) transposition": torch.nn.Sequential(
                torch.nn.ConvTranspose2d(256, 128, 4, 2, 1),
                torch.nn.GroupNorm(32, 128),
                torch.nn.LeakyReLU(0.2)
            ),
            "(4) transposition": torch.nn.Sequential(
                torch.nn.ConvTranspose2d(128, 64, 4, 2, 1),
                torch.nn.GroupNorm(32, 64),
                torch.nn.LeakyReLU(0.2)
            ),
            '(5) transposition': torch.nn.Sequential(
                torch.nn.Conv2d(64, 64, kernel_size=1),
                torch.nn.GroupNorm(32, 64),
                torch.nn.LeakyReLU(0.2)
            ),
            '(6) transposition': torch.nn.Sequential(
                torch.nn.Conv2d(64, 3, kernel_size=1),
                torch.nn.Sigmoid()
            ),
        }
        self.layer = torch.nn.ModuleDict(layer).to(self.device)
        return(True)

    def getFeedback(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # x: (b, l, d)
        x = self.layer['(1) transposition'](x)
        b, l, d = x.shape
        h, w = 16, 16
        c = 1
        assert d==h*w
        g = []
        for i, j  in zip(x, m):
            r = i[~j,:]
            r = einops.rearrange(r, 'l (c h w) -> l c h w', c=c, h=h, w=w)
            r = self.layer['(2) transposition'](r)
            r = self.layer['(3) transposition'](r)
            r = self.layer['(4) transposition'](r)
            r = r + self.layer['(5) transposition'](r)
            r = self.layer['(6) transposition'](r)
            g += [r]
            continue
        g = torch.nn.utils.rnn.pad_sequence(g, batch_first=True)
        # x = einops.rearrange(x, 'b l (c h w) -> b l c h w', c=c, h=h, w=w)
        # x = einops.rearrange(x, 'b l c h w -> (b l) c h w')
        y = einops.rearrange(o, '(b l) c h w -> b l c h w', b=b, l=l)
        return(y)
    
    forward = getFeedback
    pass


# import torch
# import einops

# x = torch.randn(4, 11, 256)
# edge = Edge('cpu')
# edge.initiateLayer()
# edge(x)


