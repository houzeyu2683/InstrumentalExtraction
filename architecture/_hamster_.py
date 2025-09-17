import torch
import math
import block
import einops
class Hamster(torch.nn.Module):
    
    def __init__(self, device: str) -> None:
        super().__init__()
        self.device = device
        return
    
    def initiateLayer(self) -> bool:
        layer = {}
        if('1'): # 128 -> 64
            tunnel = (3, 64)
            tag = f'(1) convolution'
            part = block.Convolution(
                tunnel=tunnel, group=8, inverse=False, device=self.device
            )
            part.initiateLayer()
            layer[tag] = part
            tag = f'(1) embedding'
            length = 1500+1
            dimension = tunnel[-1]
            part = block.Embedding(
                length=length, dimension=dimension, device=self.device
            )
            part.initiateLayer()
            layer[tag] = part
            pass
        if('2'): # 64 -> 32
            tunnel = (64, 128)
            tag = f'(2) convolution'
            part = block.Convolution(
                tunnel=tunnel, group=8, inverse=False, device=self.device
            )
            part.initiateLayer()
            layer[tag] = part
            tag = f'(2) embedding'
            length = 1500+1
            dimension = tunnel[-1]
            part = block.Embedding(
                length=length, dimension=dimension, device=self.device
            )
            part.initiateLayer()
            layer[tag] = part
            pass
        if('3'): # 32 -> 16
            tunnel = (128, 256)
            tag = f'(3) convolution'
            part = block.Convolution(
                tunnel=tunnel, group=8, inverse=False, device=self.device
            )
            part.initiateLayer()
            layer[tag] = part
            tag = f'(3) embedding'
            length = 1500+1
            dimension = tunnel[-1]
            part = block.Embedding(
                length=length, dimension=dimension, device=self.device
            )
            part.initiateLayer()
            layer[tag] = part
            pass
        if('4'): 
            tag = f'(4) position'
            part = block.Position(
                length=256, 
                dimension=256, 
                trainable=True, 
                device=self.device
            )
            part.initiateLayer()
            layer[tag] = part
            pass
        if('5'):
            tag = f'(5) attention'
            part = block.Attention(embedding=256, head=8, causal=False, device=self.device)
            part.initiateLayer()
            layer[tag] = part
            pass
        if('6'): # 16 -> 32
            tunnel = (256, 128)
            tag = f'(6) convolution'
            part = block.Convolution(
                tunnel=tunnel, group=8, inverse=True, device=self.device
            )
            part.initiateLayer()
            layer[tag] = part
            tag = f'(6) embedding'
            length = 1500+1
            dimension = tunnel[-1]
            part = block.Embedding(
                length=length, dimension=dimension, device=self.device
            )
            part.initiateLayer()
            layer[tag] = part
            pass
        if('7'): # 32 -> 64
            tunnel = (128+128, 64)
            tag = f'(7) convolution'
            part = block.Convolution(
                tunnel=tunnel, group=8, inverse=True, device=self.device
            )
            part.initiateLayer()
            layer[tag] = part
            tag = f'(7) embedding'
            length = 1500+1
            dimension = tunnel[-1]
            part = block.Embedding(
                length=length, dimension=dimension, device=self.device
            )
            part.initiateLayer()
            layer[tag] = part
            pass
        if('8'): # 64 -> 128
            tunnel = (64+64, 3)
            tag = f'(8) convolution'
            part = block.Convolution(
                tunnel=tunnel, group=1, inverse=True, device=self.device
            )
            part.initiateLayer()
            layer[tag] = part
            tag = f'(8) embedding'
            length = 1500+1
            dimension = tunnel[-1]
            part = block.Embedding(
                length=length, dimension=dimension, device=self.device
            )
            part.initiateLayer()
            layer[tag] = part
            pass
        if('9'):
            tag = f'(9) projection'
            part = torch.nn.Conv2d(3, 3, 1)
            layer[tag] = part
            pass
        if('10'):
            tag = f'(10) projection'
            part = torch.nn.Tanh()
            layer[tag] = part
            pass
        self.layer = torch.nn.ModuleDict(layer).to(self.device)
        return(True)

    def getFeedback(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor]:
        # x: b, c, h, w
        # t: b
        m = []
        x = self.layer['(1) convolution'](x)
        x = x + self.layer['(1) embedding'](t)[:,:,None,None]
        m += [x]
        x = self.layer['(2) convolution'](x)
        x = x + self.layer['(2) embedding'](t)[:,:,None,None]
        m += [x]
        x = self.layer['(3) convolution'](x)
        x = x + self.layer['(3) embedding'](t)[:,:,None,None]
        m += [x]
        l = m.pop()
        assert isinstance(l, torch.Tensor)
        l = l.flatten(2, -1)
        # l.shape
        l = self.layer[f'(4) position'](l)
        # atten
        a = self.layer[f'(5) attention'](l)
        # add noise
        a = a + torch.randn_like(a)
        h, w = 16, 16
        x = einops.rearrange(a, 'b l (h w) -> b l h w', h=h, w=w)
        x = self.layer['(6) convolution'](x)
        x = x + self.layer['(6) embedding'](t)[:,:,None,None]
        x = torch.cat([x, m[-1]], dim=1)
        x = self.layer['(7) convolution'](x)
        x = x + self.layer['(7) embedding'](t)[:,:,None,None]
        x = torch.cat([x, m[-2]], dim=1)
        x = self.layer['(8) convolution'](x)
        x = x + self.layer['(8) embedding'](t)[:,:,None,None]
        x = x + self.layer['(9) projection'](x)
        y = self.layer['(10) projection'](x)
        return(y)
    
    forward = getFeedback
    pass



# class Digestion(torch.nn.Module):

#     def __init__(self) -> None:
#         super().__init__()
#         return
    
#     def initiateLayer(self) -> bool:
#         layer = {}
#         kernel = torch.nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=8,
#             dim_feedforward=1024,
#             dropout=0.1,
#             activation='relu',
#             batch_first=True 
#         )
#         encoder = torch.nn.TransformerEncoder(
#             kernel,
#             num_layers=2
#         )
#         layer = {
#             "encoder": encoder
#         }
#         self.layer = torch.nn.ModuleDict(layer)
#         return(True)
        
#     def getPropagation(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
#         y = self.layer['encoder'](x, mask=None, src_key_padding_mask=p)
#         return(y)
    
#     forward = getPropagation
#     pass


# # import torch
# # import math
# # import torch.nn as nn





# chew = Chew(embedding=512)
# chew.initiateLayer()
# x = torch.randn((12, 33, 3, 224, 224))
# y = chew(x)
# y.shape
# torch.randn(12, 33, 768)

# import torch
# import torch.nn as nn

# batch_size = 12
# seq_len = 33
# embed_dim = 588
# x = torch.randn(batch_size, seq_len, embed_dim)  # [batch, seq_len, embed_dim]

# # Transformer Encoder layer
# encoder_layer = nn.TransformerEncoderLayer(
#     d_model=embed_dim,
#     nhead=8,
#     dim_feedforward=1024,
#     dropout=0.1,
#     activation='relu',
#     batch_first=True 
# )

# transformer_encoder = nn.TransformerEncoder(
#     encoder_layer,
#     num_layers=2
# )

# output = transformer_encoder(x, src_key_padding_mask=padding_mask)  # [batch, seq_len, embed_dim]
# print(output.shape)  # torch.Size([12, 33, 588])
# chew = Chew(embedding=512)
# chew.initiateLayer()
# x = torch.randn(12, 33, 3, 256, 256)
# p = None
# chew.getPropagation(x)

# class Position(torch.nn.Module):

#     def __init__(self, length: int, dimension: int):
#         super().__init__()
#         self.length = length
#         self.dimension = dimension
#         # self.device = device
#         return
    
#     def initiateLayer(self, trainable: bool, device: str) -> bool:
#         layer = {}
#         code = torch.zeros(self.length, self.dimension)
#         sequence = torch.arange(
#             0, self.length, dtype=torch.float
#         ).unsqueeze(1)  # shape: [max_len, 1]
#         denominator = (-math.log(10000.0) / self.dimension)
#         term = torch.exp(
#             torch.arange(0, self.dimension, 2).float() * denominator
#         )
#         code[:, 0::2] = torch.sin(sequence * term)
#         code[:, 1::2] = torch.cos(sequence * term)
#         code = code.unsqueeze(0).to(device)
#         if(trainable):
#             layer['code'] = torch.nn.Parameter(code, requires_grad=True)
#             self.layer = torch.nn.ParameterDict(layer)
#             return(True)
#         layer['code'] = torch.nn.Parameter(code, requires_grad=False)
#         self.layer = torch.nn.ParameterDict(layer) # shape: [1, max_len, d_model]
#         return(True)
    
#     def getPropagation(self, x: torch.Tensor) -> torch.Tensor:
#         l = x.size(1)
#         y = x + self.layer['code'][:, :l, :]
#         return(y)
    
#     forward = getPropagation
#     pass