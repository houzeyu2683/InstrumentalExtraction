import torch
import math
import block

class Hamster(torch.nn.Module):
    
    def __init__(self, device: str) -> None:
        super().__init__()
        self.device = device
        return
    
    def initiateLayer(self) -> bool:
        position = block.Position(
            length=750, 
            dimension=320, 
            trainable=True, 
            device=self.device
        )
        attention = block.Attention(
            embedding=320, 
            head=8, 
            causal=True, 
            device=self.device
        )
        edge = block.Edge(device=self.device)
        position.initiateLayer()
        attention.initiateLayer()
        edge.initiateLayer()
        layer = {
            '(1) position': position,
            '(2) attention': attention,
            '(3) edge': edge
        }
        self.layer = torch.nn.ModuleDict(layer).to(self.device)
        return(True)

    def getFeedback(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        b, l, c, h, w = x.shape
        

        
        x = self.layer['(1) position'](x)
        x = x.masked_fill(m.unsqueeze(-1), 0)
        a = self.layer['(2) attention'](x, m)
        p = self.layer['(3) edge'](a, m)


        x = x.reshape(n*l, c, h, w)
        e = []
        for f in self.layer['eye']:
            x = f(x)
            e += [x]
            continue
        x = e.pop()
        # x = self.layer['encoder'](x)
        x = x.reshape(n, l, self.embedding, 1, 1)
        x = x.flatten(2)
        x = self.layer['position'](x)
        x = self.layer['norm'](x)
        x = self.layer['attention'](
            x, src_key_padding_mask=m, is_causal=None
        )
        # x = x.unsqueeze(-1).unsqueeze(-1)
        x = x.reshape(n*l, self.embedding, 1, 1)
        e.reverse()
        for i, f in enumerate(self.layer['hand']):
            x = torch.cat([f(x), e[i]], dim=1)
            continue
        x = self.layer['pixel'](x)
        y = x.reshape(n, l, c, h, w)
        # y = x
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