import torch
import flash_attn
import einops

class Attention(torch.nn.Module):

    def __init__(self, embedding: int, head: int, causal: bool, device: str) -> None:
        super().__init__()
        self.embedding = embedding
        self.head = head
        self.causal = causal
        self.device = device
        return
    
    def initiateLayer(self) -> bool:
        assert self.embedding%self.head==0
        layer = {
            '(1) projection': torch.nn.Linear(
                self.embedding, 
                self.embedding, 
                bias=False
            ),
            '(2) query': torch.nn.Linear(
                self.embedding, 
                self.embedding, 
                bias=False
            ),
            '(2) key': torch.nn.Linear(
                self.embedding, 
                self.embedding, 
                bias=False
            ),
            '(2) value': torch.nn.Linear(
                self.embedding, 
                self.embedding, 
                bias=False
            ),
            '(2) norm': torch.nn.LayerNorm(
                self.embedding
            ),
            '(3) projection': torch.nn.Linear(
                self.embedding, 
                self.embedding, 
                bias=False
            ),
            '(3) norm': torch.nn.LayerNorm(
                self.embedding
            )
        }
        self.layer = torch.nn.ModuleDict(layer).to(self.device)
        return(True)

    def getFeedback(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        c = self.layer['(1) projection'](x)
        q = self.layer['(2) query'](c) # e: (b, l, d) -> q: (b, l, d)
        k = self.layer['(2) key'](c) # e: (b, l, d) -> k: (b, l, d)
        v = self.layer['(2) value'](c) # e: (b, l, d) -> v: (b, l, d)
        #
        h = self.head
        q = einops.rearrange(q, 'b l (h d) -> b l h d', h=h)
        k = einops.rearrange(k, 'b l (h d) -> b l h d', h=h)
        v = einops.rearrange(v, 'b l (h d) -> b l h d', h=h)
        assert isinstance(q, torch.Tensor)
        assert isinstance(k, torch.Tensor)
        assert isinstance(v, torch.Tensor)
        q=q.half()
        k=k.half()
        v=v.half()
        #
        a = flash_attn.flash_attn_func(
            q=q,
            k=k,
            v=v,
            dropout_p=0,
            causal=self.causal
        )
        assert isinstance(a, torch.Tensor)
        a = einops.rearrange(a, 'b l h d -> b l (h d)')
        # a = a.masked_fill(m.unsqueeze(-1), 0)
        o = x + a
        o = self.layer['(2) norm'](o)
        #
        o = o + self.layer['(3) projection'](o)
        y = self.layer['(3) norm'](o)
        return(y)

    forward = getFeedback
    pass
