import cloud
import torch
import flash_attn
import block

hub = cloud.Hub(folder='./rocket/20250906')
hub.initiateLibrary()
data = hub.getData(batch=4, device='cuda')
batch = next(iter(data))
print(batch)
l, m, x = batch[0]

x = x[:, :, :, 0, 0]
p = torch.nn.Linear(3, 256, bias=False).to('cuda')
x = p(x)
# x = torch.randn(m.shape+(256,)).to('cuda')
x = x.masked_fill(m.unsqueeze(-1), 0)
encoder = block.attention.Encoder(embedding=256, head=8, device='cuda')
encoder.initiateLayer()
y = encoder(x, m)
print(y.shape)


# x = torch.cat(
#     [torch.zeros(5, 8), torch.randn(7, 8)],
#     dim=0
# )
# x.shape
# ln = torch.nn.LayerNorm(8)
# ln(x)==0
# import cloud
# import torch
# import flash_attn

# hub = cloud.Hub(folder='./rocket/20250906')
# hub.initiateLibrary()
# data = hub.getData(batch=4, device='cuda')
# batch = next(iter(data))
# print(batch)




# l, m, x = batch[0]
# m = m.cpu()
# l = l.cpu()

# m_list = (~m).sum(dim=1)

# e = torch.randn(m.shape+(64,))
# B, L, _ = e.shape

# embed_dim = 256
# head = 8
# emb_proj = torch.nn.Linear(64, embed_dim, bias=True)
# assert embed_dim%head==0
# q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)
# k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)
# v_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)
# out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=True)

# s = emb_proj(e)

# q = q_proj(s)
# q_seq = [q_i[:m_i] for q_i, m_i in zip(q, m_list)]
# q_seq = torch.cat(q_seq, dim=0)
# q_L = len(q_seq)
# q_seq = q_seq.reshape(q_L, head, embed_dim//head)
# q_seq.shape

# k = k_proj(s)
# k_seq = [k_i[:m_i] for k_i, m_i in zip(k, m_list)]
# k_seq = torch.cat(k_seq, dim=0)
# k_L = len(k_seq)
# k_seq = k_seq.reshape(k_L, head, embed_dim//head)
# k_seq.shape

# v = v_proj(s)
# v_seq = [v_i[:m_i] for v_i, m_i in zip(v, m_list)]
# v_seq = torch.cat(v_seq, dim=0)
# v_L = len(v_seq)
# v_seq = v_seq.reshape(v_L, head, embed_dim//head)
# v_seq.shape

# cu_seqlens = torch.cat(
#     [torch.zeros(1, dtype=torch.int32), l.cumsum(dim=0)], 
#     dim=0
# ).to(torch.int32)

# out = flash_attn.flash_attn_varlen_func(
#     q_seq.cuda().half(),
#     k_seq.cuda().half(),
#     v_seq.cuda().half(),
#     cu_seqlens_q=cu_seqlens.cuda(),
#     cu_seqlens_k=cu_seqlens.cuda(),
#     max_seqlen_q=L,
#     max_seqlen_k=L,

# )

# out.shape


# def getDeformation(e: torch.Tensor, m: torch.Tensor, h: int) -> tuple:
#     # e: (B, L, C)
#     # m: (B, L)
#     b, l, d = e.shape
    
#     r = (~m).sum(dim=1)
#     assert d%h==0
#     assert len(r)==b
#     s = [k[:v] for k, v in zip(e, r)]
#     s = torch.cat(s, dim=0)
#     # q_L = len(q_seq)
#     s = s.reshape(len(s), h, d//h)
#     # y = (s.half(), )
#     c = torch.tensor([0] + r.tolist(), dtype=torch.int32).cumsum(dim=0)
#     return

# # def getVolume()


