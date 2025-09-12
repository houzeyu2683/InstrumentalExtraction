import torch

length = 750
layer = {
    '(1) convolutional': torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
        torch.nn.GroupNorm(num_groups=8, num_channels=32),
        torch.nn.ReLU()
    ),
    '(1) residual': torch.nn.Sequential(
        torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        torch.nn.GroupNorm(num_groups=8, num_channels=32),
        torch.nn.ReLU()
    ),
    "(1) projective": torch.nn.Embedding(
        num_embeddings=length, embedding_dim=32
    ),
    '(2) convolutional': torch.nn.Sequential(
        torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        torch.nn.GroupNorm(num_groups=8, num_channels=64),
        torch.nn.ReLU()
    ),
    '(2) residual': torch.nn.Sequential(
        torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.GroupNorm(num_groups=8, num_channels=64),
        torch.nn.ReLU()
    ),
    "(2) projective": torch.nn.Embedding(
        num_embeddings=length, embedding_dim=64
    ),
    '(3) convolutional': torch.nn.Sequential(
        torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        torch.nn.GroupNorm(num_groups=8, num_channels=128),
        torch.nn.ReLU()
    ),
    '(3) residual': torch.nn.Sequential(
        torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        torch.nn.GroupNorm(num_groups=8, num_channels=128),
        torch.nn.ReLU()
    ),
    "(3) projective": torch.nn.Embedding(
        num_embeddings=length, embedding_dim=128
    ),
    '(4) convolutional': torch.nn.Sequential(
        torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        torch.nn.GroupNorm(num_groups=8, num_channels=256),
        torch.nn.ReLU()
    ),
    '(4) residual': torch.nn.Sequential(
        torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.GroupNorm(num_groups=8, num_channels=256),
        torch.nn.ReLU()
    ),
    "(4) projective": torch.nn.Embedding(
        num_embeddings=length, embedding_dim=256
    ),
    '(5) attention': None,
}
layer = torch.nn.ModuleDict(layer)

x = torch.randn(8, 3, 128, 128)
t = torch.randint(0, 20, (8, ))

w1 = x
w1 = layer['(1) convolutional'](w1)
w1 = w1 + layer['(1) residual'](w1)
t1 = layer['(1) projective'](t)
w1 = w1 + t1[:,:,None,None]

w2 = layer['(2) convolutional'](w1)
w2 = w2 + layer['(2) residual'](w2)
t2 = layer['(2) projective'](t)
w2 = w2 + t2[:,:,None,None]

w3 = layer['(3) convolutional'](w2)
w3 = w3 + layer['(3) residual'](w3)
t3 = layer['(3) projective'](t)
w3 = w3 + t3[:,:,None,None]

w4 = layer['(4) convolutional'](w3)
w4 = w4 + layer['(4) residual'](w4)
t4 = layer['(4) projective'](t)
w4 = w4 + t4[:,:,None,None]

assert isinstance(w4, torch.Tensor)
w4 = w4.flatten(2, -1)
w4.shape





'''
            # 64 -> 32
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 32 -> 16
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 16 -> 8
            nn.Conv2d(base_channels*4, base_channels*8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 8 -> 4
            nn.Conv2d(base_channels*8, base_channels*16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 4 -> 2
            nn.Conv2d(base_channels*16, base_channels*32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 2 -> 1
            nn.Conv2d(base_channels*32, base_channels*64, kernel_size=2),  # no stride needed, kernel=2
            nn.ReLU(inplace=True),
        )
'''