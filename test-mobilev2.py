import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
# from tqdm import tqdm

# 假設 frames 是 list of PIL.Image 或 numpy array
# frames = [...]  # 影片的每一幀
x = torch.randn(1, 3, 128, 128).cuda()
# 1. 載入 MobileNetV2 backbone
device = "cuda"
mobilenet = models.mobilenet_v2(pretrained=True).features.to(device)
mobilenet = torch.nn.Sequential(*list(mobilenet.children())[:-1], torch.nn.AdaptiveAvgPool2d((1,1)))

# mobilenet = torch.nn.Sequential(
#     mobilenet.features,
#     torch.nn.AdaptiveAvgPool2d((1,1))  # global average pooling
# ).to(device)
mobilenet.eval()  # 不需要梯度

# 2. 定義 preprocessing
preprocess = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


with torch.no_grad():
    feats = mobilenet(x)  # [B, 1280, H', W']
    feats = torch.mean(feats, dim=[2,3])  # global average pooling -> [B, 1280]
    pass

# # 3. 批次大小
# batch_size = 32
# all_features = []

# # 4. 分批處理 frames
# for i in tqdm(range(0, len(frames), batch_size)):
#     batch_frames = frames[i:i+batch_size]
    
#     # 將 frames 轉成 tensor
#     batch_tensors = torch.stack([preprocess(f) for f in batch_frames]).to(device)  # [B,3,H,W]
    
    
#     all_features.append(feats.cpu())

# # 5. 合併所有 batch
# all_features = torch.cat(all_features, dim=0)  # [num_frames, 1280]
# print("All frame features shape:", all_features.shape)

# # 6. 可選：得到整段影片特徵（平均 pooling）
# video_feature = all_features.mean(dim=0)  # [1280]
# print("Video feature shape:", video_feature.shape)
