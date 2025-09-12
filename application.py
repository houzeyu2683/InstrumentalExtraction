
import cloud
import architecture

device = 'cpu'
hub = cloud.Hub(folder='./resource/')
hub.initiateLibrary()
test = hub.getTest(batch=1, device=device)

hamster = architecture.Hamster(embedding=256)
hamster.initiateLayer(device=device)

framework = architecture.Framework(model=hamster)
framework.loadWeight('experiment/0828/weight/200000.pt')
# framework.initiateOptimization(rate=1e-4)
# framework.fitWeight(data, device, accumulation=64, end=-1)
video = next(iter(test))[0][0,:,:,:,:]
inference = framework.saveInference(video, path='./test.mp4', step=100, device=device)


# from moviepy.editor import ImageSequenceClip
# import numpy as np

# # 假設 frames 是一個 list，每個元素都是 np.ndarray，形狀大概是 (H, W, 3)，dtype=uint8
# # 例如: frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(100)]

# def save_video(frames, output_path="output.mp4", fps=30):
#     # 建立影片 clip
#     clip = ImageSequenceClip(frames, fps=fps)
#     # 存檔
#     clip.write_videofile(output_path, codec="libx264", audio=False)

# # 用法
# # save_video(frames, "test.mp4", fps=25)
