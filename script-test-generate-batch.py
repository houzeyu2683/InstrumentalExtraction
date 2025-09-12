import cloud

hub = cloud.Hub(folder='./rocket/20250906')
hub.initiateLibrary()
data = hub.getData(batch=4, device='cuda')
batch = next(iter(data))
print(batch)


