
import cloud
import architecture
# import framework

device = 'cuda'
hub = cloud.Hub(folder='./resource/')
hub.initiateLibrary()
data = hub.getData(batch=2, device=device)

hamster = architecture.Hamster(embedding=256)
hamster.initiateLayer(device=device)

framework = architecture.Framework(model=hamster)
framework.initiateOptimization(rate=1e-4)
framework.fitWeight(data, device, accumulation=512, end=-1, history='experiment/0828')

