import cloud
import architecture

device = 'cuda'
hub = cloud.Hub(folder='./rocket/20250906')
hub.initiateLibrary()
data = hub.getData(batch=256, device=device)

hamster = architecture.Hamster(device=device)
hamster.initiateLayer()

framework = architecture.Framework(model=hamster, device=device)
framework.initiateOptimization(rate=1e-4)
framework.fitWeight(data, accumulation=40, end=-1, history='experiment/0917')
