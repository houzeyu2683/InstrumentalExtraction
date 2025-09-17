import torch
import os
import itertools
import safetensors.torch
import bitsandbytes
import torch.utils.tensorboard
import typing
import torchvision

class Dashboard:

    def __init__(self, path: str) -> None:
        self.path = path
        return

    def openSession(self) -> bool:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.session = torch.utils.tensorboard.SummaryWriter(self.path)
        return(True)

    def closeSession(self) -> bool:
        self.session.close()
        return(True)

    def insertElement(self, tag: str, value: float, step: int) -> bool:
        self.session.add_scalar(tag, value, step)
        return(True)

    def insertGrapgh(
        self, model: torch.nn.Module, variable: typing.Any
    ) -> bool:
        self.session.add_graph(model, variable)
        return(True)
    def insertPicture(
        self, tag: str, image: torch.Tensor, step: int
    ) -> bool:
        grid = torchvision.utils.make_grid(image)
        self.session.add_image(tag, grid, step)
        return(True)

    pass

class Framework:

    def __init__(self, model: torch.nn.Module, device: str) -> None:
        self.model = model
        self.device = device
        return

    def saveWeight(self, path: str) -> bool:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        safetensors.torch.save_file(self.model.state_dict(), path)
        return(True)

    def loadWeight(self, path: str) -> bool:
        state_dict = safetensors.torch.load_file(path)
        self.model.load_state_dict(state_dict)
        return(True)

    def initiateOptimization(self, rate: float) -> bool:
        self.optimization = bitsandbytes.optim.AdamW(
            self.model.parameters(), lr=rate
        )
        return(True)

    def getLoss(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        negative: torch.Tensor,

    ) -> torch.Tensor:
        score = 0
        if('pixel loss'):
            criteria = torch.nn.L1Loss()
            score += criteria(anchor, positive)
            pass
        if('embedding loss'):
            criteria = torch.nn.TripletMarginLoss(margin=1.0, p=2)
            if(hasattr(self, 'backbone')==False):
                net = torchvision.models.mobilenet_v2(
                    weights='MobileNet_V2_Weights.IMAGENET1K_V1'
                )
                backbone = torch.nn.Sequential(
                    *list(net.features.children())[:-1], 
                    torch.nn.AdaptiveAvgPool2d((1,1)),
                    torch.nn.Flatten(1, -1)
                )
                backbone.eval()
                for param in backbone.parameters():
                    param.requires_grad = False
                    continue
                self.backbone = backbone.to(self.device)
                pass
            distance = criteria(
                self.backbone(anchor), 
                self.backbone(positive), 
                self.backbone(negative)
            )
            score += distance
            pass
        loss = score
        return(loss)

    def fitWeight(
        self, 
        data: torch.utils.data.DataLoader,
        accumulation: int, 
        end: int,
        history: str
    ) -> bool:
        dashboard = Dashboard(path=history)
        dashboard.openSession()
        gradient = torch.amp.GradScaler()
        if(True):
            checkpoint = os.path.join(history, 'weight/0.pt')
            self.saveWeight(path=checkpoint)
            pass
        self.model.train()
        iteration = enumerate(itertools.cycle(data), 1)
        for step, batch in iteration:
            # moment, future, forgery, timestep = batch
            x, p, n, t = batch
            with torch.amp.autocast(self.device):
                y = self.model(x, t)
                loss = self.getLoss(y, p-x, n-x) / accumulation  # 分攤梯度
                pass
            gradient.scale(loss).backward()
            update = (step%accumulation)==0
            if(update):
                gradient.step(self.optimization)
                gradient.update()
                self.optimization.zero_grad()
                pass
            _ = update
            value = loss.item() * accumulation
            dashboard.insertElement('Loss/Train', value, step)
            print(f"[Step: {step} | loss: {value:.4f}]", end='\r')
            snapshot = (step%5000)==0
            if(snapshot):
                dashboard.insertPicture(
                    'Train/Gradient Prediction', x[:16,:,:,:]+y[:16,:,:,:], step
                )
                dashboard.insertPicture(
                    'Train/Gradient Truth', p[:16,:,:,:], step
                )
                checkpoint = os.path.join(history, f'weight/{step}.pt')
                self.saveWeight(path=checkpoint)
                pass
            _ = snapshot
            if(end!=-1 and step==end): break
            continue
        _ = iteration
        return(True)

    # def saveInference(
    #     self, 
    #     video: torch.Tensor, 
    #     path: str,
    #     step: int, 
    #     device: str
    # ) -> bool:
    #     self.model.eval()
    #     iteration = range(step)
    #     for index in iteration:
    #         if(index==0):
    #             l = len(video)
    #             x = video[None, :, :, :, :]
    #             m = torch.tensor([[False]*l]).to(device)
    #             pass
    #         y = self.model(x, m)[:, -1, :, :, :]
    #         f = x[:, -1, :, :, :] + y
    #         l = l + 1
    #         x = torch.cat([x, f[:, None, :, :, :]], dim=1)
    #         m = torch.tensor([[False]*l]).to(device)
    #         continue
    #     _ = iteration
    #     # inference = x.squeeze(0)
    #     getImage = torchvision.transforms.ToPILImage()
    #     sequence = [numpy.array(getImage(f)) for f in x.squeeze(0)]
        
    #     inference = moviepy.ImageSequenceClip(sequence, 25)
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #     inference.write_videofile(path, codec="libx264", audio=False)
    #     return(True)

    pass

