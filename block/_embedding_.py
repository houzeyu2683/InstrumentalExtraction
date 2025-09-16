import torch

class Embedding(torch.nn.Module):

    def __init__(
        self, length: int, dimension: int, device: str
    ) -> None:
        super().__init__()
        self.length = length
        self.dimension = dimension
        self.device = device
        return

    def initiateLayer(
        self, 
    ) -> bool:
        layer = torch.nn.Embedding(
            num_embeddings=self.length,
            embedding_dim=self.dimension
        )
        self.layer = layer.to(self.device)
        return(True)
    
    def getFeedback(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layer(x)
        return(y)

    forward = getFeedback
    pass
