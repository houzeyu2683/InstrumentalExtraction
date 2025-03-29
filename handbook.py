import InstrumentalExtraction

engine = InstrumentalExtraction.Engine(
    checkpoint='./.cache/weight.pth', 
    device='cuda', 
    half=True, 
    storage='./sample'
)
_ = engine.loadModel()
engine.inferVoice('./sample/News.wav')