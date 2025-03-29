import InstrumentalExtraction

engine = InstrumentalExtraction.machine.Engine(
    checkpoint='./.cache/weight.pth', 
    device='cuda', 
    half=True, 
    storage='./sample'
)
_ = engine.loadModel()
engine.inferVoice('./sample/News.wav')