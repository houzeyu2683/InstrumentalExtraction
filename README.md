# InstrumentalExtraction

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-orange)](https://pytorch.org/)

Last updated: 2025-03-30

This project is based on the code originally created by the contributor [seanghay][1].
I have made some modifications and rewrites to make it more convenient for integration into my own Python scripts. 
I have packaged it into a format that can be installed via `pip install`,
with the hope that it will be helpful to others who may need it.

```
pip install git+https://github.com/houzeyu2683/InstrumentalExtraction.git
```

The model weights can be downloaded from [here][2].
For usage example, please refer to `handbook.py`.

```python
import InstrumentalExtraction.machine

engine = InstrumentalExtraction.machine.Engine(
    checkpoint='./.cache/weight.pth', 
    device='cuda', 
    half=True, 
    storage='sample'
)
_ = engine.loadModel()
engine.inferVoice('./sample/News.wav')
```

I hope this helps you. If you find it helpful, please give me a star.

---

[1]: https://github.com/seanghay/uvr
[2]: https://drive.google.com/file/d/1sLkDnRif3ey9YCJTjUn2ovSxmAnEOiYF/view?usp=sharing