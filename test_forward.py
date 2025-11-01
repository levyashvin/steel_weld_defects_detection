import torch
from lf_yolo import LFYOLO_WeldDefect

m = LFYOLO_WeldDefect(nc=7)
m.eval()
x = torch.randn(1,3,640,640)
with torch.no_grad():
    out = m(x)
def describe(obj, _depth=0):
    # Return a short description for tensors, lists and other objects
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return f"Tensor shape={tuple(obj.shape)} dtype={obj.dtype}"
    except Exception:
        pass
    if isinstance(obj, (list, tuple)):
        inner = ', '.join(describe(o, _depth+1) for o in obj[:3])
        if len(obj) > 3:
            inner += ', ...'
        return f"{type(obj).__name__}(len={len(obj)}) -> [{inner}]"
    return repr(obj)

print("Forward OK.")
print("Output description:", describe(out))