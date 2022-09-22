
import torch
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import wideresnet
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization

#from configs.ncsnpp import cifar10_continuous_ve as configs
from configs.vp.ddpm import cifar10_ebm as configs
config = configs.get_config()

# checkpoint = torch.load('exp/ddpm_continuous_vp.pth')
#score_model = ncsnpp.NCSNpp(config)
# score_model = ddpm_model.DDPM(config)
score_model = wideresnet.WideResNet(config)
score_model = score_model.to(config.device)

# stat(score_model, (3, 16, 16))
# breakpoint()
# score_model.load_state_dict(checkpoint)
# score_model = score_model.eval()
x = torch.ones(2, 3, 16, 16).to(config.device).float()
x.requires_grad_()
y = torch.tensor([1.] * 2, requires_grad=True).to(config.device).float()
fx = score_model(x, y)
score = torch.autograd.grad(fx.sum(), x, retain_graph=True, create_graph=True)[0]
noise = torch.randn_like(x)
losses = torch.square(score - noise)
losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
loss = torch.mean(losses)
loss.backward()
print(score, loss)