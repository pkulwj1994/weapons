from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as T
dataset = CIFAR10('data', transform=T.Compose([
    T.Pad(4, padding_mode="reflect"),
    T.RandomCrop(32),
    T.RandomHorizontalFlip(),
    T.ToTensor(), 
    T.Normalize((0.5,0.5,0.5,), (0.5,0.5,0.5,)),
    lambda x: x + 0.03 * torch.randn_like(x)]), download=True)