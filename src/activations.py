import torch
import torchvision
import torchlens as tl

def main ():
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alexnet = torchvision.models.alexnet().to(device)
    x = torch.rand(1, 3, 224, 224)
    model_history = tl.log_forward_pass(alexnet, x, layers_to_save='all')
    print(model_history)

if __name__ == '__main__':
    main()