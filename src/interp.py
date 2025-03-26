import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformer_lens import ActivationCache
from collections import defaultdict

from dataset import QuaternionDataset
from model import HookedQuatransformer
from config import SEQ_LEN, DEVICE, BATCH_SIZE, HOOK_NAMES

def get_activations(model: HookedQuatransformer, dataloader):
    model.eval()
    caches = []
    with torch.no_grad():
        for quaternions, _ in tqdm(dataloader):
            quaternions = quaternions.to(DEVICE)
            _, cache = model.run_with_cache(quaternions, names_filter=HOOK_NAMES)
            caches.append(cache)

    all_hooks = defaultdict(list)
    for cache in caches:
        for name, act in cache.items():
            all_hooks[name].append(act.cpu())

    merged = {k: torch.stack(v, dim=0).flatten(0, 1) for k, v in all_hooks.items()}
    return ActivationCache(merged, model)

def main():
    dataset = QuaternionDataset(n_samples=BATCH_SIZE * 4, sequence_length=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                    pin_memory=True, num_workers=4, persistent_workers=True)

    model = HookedQuatransformer().to(DEVICE)
    model = torch.compile(model)
    model.load_state_dict(torch.load("best_model.pt", weights_only=True), strict=False)
    model.eval()

    caches = get_activations(model, dataloader)
    print(caches.keys())

if __name__ == "__main__":
    main()