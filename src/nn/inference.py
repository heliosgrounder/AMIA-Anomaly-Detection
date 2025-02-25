import torch
from tqdm import tqdm

from src.utils.constants import DEVICE


def run(model,
        data_loader,
        device=DEVICE,
        score_threshold=0.5):
    model.eval()
    outputs_list = []

    with torch.no_grad():
        for images, uuids in tqdm(data_loader):
            images = list(img["image"].to(device) for img in images)
            outputs = model(images)

            processed = []
            for out in outputs:
                boxes = out["boxes"]
                labels = out["labels"]
                scores = out["scores"]

                keep_mask = scores >= score_threshold
                filtered = {
                    "boxes": boxes[keep_mask].cpu(),
                    "labels": labels[keep_mask].cpu(),
                    "scores": scores[keep_mask].cpu(),
                    "uuid": uuids
                }
                processed.append(filtered)
            
            outputs_list.extend(processed)
    return outputs_list
