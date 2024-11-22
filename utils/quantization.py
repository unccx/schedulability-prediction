import time
from pathlib import Path

import torch
from torch.quantization import quantize_dynamic
from torch.utils.data import DataLoader

from data import LinkPredictDataset
from metric import LinkPredictionEvaluator as Evaluator

net = torch.load(
    Path("./net.pth"), map_location=torch.device("cpu"), weights_only=False
)
pred = torch.load(
    Path("./pred.pth"), map_location=torch.device("cpu"), weights_only=False
)
net.eval()
pred.eval()

quantized_net = quantize_dynamic(model=net)
quantized_pred = quantize_dynamic(model=pred)

print(quantized_net)
print(quantized_pred)
torch.save(quantized_net, Path("./quantized_net.pth"))
torch.save(quantized_pred, Path("./quantized_pred.pth"))

# device = torch.device("cpu")

# msg_pass_ratio: float = 0.7
# test_set = LinkPredictDataset(
#     dataset_name="2024-10-16_18-13-06",
#     root_dir=Path("./data"),
#     ratio=(msg_pass_ratio, 1 - msg_pass_ratio),
#     # data_balance=False,
#     device=device,
# )
# # batch_sz = 8
# # test_loader = DataLoader(test_set, batch_size=batch_sz, shuffle=False)
# evaluator = Evaluator(
#     ["auc", "accuracy", "f1_score", "confusion_matrix"], validate_index=1
# )
# dataset = test_set
# # dataset = data_loader.dataset

# start = time.time()
# hyperedge_embedding = net(dataset.X, dataset.msg_pass_hg)
# pos_scores = pred(hyperedge_embedding, dataset.pos_hg).squeeze()
# neg_scores = pred(hyperedge_embedding, dataset.neg_hg).squeeze()
# end = time.time()

# scores = torch.cat([pos_scores, neg_scores])
# labels = torch.cat(
#     [torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])]
# ).to(device)
# eval_res = evaluator.test(labels, scores)  # type: ignore
# print(f"eval res: {eval_res}")

# print(f"Inference latency: {end-start:.5f}sec.")


# start = time.time()
# hyperedge_embedding = quantized_net(dataset.X, dataset.msg_pass_hg)
# pos_scores = quantized_pred(hyperedge_embedding, dataset.pos_hg).squeeze()
# neg_scores = quantized_pred(hyperedge_embedding, dataset.neg_hg).squeeze()
# end = time.time()

# scores = torch.cat([pos_scores, neg_scores])
# labels = torch.cat(
#     [torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])]
# ).to(device)
# eval_res = evaluator.test(labels, scores)  # type: ignore

# print(f"Quantized eval res: {eval_res}")
# print(f"Quantized Inference latency: {end-start:.5f}sec.")
