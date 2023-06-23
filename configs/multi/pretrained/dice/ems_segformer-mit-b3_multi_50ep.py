_base_ = [
    "../../../models/segformer_mit-b3_aux.py",
    "../../../datasets/ems.py",
]
name = "segformer-mit-b3_multi_imnet_auxv2_50ep"
trainer = dict(
    max_epochs=50,
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices=1,
)
data = dict(
    batch_size_train=32,
    batch_size_eval=32,
    num_workers=8,
)
evaluation = dict(
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices=1,
)
loss="dice"
