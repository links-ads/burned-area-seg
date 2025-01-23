from datetime import datetime
from functools import partial
from pathlib import Path

from argdantic import ArgField, ArgParser
from loguru import logger as log
from mmengine import Config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from baseg.datamodules import EFFISDataModule
from baseg.io import read_raster_profile, write_raster
from baseg.modules import SingleTaskRegressionModule
from baseg.tiling import SmoothTiler
from baseg.utils import exp_name_timestamp, find_best_checkpoint
from baseg.modules.unet import UnetModule
import logging

cli = ArgParser()


@cli.command()
def train(
    cfg_path: Path = ArgField("-c", description="Path to the config file."),
    keep_name: bool = ArgField(
        "-k", default=False, description="Keep the experiment name as specified in the config file."
    ),
):
    log.info(f"Loading config from: {cfg_path}")
    config = Config.fromfile(cfg_path)
    # set the experiment name
    assert "name" in config, "Experiment name not specified in config."
    exp_name = exp_name_timestamp(config["name"]) if not keep_name else config["name"]
    config["name"] = exp_name
    log.info(f"Experiment name: {exp_name}")

    # datamodule
    log.info("Preparing the data module...")
    datamodule = EFFISDataModule(**config["data"])
    modalities = config["data"]["modalities"]
    gt_label = "mask" if "mask" in modalities else "dNBR"
    img_label = "post"

        
    log_dir = config["log_dir"] if "log_dir" in config else "outputs"
    # prepare the model
    log.info("Preparing the model...")
    model_config = config["model"]
    module = UnetModule(n_channels=model_config["n_channels"], n_classes=model_config["n_classes"], img_label=img_label, gt_label=gt_label) 

    log.info("Preparing the trainer...")
    logger = TensorBoardLogger(save_dir=log_dir, name=exp_name)
    config_dir = Path(logger.log_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    config.dump(config_dir / "config.py")
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(logger.log_dir) / "weights",
            monitor="epoch",
            mode="max",
            filename="model-{epoch:02d}-{val_loss:.2f}",
            save_top_k=6,
            every_n_epochs=10,
        )
    ]
    trainer = Trainer(**config["trainer"], callbacks=callbacks, logger=logger)

    log.info("Starting the training...")
    trainer.fit(module, datamodule=datamodule)


@cli.command()
def test(
    exp_path: Path = ArgField("-e", description="Path to the experiment folder."),
    checkpoint: Path = ArgField(
        "-c",
        default=None,
        description="Path to the checkpoint file. If not specified, the best checkpoint will be loaded.",
    ),
    predict: bool = ArgField(default=False, description="Generate predictions on the test set."),
):

    log.info(f"Loading experiment from: {exp_path}")
    config_path = exp_path / "config.py"
    models_path = exp_path / "weights"
    # asserts to check the experiment folders
    assert exp_path.exists(), "Experiment folder does not exist."
    assert config_path.exists(), f"Config file not found in: {config_path}"
    assert models_path.exists(), f"Models folder not found in: {models_path}"
    # load training config
    config = Config.fromfile(config_path)

    # datamodule
    log.info("Preparing the data module...")
    datamodule = EFFISDataModule(**config["data"])
    modalities = config["data"]["modalities"]
    gt_label = "mask" if "mask" in modalities else "dNBR"
    img_label = "post"
        
    
    
    # prepare the model
    checkpoint = checkpoint or find_best_checkpoint(models_path, "val_loss", "min")
    log.info(f"Using checkpoint: {checkpoint}")
    module_opts = dict(config=config["model"])
    if predict:
        tiler = SmoothTiler(
            tile_size=config["data"]["patch_size"],
            batch_size=config["data"]["batch_size_eval"],
            channels_first=True,
            mirrored=False,
        )
        output_path = exp_path / "predictions"
        output_path.mkdir(parents=True, exist_ok=True)
        inference_fn = partial(process_inference, output_path=output_path, img_label = "post", is_effis = True)
        module_opts.update(tiler=tiler, predict_callback=inference_fn)

    # prepare the model
    log.info("Preparing the model...")
    model_config = config["model"]
    module_class = SingleTaskRegressionModule
    module = module_class.load_from_checkpoint(checkpoint, **module_opts, img_label=img_label, gt_label=gt_label)
    log_dir = config["log_dir"] if "log_dir" in config else "outputs"
    logger1 = TensorBoardLogger(save_dir=log_dir, name=config["name"], version=exp_path.stem)
    logger2 = CSVLogger(save_dir=log_dir, name=config["name"], version=exp_path.stem)

    if predict:
        log.info("Generating predictions...")
        trainer = Trainer(**config["evaluation"], logger=False)
        trainer.predict(module, datamodule=datamodule, return_predictions=False)
    else:
        log.info("Starting the testing...")
        trainer = Trainer(**config["evaluation"], logger=[logger1, logger2])
        trainer.test(module, datamodule=datamodule)


@cli.command()
def test_multi(
    root: Path = ArgField("-r", description="Path to the root folder of the experiments."),
    from_date: datetime = ArgField("-f", default=None, description="Start date for the experiments to test."),
    epoch: int = ArgField("-e", default=None, description="Number of epochs to test."),
):
    assert root.exists(), f"Root folder does not exist: {root}"
    experiments = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith("_")]

    for exp_path in experiments:
        exp_path = exp_path / "version_0"
        log.info(f"Testing experiment: {exp_path}")
        config_path = exp_path / "config.py"
        weights_path = exp_path / "weights"

        if not config_path.exists():
            log.warning(f"Config file not found in: {config_path}")
            continue
        if not weights_path.exists():
            log.warning(f"Models folder not found in: {weights_path}")
            continue

        # parse timestamp with format <name_with_underscores>_<date>_<time>, exclude the name
        timestamp = "_".join(exp_path.parent.stem.split("_")[-2:])
        timestamp = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        if from_date and timestamp < from_date:
            log.info(f"Skipping experiment from {timestamp}, too old.")
            continue

        checkpoint = None
        if epoch is not None:
            checkpoint = list(weights_path.glob(f"model-epoch={epoch}*.ckpt"))
            if not checkpoint:
                log.warning(f"Checkpoint not found for epoch {epoch} in: {weights_path}")
                continue
            checkpoint = checkpoint[0]

        test.callback(exp_path, checkpoint=checkpoint, predict=False)


def process_inference(
    batch: dict,
    output_path: Path,
    img_label: str = "post",
    is_effis: bool = True,
):
    assert output_path.exists(), f"Output path does not exist: {output_path}"
    # for binary segmentation
    prediction = (batch["pred"] > 0.5).int().unsqueeze(0)
    prediction = prediction.cpu().numpy()
    # store the prediction as a GeoTIFF, reading the spatial information from the input image
    
    image_path = Path(batch["metadata"][img_label][0])
    input_profile = read_raster_profile(image_path)
    output_profile = input_profile.copy()
    output_profile.update(dtype="uint8", count=1)
    if is_effis:
        output_file = output_path / f"{str(image_path).split('/')[-3]}_{image_path.stem}.tif"
    else:
       output_file = output_path / f"{image_path.stem}.tif"
    write_raster(path=output_file, data=prediction, profile=output_profile)


if __name__ == "__main__":
    seed_everything(95, workers=True)
    cli()
