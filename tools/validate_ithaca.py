import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from prettytable import PrettyTable
from torchmetrics import F1Score, JaccardIndex, Precision, Recall
from tqdm import tqdm

del_palette = {
    0: (64, 64, 64),
    1: (255, 255, 200),
    255: (0, 0, 0),
    2: (255, 0, 0),
    3: (0, 255, 0)
}


def mask2rgb(image: np.ndarray, palette: dict) -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for k, v in palette.items():
        lut[k, :] = v
    return lut[image]


def read_raster_sentinel(path):
    with rasterio.open(path) as src:
        data = src.read()
        profile = src.profile
    return data, profile


def read_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile
    return data.astype(np.uint8), profile


def parse_args():
    Parser = argparse.ArgumentParser(description="Parse arguments")
    Parser.add_argument("--effis_folder",
                        type=Path,
                        help="Path to the  folder of EFFIS data.")
    Parser.add_argument("--ithaca_folder",
                        type=Path,
                        help="Path to the folder of Ithaca data.")
    Parser.add_argument("--output_folder",
                        type=Path,
                        help="Path to the output folder.")
    Parser.add_argument("--model_to_test",
                        type=str,
                        default="upernet-rn50_multi_ssl4eo_50ep")
    return Parser.parse_args()


def main():
    args = parse_args()
    assert args.effis_folder.exists(), "EFFIS folder does not exist."
    assert args.ithaca_folder.exists(), "Ithaca folder does not exist."
    args.output_folder.mkdir(exist_ok=True, parents=True)
    pixels_iou_effis_vs_ithaca = JaccardIndex(task="binary")
    pixels_iou_model_vs_ithaca = JaccardIndex(task="binary")
    pixels_precision_effis_vs_ithaca = Precision(task="binary")
    pixels_precision_model_vs_ithaca = Precision(task="binary")
    pixels_recall_effis_vs_ithaca = Recall(task="binary")
    pixels_recall_model_vs_ithaca = Recall(task="binary")
    pixels_f1_effis_vs_ithaca = F1Score(task="binary")
    pixels_f1_model_vs_ithaca = F1Score(task="binary")

    files = [f for f in args.ithaca_folder.glob("*/*/*_DEL.tif")]
    images_iou_effis_vs_ithaca = []
    images_iou_model_vs_ithaca = []
    images_precision_effis_vs_ithaca = []
    images_precision_model_vs_ithaca = []
    images_recall_effis_vs_ithaca = []
    images_recall_model_vs_ithaca = []
    images_f1_effis_vs_ithaca = []
    images_f1_model_vs_ithaca = []

    effis_imgs = []
    ithaca_imgs = []
    model_imgs = []

    tab_images = PrettyTable(['Metric', 'EFFIS vs ITHACA', 'MODEL vs ITHACA'])
    tab_pixels = PrettyTable(['Metric', 'EFFIS vs ITHACA', 'MODEL vs ITHACA'])

    for file in tqdm(files):
        images_iou_effis_vs_ithaca_fn = JaccardIndex(task="binary")
        images_iou_model_vs_ithaca_fn = JaccardIndex(task="binary")
        images_precision_effis_vs_ithaca_fn = Precision(task="binary")
        images_precision_model_vs_ithaca_fn = Precision(task="binary")
        images_recall_effis_vs_ithaca_fn = Recall(task="binary")
        images_recall_model_vs_ithaca_fn = Recall(task="binary")
        images_f1_effis_vs_ithaca_fn = F1Score(task="binary")
        images_f1_model_vs_ithaca_fn = F1Score(task="binary")
        effis_code = file.stem.split("_")[0]
        ithaca_img, _ = read_raster(file)
        effis_img, _ = read_raster(args.effis_folder / effis_code / "mask" /
                                   "mask.tiff")
        sentinel_img, _ = read_raster_sentinel(args.effis_folder / effis_code /
                                               "post" / "map.tiff")
        assert (args.effis_folder / effis_code / "predictions" /
                args.model_to_test).exists(), "Model predictions do not exist."
        prediction_img, _ = read_raster(args.effis_folder / effis_code /
                                        "predictions" / args.model_to_test /
                                        "del.tiff")

        # compare the two images
        # compute iou

        effis_img = torch.from_numpy(effis_img)
        ithaca_img = torch.from_numpy(ithaca_img)
        prediction_img = torch.from_numpy(prediction_img)

        images_iou_effis_vs_ithaca.append(
            images_iou_effis_vs_ithaca_fn(effis_img, ithaca_img))

        images_iou_model_vs_ithaca.append(
            images_iou_model_vs_ithaca_fn(prediction_img, ithaca_img))

        images_precision_effis_vs_ithaca.append(
            images_precision_effis_vs_ithaca_fn(effis_img, ithaca_img))

        images_precision_model_vs_ithaca.append(
            images_precision_model_vs_ithaca_fn(prediction_img, ithaca_img))

        images_recall_effis_vs_ithaca.append(
            images_recall_effis_vs_ithaca_fn(effis_img, ithaca_img))

        images_recall_model_vs_ithaca.append(
            images_recall_model_vs_ithaca_fn(prediction_img, ithaca_img))
        images_f1_effis_vs_ithaca.append(
            images_f1_effis_vs_ithaca_fn(effis_img, ithaca_img))
        images_f1_model_vs_ithaca.append(
            images_f1_model_vs_ithaca_fn(prediction_img, ithaca_img))

        pixels_iou_effis_vs_ithaca.update(effis_img, ithaca_img)
        pixels_iou_model_vs_ithaca.update(prediction_img, ithaca_img)
        pixels_precision_effis_vs_ithaca.update(effis_img, ithaca_img)
        pixels_precision_model_vs_ithaca.update(prediction_img, ithaca_img)
        pixels_recall_effis_vs_ithaca.update(effis_img, ithaca_img)
        pixels_recall_model_vs_ithaca.update(prediction_img, ithaca_img)
        pixels_f1_effis_vs_ithaca.update(effis_img, ithaca_img)
        pixels_f1_model_vs_ithaca.update(prediction_img, ithaca_img)

        # create plots
        fig, axs = plt.subplots(1, 4, figsize=(60, 20))
        for a in axs:
            a.set_axis_off()
        map_img = np.clip(sentinel_img[1:4][::-1] * 2.5, 0, 1)
        axs[0].imshow(map_img.transpose(1, 2, 0))
        axs[0].set_title("S2 Post fire map")
        axs[1].imshow(mask2rgb(prediction_img, del_palette))
        axs[1].set_title("Our prediction")
        axs[2].imshow(mask2rgb(effis_img, del_palette))
        axs[2].set_title("EFFIS mask")
        axs[3].imshow(mask2rgb(ithaca_img, del_palette))
        axs[3].set_title("GT")
        plt.tight_layout()
        plt.savefig(args.output_folder / f"EFFIS_{effis_code}_comparison.png")

    images_iou_effis_vs_ithaca = np.array(images_iou_effis_vs_ithaca)
    images_iou_model_vs_ithaca = np.array(images_iou_model_vs_ithaca)
    images_precision_effis_vs_ithaca = np.array(
        images_precision_effis_vs_ithaca)
    images_precision_model_vs_ithaca = np.array(
        images_precision_model_vs_ithaca)
    images_recall_effis_vs_ithaca = np.array(images_recall_effis_vs_ithaca)
    images_recall_model_vs_ithaca = np.array(images_recall_model_vs_ithaca)
    images_f1_effis_vs_ithaca = np.array(images_f1_effis_vs_ithaca)
    images_f1_model_vs_ithaca = np.array(images_f1_model_vs_ithaca)

    print("IMAGES-LEVEL AVERAGE METRICS")
    tab_images.add_row([
        "IoU",
        np.mean(images_iou_effis_vs_ithaca),
        np.mean(images_iou_model_vs_ithaca)
    ])
    tab_images.add_row([
        "Precision",
        np.mean(images_precision_effis_vs_ithaca),
        np.mean(images_precision_model_vs_ithaca)
    ])
    tab_images.add_row([
        "Recall",
        np.mean(images_recall_effis_vs_ithaca),
        np.mean(images_recall_model_vs_ithaca)
    ])
    tab_images.add_row([
        "F1-score",
        np.mean(images_f1_effis_vs_ithaca),
        np.mean(images_f1_model_vs_ithaca)
    ])

    print(tab_images)

    print("PIXELS-LEVEL AVERAGE METRICS")
    tab_pixels.add_row([
        "IoU",
        pixels_iou_effis_vs_ithaca.compute().item(),
        pixels_iou_model_vs_ithaca.compute().item()
    ])
    tab_pixels.add_row([
        "Precision",
        pixels_precision_effis_vs_ithaca.compute().item(),
        pixels_precision_model_vs_ithaca.compute().item()
    ])
    tab_pixels.add_row([
        "Recall",
        pixels_recall_effis_vs_ithaca.compute().item(),
        pixels_recall_model_vs_ithaca.compute().item()
    ])
    tab_pixels.add_row([
        "F1-score",
        pixels_f1_effis_vs_ithaca.compute().item(),
        pixels_f1_model_vs_ithaca.compute().item()
    ])
    print(tab_pixels)
    # store tabs in file

    with open(args.output_folder / "metrics.txt", "w") as f:
        f.write("IMAGES-LEVEL AVERAGE METRICS\n")
        f.write(tab_images.get_string())
        f.write("\n")
        f.write("PIXELS-LEVEL AVERAGE METRICS\n")
        f.write(tab_pixels.get_string())


if __name__ == "__main__":
    main()
