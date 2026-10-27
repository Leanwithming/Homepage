#!/usr/bin/env python3
"""Create the homepage portrait using deterministic, traditional CV operations."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def gray_world_white_balance(image: np.ndarray) -> np.ndarray:
    image_f = image.astype(np.float32)
    channel_means = image_f.reshape(-1, 3).mean(axis=0)
    target = float(channel_means.mean())
    scales = np.clip(target / np.maximum(channel_means, 1.0), 0.82, 1.18)
    return np.clip(image_f * scales, 0, 255).astype(np.uint8)


def enhance_subject(image: np.ndarray) -> np.ndarray:
    balanced = gray_world_white_balance(image)
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=1.65, tileGridSize=(10, 10)).apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)
    enhanced = cv2.bilateralFilter(enhanced, 7, 28, 28)
    soft = cv2.GaussianBlur(enhanced, (0, 0), 1.3)
    return cv2.addWeighted(enhanced, 1.18, soft, -0.18, 0)


def crop_four_by_five(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    target_height = int(round(width * 5 / 4))
    if target_height <= height:
        # Preserve the hair and eyes; trim more from the lower shirt than the top.
        top = max(0, min(28, height - target_height))
        return image[top : top + target_height]

    target_width = int(round(height * 4 / 5))
    left = max(0, (width - target_width) // 2)
    return image[:, left : left + target_width]


def foreground_mask(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)

    # Strong background seeds in the upper corners and outer side strips.
    mask[: int(height * 0.22), : int(width * 0.25)] = cv2.GC_BGD
    mask[: int(height * 0.18), int(width * 0.82) :] = cv2.GC_BGD
    mask[int(height * 0.3) :, : int(width * 0.035)] = cv2.GC_BGD
    mask[int(height * 0.3) :, int(width * 0.965) :] = cv2.GC_BGD

    # Probable and definite foreground seeds for hair, face, neck, and shirt.
    cv2.ellipse(
        mask,
        (int(width * 0.53), int(height * 0.43)),
        (int(width * 0.39), int(height * 0.41)),
        0,
        0,
        360,
        cv2.GC_PR_FGD,
        -1,
    )
    cv2.ellipse(
        mask,
        (int(width * 0.52), int(height * 0.48)),
        (int(width * 0.27), int(height * 0.29)),
        0,
        0,
        360,
        cv2.GC_FGD,
        -1,
    )
    shirt = np.array(
        [
            [int(width * 0.06), height - 1],
            [int(width * 0.12), int(height * 0.78)],
            [int(width * 0.38), int(height * 0.68)],
            [int(width * 0.66), int(height * 0.68)],
            [int(width * 0.93), int(height * 0.78)],
            [width - 1, height - 1],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [shirt], cv2.GC_PR_FGD)
    cv2.ellipse(
        mask,
        (int(width * 0.52), int(height * 0.9)),
        (int(width * 0.34), int(height * 0.18)),
        0,
        0,
        360,
        cv2.GC_FGD,
        -1,
    )

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, None, bg_model, fg_model, 7, cv2.GC_INIT_WITH_MASK)
    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.GaussianBlur(binary, (0, 0), 7.5)


def colorize_background(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), 13)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = np.power(gray, 0.82)

    shadow = np.array([26, 13, 7], dtype=np.float32)  # BGR: deep navy
    mid = np.array([83, 48, 20], dtype=np.float32)  # muted blue
    light = np.array([145, 53, 75], dtype=np.float32)  # violet highlight
    first = shadow + (mid - shadow) * np.minimum(gray[..., None] * 2, 1)
    second_mix = np.maximum(gray[..., None] * 2 - 1, 0)
    toned = first * (1 - second_mix) + light * second_mix

    # Cyan/magenta edge echoes, created only from the original pixels.
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 64, 142)
    edges = cv2.GaussianBlur(edges, (0, 0), 1.1).astype(np.float32) / 255.0
    cyan = np.roll(edges, 5, axis=1)[..., None] * np.array([110, 68, 0], dtype=np.float32)
    magenta = np.roll(edges, -5, axis=1)[..., None] * np.array([48, 0, 88], dtype=np.float32)
    toned += cyan + magenta

    height, width = gray.shape
    overlay = np.zeros_like(toned)
    # Perspective-free technical grid and scanlines.
    for x in range(0, width, 56):
        overlay[:, x : x + 1] += np.array([28, 20, 4])
    for y in range(0, height, 56):
        overlay[y : y + 1, :] += np.array([28, 20, 4])
    overlay[::6, :, :] += np.array([8, 5, 2])
    toned += overlay

    # A diagonal cyan light band gives the image a stronger editorial focal line.
    yy, xx = np.mgrid[:height, :width]
    band = np.exp(-((xx - 0.18 * width - 0.23 * yy) ** 2) / (2 * (0.12 * width) ** 2))
    toned += band[..., None] * np.array([34, 25, 0], dtype=np.float32)
    return np.clip(toned, 0, 255).astype(np.uint8)


def add_silhouette_glow(canvas: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    hard = (alpha > 0.5).astype(np.uint8) * 255
    outer = cv2.dilate(hard, np.ones((21, 21), np.uint8))
    inner = cv2.erode(hard, np.ones((9, 9), np.uint8))
    rim = cv2.GaussianBlur(cv2.subtract(outer, inner), (0, 0), 7).astype(np.float32) / 255.0
    glow = np.zeros_like(canvas, dtype=np.float32)
    glow += rim[..., None] * np.array([90, 72, 6], dtype=np.float32)
    return np.clip(canvas.astype(np.float32) + glow, 0, 255).astype(np.uint8)


def process(input_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Unable to read image: {input_path}")

    image = crop_four_by_five(image)
    image = cv2.resize(image, (1200, 1500), interpolation=cv2.INTER_LANCZOS4)
    subject = enhance_subject(image)
    alpha_u8 = foreground_mask(image)
    alpha = alpha_u8.astype(np.float32) / 255.0
    background = colorize_background(image)
    background = add_silhouette_glow(background, alpha)

    # Preserve the subject pixels; all stylization lives mostly behind the person.
    composite = (
        subject.astype(np.float32) * alpha[..., None]
        + background.astype(np.float32) * (1.0 - alpha[..., None])
    )

    # Restrained global grade ties foreground and background together.
    composite[..., 0] *= 1.025
    composite[..., 1] *= 1.005
    composite[..., 2] *= 0.99
    composite = np.clip(composite, 0, 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise SystemExit(f"Unable to write image: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    process(args.input, args.output)


if __name__ == "__main__":
    main()
