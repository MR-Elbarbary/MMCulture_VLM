"""Run a small MMCultureQA image QA sample on Modal with Gemma 4."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import modal


APP_NAME = "mmculture-gemma-inference"
DEFAULT_MODEL = "google/gemma-4-E2B-it"
SMALL_MODELS = {DEFAULT_MODEL, "google/gemma-4-E4B-it"}
LARGE_MODEL = "google/gemma-4-31B-it"
DEFAULT_DATASET = "QCRI/MMCQA-SemEval27"
DEFAULT_CONFIG = "demo_mena_en_qa"

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("mmculture-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "datasets",
        "pillow",
    )
    .env({"HF_HOME": "/cache/huggingface"})
)


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    memory=32768,
    volumes={"/cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    max_containers=1,
)
def infer_dataset(
    model_id: str,
    dataset_id: str,
    config: str,
    split: str,
    offset: int,
    limit: int,
    max_new_tokens: int,
) -> list[dict[str, str]]:
    """Load model and dataset remotely; return only compact text results."""
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, token=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        token=True,
    )
    model.eval()

    # The demo config embeds its small set of images in the parquet data. Full
    # configs fetch the matching image archive in the remote Modal container.
    rows = load_dataset(dataset_id, config, split=split)
    end = min(len(rows), offset + limit)
    predictions: list[dict[str, str]] = []

    # Full task configs store relative image paths and publish the images in a
    # split archive. Fetch that archive in Modal only for non-demo configs.
    image_root = Path("/tmp/mmcqa")
    if config.startswith("sqa_"):
        raise ValueError("This entry point handles textual QA configs; spoken QA needs ASR.")
    if not config.startswith("demo_"):
        if not config.startswith("qa_"):
            raise ValueError("Use a qa_<track> config or demo_mena_en_qa.")
        track = config.removeprefix("qa_").removeprefix("sqa_")
        region = track.split("_", maxsplit=1)[0]
        archive_track = region if region == "mena" else track
        archive_name = f"images_{archive_track}_{split}.zip"
        archive_path = hf_hub_download(
            repo_id=dataset_id,
            filename=f"archives/{archive_name}",
            repo_type="dataset",
            token=True,
            cache_dir="/tmp/mmcqa-hf",
        )
        image_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(image_root)

    for index in range(offset, end):
        row = rows[index]
        image_item = row["image"]
        if isinstance(image_item, str):
            with Image.open(image_root / image_item) as opened:
                image_item = opened.convert("RGB")
        question = row["question"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_item},
                    {
                        "type": "text",
                        "text": (
                            "Answer the visual question briefly and in the language "
                            "used by the question. Do not add explanations.\n\n"
                            f"Question: {question}"
                        ),
                    },
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
        new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
        prediction = processor.decode(new_tokens, skip_special_tokens=True).strip()
        predictions.append(
            {
                "id": str(row.get("id", index)),
                "question": question,
                "prediction": prediction,
                "reference": str(row.get("answer", "")),
            }
        )
        del inputs, generated

    hf_cache.commit()
    return predictions


@app.local_entrypoint()
def main(
    model_id: str = DEFAULT_MODEL,
    config: str = DEFAULT_CONFIG,
    split: str = "train",
    offset: int = 0,
    limit: int = 5,
    max_new_tokens: int = 128,
    output: str = "outputs/predictions.jsonl",
):
    """Run a bounded sample and save only predictions to the local machine."""
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if model_id not in SMALL_MODELS | {LARGE_MODEL}:
        raise ValueError(f"Unsupported model: {model_id}")
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    if not 1 <= max_new_tokens <= 512:
        raise ValueError("max_new_tokens must be between 1 and 512")

    inference_function = infer_dataset
    if model_id == LARGE_MODEL:
        # The 31B weights need a high-memory accelerator; keep the default
        # E2B/E4B path on L4 to avoid paying for a large GPU on normal runs.
        inference_function = infer_dataset.with_options(gpu="H100", memory=65536)
    predictions = inference_function.remote(
        model_id,
        DEFAULT_DATASET,
        config,
        split,
        offset,
        limit,
        max_new_tokens,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for prediction in predictions:
            output_file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(f"Saved {len(predictions)} predictions to {output_path}")
