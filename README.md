# MMCulture VLM inference on Modal

This starter runs Gemma 4 vision inference remotely on a small sample of the
MMCultureQA SemEval 2027 dataset. The default is the 2B-class Gemma 4 E2B model
and the dataset's 24-example `demo_mena_en_qa` config. It returns only IDs,
questions, predictions, and references to your computer as a JSONL file.

## Network and storage

- Model and dataset downloads happen in Modal, not on your computer. The first
  run still needs internet access from the Modal container to Hugging Face.
- Gemma weights are cached in the remote Modal Volume `mmculture-hf-cache` so
  later runs do not download them again. This volume stays in your Modal account
  and may incur storage charges. Remove it when you no longer need the cache:

  ```powershell
  modal volume delete mmculture-hf-cache
  ```
- The small demo includes images in its dataset files. Full `qa_<track>` configs
  require downloading that split's image archive into temporary Modal disk.
  That archive is not copied to your computer or kept in the persistent cache.
  Full splits can be several gigabytes, so begin with the demo.
- Only the compact prediction JSONL is written locally. It is ignored by Git.

## Setup

1. Install Modal locally and authenticate:

   ```powershell
   python -m pip install modal
   modal setup
   ```

2. Accept Google's Gemma model terms on Hugging Face and create a read token.
   Store it as a Modal secret (do not put it in this repository):

   ```powershell
   modal secret create huggingface-secret HF_TOKEN=hf_your_token
   ```

3. Run three demo examples:

   ```powershell
   modal run modal_inference.py --limit 3
   ```

   Results are saved to `outputs/predictions.jsonl`. Modal needs an available
   L4 GPU and the account must be enabled for GPU workloads.

## Run a dataset split

Textual configs use `qa_<track>` names. For example, run five MENA English dev
examples, starting at row zero:

```powershell
modal run modal_inference.py --config qa_mena_en --split dev --limit 5
```

The task page lists tracks such as `mena_msa`, `mena_arz`, `india_hi`, and
`brazil_pt`. Use `--offset` to select a later range, for example
`--offset 20 --limit 10`. Each run is limited to 1,000 rows. Spoken `sqa_`
configs are not handled because they need a speech recognition step first.

To try the 4B-class family member instead, pass
`--model-id google/gemma-4-E4B-it`. The default L4 is intended for the smaller
models. For 31B, pass `--model-id google/gemma-4-31B-it`; the entry point then
requests an H100 and 64 GiB of host memory. That run will cost substantially
more than the default L4 run and still depends on H100 availability.

You can choose another local output filename with `--output`, and adjust
generation length with `--max-new-tokens` (1–512, default 128).

## Dataset use

The dataset is licensed CC BY-NC 4.0 for non-commercial research. Follow the
dataset card's citation and task submission rules. The `dev` split includes
references; `devtest` is for leaderboard evaluation and has no reference
answers.
