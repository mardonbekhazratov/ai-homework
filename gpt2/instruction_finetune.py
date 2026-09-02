"""Instruction finetuning of pretrained GPT-2 small (124M) on
`instruction-data-with-response.json`.

Everything that matters for training -- prompt formatting, the Dataset, the
padding/masking collate function, the loss, the optimization loop and the
sampling -- is plain PyTorch, following "Build a Large Language Model
(From Scratch)" (ch. 7). `transformers` is used only to download the pretrained
weights and the BPE tokenizer.

Usage:
    python instruction_finetune.py                 # full run
    python instruction_finetune.py --test-mode     # quick smoke run
    python instruction_finetune.py --mask-prompt   # train on response tokens only
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from gpt_model import (
    GPT_CONFIG_124M,
    load_pretrained_gpt2,
    generate,
    text_to_token_ids,
    token_ids_to_text,
)

HERE = Path(__file__).parent
PAD_TOKEN_ID = 50256  # <|endoftext|>, GPT-2 has no dedicated padding token
IGNORE_INDEX = -100   # cross_entropy skips these positions


# --------------------------------------------------------------------------
# 1. Prompt formatting (Alpaca style)
# --------------------------------------------------------------------------

def format_input(entry):
    """Turn one JSON record into the prompt half of the training example."""
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text


def format_response(entry):
    return f"\n\n### Response:\n{entry['output']}"


# --------------------------------------------------------------------------
# 2. Dataset + collate function
# --------------------------------------------------------------------------

class InstructionDataset(Dataset):
    """Pre-tokenizes every `prompt + response` pair once, up front."""

    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        self.prompt_lengths = []  # only needed when masking the prompt

        for entry in data:
            prompt = format_input(entry)
            full_text = prompt + format_response(entry)
            self.encoded_texts.append(tokenizer.encode(full_text))
            self.prompt_lengths.append(len(tokenizer.encode(prompt)))

    def __getitem__(self, index):
        return {
            "ids": self.encoded_texts[index],
            "prompt_len": self.prompt_lengths[index],
        }

    def __len__(self):
        return len(self.data)


def custom_collate_fn(
    batch,
    pad_token_id=PAD_TOKEN_ID,
    ignore_index=IGNORE_INDEX,
    allowed_max_length=None,
    mask_prompt=False,
    device="cpu",
):
    """Pad a batch to its longest sequence and build shifted-by-one targets.

    Three things happen here:
      * every sequence gets one trailing <|endoftext|> so the model learns to
        stop, then is padded to the batch maximum;
      * `targets` are `inputs` shifted left by one position;
      * padding in the targets is replaced by `ignore_index` -- except the very
        first pad token, which is the end-of-text the model should predict.
    """
    batch_max_length = max(len(item["ids"]) + 1 for item in batch)

    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item["ids"] + [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # Keep the first <|endoftext|> as a real target, ignore the rest.
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Optional: don't compute a loss on the instruction itself.
        # targets[i] predicts padded[i + 1], so the response starts at i = prompt_len - 1.
        if mask_prompt:
            targets[: item["prompt_len"] - 1] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


# --------------------------------------------------------------------------
# 3. Loss + training loop
# --------------------------------------------------------------------------

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    # flatten (b, t, vocab) -> (b*t, vocab) so cross_entropy sees a plain batch
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    num_batches = min(num_batches or len(data_loader), len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        with torch.no_grad():
            total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    token_ids = generate(
        model=model, idx=encoded, max_new_tokens=50,
        context_size=context_size, eos_id=PAD_TOKEN_ID,
    )
    decoded = token_ids_to_text(token_ids, tokenizer)
    print(decoded[len(start_context):].replace("\n", " "))
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    """The classic loop: forward, loss, backward, step -- nothing hidden."""
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()                                   # reset gradients
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()                                          # backprop
            optimizer.step()                                         # update weights
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, out_path):
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(__import__("matplotlib").ticker.MaxNLocator(integer=True))

    ax2 = ax1.twiny()  # second x-axis showing tokens processed
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.savefig(out_path)
    print(f"Loss plot saved to {out_path}")


# --------------------------------------------------------------------------
# 4. Main
# --------------------------------------------------------------------------

def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(args):
    torch.manual_seed(123)
    device = pick_device(args.device)
    print(f"Device: {device}")

    # ---- data -------------------------------------------------------------
    with open(HERE / args.data, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Number of entries: {len(data)}")

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.10)
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    if args.test_mode:  # tiny run just to check the plumbing
        train_data, val_data, test_data = train_data[:8], val_data[:2], test_data[:2]

    print(f"Train/Val/Test: {len(train_data)}/{len(val_data)}/{len(test_data)}")

    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    collate = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=GPT_CONFIG_124M["context_length"],
        mask_prompt=args.mask_prompt,
    )
    loader_kwargs = dict(batch_size=args.batch_size, collate_fn=collate, num_workers=0)
    train_loader = DataLoader(InstructionDataset(train_data, tokenizer),
                              shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(InstructionDataset(val_data, tokenizer),
                            shuffle=False, drop_last=False, **loader_kwargs)

    # ---- model ------------------------------------------------------------
    model = load_pretrained_gpt2("gpt2")
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded pretrained GPT-2 small: {n_params / 1e6:.1f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    # ---- loss before finetuning ------------------------------------------
    with torch.no_grad():
        base_train = calc_loss_loader(train_loader, model, device, num_batches=5)
        base_val = calc_loss_loader(val_loader, model, device, num_batches=5)
    print(f"Initial losses -> train {base_train:.3f}, val {base_val:.3f}")

    # ---- train ------------------------------------------------------------
    start_context = format_input(val_data[0]) + "\n\n### Response:\n"
    start_time = time.time()
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.epochs, eval_freq=5, eval_iter=5,
        start_context=start_context, tokenizer=tokenizer,
    )
    print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes")

    if train_losses:
        epochs_tensor = torch.linspace(0, args.epochs, len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses,
                    HERE / "loss-plot.pdf")

    # ---- save -------------------------------------------------------------
    torch.save(model.state_dict(), HERE / args.out_model)
    print(f"Model saved as {args.out_model}")

    # ---- generate responses for the held-out test set ---------------------
    print("\nGenerating test-set responses...")
    for entry in test_data:
        prompt = format_input(entry) + "\n\n### Response:\n"
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(prompt, tokenizer).to(device),
            max_new_tokens=args.max_new_tokens,
            context_size=GPT_CONFIG_124M["context_length"],
            eos_id=PAD_TOKEN_ID,
        )
        generated = token_ids_to_text(token_ids, tokenizer)[len(prompt):]
        entry["gpt2_finetuned_response"] = generated.strip()
        print("-" * 60)
        print(format_input(entry))
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nModel response:\n>> {entry['gpt2_finetuned_response']}")

    with open(HERE / args.out_responses, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4)
    print(f"\nResponses saved as {args.out_responses}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="instruction-data-with-response.json")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--mask-prompt", action="store_true",
                        help="compute the loss on response tokens only")
    parser.add_argument("--out-model", default="gpt2-small124M-sft.pth")
    parser.add_argument("--out-responses", default="test-set-responses.json")
    parser.add_argument("--test-mode", action="store_true",
                        help="tiny subset + short generations, to check the code runs")
    args = parser.parse_args()

    if args.test_mode:
        args.epochs, args.batch_size, args.max_new_tokens = 1, 2, 32

    main(args)
