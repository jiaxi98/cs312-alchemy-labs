"""Small declared A1 interventions around the unmodified course training loop."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path


def edit_tokens(dataset, *, after_steps, count, position, batch_size, vocab_size):
    row = int(after_steps) * int(batch_size)
    count, position = int(count), int(position)
    if after_steps < 0 or count < 1 or position < 0:
        raise ValueError("Invalid token-edit specification")
    if row >= len(dataset) or position + count > dataset.seq_len:
        raise ValueError("Token edit falls outside the selected training data")
    tokens = dataset[row]["input_ids"]
    before = [int(v) for v in tokens[position:position + count]]
    after = [17 if v != 17 else 18 for v in before]
    if max(after) >= vocab_size:
        raise ValueError("Edit token IDs do not fit this vocabulary")
    tokens[position:position + count] = after
    return dict(after_steps=int(after_steps), first_consumed_step=int(after_steps) + 1,
                row=row, position=position, count=count, before=before, after=after)


@contextmanager
def token_edit_context(spec, config, output):
    if not spec:
        yield
        return
    from data import PreprocessedTokenDataset
    from model_config import resolve_model_config
    if config.perturb_one_token or config.num_epochs != 1 or config.num_micro_batches != 1:
        raise ValueError("Timed token editing requires one epoch, one microbatch and no other edit")
    original = PreprocessedTokenDataset.shuffle
    edits = []
    vocab = resolve_model_config(model_name=config.model_name, model_config=config.model_config).vocab_size

    def shuffle(dataset, seed=None):
        result = original(dataset, seed=seed)
        if edits:
            raise RuntimeError("Token-edit run shuffled its training data more than once")
        # The course's selected/shuffled mmap is copy-on-write or an ephemeral
        # file. This edit cannot modify the shared immutable token release.
        audit = edit_tokens(result, **spec, batch_size=config.batch_size, vocab_size=vocab)
        edits.append(audit)
        Path(output, "token_edit.json").write_text(json.dumps(audit, indent=2) + "\n")
        return result

    PreprocessedTokenDataset.shuffle = shuffle
    try:
        yield
        if len(edits) != 1:
            raise RuntimeError("Declared token edit was not applied")
    finally:
        PreprocessedTokenDataset.shuffle = original


def log_fixed_validation_rms(ctx):
    """The handout p18 hook: FP32 forward on fixed eight validation sequences."""
    import torch
    input_ids = next(iter(ctx.val_batches["val"]))[:8]
    input_ids = input_ids.to(next(ctx.model.parameters()).device)
    stats, handles = {}, []
    for index in (0, 2, 5):
        def hook(module, inputs, output, index=index):
            stats[f"fixed_val/layer_{index + 1}_residual_rms"] = output.detach().float().square().mean().sqrt()
        handles.append(ctx.model.model.layers[index].register_forward_hook(hook))
    was_training = ctx.model.training
    ctx.model.eval()
    try:
        with torch.no_grad():
            ctx.model(input_ids=input_ids)
    finally:
        for handle in handles:
            handle.remove()
        ctx.model.train(was_training)
    return stats
