"""The pooling of a dense encoder's output into one embedding per text; used by
the retrievers (query side) and the index builder (document side), so both
sides embed the same way."""

import torch


def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask=None,
    pooling_method="mean",
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    elif pooling_method == "last_token":
        # Qwen3-style: use hidden state at the last real (EOS) token position.
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_state[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            return last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device),
                sequence_lengths,
            ]
    else:
        raise NotImplementedError("Pooling method not implemented!")
