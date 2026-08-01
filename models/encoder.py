"""
Token encoder module.

Responsibility:
- Load a pretrained language model
- Convert raw tokenized input into token-level embeddings
- Provide semantic baseline representations

IMPORTANT:
- This module does NOT learn role similarity
- This module is frozen by default
- All role learning happens downstream
"""

from typing import List, Tuple

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel


class TokenEncoder(nn.Module):
    """
    Wrapper around a pretrained Transformer encoder
    that returns token-level embeddings.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 64,
        freeze: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Encoder model (no decoder, no LM head)
        self.encoder = AutoModel.from_pretrained(model_name)

        # Freeze encoder parameters if requested
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.hidden_size = self.encoder.config.hidden_size

    def forward(
        self,
        sentences: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of sentences.

        Returns:
            token_embeddings:
                Tensor of shape (batch_size, seq_len, hidden_dim)

            attention_mask:
                Tensor of shape (batch_size, seq_len)
        """

        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        token_embeddings = outputs.last_hidden_state

        return token_embeddings, attention_mask

    @torch.no_grad()
    def encode_with_tokens(
        self,
        sentence: str,
    ):
        """
        Encode a single sentence and return aligned
        (tokens, embeddings).

        Returns
        -------
        tokens : List[str]
            Tokens exactly matching the returned embeddings.

        embeddings : torch.Tensor
            Shape: (num_real_tokens, hidden_size)
        """

        encoded = self.tokenizer(
            sentence,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        token_embeddings = outputs.last_hidden_state[0]

        all_tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids[0]
        )

        tokens = []
        embeddings = []

        for tok, emb, mask in zip(
            all_tokens,
            token_embeddings,
            attention_mask[0],
        ):
            if mask.item() == 0:
                continue

            if tok in self.tokenizer.all_special_tokens:
                continue

            tokens.append(tok)
            embeddings.append(emb)

        embeddings = torch.stack(embeddings)

        return tokens, embeddings