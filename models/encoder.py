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

        Args:
            sentences: list of raw text strings

        Returns:
            token_embeddings:
                Tensor of shape (batch_size, seq_len, hidden_dim)

            attention_mask:
                Tensor of shape (batch_size, seq_len)
        """

        # Tokenize
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Forward through transformer
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Token-level embeddings
        token_embeddings = outputs.last_hidden_state

        return token_embeddings, attention_mask
