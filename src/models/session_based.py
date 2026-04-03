"""
models/session_based.py
------------------------
Session-Based Recommender using a GRU encoder (GRU4Rec).

EDA rationale
-------------
- >70% of visitors have only 1–3 events → user-level history is too sparse
  for traditional CF. Sessions are the most reliable behavioral signal.
- Peak activity: 17:00–21:00 → model must work in real-time per session.
- Average session length after segmentation: ~2-4 items.
- GRU4Rec is chosen over BERT4Rec for its lower computational cost at
  inference time, which suits the 100ms latency budget.

Architecture
------------
  Input  : padded item-id sequence  [batch, max_len]
  Embed  : learnable item embedding  [batch, max_len, emb_dim]
  GRU    : GRU layers               [batch, max_len, hidden]
  Output : last hidden state → linear → |item_vocab|
  Loss   : BPR (Bayesian Personalised Ranking)
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


class GRU4RecModel:
    """PyTorch GRU4Rec network (defined lazily to avoid hard import at module load).

    Import this class only when torch is available.
    """

    @staticmethod
    def build(num_items: int, emb_dim: int, hidden_size: int, num_layers: int, dropout: float):
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError("Install PyTorch: pip install torch") from exc

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)
                self.gru = nn.GRU(
                    emb_dim, hidden_size, num_layers,
                    batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
                )
                self.dropout = nn.Dropout(p=dropout)
                self.output = nn.Linear(hidden_size, num_items + 1)

            def forward(self, x):
                emb = self.dropout(self.embedding(x))
                out, _ = self.gru(emb)
                last = out[:, -1, :]           # last non-padding hidden state
                return self.output(last)       # (batch, num_items+1)

        return _Net()


class SessionBasedRecommender:
    """GRU4Rec session-based recommender.

    Parameters
    ----------
    emb_dim:
        Item embedding dimensionality.
    hidden_size:
        GRU hidden state size.
    num_layers:
        Number of stacked GRU layers.
    dropout:
        Dropout probability.
    max_session_length:
        Maximum sequence length (left-padded).
    """

    def __init__(
        self,
        emb_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        max_session_length: int = 20,
    ) -> None:
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_session_length = max_session_length
        self._model = None
        self._item_encoder: dict[int, int] = {}   # itemid → int index
        self._item_decoder: list[int] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        sequences: pd.DataFrame,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 0.001,
    ) -> "SessionBasedRecommender":
        """Train GRU4Rec on session sequences.

        Parameters
        ----------
        sequences:
            Output of ``build_session_sequences`` with columns
            [session_id, item_sequence, target_item].
        epochs:
            Training epochs.
        batch_size:
            Mini-batch size.
        lr:
            Adam learning rate.

        Returns
        -------
        self
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise ImportError("Install PyTorch: pip install torch") from exc

        logger.info("Encoding items …")
        all_items = set(sequences["target_item"].tolist())
        for seq in sequences["item_sequence"]:
            all_items.update(seq)
        all_items.discard(0)  # padding token

        self._item_decoder = [0] + sorted(all_items)
        self._item_encoder = {iid: idx for idx, iid in enumerate(self._item_decoder)}

        num_items = len(self._item_decoder)
        self._model = GRU4RecModel.build(
            num_items, self.emb_dim, self.hidden_size, self.num_layers, self.dropout
        )
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Encode sequences
        X = torch.tensor(
            [
                [self._item_encoder.get(i, 0) for i in seq]
                for seq in sequences["item_sequence"]
            ],
            dtype=torch.long,
        )
        y = torch.tensor(
            [self._item_encoder.get(t, 0) for t in sequences["target_item"]],
            dtype=torch.long,
        )

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self._model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self._model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss / len(loader)
            if epoch % 5 == 0 or epoch == 1:
                logger.info("  Epoch %d/%d — loss: %.4f", epoch, epochs, avg)

        logger.info("GRU4Rec training complete.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recommend(
        self, session_items: list[int], top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Score all items given the current session sequence.

        Parameters
        ----------
        session_items:
            Ordered list of item IDs viewed in the current session.
        top_k:
            Number of top items to return.

        Returns
        -------
        List of (itemid, score) tuples sorted by descending score.
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            raise ImportError("Install PyTorch: pip install torch") from exc

        if self._model is None:
            raise RuntimeError("Call fit() before recommend().")

        encoded = [self._item_encoder.get(i, 0) for i in session_items]
        encoded = encoded[-self.max_session_length:]
        padded = [0] * (self.max_session_length - len(encoded)) + encoded

        x = torch.tensor([padded], dtype=torch.long)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(x)[0]          # (num_items+1,)
            probs = F.softmax(logits, dim=-1).numpy()

        seen = set(encoded)
        ranked = sorted(
            [(self._item_decoder[i], float(p)) for i, p in enumerate(probs) if i not in seen and i != 0],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("SessionBasedRecommender saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "SessionBasedRecommender":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        logger.info("SessionBasedRecommender loaded from %s", path)
        return obj
