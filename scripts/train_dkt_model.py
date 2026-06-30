"""
Task 4: Deep Knowledge Tracing (DKT) with LSTM/GRU

Trains a deep learning model to predict student knowledge states using LSTM/GRU architecture.
- Sequences per student over time
- Embedding layer for skills and other features
- LSTM/GRU encoder for temporal dynamics
- Output layer for knowledge prediction
- Trained on interaction dataset with proper sequence batching
"""

import json
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
except ImportError:
    print("[DKT] Warning: scikit-learn not available")
    raise


class InteractionSequenceDataset(Dataset):
    """PyTorch Dataset for student interaction sequences."""
    
    def __init__(
        self,
        events: List[Dict],
        skill_vocab: Dict[str, int],
        topic_vocab: Dict[str, int],
        difficulty_vocab: Dict[str, int],
        max_seq_len: int = 50,
        min_seq_len: int = 2
    ):
        """
        Args:
            events: List of interaction events
            skill_vocab: Mapping skill -> index
            topic_vocab: Mapping topic -> index
            difficulty_vocab: Mapping difficulty -> index
            max_seq_len: Maximum sequence length
            min_seq_len: Minimum sequence length to include
        """
        self.skill_vocab = skill_vocab
        self.topic_vocab = topic_vocab
        self.difficulty_vocab = difficulty_vocab
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        
        # Organize sequences by student
        sequences_dict = defaultdict(list)
        sorted_events = sorted(events, key=lambda e: (e['student_key'], e['answered_at']))
        
        for event in sorted_events:
            sequences_dict[event['student_key']].append(event)
        
        # Create dataset
        self.sequences = []
        for student_id, student_events in sequences_dict.items():
            if len(student_events) >= min_seq_len:
                # Create overlapping subsequences
                for start in range(0, len(student_events), max(1, len(student_events) // 3)):
                    end = min(start + max_seq_len, len(student_events))
                    if end - start >= min_seq_len:
                        self.sequences.append(student_events[start:end])
        
        print(f"[DKT] Dataset created: {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        seq_len = len(sequence)
        
        # Encode sequence
        skills = torch.tensor([self.skill_vocab.get(e['topic'], 0) for e in sequence], dtype=torch.long)
        topics = torch.tensor([self.topic_vocab.get(e['topic'], 0) for e in sequence], dtype=torch.long)
        difficulties = torch.tensor([self.difficulty_vocab.get(e['difficulty'], 0) for e in sequence], dtype=torch.long)
        
        # Correctness (target)
        correctness = torch.tensor([int(e['is_correct']) for e in sequence], dtype=torch.float32)
        
        # Response time (normalized)
        response_times = torch.tensor([e['time_spent_sec'] / 100.0 for e in sequence], dtype=torch.float32)
        
        # Pad if necessary
        padded_skills = torch.zeros(self.max_seq_len, dtype=torch.long)
        padded_topics = torch.zeros(self.max_seq_len, dtype=torch.long)
        padded_difficulties = torch.zeros(self.max_seq_len, dtype=torch.long)
        padded_correctness = torch.zeros(self.max_seq_len, dtype=torch.float32)
        padded_times = torch.zeros(self.max_seq_len, dtype=torch.float32)
        
        padded_skills[:seq_len] = skills
        padded_topics[:seq_len] = topics
        padded_difficulties[:seq_len] = difficulties
        padded_correctness[:seq_len] = correctness
        padded_times[:seq_len] = response_times
        
        return {
            'skills': padded_skills,
            'topics': padded_topics,
            'difficulties': padded_difficulties,
            'correctness': padded_correctness,
            'times': padded_times,
            'seq_len': seq_len
        }


class DKTModel(nn.Module):
    """Deep Knowledge Tracing model with LSTM encoder."""
    
    def __init__(
        self,
        num_skills: int,
        num_topics: int,
        num_difficulties: int,
        embedding_dim: int = 64,
        rnn_dim: int = 128,
        rnn_layers: int = 2,
        dropout: float = 0.3,
        use_gru: bool = False
    ):
        """
        Args:
            num_skills: Number of unique skills
            num_topics: Number of unique topics
            num_difficulties: Number of difficulty levels
            embedding_dim: Embedding dimension
            rnn_dim: RNN hidden dimension
            rnn_layers: Number of RNN layers
            dropout: Dropout rate
            use_gru: Use GRU instead of LSTM
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.rnn_dim = rnn_dim
        
        # Embeddings
        self.skill_embed = nn.Embedding(num_skills + 1, embedding_dim, padding_idx=0)
        self.topic_embed = nn.Embedding(num_topics + 1, embedding_dim, padding_idx=0)
        self.difficulty_embed = nn.Embedding(num_difficulties + 1, embedding_dim // 2, padding_idx=0)
        
        # Input projection
        input_dim = embedding_dim + embedding_dim + embedding_dim // 2 + 1  # skill + topic + difficulty + time
        
        # RNN
        rnn_class = nn.GRU if use_gru else nn.LSTM
        self.rnn = rnn_class(
            input_size=input_dim,
            hidden_size=rnn_dim,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_fc = nn.Sequential(
            nn.Linear(rnn_dim, rnn_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, skills, topics, difficulties, times, seq_lens):
        """
        Args:
            skills: (batch, seq_len) skill indices
            topics: (batch, seq_len) topic indices
            difficulties: (batch, seq_len) difficulty indices
            times: (batch, seq_len) response times
            seq_lens: (batch,) actual sequence lengths
        
        Returns:
            predictions: (batch, seq_len) predicted correctness
        """
        # Embeddings
        skill_emb = self.skill_embed(skills)  # (batch, seq_len, embed_dim)
        topic_emb = self.topic_embed(topics)  # (batch, seq_len, embed_dim)
        diff_emb = self.difficulty_embed(difficulties)  # (batch, seq_len, embed_dim//2)
        
        # Combine embeddings
        times_unsqueezed = times.unsqueeze(-1)  # (batch, seq_len, 1)
        combined = torch.cat([skill_emb, topic_emb, diff_emb, times_unsqueezed], dim=-1)
        
        # Pack sequences (handle variable lengths)
        packed = nn.utils.rnn.pack_padded_sequence(
            combined, seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # RNN
        rnn_out, _ = self.rnn(packed)
        
        # Unpack sequences
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        
        # Output projection
        output = self.output_fc(self.dropout(unpacked))  # (batch, seq_len, 1)
        
        return output.squeeze(-1)  # (batch, seq_len)


def load_latest_dataset() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load latest interaction dataset."""
    dataset_dir = Path("backend/data/ml/interaction_datasets")
    latest_version_file = dataset_dir / "LATEST_VERSION"
    
    if not latest_version_file.exists():
        raise FileNotFoundError("No interaction dataset found.")
    
    latest_version = latest_version_file.read_text().strip()
    latest_dir = dataset_dir / latest_version
    
    events_file = latest_dir / "events.jsonl"
    all_events = []
    with open(events_file, 'r') as f:
        for line in f:
            if line.strip():
                all_events.append(json.loads(line))
    
    train_events = [e for e in all_events if e.get('split') == 'train']
    val_events = [e for e in all_events if e.get('split') == 'val']
    test_events = [e for e in all_events if e.get('split') == 'test']
    
    print(f"[DKT] Loaded: train={len(train_events)}, val={len(val_events)}, test={len(test_events)}")
    return train_events, val_events, test_events


def build_vocabs(events: List[Dict]) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Build vocabularies for skills, topics, difficulties."""
    skills = set()
    topics = set()
    difficulties = set()
    
    for event in events:
        skills.add(event['topic'])
        topics.add(event['topic'])
        difficulties.add(event['difficulty'])
    
    skill_vocab = {skill: idx + 1 for idx, skill in enumerate(sorted(skills))}
    topic_vocab = {topic: idx + 1 for idx, topic in enumerate(sorted(topics))}
    difficulty_vocab = {diff: idx + 1 for idx, diff in enumerate(sorted(difficulties))}
    
    print(f"[DKT] Vocabularies: {len(skill_vocab)} skills, {len(topic_vocab)} topics, {len(difficulty_vocab)} difficulties")
    
    return skill_vocab, topic_vocab, difficulty_vocab


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    batch_count = 0
    
    for batch in train_loader:
        skills = batch['skills'].to(device)
        topics = batch['topics'].to(device)
        difficulties = batch['difficulties'].to(device)
        correctness = batch['correctness'].to(device)
        times = batch['times'].to(device)
        seq_lens = batch['seq_len'].to(device)
        
        # Forward pass
        predictions = model(skills, topics, difficulties, times, seq_lens)
        
        # Mask loss (only compute for non-padded positions)
        batch_size, seq_len = predictions.shape
        mask = torch.arange(seq_len, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
        
        loss = criterion(predictions[mask], correctness[mask])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
    
    return total_loss / max(1, batch_count)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on dataset."""
    model.eval()
    predictions_list = []
    actuals_list = []
    losses = []
    criterion = nn.BCELoss()
    
    for batch in data_loader:
        skills = batch['skills'].to(device)
        topics = batch['topics'].to(device)
        difficulties = batch['difficulties'].to(device)
        correctness = batch['correctness'].to(device)
        times = batch['times'].to(device)
        seq_lens = batch['seq_len'].to(device)
        
        predictions = model(skills, topics, difficulties, times, seq_lens)
        
        # Mask loss
        batch_size, seq_len = predictions.shape
        mask = torch.arange(seq_len, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
        
        loss = criterion(predictions[mask], correctness[mask])
        losses.append(loss.item())
        
        predictions_list.extend(predictions[mask].cpu().numpy())
        actuals_list.extend(correctness[mask].cpu().numpy())
    
    predictions_arr = np.array(predictions_list)
    actuals_arr = np.array(actuals_list)
    
    eps = 1e-10
    predictions_arr = np.clip(predictions_arr, eps, 1 - eps)
    
    metrics = {
        'loss': np.mean(losses),
        'auc': roc_auc_score(actuals_arr, predictions_arr),
        'bce': log_loss(actuals_arr, predictions_arr),
        'rmse': np.sqrt(mean_squared_error(actuals_arr, predictions_arr))
    }
    
    return metrics


def main():
    print("[DKT] ========== TASK 4: DEEP KNOWLEDGE TRACING ==========\n")
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DKT] Device: {device}")
    
    batch_size = 32
    max_seq_len = 50
    min_seq_len = 2
    epochs = 15
    learning_rate = 0.001
    embedding_dim = 64
    rnn_dim = 128
    rnn_layers = 2
    dropout = 0.3
    use_gru = False  # LSTM by default
    num_workers = max(0, min(os.cpu_count() or 2, 4))
    
    # Load data
    print("[DKT] Loading interaction dataset...")
    train_events, val_events, test_events = load_latest_dataset()
    
    # Build vocabularies
    all_events = train_events + val_events + test_events
    skill_vocab, topic_vocab, difficulty_vocab = build_vocabs(all_events)
    
    # Create datasets
    print("[DKT] Creating datasets...")
    train_dataset = InteractionSequenceDataset(
        train_events, skill_vocab, topic_vocab, difficulty_vocab,
        max_seq_len=max_seq_len, min_seq_len=min_seq_len
    )
    val_dataset = InteractionSequenceDataset(
        val_events, skill_vocab, topic_vocab, difficulty_vocab,
        max_seq_len=max_seq_len, min_seq_len=min_seq_len
    )
    test_dataset = InteractionSequenceDataset(
        test_events, skill_vocab, topic_vocab, difficulty_vocab,
        max_seq_len=max_seq_len, min_seq_len=min_seq_len
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    
    # Create model
    print("[DKT] Building DKT model...")
    model = DKTModel(
        num_skills=len(skill_vocab),
        num_topics=len(topic_vocab),
        num_difficulties=len(difficulty_vocab),
        embedding_dim=embedding_dim,
        rnn_dim=rnn_dim,
        rnn_layers=rnn_layers,
        dropout=dropout,
        use_gru=use_gru
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[DKT] Model parameters: {total_params:,}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    # Training loop
    print(f"\n[DKT] Training for {epochs} epochs...")
    best_val_auc = 0.0
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"[DKT] Epoch {epoch + 1}/{epochs}")
        print(f"      Train Loss: {train_loss:.4f}")
        print(f"      Val AUC: {val_metrics['auc']:.4f}, Val BCE: {val_metrics['bce']:.4f}")
        
        # Early stopping
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[DKT] Early stopping at epoch {epoch + 1}")
                break
    
    # Evaluate on test set
    print("\n[DKT] Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"[DKT] Test Metrics:")
    print(f"      Loss: {test_metrics['loss']:.4f}")
    print(f"      AUC:  {test_metrics['auc']:.4f}")
    print(f"      BCE:  {test_metrics['bce']:.4f}")
    print(f"      RMSE: {test_metrics['rmse']:.4f}")
    
    # Save model
    print("\n[DKT] Saving model artifact...")
    model_dir = Path("backend/models/dkt")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = model_dir / f"dkt_model_{timestamp}.pt"
    
    torch.save({
        'model_state': model.state_dict(),
        'model_config': {
            'num_skills': len(skill_vocab),
            'num_topics': len(topic_vocab),
            'num_difficulties': len(difficulty_vocab),
            'embedding_dim': embedding_dim,
            'rnn_dim': rnn_dim,
            'rnn_layers': rnn_layers,
            'dropout': dropout,
            'use_gru': use_gru,
            'max_seq_len': max_seq_len
        },
        'vocabs': {
            'skill_vocab': skill_vocab,
            'topic_vocab': topic_vocab,
            'difficulty_vocab': difficulty_vocab
        },
        'metrics': {
            'train_size': len(train_events),
            'val_size': len(val_events),
            'test_size': len(test_events),
            'test_auc': test_metrics['auc'],
            'test_bce': test_metrics['bce'],
            'test_rmse': test_metrics['rmse'],
            'best_val_auc': best_val_auc
        }
    }, model_path)
    
    print(f"[DKT] Model saved: {model_path}")
    
    # Save latest symlink
    latest_path = model_dir / "dkt_model_latest.pt"
    torch.save({
        'model_state': model.state_dict(),
        'model_config': {
            'num_skills': len(skill_vocab),
            'num_topics': len(topic_vocab),
            'num_difficulties': len(difficulty_vocab),
            'embedding_dim': embedding_dim,
            'rnn_dim': rnn_dim,
            'rnn_layers': rnn_layers,
            'dropout': dropout,
            'use_gru': use_gru,
            'max_seq_len': max_seq_len
        },
        'vocabs': {
            'skill_vocab': skill_vocab,
            'topic_vocab': topic_vocab,
            'difficulty_vocab': difficulty_vocab
        },
        'metrics': {
            'train_size': len(train_events),
            'val_size': len(val_events),
            'test_size': len(test_events),
            'test_auc': test_metrics['auc'],
            'test_bce': test_metrics['bce'],
            'test_rmse': test_metrics['rmse'],
            'best_val_auc': best_val_auc
        }
    }, latest_path)
    
    print(f"[DKT] Latest saved: {latest_path}")
    
    # Save metrics
    metrics_path = model_dir / f"dkt_metrics_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'train_size': len(train_events),
            'val_size': len(val_events),
            'test_size': len(test_events),
            'num_skills': len(skill_vocab),
            'num_topics': len(topic_vocab),
            'num_difficulties': len(difficulty_vocab),
            'test_metrics': test_metrics,
            'best_val_auc': best_val_auc,
            'epochs_trained': min(epoch + 1, epochs),
            'model_config': {
                'embedding_dim': embedding_dim,
                'rnn_dim': rnn_dim,
                'rnn_layers': rnn_layers,
                'dropout': dropout,
                'use_gru': use_gru
            }
        }, f, indent=2)
    
    print(f"[DKT] Metrics saved: {metrics_path}")
    
    print("\n[DKT] ========== TASK 4 COMPLETE ==========\n")
    
    return model


if __name__ == "__main__":
    main()
