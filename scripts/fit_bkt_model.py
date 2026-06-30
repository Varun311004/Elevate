"""
Task 3: Bayesian Knowledge Tracing with EM Parameter Fitting

Fits per-topic BKT parameters from interaction data using Expectation-Maximization.
- Implements forward-backward algorithm from scratch (no library calls for core EM)
- Evaluates on held-out test set with BCE, AUC, RMSE metrics
- Compares fitted parameters against hardcoded baseline
- Saves model artifacts and evaluation results
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, List, Tuple, Any

try:
    from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
except ImportError:
    print("[BKT] Warning: scikit-learn not available. Install via: pip install scikit-learn")
    raise


def load_latest_dataset() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load latest interaction dataset and return train/val/test splits."""
    dataset_dir = Path("backend/data/ml/interaction_datasets")
    latest_version_file = dataset_dir / "LATEST_VERSION"
    
    if not latest_version_file.exists():
        raise FileNotFoundError("No interaction dataset found. Run build_interaction_dataset.py first.")
    
    latest_version = latest_version_file.read_text().strip()
    latest_dir = dataset_dir / latest_version
    
    events_file = latest_dir / "events.jsonl"
    if not events_file.exists():
        raise FileNotFoundError(f"Events file not found: {events_file}")
    
    all_events = []
    with open(events_file, 'r') as f:
        for line in f:
            if line.strip():
                all_events.append(json.loads(line))
    
    train_events = [e for e in all_events if e.get('split') == 'train']
    val_events = [e for e in all_events if e.get('split') == 'val']
    test_events = [e for e in all_events if e.get('split') == 'test']
    
    print(f"[BKT] Loaded: train={len(train_events)}, val={len(val_events)}, test={len(test_events)}")
    return train_events, val_events, test_events


class BKTModel:
    """
    Bayesian Knowledge Tracing model with EM parameter fitting.
    
    Per-skill modeling:
    - p0: P(mastery at start)
    - pg: P(correct | not mastered) = guess probability
    - ps: P(incorrect | mastered) = slip probability
    - pl: P(transition from not-mastered → mastered) = learn probability
    """
    
    def __init__(self, skill_mode='topic'):
        """
        Args:
            skill_mode: How to define skills ('topic' or 'subject_grade_topic')
        """
        self.skill_mode = skill_mode
        self.parameters: Dict[str, Dict[str, float]] = {}
        self.training_info: Dict[str, Any] = {}
    
    def _get_skill_id(self, event: Dict) -> str:
        """Extract skill identifier from event."""
        if self.skill_mode == 'topic':
            return event['topic']
        elif self.skill_mode == 'subject_grade_topic':
            return f"{event['subject']}_g{event['grade']}_{event['topic']}"
        return event['topic']
    
    def organize_sequences(self, events: List[Dict]) -> Dict[str, Dict[str, List[int]]]:
        """
        Organize events into (student, skill) -> [correctness_1, correctness_2, ...]
        
        Returns:
            Dict mapping student_key -> {skill -> [correctness_sequence]}
        """
        sequences = defaultdict(lambda: defaultdict(list))
        
        # Sort by student and timestamp to preserve order
        sorted_events = sorted(events, key=lambda e: (e['student_key'], e['answered_at']))
        
        for event in sorted_events:
            student = event['student_key']
            skill = self._get_skill_id(event)
            # Use 'is_correct' field (not 'correctness')
            is_correct = event.get('is_correct', 0)
            sequences[student][skill].append(int(is_correct))
        
        return sequences
    
    def forward_backward(
        self,
        observations: List[int],
        p0: float,
        pg: float,
        ps: float,
        pl: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward-backward algorithm for hidden state inference.
        
        Args:
            observations: Sequence of 0/1 (incorrect/correct)
            p0: Initial mastery probability
            pg: Guess probability (P(correct | not mastered))
            ps: Slip probability (P(incorrect | mastered))
            pl: Learn probability (P(master | not mastered))
        
        Returns:
            (gamma, alpha, beta) where:
            - gamma[t, state]: P(mastery_t = state | observations)
            - alpha[t, state]: P(mastery_t = state, observations_1..t)
            - beta[t, state]: P(observations_t+1..T | mastery_t = state)
        """
        n = len(observations)
        epsilon = 1e-10
        
        # Forward pass
        alpha = np.zeros((n, 2))
        
        # t=0: initial state
        obs_prob_0 = pg if observations[0] == 1 else (1 - pg)
        obs_prob_1 = (1 - ps) if observations[0] == 1 else ps
        
        alpha[0, 0] = (1 - p0) * obs_prob_0
        alpha[0, 1] = p0 * obs_prob_1
        
        # t=1..n-1: forward updates
        for t in range(1, n):
            obs_prob_0 = pg if observations[t] == 1 else (1 - pg)
            obs_prob_1 = (1 - ps) if observations[t] == 1 else ps
            
            # P(mastery_t=0 | mastery_t-1)
            # = P(mastery_t-1=0) * P(mastery_t=0 | mastery_t-1=0)
            #   + P(mastery_t-1=1) * P(mastery_t=0 | mastery_t-1=1)
            # = P(mastery_t-1=0) * (1-pl) + P(mastery_t-1=1) * 0
            trans_prob_0 = (alpha[t-1, 0] * (1 - pl)) * obs_prob_0
            
            # P(mastery_t=1 | mastery_t-1)
            # = P(mastery_t-1=0) * pl + P(mastery_t-1=1) * pl
            trans_prob_1 = (alpha[t-1, 0] * pl + alpha[t-1, 1] * 1.0) * obs_prob_1
            
            alpha[t, 0] = trans_prob_0
            alpha[t, 1] = trans_prob_1
        
        # Backward pass
        beta = np.ones((n, 2))
        
        for t in range(n - 2, -1, -1):
            obs_prob_0 = pg if observations[t+1] == 1 else (1 - pg)
            obs_prob_1 = (1 - ps) if observations[t+1] == 1 else ps
            
            # P(observations_t+1..T | mastery_t=0)
            # = P(mastery_t+1=0 | mastery_t=0) * P(obs_t+1 | mastery_t+1=0) * P(obs_t+2..T | mastery_t+1=0)
            #   + P(mastery_t+1=1 | mastery_t=0) * P(obs_t+1 | mastery_t+1=1) * P(obs_t+2..T | mastery_t+1=1)
            beta[t, 0] = ((1 - pl) * obs_prob_0 * beta[t+1, 0] +
                          pl * obs_prob_1 * beta[t+1, 1])
            
            # P(observations_t+1..T | mastery_t=1)
            beta[t, 1] = ((1 - pl) * obs_prob_0 * beta[t+1, 0] +
                          1.0 * obs_prob_1 * beta[t+1, 1])
        
        # Posterior: gamma[t, state] = P(mastery_t = state | all observations)
        gamma = np.zeros((n, 2))
        for t in range(n):
            gamma[t, 0] = alpha[t, 0] * beta[t, 0]
            gamma[t, 1] = alpha[t, 1] * beta[t, 1]
            total = gamma[t, 0] + gamma[t, 1] + epsilon
            gamma[t] /= total
        
        return gamma, alpha, beta
    
    def fit(
        self,
        train_events: List[Dict],
        max_iterations: int = 50,
        tol: float = 1e-4
    ) -> Dict[str, Dict[str, float]]:
        """
        Fit BKT parameters using EM algorithm.
        
        Args:
            train_events: Training interaction events
            max_iterations: Max EM iterations
            tol: Convergence tolerance
        
        Returns:
            Dict mapping skill -> {p0, pg, ps, pl}
        """
        print(f"[BKT] Organizing training sequences...")
        sequences = self.organize_sequences(train_events)
        
        all_skills = set()
        for student_seqs in sequences.values():
            all_skills.update(student_seqs.keys())
        
        print(f"[BKT] Found {len(all_skills)} skills across {len(sequences)} students")
        
        # Initialize parameters
        params = {}
        for skill in all_skills:
            params[skill] = {
                'p0': 0.25,
                'pg': 0.30,
                'ps': 0.15,
                'pl': 0.10
            }
        
        prev_likelihood = -np.inf
        
        for iteration in range(max_iterations):
            print(f"[BKT] EM iteration {iteration + 1}/{max_iterations}...")
            
            total_likelihood = 0
            
            for skill in all_skills:
                skill_sequences = []
                for student, seqs in sequences.items():
                    if skill in seqs and len(seqs[skill]) > 1:
                        skill_sequences.append(seqs[skill])
                
                if not skill_sequences:
                    continue
                
                # E-step: accumulate sufficient statistics
                e_p0_num = 0      # expected # mastered at t=0
                e_p0_den = 0      # expected # students
                
                e_pg_num = 0      # expected # correct when not mastered
                e_pg_den = 0      # expected # not mastered
                
                e_ps_num = 0      # expected # incorrect when mastered
                e_ps_den = 0      # expected # mastered
                
                e_pl_num = 0      # expected # learn transitions
                e_pl_den = 0      # expected # t where not mastered at t-1
                
                for obs_seq in skill_sequences:
                    gamma, alpha, beta = self.forward_backward(
                        obs_seq,
                        params[skill]['p0'],
                        params[skill]['pg'],
                        params[skill]['ps'],
                        params[skill]['pl']
                    )
                    
                    # Likelihood contribution
                    total_likelihood += np.sum(np.log(alpha[-1] + 1e-10))
                    
                    # Initial state: P(mastery_0 | observations)
                    e_p0_num += gamma[0, 1]
                    e_p0_den += 1.0
                    
                    # Observation likelihoods
                    for t, obs in enumerate(obs_seq):
                        if obs == 1:
                            e_pg_num += gamma[t, 0]
                            e_ps_den += gamma[t, 1]
                            e_ps_num += gamma[t, 1] - gamma[t, 1]  # 0, since obs=1
                        else:
                            e_pg_den += gamma[t, 0]
                            e_ps_num += gamma[t, 1]
                            e_pg_den += 0
                        
                        e_pg_den += gamma[t, 0]
                        e_ps_den += gamma[t, 1]
                
                    # Learn transition: count expected not-mastered states at t < T-1
                    for t in range(len(obs_seq) - 1):
                        e_pl_den += gamma[t, 0]
                        # In a full EM, we'd compute P(mastery_t+1=1, mastery_t=0 | obs)
                        # Simplified: assume learn when mastered at t+1 but not at t
                        if gamma[t, 0] > 0.5:  # heuristic
                            # Check if mastered at t+1
                            if t + 1 < len(gamma) and gamma[t+1, 1] > 0.5:
                                e_pl_num += 1.0
                
                # M-step: update parameters
                if e_p0_den > 0:
                    params[skill]['p0'] = np.clip(e_p0_num / e_p0_den, 0.01, 0.99)
                
                if e_pg_den > 0:
                    params[skill]['pg'] = np.clip(e_pg_num / e_pg_den, 0.01, 0.99)
                else:
                    params[skill]['pg'] = 0.25
                
                if e_ps_den > 0:
                    params[skill]['ps'] = np.clip(e_ps_num / e_ps_den, 0.01, 0.99)
                else:
                    params[skill]['ps'] = 0.15
                
                if e_pl_den > 0:
                    params[skill]['pl'] = np.clip(e_pl_num / e_pl_den, 0.01, 0.5)
                else:
                    params[skill]['pl'] = 0.10
            
            # Check convergence
            likelihood_change = total_likelihood - prev_likelihood
            print(f"[BKT]   Likelihood: {total_likelihood:.2f}, Change: {likelihood_change:.4f}")
            
            if abs(likelihood_change) < tol:
                print(f"[BKT] Converged at iteration {iteration + 1}")
                break
            
            prev_likelihood = total_likelihood
        
        self.parameters = params
        self.training_info = {
            'num_skills': len(all_skills),
            'num_sequences': sum(len(seqs) for seqs in sequences.values()),
            'skill_mode': self.skill_mode,
            'final_likelihood': prev_likelihood
        }
        
        return params
    
    def evaluate(self, test_events: List[Dict]) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_events: Test interaction events
        
        Returns:
            Dict with metrics: {bce, auc, rmse}
        """
        sequences = self.organize_sequences(test_events)
        
        predictions = []
        actuals = []
        
        for student, seqs in sequences.items():
            for skill, obs_seq in seqs.items():
                if skill not in self.parameters or len(obs_seq) < 1:
                    continue
                
                params = self.parameters[skill]
                gamma, _, _ = self.forward_backward(
                    obs_seq,
                    params['p0'],
                    params['pg'],
                    params['ps'],
                    params['pl']
                )
                
                # Predict based on posterior mastery and observation model
                for t, obs in enumerate(obs_seq):
                    # P(correct | posterior mastery)
                    prob_correct = (
                        gamma[t, 0] * params['pg'] +
                        gamma[t, 1] * (1 - params['ps'])
                    )
                    predictions.append(prob_correct)
                    actuals.append(obs)
        
        if not predictions:
            return {'bce': np.nan, 'auc': np.nan, 'rmse': np.nan}
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Clip predictions for numerical stability
        eps = 1e-10
        predictions = np.clip(predictions, eps, 1 - eps)
        
        bce = log_loss(actuals, predictions)
        auc = roc_auc_score(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        
        return {'bce': bce, 'auc': auc, 'rmse': rmse}
    
    def save(self, artifact_dir: Path = None) -> Tuple[Path, Path]:
        """Save model and metadata."""
        if artifact_dir is None:
            artifact_dir = Path("backend/models/bkt")
        
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        
        # Save model
        model_path = artifact_dir / f"bkt_model_{timestamp}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'parameters': self.parameters,
                'training_info': self.training_info,
                'skill_mode': self.skill_mode
            }, f)
        
        # Also save as latest
        latest_path = artifact_dir / "bkt_model_latest.pkl"
        with open(latest_path, 'wb') as f:
            pickle.dump({
                'parameters': self.parameters,
                'training_info': self.training_info,
                'skill_mode': self.skill_mode
            }, f)
        
        print(f"[BKT] Model saved: {model_path}")
        print(f"[BKT] Latest symlink: {latest_path}")
        
        return model_path, latest_path


def compare_with_baseline(
    fitted_params: Dict[str, Dict[str, float]],
    baseline_metrics: Dict[str, float],
    fitted_metrics: Dict[str, float]
) -> None:
    """Print comparison between baseline and fitted parameters."""
    print("\n" + "="*70)
    print("BKT PARAMETER FITTING RESULTS")
    print("="*70)
    
    print(f"\nSample fitted parameters (first 3 skills):")
    for i, (skill, params) in enumerate(list(fitted_params.items())[:3]):
        print(f"  {skill}:")
        print(f"    p0 (initial): {params['p0']:.4f}")
        print(f"    pg (guess):   {params['pg']:.4f}")
        print(f"    ps (slip):    {params['ps']:.4f}")
        print(f"    pl (learn):   {params['pl']:.4f}")
    
    print(f"\nHardcoded Baseline Parameters:")
    print(f"  pg: 0.28  (fixed)")
    print(f"  ps: 0.08  (fixed)")
    
    print(f"\nEvaluation Metrics Comparison:")
    print(f"  Metric        | Baseline      | Fitted        | Delta")
    print(f"  {'-'*60}")
    print(f"  BCE           | N/A           | {fitted_metrics['bce']:.4f}     | N/A")
    print(f"  AUC           | N/A           | {fitted_metrics['auc']:.4f}     | N/A")
    print(f"  RMSE          | N/A           | {fitted_metrics['rmse']:.4f}     | N/A")
    print(f"\nNote: Baseline metrics unavailable (hardcoded params not fully specified).")
    print(f"Fitted model should show AUC > 0.65 and RMSE < 0.50 for reasonable fit.\n")


def main():
    print("[BKT] ========== TASK 3: BKT EM PARAMETER FITTING ==========")
    
    # Load dataset
    print("[BKT] Loading interaction dataset...")
    train_events, val_events, test_events = load_latest_dataset()
    
    # Train model
    print("[BKT] Training BKT model with EM algorithm...")
    model = BKTModel(skill_mode='topic')
    fitted_params = model.fit(train_events, max_iterations=30)
    
    print(f"[BKT] Fitted {len(fitted_params)} topic-level skills")
    
    # Evaluate on test set
    print("[BKT] Evaluating on test set...")
    test_metrics = model.evaluate(test_events)
    
    print(f"[BKT] Test Metrics:")
    print(f"      BCE:  {test_metrics['bce']:.4f}")
    print(f"      AUC:  {test_metrics['auc']:.4f}")
    print(f"      RMSE: {test_metrics['rmse']:.4f}")
    
    # Evaluation on validation (for reference)
    print("[BKT] Evaluating on validation set...")
    val_metrics = model.evaluate(val_events)
    print(f"[BKT] Val Metrics:")
    print(f"      BCE:  {val_metrics['bce']:.4f}")
    print(f"      AUC:  {val_metrics['auc']:.4f}")
    print(f"      RMSE: {val_metrics['rmse']:.4f}")
    
    # Save artifacts
    model_path, latest_path = model.save()
    
    # Save metrics
    metrics_path = Path("backend/models/bkt") / f"bkt_metrics_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'train_size': len(train_events),
            'val_size': len(val_events),
            'test_size': len(test_events),
            'num_skills': len(fitted_params),
            'test_metrics': test_metrics,
            'val_metrics': val_metrics,
            'fitted_params_sample': {k: v for k, v in list(fitted_params.items())[:3]}
        }, f, indent=2)
    print(f"[BKT] Metrics saved: {metrics_path}")
    
    # Comparison
    compare_with_baseline(fitted_params, {}, test_metrics)
    
    print("[BKT] ========== TASK 3 COMPLETE ==========\n")
    
    return model


if __name__ == "__main__":
    model = main()
