"""
HiveSTA v2: Advanced Non-Neural Token Prediction Engine
======================================================
Architecture combining Variable-Order Sparse Context Automata (PPM/Backoff principles),
Discrete Associative Memory (HashRAG v2), and Bitwise Hash State Registers.
Designed for extreme low-power, microsecond latency, and zero catastrophic forgetting.
"""

import math
import time
import hashlib
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


class VariableContextAutomaton:
    """
    Automaton that learns variable-length token sequences (from 1-gram to max_order).
    Uses absolute discounting and recursive backoff (similar to Kneser-Ney / PPM)
    to calculate true probability distributions over candidate next tokens.
    """
    def __init__(self, name: str, max_order: int = 4, discount: float = 0.6):
        self.name = name
        self.max_order = max_order
        self.discount = discount
        # context tuple -> {next_token: count}
        self.counts: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # context tuple -> total occurrences
        self.context_totals: Dict[Tuple[str, ...], int] = defaultdict(int)
        # unigram counts for global fallback
        self.unigram_counts: Dict[str, int] = defaultdict(int)
        self.total_tokens: int = 0
        self.vocab: set = set()

    def train_sequence(self, tokens: List[str]):
        """Ingest a sequence of tokens and update n-gram counts across all orders."""
        n = len(tokens)
        for i in range(n):
            token = tokens[i]
            self.unigram_counts[token] += 1
            self.total_tokens += 1
            self.vocab.add(token)

            for order in range(1, self.max_order + 1):
                if i >= order:
                    context = tuple(tokens[i - order:i])
                    self.counts[context][token] += 1
                    self.context_totals[context] += 1

    def predict_distribution(self, context_tokens: List[str]) -> Dict[str, float]:
        """
        Calculates next-token probability distribution using variable-order backoff.
        P(w | context) combines higher-order observations with lower-order fallbacks.
        """
        if not self.vocab:
            return {}

        # Default uniform/unigram fallback
        probs: Dict[str, float] = {}
        for w, c in self.unigram_counts.items():
            probs[w] = (c + 0.1) / (self.total_tokens + 0.1 * len(self.vocab))

        # Backoff interpolation from order 1 up to max_order
        for order in range(1, self.max_order + 1):
            if len(context_tokens) >= order:
                sub_ctx = tuple(context_tokens[-order:])
                if sub_ctx in self.counts:
                    total = self.context_totals[sub_ctx]
                    num_distinct = len(self.counts[sub_ctx])
                    # Interpolation weight for higher order
                    lambda_ctx = max(0.1, min(0.9, 1.0 - (num_distinct * self.discount) / total))

                    higher_order_probs = {}
                    for tok, count in self.counts[sub_ctx].items():
                        higher_order_probs[tok] = max(0.0, count - self.discount) / total

                    # Blend higher order with previous lower order
                    for w in self.vocab:
                        p_higher = higher_order_probs.get(w, 0.0)
                        probs[w] = (1.0 - lambda_ctx) * probs.get(w, 0.0) + lambda_ctx * p_higher

        # Normalize distribution
        total_p = sum(probs.values())
        if total_p > 0:
            for w in probs:
                probs[w] /= total_p

        return probs


class DynamicBitState:
    """
    Maintains a compact 128-bit discrete register updated via bitwise hash operations.
    Acts as a non-recurrent working memory tracking recent topic / syntactic signatures.
    """
    def __init__(self, size: int = 8):
        self.size = size
        self.state = [0] * size

    def update(self, token: str):
        # 64-bit integer hash from token
        h = int(hashlib.blake2b(token.encode(), digest_size=8).hexdigest(), 16)
        for i in range(self.size):
            shifted = (h >> (i * 7)) & 0xFFFF
            self.state[i] = ((self.state[i] ^ shifted) + (h & 0xFF)) & 0xFFFF

    def reset(self):
        self.state = [0] * self.size

    def get_bias(self, token: str) -> float:
        """Computes a lightweight deterministic bias between state and candidate token."""
        h = int(hashlib.blake2b(token.encode(), digest_size=4).hexdigest(), 16)
        match = sum(1 for i in range(self.size) if (self.state[i] ^ h) & 0x0F == 0)
        return (match / self.size) * 0.05


class AssociativeHashRAG:
    """
    Discrete Associative Memory using an inverted index with dynamic token co-occurrence
    and keyword association windows (no floating-point vectors required).
    """
    def __init__(self):
        self.inverted_index: Dict[str, List[int]] = defaultdict(list)
        self.co_occurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        self.documents: Dict[int, List[str]] = {}

    def add_document(self, doc_id: int, tokens: List[str]):
        self.documents[doc_id] = tokens
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self.inverted_index[token].append(doc_id)

        # Sliding co-occurrence window of size 4
        for i in range(len(tokens)):
            for j in range(i + 1, min(i + 5, len(tokens))):
                pair = tuple(sorted([tokens[i], tokens[j]]))
                self.co_occurrence[pair] += 1

    def retrieve_associated_tokens(self, query_tokens: List[str], top_k: int = 5) -> Dict[str, float]:
        """Returns tokens strongly associated with the query context."""
        doc_scores: Dict[int, float] = defaultdict(float)
        for q in query_tokens:
            for doc_id in self.inverted_index.get(q, []):
                doc_scores[doc_id] += 1.0

        if not doc_scores:
            return {}

        best_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        token_boosts: Dict[str, float] = defaultdict(float)

        for doc_id, doc_score in best_docs:
            doc_toks = self.documents.get(doc_id, [])
            for tok in doc_toks:
                token_boosts[tok] += doc_score * 0.02

        return token_boosts


class HiveSTA_v2:
    """
    Hive Sparse Token Automata (v2) Coordinator:
    Combines specialized variable-order automata, non-vector associative RAG,
    and discrete bitwise registers into a unified token prediction engine.
    """
    def __init__(self, max_order: int = 4):
        self.local_automaton = VariableContextAutomaton("local_context", max_order=max_order, discount=0.55)
        self.global_automaton = VariableContextAutomaton("global_priors", max_order=2, discount=0.7)
        self.bit_state = DynamicBitState(size=8)
        self.associative_rag = AssociativeHashRAG()
        self.doc_counter = 0

    def add_text(self, text: str, track_in_rag: bool = True):
        """Learns text online in O(L) time without retraining or backpropagation."""
        tokens = self.tokenize(text)
        if not tokens:
            return

        self.local_automaton.train_sequence(tokens)
        self.global_automaton.train_sequence(tokens)

        if track_in_rag:
            self.associative_rag.add_document(self.doc_counter, tokens)
            self.doc_counter += 1

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple deterministic whitespace and punctuation tokenizer."""
        clean = ""
        for char in text.lower():
            if char.isalnum() or char.isspace():
                clean += char
            else:
                clean += f" {char} "
        return [t for t in clean.split() if t]

    def predict_next(self, context: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predicts top candidate tokens given context text.
        Returns list of (token, probability) sorted descending.
        """
        tokens = self.tokenize(context)
        if not tokens and not self.local_automaton.vocab:
            return []

        # 1. Probabilities from variable-order automaton
        p_local = self.local_automaton.predict_distribution(tokens)
        p_global = self.global_automaton.predict_distribution(tokens)

        # 2. Associated token boosts from HashRAG
        rag_boosts = self.associative_rag.retrieve_associated_tokens(tokens[-4:])

        # 3. Dynamic bitwise bias
        vocab = self.local_automaton.vocab
        combined_scores: Dict[str, float] = {}

        for w in vocab:
            score = 0.75 * p_local.get(w, 0.0) + 0.25 * p_global.get(w, 0.0)
            score += rag_boosts.get(w, 0.0)
            score += self.bit_state.get_bias(w)
            combined_scores[w] = score

        # Normalize to valid probabilities
        total = sum(combined_scores.values())
        if total > 0:
            for w in combined_scores:
                combined_scores[w] /= total

        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def evaluate_sequence(self, tokens: List[str]) -> Tuple[float, float, Dict[int, float]]:
        """
        Evaluates cross-entropy loss, perplexity, and Top-K accuracy on a sequence.
        Returns (cross_entropy_bits, perplexity, {1: top1_acc, 3: top3_acc, 5: top5_acc}).
        """
        if len(tokens) < 2:
            return 0.0, 1.0, {1: 1.0, 3: 1.0, 5: 1.0}

        total_log_prob = 0.0
        n_eval = 0
        hits = {1: 0, 3: 0, 5: 0}
        eps = 1e-7

        for i in range(1, len(tokens)):
            ctx = tokens[:i]
            target = tokens[i]

            predictions = self.predict_next(" ".join(ctx[-5:]), top_k=10)
            pred_dict = dict(predictions)
            pred_tokens = [tok for tok, _ in predictions]

            prob = pred_dict.get(target, eps)
            total_log_prob += -math.log2(max(prob, eps))
            n_eval += 1

            for k in [1, 3, 5]:
                if target in pred_tokens[:k]:
                    hits[k] += 1

        cross_entropy = total_log_prob / n_eval if n_eval > 0 else 0.0
        perplexity = 2.0 ** min(cross_entropy, 20.0)  # cap for display stability
        accuracies = {k: hits[k] / n_eval if n_eval > 0 else 0.0 for k in hits}

        return cross_entropy, perplexity, accuracies

    def generate(self, prompt: str, max_tokens: int = 15, deterministic: bool = True) -> str:
        """Autoregressively generates continuation tokens."""
        tokens = self.tokenize(prompt)
        current_tokens = list(tokens)
        self.bit_state.reset()

        for t in current_tokens:
            self.bit_state.update(t)

        for _ in range(max_tokens):
            context_str = " ".join(current_tokens[-6:])
            candidates = self.predict_next(context_str, top_k=5)
            if not candidates:
                break

            if deterministic:
                next_tok = candidates[0][0]
            else:
                toks, probs = zip(*candidates)
                r = time.time() % 1.0
                cum = 0.0
                next_tok = toks[0]
                for tok, p in zip(toks, probs):
                    cum += p
                    if r <= cum:
                        next_tok = tok
                        break

            current_tokens.append(next_tok)
            self.bit_state.update(next_tok)

        return " ".join(current_tokens)


if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("🚀 HiveSTA v2 - Moteur de Prédiction Discret Sans Neurones")
    print("=" * 60)
    print("Initialisation du modèle de test...")
    engine = HiveSTA_v2(max_order=4)
    engine.add_text("operating systems manage hardware resources like memory cpu and storage devices")
    engine.add_text("discrete automata process tokens with microsecond deterministic latency")
    engine.add_text("neural networks require massive matrix multiplications on gpu clusters")

    prompt = "operating systems manage"
    preds = engine.predict_next(prompt, top_k=3)
    gen = engine.generate(prompt, max_tokens=6)

    print(f"Prompt de test : '{prompt}'")
    print(f"Top-3 prédictions de tokens : {preds}")
    print(f"Génération autorégressive : '{gen}'")
    print("\n💡 Pour lancer le Chatbot interactif complet, exécutez :")
    print("   python3 code/chatbot.py")
    print("=" * 60)

