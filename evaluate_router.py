"""
evaluate_router.py
==================
Measure how well each router sends messages to the right agent.

  python evaluate_router.py          development: 5-fold cross-validation on the
                                     TRAIN set only (the held-out test is not touched)
  python evaluate_router.py --test   final: train on TRAIN, score on the held-out TEST

Routers compared: the hand-written rules, the learned classifier, and the
hybrid the app uses (learned when confident, otherwise rules).
"""

import argparse
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedKFold

from agents.router_data import ROUTES, TEST, TRAIN
from agents.routing import HybridRouter, LearnedRouter, route_message
from rag_core.embeddings import get_embeddings


def flatten(data):
    return [(text, route) for route, texts in data.items() for text in texts]


def report(title, texts, truth, predicted, confidences=None, show_errors=True):
    correct = [p == t for p, t in zip(predicted, truth)]
    print(f"\n{title}: {sum(correct)}/{len(correct)} correct = {100 * np.mean(correct):.1f}%")
    for route in ROUTES:
        idx = [i for i, t in enumerate(truth) if t == route]
        got = sum(correct[i] for i in idx)
        print(f"    {route:11} {got:3}/{len(idx):3}  {100 * got / max(len(idx), 1):5.1f}%")
    if show_errors:
        for i, ok in enumerate(correct):
            if not ok:
                conf = f" ({confidences[i]:.2f})" if confidences is not None else ""
                print(f"      wrong: {texts[i]!r}: expected {truth[i]}, got {predicted[i]}{conf}")


def hybrid_predictions(learned_pred, learned_conf, rule_pred, threshold):
    return [l if c >= threshold else r for l, c, r in zip(learned_pred, learned_conf, rule_pred)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="score on the held-out test set")
    parser.add_argument("--threshold", type=float, default=HybridRouter().threshold)
    args = parser.parse_args()
    embeddings = get_embeddings()

    if args.test:
        train, test = flatten(TRAIN), flatten(TEST)
        texts, truth = [t for t, _ in test], [r for _, r in test]
        learned = LearnedRouter(train, embeddings)
        probs = learned.predict_proba(texts)
        learned_pred = [max(p, key=p.get) for p in probs]
        learned_conf = [max(p.values()) for p in probs]
        rule_pred = [route_message(t).name for t in texts]
        print(f"Held-out test set: {len(texts)} messages (train set: {len(train)})")
        report("RULES  ", texts, truth, rule_pred)
        report("LEARNED", texts, truth, learned_pred, learned_conf)
        report("HYBRID ", texts, truth, hybrid_predictions(learned_pred, learned_conf, rule_pred, args.threshold))
        fallbacks = sum(c < args.threshold for c in learned_conf)
        print(f"\nHybrid fell back to the rules on {fallbacks}/{len(texts)} messages (threshold {args.threshold}).")
        return

    data = flatten(TRAIN)
    texts, truth = [t for t, _ in data], np.array([r for _, r in data])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    learned_pred = np.empty(len(texts), dtype=object)
    learned_conf = np.zeros(len(texts))
    for fit_idx, val_idx in skf.split(texts, truth):
        model = LearnedRouter([data[i] for i in fit_idx], embeddings)
        probs = model.predict_proba([texts[i] for i in val_idx])
        for i, p in zip(val_idx, probs):
            learned_pred[i] = max(p, key=p.get)
            learned_conf[i] = max(p.values())
    rule_pred = [route_message(t).name for t in texts]
    print(f"5-fold cross-validation on the TRAIN set ({len(texts)} messages); test set untouched")
    report("RULES  ", texts, list(truth), rule_pred, show_errors=False)
    report("LEARNED", texts, list(truth), list(learned_pred), learned_conf)
    hybrid = hybrid_predictions(list(learned_pred), learned_conf, rule_pred, args.threshold)
    report("HYBRID ", texts, list(truth), hybrid, show_errors=False)
    print("\nHybrid accuracy by fallback threshold (0 = never fall back to the rules):")
    for threshold in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6):
        pred = hybrid_predictions(list(learned_pred), learned_conf, rule_pred, threshold)
        fallbacks = int(sum(learned_conf < threshold))
        print(f"    threshold {threshold:.1f}: {100 * np.mean([p == t for p, t in zip(pred, truth)]):.1f}%  "
              f"(rules used on {fallbacks}/{len(truth)})")
    print("\nConfidence of correct vs wrong learned predictions:")
    ok = [c for c, p, t in zip(learned_conf, learned_pred, truth) if p == t]
    bad = [c for c, p, t in zip(learned_conf, learned_pred, truth) if p != t]
    print(f"    correct: mean {np.mean(ok):.2f}   wrong: mean {np.mean(bad) if bad else float('nan'):.2f}")
    print(f"    labels in train: {dict(Counter(truth))}")


if __name__ == "__main__":
    main()
