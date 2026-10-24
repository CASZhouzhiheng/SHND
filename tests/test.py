from __future__ import annotations
import csv
import io
import json
import math
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from shnd.construction import (GRAPH_SCHEMA, normalized_derivatives, signed_cofluctuation,
                               test_candidate_edges as permutation_test, finalize_edges, make_features)
from shnd.data import stratified_tenfold, cross_dataset_split, HypergraphDataset
from shnd.incidence import IncidenceOperator
from shnd.model import SHNDModel
from shnd.losses import contrastive_loss, contrastive_masks, dual_contrastive_losses
from shnd.metrics import risk_at_coverages, coverage_details
from shnd.engine import train_run, ABLATIONS
from shnd.experiment import staged_search

torch.set_num_threads(1)


class PaperTests(unittest.TestCase):
    def test_derivative_not_centered(self):
        raw = np.array([[0, 0], [1, 1], [3, 4], [7, 6]], np.float32)
        expected = np.diff(raw, axis=0) / np.diff(raw, axis=0).std(axis=0)
        np.testing.assert_allclose(normalized_derivatives(raw), expected)
        self.assertTrue((normalized_derivatives(raw).mean(0) != 0).all())
        self.assertEqual(make_features(raw)[0].shape, (2, 3))
        np.testing.assert_equal(normalized_derivatives(np.ones((4, 2))), 0)

    def test_signed_mtd_against_numpy(self):
        q = np.array([[1, 2, -1], [-2, -3, -2], [2, -1, 2], [4, 5, 3]], np.float32)
        for edge in ((0, 1), (0, 1, 2)):
            values = q[:, edge]
            product = values.prod(1)
            xi = (product-product.mean())/product.std()
            sign = np.where((values >= 0).all(1), 1, np.where((values <= 0).all(1), -1, 0))
            actual = signed_cofluctuation(torch.tensor(q), torch.tensor([edge]))[0]
            np.testing.assert_allclose(actual.numpy(), sign*np.abs(xi), rtol=2e-6)
            self.assertEqual(actual[2], 0)
        batched = signed_cofluctuation(torch.tensor(np.stack([q, q])), torch.tensor([[0, 1]]))
        self.assertEqual(tuple(batched.shape), (2, 1, 4))

    def test_null_and_pvalue_against_independent_reference(self):
        q = np.random.default_rng(4).normal(size=(8, 4)).astype(np.float32)
        kw = dict(order=3, permutations=19, alpha=.2, seed=31, device="cpu")
        selected, result = permutation_test(q, edge_batch_size=1, permutation_chunk=4, **kw)
        selected2, result2 = permutation_test(q, edge_batch_size=8, permutation_chunk=7, **kw)
        self.assertEqual(result, result2)
        generator = torch.Generator().manual_seed(31)
        indices = torch.rand((19, 4, 8), generator=generator).argsort(-1).transpose(1, 2).numpy()
        for edge, p in result:
            def strength(a):
                v = a[:, edge]; product = v.prod(1)
                xi = (product-product.mean())/product.std()
                gate = (v >= 0).all(1) | (v <= 0).all(1)
                return (np.abs(xi)*gate).mean()
            null = [strength(np.take_along_axis(q, ix, axis=0)) for ix in indices]
            self.assertAlmostEqual(p, (1+sum(x >= strength(q) for x in null))/20)
        self.assertTrue(all(p < .2 for p in selected.values()))
        self.assertEqual(finalize_edges(selected, result, 4), sorted(selected.items(), key=lambda x: (x[1], len(x[0]), x[0])))
        self.assertEqual(finalize_edges({}, result, 4), [])

    def test_gradient_and_adjoint_dense(self):
        edges = [(0, 1), (1, 2, 3)]
        op = IncidenceOperator.from_edge_list(edges, 5)
        degree = np.array([1, 2, 1, 1, 1])
        dense = []
        for edge in edges:
            for v in edge:
                row = np.zeros(5)
                row[v] += 1/np.sqrt(degree[v])
                for u in edge:
                    row[u] -= 1/(len(edge)*np.sqrt(degree[u]))
                dense.append(row)
        G = torch.tensor(np.array(dense), dtype=torch.float32)
        x = torch.randn(5, 3); flux = torch.randn(5, 3)
        torch.testing.assert_close(op.gradient(x), G@x)
        torch.testing.assert_close(op.divergence(flux), G.T@flux)
        torch.testing.assert_close((op.gradient(x)*flux).sum(), (x*op.divergence(flux)).sum())
        torch.testing.assert_close(op.gradient(torch.tensor(np.sqrt(degree), dtype=torch.float32)[:, None]), torch.zeros(5, 1))
        with self.assertRaises(ValueError):
            IncidenceOperator.from_edge_list([], 5)

    def test_encoder_equation_and_rejection_readout(self):
        torch.manual_seed(7)
        op = IncidenceOperator.from_edge_list([(0, 1), (1, 2, 3)], 4)
        model = SHNDModel(5, 6, 3)
        x = torch.randn(4, 5)
        output = model([{"x": x, "incidence": op}])
        a = output["incidence_attention"]
        sums = torch.zeros(4).index_add_(0, op.node_ids, a)
        torch.testing.assert_close(sums, torch.ones(4))
        G = op.gradient(torch.eye(4))
        z = torch.relu(model.encoder.input_projection(x))
        for _ in range(3):
            z = z-G.T@torch.diag(a)@G@z
        torch.testing.assert_close(z, output["node_embeddings"])
        edge_means = torch.stack([z[:2].mean(0), z[1:].mean(0)])
        beta = torch.stack([a[:2].mean(), a[2:].mean()]).softmax(0)
        torch.testing.assert_close(output["rejection_embedding"][0], (beta[:, None]*edge_means).sum(0))
        self.assertFalse(any("step_logits" in name for name, _ in model.named_parameters()))
        self.assertIsNone(model.predictive_head.bias)

    def test_batch_matches_single_graph(self):
        model = SHNDModel(4, 5, 2)
        batch = [{"x": torch.randn(4, 4), "incidence": IncidenceOperator.from_edge_list(e, 4)}
                 for e in [[(0, 1), (1, 2, 3)], [(0, 2, 3)]]]
        merged = model(batch)
        for key in ("logits", "reliability_logits", "predictive_embedding", "rejection_embedding"):
            torch.testing.assert_close(merged[key], torch.cat([model([b])[key] for b in batch]))

    def test_contrastive_masks(self):
        y = torch.tensor([0, 0, 1, 1]); pred = torch.tensor([0, 0, 0, 1])
        g = torch.tensor([.1, .15, .1, .5])
        pf, nf, pg, ng = contrastive_masks(y, pred, g, .1)
        self.assertEqual(pf[0].tolist(), [False, True, False, False])
        self.assertEqual(nf[0].tolist(), [False, False, True, False])
        self.assertEqual(pg[0].tolist(), [False, True, False, False])
        self.assertEqual(ng[0].tolist(), [False, False, True, True])
        self.assertFalse((pg & ng).any())
        self.assertEqual(int((pg | ng).sum()), 12)

    def test_full_loss_keeps_all_anchors_and_positive_only(self):
        h = torch.ones(3, 2, requires_grad=True)
        p = torch.zeros(3, 3, dtype=torch.bool); p[0, 1:] = True
        n = torch.zeros_like(p)
        weights = torch.tensor([.5, .7, .9], requires_grad=True)
        actual = contrastive_loss(h, p, n, weights, .2, top_k=0)
        self.assertAlmostEqual(actual.item(), .5*math.log(2)/3, places=6)
        actual.backward()
        self.assertAlmostEqual(weights.grad[0].item(), math.log(2)/3, places=6)
        self.assertTrue(torch.isfinite(h.grad).all())

    def test_confidence_gradient_switch(self):
        for detach in (False, True):
            logits = torch.tensor([[2., 0], [2., 0], [2., 0], [0., 2]], requires_grad=True)
            reliability = torch.zeros(4, requires_grad=True)
            outputs = dict(logits=logits, reliability_logits=reliability,
                           predictive_embedding=torch.randn(4, 3, requires_grad=True),
                           rejection_embedding=torch.randn(4, 3, requires_grad=True))
            cf, cr = dual_contrastive_losses(outputs, torch.tensor([0, 0, 1, 1]), top_k=0, detach_confidence=detach)
            (cf+cr).backward()
            if detach:
                self.assertIsNone(logits.grad)
            else:
                self.assertGreater(logits.grad.abs().sum().item(), 0)
                self.assertGreater(reliability.grad.abs().sum().item(), 0)

    def test_tenfold_no_leakage_and_reproducibility(self):
        rows = [dict(dataset="A", subject_id=str(i), label=i%2) for i in range(43)]
        folds = list(stratified_tenfold(rows, 9))
        self.assertEqual(folds, list(stratified_tenfold(list(reversed(rows)), 9)))
        seen = []
        for _, train, val, test in folds:
            groups = [set(r["subject_id"] for r in part) for part in (train, val, test)]
            self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
            self.assertEqual(sum(map(len, groups)), len(rows))
            seen.extend(r["subject_id"] for r in test)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), set(r["subject_id"] for r in rows))
        target = [dict(dataset="B", subject_id=str(i), label=i%2) for i in range(12)]
        train, val, test = cross_dataset_split(rows+target, "A", "B", 9)
        self.assertEqual(test, target)
        self.assertEqual(len(train)+len(val), len(rows))

    def test_coverage_percent_and_all_subjects(self):
        rows = [dict(label=0, prediction=int(i >= 5), score=10-i) for i in range(10)]
        risks = risk_at_coverages(rows)
        self.assertIn(100, risks); self.assertNotIn(99, risks)
        self.assertEqual(risks[50], 0); self.assertEqual(risks[100], 50)
        self.assertEqual(coverage_details(rows)[-1]["accepted_count"], 10)

    def test_search_never_receives_test_rows(self):
        calls = []
        def fake(train, val, test, config, method, seed, directory):
            calls.append((train, val, test))
            return {"validation_score": config["hidden_dim"]}
        with tempfile.TemporaryDirectory() as temp, patch("shnd.experiment.train_run", fake):
            result = staged_search(["train"], ["val"], {"hidden_dim": 64}, "shnd", 7,
                                  Path(temp), [{"hidden_dim": [32, 16]}])
        self.assertEqual(result["hidden_dim"], 16)
        self.assertTrue(all(c == (["train"], ["val"], None) for c in calls))

    def test_training_all_ablations_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp); rows = []
            for i in range(12):
                path = root/f"g{i}.pt"
                torch.save(dict(schema=GRAPH_SCHEMA, x_derivative=torch.randn(4, 5),
                                edge_list=[(0, 1), (1, 2), (1, 2, 3)]), path)
                rows.append(dict(pt_path=str(path), label=i%2, dataset="test", subject_id=str(i)))
            cfg = dict(feature="x_derivative", input_dim=5, hidden_dim=6, num_layers=2, num_classes=2,
                       epochs=3, warmup_epochs=1, batch_size=4, learning_rate=.0005, weight_decay=.0001,
                       lambda_c=.5, lambda_f=.05, lambda_g=.05, tau_f=.2, tau_g=.2, gamma=.1,
                       contrastive_k=3, eval_every=2, selection_metric="mean_risk", gradient_clip=0, device="cpu")
            for method in sorted(ABLATIONS):
                directory = root/method
                risks = train_run(rows[:8], rows[8:10], rows[10:], cfg, method, 7, directory)
                self.assertTrue(all(math.isfinite(v) for v in risks.values()))
                checkpoint = torch.load(directory/"best_model.pt", weights_only=False)
                self.assertEqual(checkpoint["schema"], "shnd-paper-model-v1")
                SHNDModel(**checkpoint["model_config"]).load_state_dict(checkpoint["model_state"])
                if method == "shnd":
                    with (directory/"coverage_risk_msp.csv").open(encoding="utf-8-sig") as handle:
                        msp = list(csv.DictReader(handle))
                    self.assertEqual(float(msp[-1]["risk_pct"]), risks[100])
            old = root/"old.pt"; torch.save(dict(x_derivative=torch.ones(4,5), edge_list=[(0,1)]), old)
            with self.assertRaisesRegex(ValueError, "legacy graph"):
                HypergraphDataset([dict(rows[0], pt_path=str(old))])[0]
            chosen = staged_search(rows[:8], rows[8:10], cfg, "shnd", 7, root/"real_search",
                                   [{"hidden_dim": [4, 6]}])
            self.assertIn(chosen["hidden_dim"], [4, 6])
            for trial in (1, 2):
                trial_dir = root/f"real_search/stage_1/trial_{trial}"
                self.assertTrue((trial_dir/"best_model.pt").exists())
                self.assertFalse((trial_dir/"predictions.csv").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
