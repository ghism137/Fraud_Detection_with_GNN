"""
run_gnn.py — Orchestrator: Pipeline B → Graph Build → GAT (SKELETON)
=====================================================================
TRẠNG THÁI: SKELETON — Chờ implement Tuần 6-7

Các module phụ thuộc chưa được implement:
  - src/graph/builder_3tier.py  (3-Tier Heuristic Graph Construction)
  - src/graph/validator.py      (Graph Quality Validation, max_degree < 500)
  - src/models/gat_layer.py     (GAT Model với LayerNorm + Residual)

Luồng dự kiến (sẽ điền vào sau):
  [1] Load & Merge raw CSV
  [2] Load top50_features.json từ cache (bắt buộc — GNN chạy sau LGBM/MLP)
  [3] MLPPipelinePreprocessor (Pipeline B) → node features
  [4] GraphBuilder3Tier.build() → edge_index (3 tầng: uid, temporal, device)
  [5] GraphValidator.validate(max_degree=500) → cảnh báo Super-nodes
  [6] GATModel.fit() → train trên PyG Data object
  [7] Evaluate (AUC-PR + Cost Matrix)
  [8] np.save("data/processed/gnn_proba.npy", y_prob) → Late Fusion

Sử dụng (sau khi implement xong):
  python src/pipelines/run_gnn.py --mode debug --model gat

Tham khảo kiến trúc:
  - Dou et al. 2020 — CARE-GNN (arxiv:2008.08692)
  - Kipf & Welling 2017 — GCN (arxiv:1609.02907)
  - PyG GAT: torch_geometric.nn.GATConv
"""

import os
import sys
import argparse

# ── Setup project root path ───────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from src import config as cfg


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="GNN Orchestrator — Fraud Detection (SKELETON)"
    )
    parser.add_argument(
        "--mode", choices=["full", "debug"], default="debug",
    )
    parser.add_argument(
        "--model", choices=["gat", "gcn"], default="gat",
        help="Kiến trúc GNN sử dụng. Mặc định: GAT."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print(f"  GNN Pipeline [{args.model.upper()}]  |  mode={args.mode}")
    print("  TRẠNG THÁI: SKELETON — Implement ở Tuần 6-7")
    print("=" * 60)

    # TODO [1]: Load & merge data
    # df, y = feature_selection.load_and_merge(cfg.DATA_DIR, ...)

    # TODO [2]: Load top50 từ cache (GNN dùng cùng feature set với MLP/LGBM)
    # Bắt buộc phải có cache — chạy run_lgbm.py hoặc run_mlp.py trước.
    # top50_names = _load_top50_from_cache(cfg.PROCESSED_DIR)

    # TODO [3]: Pipeline B — node features
    # from src.preprocessing.pipeline_b_mlp import MLPPipelinePreprocessor
    # preprocessor = MLPPipelinePreprocessor()
    # X_nodes = preprocessor.fit_transform(X_top50)

    # TODO [4]: Graph Construction (3-Tier Heuristic)
    # from src.graph.builder_3tier import GraphBuilder3Tier
    # builder = GraphBuilder3Tier(
    #     temporal_window_days=cfg.GRAPH_TEMPORAL_WINDOW_DAYS
    # )
    # edge_index = builder.build(df)

    # TODO [5]: Graph Validation
    # from src.graph.validator import GraphValidator
    # GraphValidator.validate(edge_index, max_degree=cfg.GRAPH_MAX_DEGREE)

    # TODO [6]: Train GNN
    # from src.models.gat_layer import GATFraudModel
    # model = GATFraudModel(
    #     in_channels=X_nodes.shape[1],
    #     hidden_channels=cfg.GNN_HIDDEN_DIM,
    #     heads=cfg.GNN_HEADS,
    #     dropout=cfg.GNN_DROPOUT,
    # )
    # model.fit(X_nodes, edge_index, y, epochs=cfg.GNN_EPOCHS, lr=cfg.GNN_LR)

    # TODO [7]: Evaluate + [8]: Save
    # y_prob = model.predict_proba(X_nodes_val)
    # np.save(os.path.join(cfg.PROCESSED_DIR, "gnn_proba.npy"), y_prob)

    print("\n[TODO] Implement các bước trên sau khi hoàn thành Tuần 6-7.")
    print(f"       Config đã sẵn sàng: GNN_HIDDEN_DIM={cfg.GNN_HIDDEN_DIM}, HEADS={cfg.GNN_HEADS}")


if __name__ == "__main__":
    main()
