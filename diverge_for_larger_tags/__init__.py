"""
Two-stage DIVERGE for large TAGs (computer, arxiv, ...), where holding all 5 LoRA-init
embedding caches on the GPU at once exceeds VRAM:

  1) train_gnn_export_logits.py  one init approach at a time: load its embeddings,
                                 train the GNN, save the best-epoch logits (+ weights),
                                 free memory.
  2) diverge_from_logits.py      DIVERGE ensembles from the saved logits.
     ablation_from_logits.py     M-ablation from the same saved logits.

Driver: diverge_for_larger_tags/run_diverge_large_tags.bat
"""
