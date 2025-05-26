# Robust VLM

### Train
- Train all LoRA adapters together
    ```bash
    python train.py
    ```
- Train LoRA step by step
    ```bash
    python train_steps.py
    ```

- Train LoRA with descriptional similarity
    ```bash
    python train_desc.py
    ```

- Train LoRA with descriptional similarity (multiple GPU)
    ```bash
    python train_desc_multi.py
    ```

### Evaluate
```bash
python eval.py --save_dir "./experiments/models/CLIP@LoRA@q_v@r4"
```