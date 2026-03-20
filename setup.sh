#!/bin/bash
set -e

echo "=== Checking PyTorch ==="
python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null || {
    echo "Installing PyTorch..."
    pip install torch --quiet
}
pip install numpy matplotlib tqdm sentencepiece datasets --quiet

echo ""
echo "=== Preparing WikiText-103 BPE splits ==="
python utils/data.py --prepare-wikitext103

echo ""
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)')
print(f'SDPA: {hasattr(torch.nn.functional, \"scaled_dot_product_attention\")}')
"

echo ""
echo "=== Ready ==="
echo "  Train:   bash run_experiments.sh"
echo "  Resume:  python phase1/train.py --config baseline --resume"
