#!/bin/bash
set -e

echo "=== Checking PyTorch ==="
python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null || {
    echo "Installing PyTorch..."
    pip install torch --quiet
}
pip install numpy matplotlib tqdm --quiet

echo ""
echo "=== Downloading enwik8 ==="
mkdir -p data
if [ ! -f data/enwik8 ]; then
    wget -q http://mattmahoney.net/dc/enwik8.zip -O data/enwik8.zip
    cd data && unzip -o enwik8.zip && rm enwik8.zip && cd ..
    echo "Downloaded: $(wc -c < data/enwik8) bytes"
else
    echo "Already exists: $(wc -c < data/enwik8) bytes"
fi

echo ""
echo "=== Preparing splits ==="
python utils/data.py --prepare

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
echo "  Train:  python phase1/train.py --config baseline"
echo "  Resume: python phase1/train.py --config baseline --resume"
