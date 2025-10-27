# SentimentScope: Sentiment Analysis using Transformers

This project fine-tunes a minimalist transformer classifier ("DemoGPT") for binary sentiment analysis on the IMDB `aclImdb` dataset. The workflow is implemented in the notebook `SentimentScope_starter_final.ipynb`.

- **Notebook**: `SentimentScope_starter_final.ipynb`
- **Dataset**: IMDB reviews (`aclImdb`)
- **Goal**: Achieve >75% test accuracy
- **Your result**: Val 77.68%, Test 75.46%

## Project structure
- `SentimentScope_starter_final.ipynb` — complete end-to-end workflow
- `aclImdb/` — dataset folder (not tracked here; add locally or via Kaggle inputs)

## Environment setup
You can run the notebook locally (CPU) or on Kaggle. Install dependencies inside the notebook or beforehand.

### Install (local)
```bash
pip install pandas transformers torch torchvision torchaudio matplotlib scikit-learn
```

### Install (in-notebook)
The notebook includes cells like:
```python
%pip install pandas transformers torch torchvision torchaudio
```
After installation, restart the kernel if imports fail.

## Data setup
Place or mount the IMDB dataset so it contains paths like:
```
aclImdb/
  train/pos  train/neg
  test/pos   test/neg
```

### Recommended portable paths (pathlib)
In the notebook, configure dataset paths using `pathlib` and relative locations:
```python
from pathlib import Path

# Example: dataset lives under project root
project_root = Path.cwd()
data_dir = project_root / 'aclImdb'

train_pos_path = str(data_dir / 'train' / 'pos')
train_neg_path = str(data_dir / 'train' / 'neg')
test_pos_path  = str(data_dir / 'test'  / 'pos')
test_neg_path  = str(data_dir / 'test'  / 'neg')
```

### Kaggle tip
If you added the dataset via Kaggle "Add data", the base will be under `/kaggle/input/<slug>/aclImdb`. Example:
```python
from pathlib import Path
base = Path('/kaggle/input/imdb-dataset-of-50k-movie-reviews/aclImdb')
```

## What the notebook covers
- Load, explore, and prepare the dataset into Pandas DataFrames
- Tokenize with `AutoTokenizer('bert-base-uncased')`
- Implement `IMDBDataset` and `DataLoader`
- Build a compact transformer (`DemoGPT`) for classification:
  - token + positional embeddings
  - transformer blocks
  - mean pooling
  - linear classifier head
- Train with `AdamW` + `CrossEntropyLoss`
- Evaluate via `calculate_accuracy()` on validation and test sets

## Quick start
1. Open `SentimentScope_starter_final.ipynb`.
2. Run the install cells if needed.
3. Set the dataset paths as shown above and verify they exist.
4. Run all cells.

## Reproducing reported metrics
Your run achieved:
- Validation accuracy: 77.68%
- Test accuracy: 75.46%

If you see lower accuracy, try:
- Increase epochs (e.g., 5–10)
- Increase model capacity (`d_embed`, `layers_num`) ensuring `heads_num * head_size = d_embed`
- Adjust `MAX_LENGTH` (e.g., 256) if memory permits
- Tune `dropout_rate` and learning rate

## Troubleshooting
- "PyTorch not installed" with `return_tensors='pt'` — install `torch torchvision torchaudio` and restart the kernel, or temporarily set `return_tensors='np'` for quick tokenizer tests.
- Path not found — print paths and ensure `train/pos`, `train/neg`, `test/pos`, `test/neg` exist.

## License
This project is for educational purposes. Please respect the IMDB dataset license and Kaggle terms if using Kaggle resources.
