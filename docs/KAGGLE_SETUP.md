# Kaggle GPU Training Setup

Train Symbolic Regression models on Kaggle GPU for 10-100x faster formula discovery.

## Quick Start

### 1. Connect Kaggle to GitHub

1. Go to [kaggle.com/account](https://www.kaggle.com/account)
2. Scroll to "GitHub" section
3. Click "Link to GitHub"
4. Authorize Kaggle

### 2. Create Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click "New Notebook"
3. **Enable GPU**:
   - Click Settings (right panel)
   - Accelerator → **GPU P100** or **GPU T4**
4. Upload `notebooks/kaggle_sr_training.ipynb`

### 3. Or Import from GitHub

1. Click "File" → "Import Notebook"
2. Select "GitHub"
3. Enter: `Aicua/Aicua_SR`
4. Select `notebooks/kaggle_sr_training.ipynb`

---

## Kaggle Free Tier

- **GPU Hours**: 30 hours/week
- **Session Limit**: 9 hours max per session
- **Storage**: 100GB workspace
- **Internet**: Enabled for packages

---

## Training Workflow

### Option A: Clone Repo in Notebook (Recommended)

```python
# In Kaggle notebook
!git clone https://github.com/Aicua/Aicua_SR.git
%cd Aicua_SR
!pip install pysr pandas numpy
!python scripts/generate_spline_dataset.py
# ... train models
```

### Option B: Upload Dataset Directly

1. Download `data/generated/*.csv` from GitHub
2. Upload to Kaggle Dataset
3. Reference in notebook: `/kaggle/input/your-dataset/`

---

## Export Trained Models

### Method 1: Download JSON

After training, notebook saves:
```
data/generated/sr_discovered_formulas.json
data/generated/petal_spline_formulas.py
```

Download and commit to GitHub:
```bash
git add data/generated/
git commit -m "Add SR formulas from Kaggle training"
git push
```

### Method 2: Push from Kaggle (Advanced)

Set up Git credentials in notebook:
```python
!git config user.name "Your Name"
!git config user.email "your@email.com"
# Use Personal Access Token for auth
!git remote set-url origin https://{TOKEN}@github.com/Aicua/Aicua_SR.git
!git push
```

---

## Training Parameters

### Fast Training (Demo)
```python
PySRRegressor(
    niterations=50,
    populations=10,
    population_size=30,
)
# ~5-10 minutes per target
```

### Production Training
```python
PySRRegressor(
    niterations=200,
    populations=30,
    population_size=100,
    maxsize=30,
)
# ~30-60 minutes per target (better formulas)
```

### Heavy Training
```python
PySRRegressor(
    niterations=500,
    populations=50,
    population_size=200,
    maxsize=40,
)
# ~2-4 hours per target (best formulas)
```

---

## Estimated Training Times

| Targets | Iterations | GPU Time | Quality |
|---------|-----------|----------|---------|
| 11 (petal) | 50 | ~1-2 hours | Good |
| 11 (petal) | 200 | ~5-8 hours | Better |
| 20+ (all) | 100 | ~8-12 hours | Full |

---

## Tips

1. **Save intermediate results**: Checkpoint models during long training
2. **Monitor GPU usage**: Kaggle shows remaining GPU hours
3. **Batch training**: Train multiple targets in one session
4. **Version control**: Tag each training run with date/config

---

## Troubleshooting

### "GPU not available"
- Check Settings → Accelerator → GPU
- GPU quota might be exhausted (wait for reset)

### "Session timeout"
- Max 9 hours per session
- Save results before timeout

### "Import error"
```python
!pip install pysr --upgrade
# Restart kernel if needed
```

---

## Integration with Aicua_SR

After training:
1. Download generated formulas
2. Place in `data/generated/`
3. Run `python scripts/generate_cot_cli.py`
4. CLI will use SR-discovered formulas automatically!

```bash
# Auto-uses trained formulas if available
python scripts/generate_spline_cli.py --size 2.0 --layers 3
```
