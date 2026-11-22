# Dataset instructions

`news.csv` is a small illustrative sample included only to verify that the
project runs. Replace it with a sufficiently large, reliable labeled dataset
before reporting model performance.

Required CSV columns:
- `text`
- `label` (`fake` or `true`)

To combine separate `Fake.csv` and `True.csv` files:

```python
import pandas as pd

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
fake["label"] = "fake"
true["label"] = "true"
combined = pd.concat([fake, true], ignore_index=True)
combined = combined[["text", "label"]].dropna()
combined.to_csv("news.csv", index=False)
```

If the source uses different column names, rename them before combining.
