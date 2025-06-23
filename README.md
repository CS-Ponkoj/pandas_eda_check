# pandas_eda_check

A lightweight Python utility for quick dataset interpretation using `pandas`.

## 🔍 What it does

Provides a quick summary of:
- Unique values per column
- Missing values (count and %)
- Total data shape

Ideal for exploratory data analysis (EDA) and dataset cleaning.

## 📦 Installation

```bash
pip install pandas_eda_check
```

## 💻 Usage

```python
from pandas_eda_check import check
import pandas as pd

df = pd.read_csv('your_dataset.csv')
summary = check(df)
print(summary)
```

## 👤 Author

Developed by [Ponkoj Shill](https://github.com/CS-Ponkoj)  
Email: csponkoj@gmail.com

## 📝 License

MIT License