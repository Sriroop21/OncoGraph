import pandas as pd
import json

print('='*70)
print('MODEL ACCURACY - OBJECTIVE 1')
print('='*70)

# From classification report
with open('backend/results/test_classification_report.json', 'r') as f:
    report = json.load(f)

print(f'\nOVERALL ACCURACY: {report["accuracy"]*100:.2f}%')
print(f'\nMacro Precision:  {report["macro avg"]["precision"]*100:.2f}%')
print(f'Macro Recall:     {report["macro avg"]["recall"]*100:.2f}%')
print(f'Macro F1-Score:   {report["macro avg"]["f1-score"]*100:.2f}%')

# From per-class metrics
df = pd.read_csv('backend/results/test_per_class_metrics.csv')

# Show first few rows to see column names
print('\nPer-Cancer Performance (Top 10):')
print(df.head(10).to_string(index=False))

print('='*70)
