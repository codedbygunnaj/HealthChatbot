import pandas as pd

# Read both files
train = pd.read_csv('Training.csv')
test = pd.read_csv('Testing.csv')

# Fix the disease names in Training.csv
train.iloc[:, -1] = train.iloc[:, -1].replace({
    'Peptic ulcer diseae': 'Peptic ulcer disease',
    'Paroymsal  Positional Vertigo': '(vertigo) Paroxysmal Positional Vertigo',
    '(vertigo) Paroymsal  Positional Vertigo': '(vertigo) Paroxysmal Positional Vertigo',
    'Osteoarthristis': 'Osteoarthritis',
    'Diabetes ': 'Diabetes',
    'Hypertension ': 'Hypertension'
})

# Fix the disease names in Testing.csv
test.iloc[:, -1] = test.iloc[:, -1].replace({
    'Diabetes ': 'Diabetes',
    'Hypertension ': 'Hypertension'
})

# Save the corrected files
train.to_csv('Training.csv', index=False)
test.to_csv('Testing.csv', index=False)

print("✅ Fixed Training.csv and Testing.csv")
EOF
