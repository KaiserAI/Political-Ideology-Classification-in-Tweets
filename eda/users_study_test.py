import pandas as pd

# Cargar datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('testCleaned.csv')

# Estadísticas base
total_train = len(train_df)
total_test = len(test_df)

# 1. Usuarios de test presentes en train por label
common_labels = test_df[test_df['label'].isin(train_df['label'])]
n_common = len(common_labels)
pct_common = (n_common / total_test) * 100

# 2. De los comunes, cuántos tienen misma combinación gender-profession
train_combinations = set(zip(train_df['gender'], train_df['profession']))
common_both = common_labels[common_labels.apply(lambda x: (x['gender'], x['profession']) in train_combinations, axis=1)]
n_common_both = len(common_both)
pct_common_both = (n_common_both / n_common * 100) if n_common > 0 else 0

# Mostrar resultados
print(f"\n→ {n_common} de {total_test} usuarios en Test están en Train ({pct_common:.1f}%)")
print(f"→ De esos, {n_common_both} tienen misma combinación gender+profession ({pct_common_both:.1f}%)\n")
print(f"Detalle totales:")
print(f"- Train: {total_train} usuarios")
print(f"- Test:  {total_test} usuarios")