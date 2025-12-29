import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def explore_dataset(csv_file, bias_column='bias', target_column='target', plot_corr=True):
    """
    Explores and displays relevant information about a CSV dataset.

    Parameters:
      csv_file (str): Path to the CSV file.
      bias_column (str): Name of the bias column (if exists).
      target_column (str): Name of the target variable.
      plot_corr (bool): If True, generates a correlation matrix plot.

    The function displays:
    - DataFrame dimensions
    - Variable types and number of missing values
    - Descriptive statistics
    - Distribution of the bias column (if exists)
    - Distribution of the target variable
    - Cross-tabulation with total percentages
    - Correlation matrix
    """
    # Load the dataset
    df = pd.read_csv(csv_file)
    print("== Basic Information ==")
    print(f"Dimensions: {df.shape[0]} rows, {df.shape[1]} columns\n")

    # Data types and missing values
    print("== Data Types and Missing Values ==")
    missing_summary = df.isnull().sum().sort_values(ascending=False)
    info = pd.DataFrame({'Type': df.dtypes, 'Missing': missing_summary})
    print(info[info['Missing'] > 0])
    print("\n")

    # Numerical descriptive statistics
    print("== Numerical Descriptive Statistics ==")
    print(df.describe())
    print("\n")

    # Categorical descriptive statistics
    try:
        cat_desc = df.describe(include=['object'])
        if cat_desc.empty:
            print("== No categorical columns to describe ==")
        else:
            print("== Categorical Descriptive Statistics ==")
            print(cat_desc)
    except ValueError:
        print("== No categorical columns to describe ==")
    print("\n")

    # Distributions
    if target_column in df.columns:
        print(f"== Distribution of '{target_column}' ==")
        print(df[target_column].value_counts(dropna=False))
        print("\n")
    if bias_column in df.columns:
        print(f"== Distribution of '{bias_column}' ==")
        print(df[bias_column].value_counts(dropna=False))
        print("\n")

    # Cross-tabulation and percentages
    if bias_column in df.columns and target_column in df.columns:
        print(f"== Cross-tabulation: '{bias_column}' vs '{target_column}' ==")
        ct = pd.crosstab(df[bias_column], df[target_column], margins=True)
        print(ct)
        total = ct.loc['All', 'All']
        pct = (ct / total * 100).round(2)
        print("\n== Percentages of total (%) ==")
        print(pct)
        print("\n")

    # Correlation matrix
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        corr = df[numeric_cols].corr()
        print("== Correlation Matrix (numerical variables) ==")
        print(corr)
        print("\n")
        # Target correlations if numeric
        if target_column in numeric_cols:
            target_corr = corr[target_column].sort_values(ascending=False)
            print(f"== Correlation of '{target_column}' with other variables ==")
            print(target_corr)
            print("\n")
    else:
        print("No numerical variables to calculate correlations.\n")

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    if target_column in df.columns:
        sns.countplot(x=target_column, data=df, ax=axes[0])
        axes[0].set_title(f"Distribution of {target_column}")
    if bias_column in df.columns:
        sns.countplot(x=bias_column, data=df, ax=axes[1])
        axes[1].set_title(f"Distribution of {bias_column}")
    if len(numeric_cols) > 0:
        col = numeric_cols[0]
        sns.histplot(df[col].dropna(), kde=True, ax=axes[2])
        axes[2].set_title(f"Histogram of {col}")
    if (target_column in df.columns) and (bias_column in df.columns):
        sns.boxplot(x=bias_column, y=target_column, data=df, ax=axes[3])
        axes[3].set_title(f"{target_column} by {bias_column}")

    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    if plot_corr and len(numeric_cols) > 0:
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
        plt.title("Numerical Variables Correlation Matrix")
        plt.show()


def convert_to_numeric(df, one_hot=False):
    """
    Converts all columns in a DataFrame to numeric values.
    - Replaces range strings like "[6-8]" with their mean.
    - Encodes categorical variables using LabelEncoder or One-Hot Encoding.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - one_hot (bool): If True, uses One-Hot Encoding; otherwise uses Label Encoding.

    Returns:
    - pd.DataFrame: A fully numeric DataFrame.
    """
    df = df.copy()

    # Step 1: Convert ranges like "[6-8]" to their mean (e.g., 7.0)
    def range_to_mean(x):
        if isinstance(x, str) and re.match(r"\[\d+-\d+\]", x):
            a, b = map(int, x[1:-1].split("-"))
            return (a + b) / 2
        return x

    df = df.applymap(range_to_mean)

    # Step 2: Identify categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    if one_hot:
        # Apply One-Hot Encoding
        df = pd.get_dummies(df, columns=cat_cols)
    else:
        # Apply Label Encoding
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Step 3: Ensure all remaining columns are numeric
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            # If still not numeric, apply Label Encoding as fallback
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return df


def create_csv(data, file_name):
    data = convert_to_numeric(data)

    data.drop(columns=["id", "tweet"], inplace=True)
    data_wo_ideology_binary = data.drop(columns=["ideology_binary"])
    data_wo_label_ib = data_wo_ideology_binary.drop(columns=["label"])
    data_wo_label = data.drop(columns=["label"])

    data.to_csv(f'dataset/{file_name}_numeric.csv', index=False)
    data_wo_ideology_binary.to_csv(f'dataset/{file_name}_wo_ideology_binary_numeric.csv', index=False)
    data_wo_label_ib.to_csv(f'dataset/{file_name}_wo_ideology_binary_label_numeric.csv', index=False)
    data_wo_label.to_csv(f'dataset/{file_name}_wo_label_numeric.csv', index=False)

def tweet_token_stats_by_ideology(df):
    """
    Calcula estadísticas de longitud de tweets por clase de 'ideology_multiclass'.

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene las columnas 'tweet' e 'ideology_multiclass'.

    Retorna:
        pd.DataFrame: Tabla con estadísticas por clase ideológica.
    """
    if 'tweet' not in df.columns or 'ideology_multiclass' not in df.columns:
        raise ValueError("El DataFrame debe contener las columnas 'tweet' e 'ideology_multiclass'.")

    # Calcular la longitud de cada tweet en número de tokens
    df = df.copy()
    df['n_tokens'] = df['tweet'].astype(str).apply(lambda x: len(x.split()))

    # Agrupar por clase ideológica y calcular estadísticas
    stats = df.groupby('ideology_multiclass')['n_tokens'].agg(
        media='mean',
        desviación='std',
        mínimo='min',
        máximo='max'
    ).reset_index()

    # Redondear los valores a dos decimales
    stats = stats.round(2)

    return stats


def top_n_words_by_class(df: pd.DataFrame,
                              tweet_col: str = 'tweet',
                              class_col: str = 'ideology_multiclass',
                              n: int = 10,
                              lang: str = 'spanish',
                              extra_stopwords: set = None) -> pd.DataFrame:
    """
    Devuelve un DataFrame ancho con las n palabras más frecuentes
    de cada clase ideológica. Cada columna es una clase, y cada fila
    i contiene la i-ésima palabra más frecuente (con su count) para esa clase.

    Parámetros:
    - df: DataFrame con columnas tweet_col y class_col.
    - tweet_col: nombre de la columna con el texto del tweet.
    - class_col: nombre de la columna con la clase ideológica.
    - n: número de palabras top a extraer.
    - lang: idioma para las stopwords de NLTK.
    - extra_stopwords: conjunto de tokens extra a añadir como stopwords.
    """
    if tweet_col not in df or class_col not in df:
        raise ValueError(f"El DataFrame debe contener '{tweet_col}' y '{class_col}'.")

    # Stopwords de NLTK para el idioma, más extras
    base_sw = set(stopwords.words(lang))
    extra_sw = extra_stopwords or {'q', 'si'}
    sw = base_sw.union(extra_sw)

    result = {}

    for cls, group in df.groupby(class_col):
        text = " ".join(group[tweet_col].astype(str)).lower()
        tokens = word_tokenize(text)
        filtered = [t for t in tokens if t.isalpha() and t not in sw]
        counts = Counter(filtered).most_common(n)
        result[cls] = [f"{word} ({cnt})" for word, cnt in counts]

    # Rellenar hasta n elementos si alguna clase tiene menos palabras
    for cls, lst in result.items():
        if len(lst) < n:
            result[cls] = lst + [None] * (n - len(lst))

    return pd.DataFrame(result, index=[f"Top {i+1}" for i in range(n)])



train = pd.read_csv(os.path.join("dataset", "train.csv"))
test = pd.read_csv(os.path.join("dataset", "development.csv"))

create_csv(train, "train")
create_csv(test, "test")

#explore_dataset(os.path.join("dataset", "test_numeric.csv"), "profession", "ideology_multiclass")
#print("Número de usuarios únicos en el dataset de train", len(train['label'].unique()))
print("  == Cantidad de tokens en los tweets por ideología ==")
print(tweet_token_stats_by_ideology(train))

print()

print("              == Palabras más usadas por clase ==")
print(top_n_words_by_class(train, n=10, lang='spanish'))