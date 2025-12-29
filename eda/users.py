import pandas as pd

# Cargar los datos
data_train = pd.read_csv('dataset/train.csv')
data_test = pd.read_csv('dataset/development.csv')


def analizar_consistencia(data, nombre_dataset):
    print(f"=== Análisis de consistencia ideológica por usuario - {nombre_dataset} ===\n")

    # Agrupar por usuario y contar sus diferentes ideologías
    user_ideology_counts = data.groupby('label')['ideology_multiclass'].value_counts().unstack(fill_value=0)

    # Añadir columna con el total de ejemplos por usuario
    user_ideology_counts['total_ejemplos'] = user_ideology_counts.sum(axis=1)

    # Añadir columna con la ideología más frecuente y su conteo
    user_ideology_counts['ideologia_dominante'] = user_ideology_counts.iloc[:, :-1].idxmax(axis=1)
    user_ideology_counts['conteo_dominante'] = user_ideology_counts.iloc[:, :-1].max(axis=1)

    # Calcular el porcentaje de consistencia
    user_ideology_counts['porcentaje_consistencia'] = (
            user_ideology_counts['conteo_dominante'] / user_ideology_counts['total_ejemplos'] * 100
    ).round(2)

    # Ordenar por total de ejemplos
    user_ideology_counts_sorted = user_ideology_counts.sort_values('total_ejemplos', ascending=False)

    # Mostrar resultados
    print("\nDetalle por usuario:")
    print(user_ideology_counts_sorted)

    # Estadísticas generales
    print("\n=== Estadísticas generales ===")
    print(f"Número total de usuarios: {len(user_ideology_counts)}")
    print(f"Promedio de consistencia: {user_ideology_counts['porcentaje_consistencia'].mean().round(2)}%")

    # Contar usuarios completamente consistentes (100%)
    usuarios_consistentes = len(user_ideology_counts[user_ideology_counts['porcentaje_consistencia'] == 100])
    print(f"Usuarios completamente consistentes: {usuarios_consistentes} "
          f"({round(usuarios_consistentes / len(user_ideology_counts) * 100, 2)}% del total)")

    # Distribución de consistencia
    print("\n=== Distribución de niveles de consistencia ===")
    bins = [0, 50, 75, 90, 100]
    labels = ['0-50%', '51-75%', '76-90%', '91-100%']
    consistency_distribution = pd.cut(user_ideology_counts['porcentaje_consistencia'],
                                      bins=bins, labels=labels).value_counts().sort_index()
    print(consistency_distribution)

    # Usuarios con más variabilidad
    print("\n=== Top 10 usuarios con mayor variabilidad ideológica ===")
    menos_consistentes = user_ideology_counts_sorted[
        user_ideology_counts_sorted['total_ejemplos'] >= 5
        ].nsmallest(10, 'porcentaje_consistencia')
    print(menos_consistentes[['total_ejemplos', 'ideologia_dominante', 'porcentaje_consistencia']])

    return set(user_ideology_counts.index)


# Ejecutar el análisis para ambos datasets
usuarios_train = analizar_consistencia(data_train, "Train")
usuarios_development = analizar_consistencia(data_test, "Development")

# Comparación entre usuarios de ambos datasets
print("\n=== Comparación de usuarios entre Train y Development ===")
usuarios_comunes = usuarios_train & usuarios_development
solo_en_train = usuarios_train - usuarios_development
solo_en_development = usuarios_development - usuarios_train

print(f"Usuarios comunes: {len(usuarios_comunes)}")
print(f"Usuarios solo en Train: {len(solo_en_train)}")
print(f"Usuarios solo en Development: {len(solo_en_development)}")

if solo_en_train:
    print("\nUsuarios presentes solo en Train:")
    print(solo_en_train)

if solo_en_development:
    print("\nUsuarios presentes solo en Development:")
    print(solo_en_development)
