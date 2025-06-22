import vaex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.cluster import (MiniBatchKMeans,KMeans,DBSCAN,OPTICS,AgglomerativeClustering
)
from sklearn.decomposition import PCA
from kmodes.kmodes import KModes
from umap import UMAP


FILE_PATH = "processed_data.hdf5"
N_ROWS = 10000
GROUP_CODES = ['A', 'B', 'C', 'D', 'E', 'F']
CLUSTER_METHOD = 'MiniBatchKMeans'  # 'MiniBatchKMeans', 'KMeans', 'KModes', 'DBSCAN', 'OPTICS', 'Agglomerative'
REDUCTION_METHOD = 'UMAP'  # 'PCA', 'UMAP'
N_CLUSTERS = 8

df_vaex = vaex.open(FILE_PATH)
df = df_vaex.head(N_ROWS).to_pandas_df()


for col in ['revenue', 'expenditure', 'employees_count']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['profitability'] = df['revenue'] / df['expenditure']
df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
df['lifetime_days'] = (df['end_date'] - df['start_date']).dt.days
df['lifetime_days'] = df['lifetime_days'].fillna((pd.Timestamp.now() - df['start_date']).dt.days)


def encode_categoricals(df_subset):
    df_subset = df_subset.copy()
    cat_cols = df_subset.select_dtypes(include=['object', 'category']).columns
    df_cat = df_subset[cat_cols].fillna("unknown").astype(str)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df_cat)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df_cat.index)
    df_subset = df_subset.drop(columns=cat_cols).join(encoded_df)
    return df_subset


def prepare_group(df, group):
    if group == 'A':
        df_sub = df[['kind', 'category', 'year', 'start_date', 'end_date']].copy()
        df_sub['start_year'] = df_sub['start_date'].dt.year
        df_sub['end_year'] = df_sub['end_date'].dt.year
        df_sub = df_sub.drop(columns=['start_date', 'end_date'])
    elif group == 'B':
        df_sub = df[['region', 'region_code', 'settlement_type', 'lat', 'lon']].copy()
    elif group == 'C':
        df_sub = df[['activity_code_main', 'category', 'region']].copy()
    elif group == 'D':
        df_sub = df[['revenue', 'expenditure', 'profitability']].copy()
    elif group == 'E':
        df_sub = df[['employees_count', 'category', 'region', 'activity_code_main']].copy()
    elif group == 'F':
        df_sub = df[['year', 'start_date', 'end_date', 'lifetime_days']].copy()
        df_sub['start_month'] = df_sub['start_date'].dt.month
        df_sub['end_month'] = df_sub['end_date'].dt.month
        df_sub = df_sub.drop(columns=['start_date', 'end_date'])
    else:
        raise ValueError("Что-то не то")

    if group != 'D':
        df_sub = encode_categoricals(df_sub)

    df_sub = df_sub.replace([np.inf, -np.inf], np.nan).fillna(df_sub.median(numeric_only=True))
    return df_sub


def cluster_data(X, method, n_clusters=8):
    if method == 'MiniBatchKMeans':
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X)
    elif method == 'KMeans':
        model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        labels = model.fit_predict(X)
    elif method == 'KModes':
        model = KModes(n_clusters=n_clusters, init='Huang', n_init=5, verbose=0)
        labels = model.fit_predict(X)
    elif method == 'DBSCAN':
        model = DBSCAN(eps=1.5, min_samples=5)
        labels = model.fit_predict(X)
    elif method == 'OPTICS':
        model = OPTICS(min_samples=5)
        labels = model.fit_predict(X)
    elif method == 'Agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)
    else:
        raise ValueError("Неизвестный метод кластеризации")
    return labels


def evaluate_clustering(X, labels):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    if n_clusters < 2:
        print("Недостаточно кластеров для расчёта метрик.")
        return

    silhouette = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_score = davies_bouldin_score(X, labels)

    print(f"Кол-во кластеров: {n_clusters}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Calinski-Harabasz Index: {ch_score:.2f}")
    print(f"Davies-Bouldin Index: {db_score:.4f}")


def visualize_clusters(X_scaled, labels, group_name, reduction='PCA'):
    if reduction == 'PCA':
        reducer = PCA(n_components=2)
    elif reduction == 'UMAP':
        reducer = UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Метод снижения размерности не поддерживается")

    X_2d = reducer.fit_transform(X_scaled)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    plt.colorbar(scatter, label='Кластер')
    plt.title(f"Группа {group_name} — кластеризация ({reduction})")
    plt.xlabel(f"{reduction} 1")
    plt.ylabel(f"{reduction} 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def correlation_analysis(df):
    print("\nКорреляционный анализ:")

    #Зависимость между выручкой и числом сотрудников
    corr_revenue_employees = df[['revenue', 'employees_count']].corr().loc['revenue', 'employees_count']
    print(f"Корреляция между выручкой и числом сотрудников: {corr_revenue_employees:.2f}")

    #Зависимость между регионом и финансовыми показателями
    region_financials = df.groupby('region')[['revenue', 'expenditure']].median().reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(x='region', y='revenue', data=region_financials, order=region_financials.sort_values(by='revenue', ascending=False)['region'].unique(), palette='viridis')
    plt.xticks(rotation=90)
    plt.title('Средняя выручка по регионам')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='region', y='expenditure', data=region_financials, order=region_financials.sort_values(by='expenditure', ascending=False)['region'].unique(), palette='viridis')
    plt.xticks(rotation=90)
    plt.title('Средние затраты по регионам')
    plt.tight_layout()
    plt.show()

    #Зависимость между ОКВЭД и рентабельностью
    okved_profitability = df.groupby('activity_code_main')['profitability'].median().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=okved_profitability.values, y=okved_profitability.index, palette='viridis')
    plt.title('Топ 10 ОКВЭД по рентабельности')
    plt.xlabel('Медианная рентабельность')
    plt.ylabel('ОКВЭД')
    plt.tight_layout()
    plt.show()


results = {}

for code in GROUP_CODES:
    print(f"\nОбработка группы {code}...")
    group_df = prepare_group(df, code)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(group_df)

    print(f"Кластеризация методом {CLUSTER_METHOD}...")
    labels = cluster_data(X_scaled, CLUSTER_METHOD, n_clusters=N_CLUSTERS)
    results[code] = labels

    print(f"Метрики качества кластеризации для группы {code}:")
    evaluate_clustering(X_scaled, labels)

    print(f"Визуализация через {REDUCTION_METHOD}...")
    visualize_clusters(X_scaled, labels, code, reduction=REDUCTION_METHOD)


print("\nПроведение корреляционного анализа...")
correlation_analysis(df)

def plot_correlation_heatmap(df):
    print("\nПостроение тепловой карты корреляций...")


    numeric_cols = ['revenue','expenditure','employees_count','profitability','lifetime_days'
    ]

    available_cols = [col for col in numeric_cols if col in df.columns]

    if len(available_cols) < 2:
        print("Недостаточно числовых признаков для построения тепловой карты.")
        return


    df_corr = df[available_cols].copy()


    correlation_matrix = df_corr.corr(method='pearson')


    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt='.2f',
        linewidths=0.5,
        linecolor='lightgray'
    )
    plt.title('Тепловая карта корреляций')
    plt.tight_layout()
    plt.show()

plot_correlation_heatmap(df)