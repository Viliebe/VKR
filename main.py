import pandas as pd
import vaex
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn.cluster import (KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, OPTICS)
from kmodes.kmodes import KModes
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, OrdinalEncoder)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, davies_bouldin_score)
import hdbscan
warnings.filterwarnings("ignore")




# # ОТКРЫТЬ ФАЙЛ HDF5
#file_path = 'panel.hdf5'
#file_path = 'results/result.hdf5'
file_path = 'processed_data.hdf5'
df = vaex.open(file_path)
df_pd = df.head(100000).to_pandas_df()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 0)
print(df_pd)



# file_path = 'processed_data.hdf5'
# file_path = 'processed_encoded.hdf5'
# df = vaex.open(file_path)
#
# batch_size = 1000000  # размер пакета
# for i in range(0, len(df), batch_size):
#     df_batch = df[i:i + batch_size].to_pandas_df()
#
#     # здесь можно выполнять обработку текущего пакета
#     print(f"Обработаны строки {i}–{i + batch_size}")


 # file_path = 'panel.hdf5'
# file_path = 'results/result.hdf5'
# df = vaex.open(file_path)
#
# batch_size = 1000000
# total_rows = len(df)
#
# total_missing = None
# numeric_cols = None
#
# for i in range(0, total_rows, batch_size):
#     print(f"\nОбрабатывается пакет {i}–{min(i + batch_size, total_rows)}")
#

#     df_batch = df[i:i + batch_size].to_pandas_df()
#

#     missing = df_batch.isnull().sum()
#     if total_missing is None:
#         total_missing = missing
#     else:
#         total_missing += missing
#

#     if numeric_cols is None:
#         numeric_cols = df_batch.select_dtypes(include=['number']).columns
#

#     df_cleaned = df_batch.dropna()
#     print(f"Удалено строк в этом пакете: {len(df_batch) - len(df_cleaned)}")
#

#     df_filled_median = df_batch.copy()
#     df_filled_median[numeric_cols] = df_filled_median[numeric_cols].fillna(df_filled_median[numeric_cols].median())
#

#     df_filled_zero = df_batch.fillna(0)
#

#     df_interpolated = df_batch.copy()
#     df_interpolated[numeric_cols] = df_interpolated[numeric_cols].interpolate(method='linear')
#

#     df_filled_median.to_csv('processed_data.csv', mode='a', index=False, header=(i == 0))
#
# print("\nОбщее количество пропущенных значений:")
# print(total_missing)


#ПРОСМОТР ДАННЫХ

# print("Первые строки данных:")
# print(df_pd.head(10))
#

# print("\nИнформация о данных:")
# print(df_pd.info())
#
# print("\nОписательная статистика:")
# print(df_pd.describe())



# АНАЛИЗ РАСПРЕДЕЛЕНИЙ
# numeric_cols = df_pd.select_dtypes(include=['number']).columns
#
# for col in numeric_cols:
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     sns.histplot(df_pd[col], kde=True)
#     plt.title(f'Гистограмма {col}')
#
#     plt.subplot(1, 2, 2)
#     sns.boxplot(x=df_pd[col])
#     plt.title(f'Boxplot {col}')
#
#     plt.tight_layout()
#     plt.show()
#
# if len(numeric_cols) >= 2:
#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(x=df_pd[numeric_cols[0]], y=df_pd[numeric_cols[1]])
#     plt.title(f'Scatter plot: {numeric_cols[0]} vs {numeric_cols[1]}')
#     plt.show()



# КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
# corr_matrix = df_pd.corr(numeric_only=True)
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Корреляционная матрица')
# plt.show()



# # ОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ
# print("\nПроверка пропусков:")
# print(df_pd.isnull().sum())
#
# df_cleaned = df_pd.dropna()
# print("\nКоличество удалённых строк из-за пропусков:", len(df_pd) - len(df_cleaned))
#
# df_filled_median = df_pd.copy()
# numeric_cols = df_filled_median.select_dtypes(include=['number']).columns
# df_filled_median[numeric_cols] = df_filled_median[numeric_cols].fillna(df_filled_median[numeric_cols].median())
# print("\nПример заполнения пропусков медианой:")
# print(df_filled_median.head())
#
# df_filled_zero = df_pd.fillna(0)
# print("\nПример заполнения пропусков нулём:")
# print(df_filled_zero.head())
#
# df_interpolated = df_pd.copy()
# df_interpolated[numeric_cols] = df_interpolated[numeric_cols].interpolate(method='linear')
# print(df_interpolated.head())



# ОБРАБОТКА ВЫБРОСОВ
# numeric_cols = df_pd.select_dtypes(include=['number']).columns
#
# for col in numeric_cols:
#     plt.figure(figsize=(8, 4))
#     sns.boxplot(x=df_pd[col])
#     plt.title(f'Boxplot для {col}')
#     plt.show()

# def remove_outliers_iqr(df, column):
#     Q1 = df[column].quantile(0.25)
#     Q3 = df[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
#
#
# df_no_outliers = df_pd.copy()
# for col in numeric_cols:
#     df_no_outliers = remove_outliers_iqr(df_no_outliers, col)
#
# print(f"\nКоличество строк до удаления выбросов: {len(df_pd)}")
# print(f"Количество строк после удаления выбросов: {len(df_no_outliers)}")
#

# def cap_outliers_iqr(df, column):
#     Q1 = df[column].quantile(0.25)
#     Q3 = df[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
#     return df
#
#
# df_capped = df_pd.copy()
# for col in numeric_cols:
#     df_capped = cap_outliers_iqr(df_capped, col)
#
# print("\nПример замены выбросов на границы IQR:")
# print(df_capped.head())

# for col in numeric_cols:
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     sns.histplot(df_pd[col], kde=True)
#     plt.title(f'{col} (до обработки выбросов)')
#
#     plt.subplot(1, 2, 2)
#     sns.histplot(df_no_outliers[col], kde=True)
#     plt.title(f'{col} (после удаления выбросов)')
#     plt.tight_layout()
#     plt.show()





# КОДИРОВАНИЕ КАТЕГОРИАЛЬНХ ДАННЫХ
# df_encoded = pd.get_dummies(df_pd, drop_first=True)
# print("\nПример закодированных данных:")
# print(df_encoded.head())

# from sklearn.preprocessing import MinMaxScaler
#
# numeric_cols = df_encoded.select_dtypes(include=['number']).columns
# minmax_scaler = MinMaxScaler()
# df_encoded[numeric_cols] = minmax_scaler.fit_transform(df_encoded[numeric_cols])
# print("\nПример отмасштабированных данных:")
# print(df_encoded.head())

# try:
#     from imblearn.over_sampling import SMOTE
#     from sklearn.model_selection import train_test_split
#
#     # Замените 'target' на имя реальной целевой переменной
#     X = df_encoded.drop('target', axis=1)
#     y = df_encoded['target']
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#





# file_path = 'processed_data.hdf5'
# df = vaex.open(file_path)

# df_pd = df.head(100).to_pandas_df()

# columns_to_drop = [
#     "tin", "year", "reg_number", "kind", "category",
#     "org_name", "org_short_name", "activity_code_main",
#     "region_iso_code", "region_code", "area", "settlement",
#     "oktmo", "lat", "lon", "start_date", "end_date"
# ]

# df_pd.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')
#

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.width', 0)
#
# print("Первые 100 строк после удаления столбцов:")
# print(df_pd)







# INPUT_FILE = 'processed_data.hdf5'
# OUTPUT_FILE = 'processed_encoded111.hdf5'
# CHUNK_SIZE = 500000

# NUMERIC_FEATURES = ['revenue', 'expenditure', 'employees_count']
# CATEGORICAL_FEATURES = ['region', 'settlement_type']
# # COLUMNS_TO_DROP = [
# #     "tin", "year", "reg_number", "kind", "category",
# #     "org_name", "org_short_name", "activity_code_main",
# #     "region_iso_code", "region_code", "area", "settlement",
# #     "oktmo", "lat", "lon", "start_date", "end_date"
# # ]
#
#
# def process_chunk(df_chunk: vaex.dataframe.DataFrame) -> pd.DataFrame:
#     df_pd = df_chunk.to_pandas_df()
#

#     # df_pd.drop(columns=COLUMNS_TO_DROP, inplace=True, errors='ignore')
#

#     available_columns = df_pd.columns.tolist()
#     kept_columns = [col for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES if col in available_columns]
#     df_pd = df_pd[kept_columns]
#
#

#     for col in NUMERIC_FEATURES:
#         if col in df_pd.columns:
#             df_pd[col] = pd.to_numeric(df_pd[col], errors='coerce')  # ← вот ключ
#             median = df_pd[col].median()
#             df_pd[col] = df_pd[col].fillna(median if pd.notna(median) else 0)
#

#     for col in CATEGORICAL_FEATURES:
#         if col in df_pd.columns:
#             df_pd[col] = df_pd[col].fillna('unknown').astype(str)
#

#     for col in CATEGORICAL_FEATURES:
#         if col in df_pd.columns:
#             top_values = df_pd[col].value_counts().nlargest(10).index
#             for val in top_values:
#                 safe_name = f"{col}_{str(val).replace(' ', '_').replace('/', '_')}"
#                 df_pd[safe_name] = (df_pd[col] == val).astype(int)
#             df_pd.drop(columns=[col], inplace=True)
#
#     return df_pd
# #
#
# def batch_process_vaex(input_path, output_path, chunk_size=CHUNK_SIZE):
#     df = vaex.open(input_path)
#     total_rows = len(df)
#     print(f"Всего строк в датасете: {total_rows:,}")
#
#     processed_batches = []
#
#     for start in range(0, total_rows, chunk_size):
#         end = min(start + chunk_size, total_rows)
#         print(f"\nОбработка строк {start:,} — {end:,}...")
#         df_chunk = df[start:end]
#         df_processed_pd = process_chunk(df_chunk)
#         df_processed_vaex = vaex.from_pandas(df_processed_pd)
#         processed_batches.append(df_processed_vaex)
#

#     df_final = vaex.concat(processed_batches)
#     df_final.export_hdf5(output_path, progress=True)
#     print(f"\n Обработанный датасет сохранён в: {output_path}")




# PCA

# def run_clustering_with_metrics(input_path, k=4, sample_size=100000):
#     print(f"\n Загрузка обработанных данных из {input_path}")
#     df = vaex.open(input_path)
#

#     df_sample = df.sample(n=sample_size)
#     df_pd = df_sample.to_pandas_df()
#

#     numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
#     X = df_pd[numeric_cols].fillna(0).values
#

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     print(f"\n Запуск MiniBatchKMeans (k={k})...")
#     kmeans = MiniBatchKMeans(n_clusters=k, batch_size=10000, max_iter=100, random_state=42)
#     kmeans.fit(X_scaled)
#     labels = kmeans.labels_
#

#     print(f"\n Метрики кластеризации:")
#     print(f"- Inertia: {kmeans.inertia_:.2f}")
#     print(f"- Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
#     print(f"- Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, labels):.2f}")
#     print(f"- Davies-Bouldin Score: {davies_bouldin_score(X_scaled, labels):.4f}")
#

#     print("\n Визуализация кластеров через PCA...")
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X_scaled)
#
#     plt.figure(figsize=(10, 7))
#     scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6, s=10)
#     plt.colorbar(scatter, label="Кластер")
#     plt.title(f"Кластеры по PCA ({sample_size:,} строк)")
#     plt.xlabel("Главная компонента 1")
#     plt.ylabel("Главная компонента 2")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
#
# def run_kmeans_greedypp_with_metrics(input_path, k=4, sample_size=100000):
#     print(f"\n Загрузка обработанных данных из {input_path}")
#     df = vaex.open(input_path)
#

#     df_sample = df.sample(n=sample_size)
#     df_pd = df_sample.to_pandas_df()
#

#     numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
#     X = df_pd[numeric_cols].fillna(0).values
#

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     print(f"\n Запуск KMeans (инициализация: k-means++) с k={k}...")
#     kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
#     kmeans.fit(X_scaled)
#     labels = kmeans.labels_
#

#     print(f"\nМетрики кластеризации (KMeans k-means++):")
#     print(f"- Inertia: {kmeans.inertia_:.2f}")
#     print(f"- Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
#     print(f"- Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, labels):.2f}")
#     print(f"- Davies-Bouldin Score: {davies_bouldin_score(X_scaled, labels):.4f}")
#

#     print("Визуализация кластеров через PCA...")
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X_scaled)
#
#     plt.figure(figsize=(10, 7))
#     scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
#     plt.colorbar(scatter, label="Кластер")
#     plt.title(f"KMeans (k-means++) по PCA ({sample_size:,} строк)")
#     plt.xlabel("Главная компонента 1")
#     plt.ylabel("Главная компонента 2")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#

# def run_kmodes_with_metrics(input_path, k=4, sample_size=100000):
#     print(f"\nЗагрузка обработанных данных из {input_path}")
#     df = vaex.open(input_path)
#
#     df_sample = df.sample(n=sample_size)
#     df_pd = df_sample.to_pandas_df()
#

#     categorical_cols = df_pd.select_dtypes(include=['object', 'category']).columns.tolist()
#     if not categorical_cols:
#         print(" Нет категориальных признаков для K-Modes")
#         return
#
#     X_cat = df_pd[categorical_cols].fillna('unknown').astype(str)
#
#     print(f"\n Запуск K-Modes (k={k}) на категориальных признаках: {categorical_cols}")
#     km = KModes(n_clusters=k, init='Huang', n_init=5, verbose=1, random_state=42)
#     labels = km.fit_predict(X_cat)
#
#     from sklearn.preprocessing import OrdinalEncoder
#     X_encoded = OrdinalEncoder().fit_transform(X_cat)
#
#     print(f"\n Метрики кластеризации (K-Modes):")
#     print(f"- Silhouette Score: {silhouette_score(X_encoded, labels):.4f}")
#     print(f"- Calinski-Harabasz Score: {calinski_harabasz_score(X_encoded, labels):.2f}")
#     print(f"- Davies-Bouldin Score: {davies_bouldin_score(X_encoded, labels):.4f}")
#
#     print(" Визуализация кластеров через PCA...")
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X_encoded)
#
#     plt.figure(figsize=(10, 7))
#     scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
#     plt.colorbar(scatter, label="Кластер")
#     plt.title(f"K-Modes по PCA ({sample_size:,} строк)")
#     plt.xlabel("Главная компонента 1")
#     plt.ylabel("Главная компонента 2")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()




# def run_kmodes_with_metrics(input_path, k=4, sample_size=100000):
#     print(f"\nЗагрузка данных из {input_path}")
#     df = vaex.open(input_path)
#
#     CATEGORICAL_FEATURES = ['region', 'settlement_type']
#
#     missing_cols = [col for col in CATEGORICAL_FEATURES if col not in df.get_column_names()]
#     if missing_cols:
#         print(f" Отсутствуют необходимые столбцы: {missing_cols}")
#         return
#
#
#     df_sample = df.sample(n=sample_size)
#     df_pd = df_sample.to_pandas_df()
#
#     X_cat = df_pd[CATEGORICAL_FEATURES].fillna('unknown').astype(str)
#
#     print(f"\n Запуск K-Modes (k={k}) на столбцах: {CATEGORICAL_FEATURES}")
#     km = KModes(n_clusters=k, init='Huang', n_init=5, verbose=1, random_state=42)
#     labels = km.fit_predict(X_cat)
#
#     encoder = OrdinalEncoder()
#     X_encoded = encoder.fit_transform(X_cat)
#
#     print(f"\n Метрики кластеризации:")
#     print(f"- Silhouette Score: {silhouette_score(X_encoded, labels):.4f}")
#     print(f"- Calinski-Harabasz Score: {calinski_harabasz_score(X_encoded, labels):.2f}")
#     print(f"- Davies-Bouldin Score: {davies_bouldin_score(X_encoded, labels):.4f}")
#
#     print("\nВизуализация кластеров...")
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X_encoded)
#
#     plt.figure(figsize=(10, 7))
#     scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
#     plt.colorbar(scatter, label="Кластер")
#     plt.title(f"K-Modes кластеризация (k={k})\nПризнаки: {CATEGORICAL_FEATURES}")
#     plt.xlabel("Главная компонента 1")
#     plt.ylabel("Главная компонента 2")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
#     return labels
#
# def run_dbscan_with_metrics(input_path, sample_size=100000, eps=0.5, min_samples=10):
#     print(f"\nЗагрузка обработанных данных из {input_path}")
#     df = vaex.open(input_path)
#
#     df_sample = df.sample(n=sample_size)
#     df_pd = df_sample.to_pandas_df()
#
#     numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
#     X = df_pd[numeric_cols].fillna(0).values
#
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     print(f"\n Запуск DBSCAN (eps={eps}, min_samples={min_samples})...")
#     db = DBSCAN(eps=eps, min_samples=min_samples)
#     labels = db.fit_predict(X_scaled)
#
#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise = list(labels).count(-1)
#
#     print(f"\n Метрики кластеризации (DBSCAN):")
#     print(f"- Количество кластеров (без шума): {n_clusters}")
#     print(f"- Количество шумовых точек: {n_noise}")
#
#     if n_clusters > 1:
#         print(f"- Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
#         print(f"- Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, labels):.2f}")
#         print(f"- Davies-Bouldin Score: {davies_bouldin_score(X_scaled, labels):.4f}")
#     else:
#         print("️ Недостаточно кластеров для расчёта метрик (нужно ≥2)")
#
#     print(" Визуализация кластеров через PCA...")
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X_scaled)
#
#     plt.figure(figsize=(10, 7))
#     scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
#     plt.colorbar(scatter, label="Кластер")
#     plt.title(f"DBSCAN по PCA ({sample_size:,} строк)")
#     plt.xlabel("Главная компонента 1")
#     plt.ylabel("Главная компонента 2")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
#
#
#
#
# def run_optics_with_metrics(input_path, sample_size=100000, min_samples=10, xi=0.05):
#     print(f"\n Загрузка обработанных данных из {input_path}")
#     df = vaex.open(input_path)
#
#     df_sample = df.sample(n=sample_size)
#     df_pd = df_sample.to_pandas_df()
#
#     numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
#     X = df_pd[numeric_cols].fillna(0).values

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     print(f"\n Запуск OPTICS (min_samples={min_samples}, xi={xi})...")
#     optics = OPTICS(min_samples=min_samples, xi=xi)
#     labels = optics.fit_predict(X_scaled)
#
#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise = list(labels).count(-1)
#
#     print(f"\n Метрики кластеризации (OPTICS):")
#     print(f"- Количество кластеров (без шума): {n_clusters}")
#     print(f"- Количество шумовых точек: {n_noise}")
#
#     if n_clusters > 1:
#         print(f"- Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
#         print(f"- Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, labels):.2f}")
#         print(f"- Davies-Bouldin Score: {davies_bouldin_score(X_scaled, labels):.4f}")
#     else:
#         print("Недостаточно кластеров для расчёта метрик ")
#

#     print("Визуализация кластеров через PCA...")
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X_scaled)
#
#     plt.figure(figsize=(10, 7))
#     scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
#     plt.colorbar(scatter, label="Кластер")
#     plt.title(f"OPTICS по PCA ({sample_size:,} строк)")
#     plt.xlabel("Главная компонента 1")
#     plt.ylabel("Главная компонента 2")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
#
#
# def run_agglomerative_with_metrics(input_path, k=4, sample_size=10000, linkage='ward'):
#     print(f"\nЗагрузка обработанных данных из {input_path}")
#     df = vaex.open(input_path)
#
#     df_sample = df.sample(n=sample_size)
#     df_pd = df_sample.to_pandas_df()
#
#     numeric_cols = df_pd.select_dtypes(include=[float, int]).columns.tolist()
#     X = df_pd[numeric_cols].fillna(0).values
#

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     print(f"\nЗапуск Agglomerative Clustering (k={k}, linkage={linkage})...")
#     agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
#     labels = agg.fit_predict(X_scaled)
#

#     print(f"\n Метрики кластеризации (Agglomerative):")
#     print(f"- Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
#     print(f"- Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, labels):.2f}")
#     print(f"- Davies-Bouldin Score: {davies_bouldin_score(X_scaled, labels):.4f}")
#
#     print("Визуализация кластеров через PCA...")
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X_scaled)
#
#     plt.figure(figsize=(10, 7))
#     scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
#     plt.colorbar(scatter, label="Кластер")
#     plt.title(f"Agglomerative Clustering (PCA, {sample_size:,} строк)")
#     plt.xlabel("PC1")
#     plt.ylabel("PC2")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
#
# def run_hdbscan_with_metrics(input_path, sample_size=100000, min_cluster_size=15):
#     print(f"\n Загрузка данных из {input_path}")
#     df = vaex.open(input_path)
#     df_sample = df.sample(n=sample_size)
#     df_pd = df_sample.to_pandas_df()
#
#     numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
#     X = df_pd[numeric_cols].fillna(0).values
#
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     print(f"\nЗапуск HDBSCAN (min_cluster_size={min_cluster_size})...")
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
#     labels = clusterer.fit_predict(X_scaled)
#
#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#     print(f"\nМетрики кластеризации (HDBSCAN):")
#     print(f"- Количество кластеров: {n_clusters}")
#     print(f"- Шумовых точек: {list(labels).count(-1)}")
#     if n_clusters > 1:
#         print(f"- Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
#         print(f"- Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, labels):.2f}")
#         print(f"- Davies-Bouldin Score: {davies_bouldin_score(X_scaled, labels):.4f}")
#     else:
#         print("Недостаточно кластеров для метрик.")
#
#     print("Визуализация через PCA...")
#     X_pca = PCA(n_components=2).fit_transform(X_scaled)
#
#     plt.figure(figsize=(10, 7))
#     scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
#     plt.colorbar(scatter, label="Кластер")
#     plt.title(f"HDBSCAN по PCA ({sample_size:,} строк)")
#     plt.xlabel("PC1")
#     plt.ylabel("PC2")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()






# UMAP
# def visualize_clusters_umap(X, labels, method_name, sample_size=5000):
#     print(f" Применение UMAP для {method_name}...")
#     reducer = umap.UMAP(n_components=2, random_state=42)
#     indices = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
#     X_sampled = X[indices]
#     labels_sampled = labels[indices]
#     X_umap = reducer.fit_transform(X_sampled)
#
#     plt.figure(figsize=(10, 7))
#     scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels_sampled, cmap='tab10', alpha=0.6, s=10)
#     plt.colorbar(scatter, label='Кластер')
#     plt.title(f'{method_name} — Визуализация через UMAP\n({len(X_sampled)} строк)')
#     plt.xlabel('UMAP 1')
#     plt.ylabel('UMAP 2')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
#
#
# #  1. MiniBatchKMeans + UMAP
# def run_clustering_with_metrics(input_path, k=4, sample_size=100000):
#     print(f"\nЗагрузка обработанных данных из {input_path}")
#     df = vaex.open(input_path)
#
#     df_sample = df.sample(n=sample_size)
#     df_pd = df_sample.to_pandas_df()
#
#     numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
#     X = df_pd[numeric_cols].fillna(0).values
#
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     print(f"\nЗапуск MiniBatchKMeans (k={k})...")
#     kmeans = MiniBatchKMeans(n_clusters=k, batch_size=10000, max_iter=100, random_state=42)
#     kmeans.fit(X_scaled)
#     labels = kmeans.labels_
#
#     print(f"\n Метрики кластеризации:")
#     print(f"- Inertia: {kmeans.inertia_:.2f}")
#     if k > 1:
#         print(f"- Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
#         print(f"- Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, labels):.2f}")
#         print(f"- Davies-Bouldin Score: {davies_bouldin_score(X_scaled, labels):.4f}")
#
#     visualize_clusters_umap(X_scaled, labels, "MiniBatchKMeans", sample_size=5000)
#
# # 2. KMeans (k-means++) + UMAP
# def run_kmeans_greedypp_with_metrics(input_path, k=4, sample_size=100000):
#     print(f"\n Загрузка обработанных данных из {input_path}")
#     df = vaex.open(input_path)
#
#     df_sample = df.sample(n=sample_size)
#     df_pd = df_sample.to_pandas_df()
#
#     numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
#     X = df_pd[numeric_cols].fillna(0).values
#
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     print(f"\nЗапуск KMeans (инициализация: k-means++) с k={k}...")
#     kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
#     kmeans.fit(X_scaled)
#     labels = kmeans.labels_
#
#     print(f"\nМетрики кластеризации (KMeans k-means++):")
#     print(f"- Inertia: {kmeans.inertia_:.2f}")
#     if k > 1:
#         print(f"- Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
#         print(f"- Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, labels):.2f}")
#         print(f"- Davies-Bouldin Score: {davies_bouldin_score(X_scaled, labels):.4f}")
#
#     visualize_clusters_umap(X_scaled, labels, "KMeans", sample_size=5000)
#
# # 3. DBSCAN + UMAP
# def run_dbscan_with_metrics(input_path, sample_size=100000, eps=0.5, min_samples=10):
#     print(f"\nЗагрузка обработанных данных из {input_path}")
#     df = vaex.open(input_path)
#

#     df_sample = df.sample(n=sample_size)
#     df_pd = df_sample.to_pandas_df()
#
#     numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
#     X = df_pd[numeric_cols].fillna(0).values
#
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     print(f"\n Запуск DBSCAN (eps={eps}, min_samples={min_samples})...")
#     db = DBSCAN(eps=eps, min_samples=min_samples)
#     labels = db.fit_predict(X_scaled)
#
#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise = list(labels).count(-1)
#
#     print(f"\nМетрики кластеризации (DBSCAN):")
#     print(f"- Количество кластеров (без шума): {n_clusters}")
#     print(f"- Количество шумовых точек: {n_noise}")
#     if n_clusters > 1:
#         print(f"- Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
#         print(f"- Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, labels):.2f}")
#         print(f"- Davies-Bouldin Score: {davies_bouldin_score(X_scaled, labels):.4f}")
#
#     visualize_clusters_umap(X_scaled, labels, "DBSCAN", sample_size=5000)
#
#
#
# # 4. OPTICS + UMAP
# def run_optics_with_metrics(input_path, sample_size=100000, min_samples=10, xi=0.05):
#     print(f"\n Загрузка обработанных данных из {input_path}")
#     df = vaex.open(input_path)
#
#     df_sample = df.sample(n=sample_size)
#     df_pd = df_sample.to_pandas_df()
#
#     numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
#     X = df_pd[numeric_cols].fillna(0).values
#
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     print(f"\nЗапуск OPTICS (min_samples={min_samples}, xi={xi})...")
#     optics = OPTICS(min_samples=min_samples, xi=xi)
#     labels = optics.fit_predict(X_scaled)
#
#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise = list(labels).count(-1)
#
#     print(f"\n Метрики кластеризации (OPTICS):")
#     print(f"- Количество кластеров (без шума): {n_clusters}")
#     print(f"- Количество шумовых точек: {n_noise}")
#     if n_clusters > 1:
#         print(f"- Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
#         print(f"- Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, labels):.2f}")
#         print(f"- Davies-Bouldin Score: {davies_bouldin_score(X_scaled, labels):.4f}")
#
#     visualize_clusters_umap(X_scaled, labels, "OPTICS", sample_size=5000)
#
# # 5. Agglomerative Clustering + UMAP
# def run_agglomerative_with_metrics(input_path, k=4, sample_size=10000, linkage='ward'):
#     print(f"\nЗагрузка обработанных данных из {input_path}")
#     df = vaex.open(input_path)
#
#     df_sample = df.sample(n=sample_size)
#     df_pd = df_sample.to_pandas_df()
#
#     numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
#     X = df_pd[numeric_cols].fillna(0).values
#
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     print(f"\nЗапуск Agglomerative Clustering (k={k}, linkage='{linkage}')...")
#     agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
#     labels = agg.fit_predict(X_scaled)
#
#     print(f"\nМетрики кластеризации (Agglomerative):")
#     print(f"- Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
#     print(f"- Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, labels):.2f}")
#     print(f"- Davies-Bouldin Score: {davies_bouldin_score(X_scaled, labels):.4f}")
#
#     visualize_clusters_umap(X_scaled, labels, "Agglomerative", sample_size=5000)
#
#
#
# # 6. K-Modes (для категориальных данных) + UMAP
# def run_kmodes_with_metrics(input_path, k=4, sample_size=100000):
#     CATEGORICAL_FEATURES = ['region', 'settlement_type']
#     print(f"\nЗагрузка данных из {input_path}")
#     df = vaex.open(input_path)
#
#     missing_cols = [col for col in CATEGORICAL_FEATURES if col not in df.get_column_names()]
#     if missing_cols:
#         print(f"Отсутствуют необходимые столбцы: {missing_cols}")
#         return
#
#     df_sample = df.sample(n=sample_size)
#     df_pd = df_sample.to_pandas_df()
#
#     X_cat = df_pd[CATEGORICAL_FEATURES].fillna('unknown').astype(str)
#
#     encoder = OrdinalEncoder()
#     X_encoded = encoder.fit_transform(X_cat)
#
#     print(f"\nЗапуск K-Modes (k={k}) на столбцах: {CATEGORICAL_FEATURES}")
#     km = KModes(n_clusters=k, init='Huang', n_init=5, verbose=1, random_state=42)
#     labels = km.fit_predict(X_encoded)
#
#     print(f"\nМетрики кластеризации (K-Modes):")
#     print(f"- Silhouette Score: {silhouette_score(X_encoded, labels):.4f}")
#     print(f"- Calinski-Harabasz Score: {calinski_harabasz_score(X_encoded, labels):.2f}")
#     print(f"- Davies-Bouldin Score: {davies_bouldin_score(X_encoded, labels):.4f}")
#
#     visualize_clusters_umap(X_encoded, labels, "K-Modes", sample_size=5000)
#
#
# # 7. HDBSCAN + UMAP
# def run_hdbscan_with_metrics(input_path, sample_size=100000, min_cluster_size=100):
#     print(f"\nЗагрузка данных из {input_path}")
#     df = vaex.open(input_path)
#     df_sample = df.sample(n=sample_size)
#     df_pd = df_sample.to_pandas_df()
#
#     numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
#     X = df_pd[numeric_cols].fillna(0).values
#
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     print(f"\nЗапуск HDBSCAN (min_cluster_size={min_cluster_size})...")
#     clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
#     labels = clusterer.fit_predict(X_scaled)
#
#     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#     print(f"\nМетрики кластеризации (HDBSCAN):")
#     print(f"- Количество кластеров: {n_clusters}")
#     print(f"- Шумовых точек: {list(labels).count(-1)}")
#     if n_clusters > 1:
#         print(f"- Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
#         print(f"- Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, labels):.2f}")
#         print(f"- Davies-Bouldin Score: {davies_bouldin_score(X_scaled, labels):.4f}")
#     else:
#         print("Недостаточно кластеров для метрик.")
#
#     print("Визуализация через UMAP...")
#     reducer = umap.UMAP(n_components=2, random_state=42)
#     X_umap = reducer.fit_transform(X_scaled)
#
#     plt.figure(figsize=(10, 7))
#     scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
#     plt.colorbar(scatter, label="Кластер")
#     plt.title(f"HDBSCAN по UMAP ({sample_size:,} строк)")
#     plt.xlabel("UMAP1")
#     plt.ylabel("UMAP2")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
#
# def run_kmeans_hdbscan_hybrid_pca(input_path, sample_size=100000, k=5, hdb_min_cluster_size=10):
#     print(f"\nЗагрузка данных из {input_path}")
#     df = vaex.open(input_path)
#     df_sample = df.sample(n=sample_size)
#     df_pd = df_sample.to_pandas_df()
#
#     numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
#     X = df_pd[numeric_cols].fillna(0).values
#
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     print(f"\nKMeans (k={k}) — начальная кластеризация...")
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans_labels = kmeans.fit_predict(X_scaled)
#
#     final_labels = np.full(shape=len(X_scaled), fill_value=-1)
#     next_label = 0
#
#     print(f"HDBSCAN внутри каждого кластера...")
#     for i in range(k):
#         idx = np.where(kmeans_labels == i)[0]
#         if len(idx) < hdb_min_cluster_size:
#             continue  # пропускаем слишком маленькие кластеры
#
#         sub_X = X_scaled[idx]
#         sub_clusterer = hdbscan.HDBSCAN(min_cluster_size=hdb_min_cluster_size)
#         sub_labels = sub_clusterer.fit_predict(sub_X)
#
#         for j, sub_label in zip(idx, sub_labels):
#             if sub_label != -1:
#                 final_labels[j] = next_label + sub_label
#         next_label += sub_labels.max() + 1 if sub_labels.max() >= 0 else 0
#
#     n_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)
#     print(f"\nИтоговое количество кластеров: {n_clusters}")
#     print(f"Шумовых точек: {list(final_labels).count(-1)}")
#
#     if n_clusters > 1:
#         print(f"- Silhouette Score: {silhouette_score(X_scaled, final_labels):.4f}")
#         print(f"- Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, final_labels):.2f}")
#         print(f"- Davies-Bouldin Score: {davies_bouldin_score(X_scaled, final_labels):.4f}")
#     else:
#         print("Недостаточно кластеров для оценки метрик")
#
#     print("Визуализация кластеров через PCA...")
#     X_pca = PCA(n_components=2).fit_transform(X_scaled)
#     plt.figure(figsize=(10, 7))
#     plt.scatter(X_pca[:, 0], X_pca[:, 1], c=final_labels, cmap='tab10', s=10, alpha=0.6)
#     plt.title("Гибрид: KMeans + HDBSCAN (PCA)")
#     plt.xlabel("PC1")
#     plt.ylabel("PC2")
#     plt.colorbar(label="Cluster")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
#
# def run_kmeans_hdbscan_hybrid_umap(input_path, sample_size=100000, k=5, hdb_min_cluster_size=10):
#     print(f"\nЗагрузка данных из {input_path}")
#     df = vaex.open(input_path)
#     df_sample = df.sample(n=sample_size)
#     df_pd = df_sample.to_pandas_df()
#
#     numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
#     if not numeric_cols:
#         print("Нет числовых признаков для кластеризации.")
#         return
#
#     X = df_pd[numeric_cols].fillna(0).values
#     X_scaled = StandardScaler().fit_transform(X)
#
#     print(f"\n KMeans (k={k}) — начальная кластеризация...")
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans_labels = kmeans.fit_predict(X_scaled)
#
#     final_labels = np.full(len(X_scaled), -1)
#     next_label = 0
#
#     print(f" HDBSCAN внутри каждого кластера...")
#     for i in range(k):
#         idx = np.where(kmeans_labels == i)[0]
#         if len(idx) < hdb_min_cluster_size:
#             continue
#         sub_X = X_scaled[idx]
#         sub_clusterer = hdbscan.HDBSCAN(min_cluster_size=hdb_min_cluster_size)
#         sub_labels = sub_clusterer.fit_predict(sub_X)
#
#         for j, sub_label in zip(idx, sub_labels):
#             if sub_label != -1:
#                 final_labels[j] = next_label + sub_label
#         next_label += sub_labels.max() + 1 if sub_labels.max() >= 0 else 0
#
#     n_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)
#     print(f"\nИтоговое количество кластеров: {n_clusters}")
#     print(f"Шумовых точек: {list(final_labels).count(-1)}")
#
#     if n_clusters > 1:
#         print(f"- Silhouette Score: {silhouette_score(X_scaled, final_labels):.4f}")
#         print(f"- Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, final_labels):.2f}")
#         print(f"- Davies-Bouldin Score: {davies_bouldin_score(X_scaled, final_labels):.4f}")
#     else:
#         print("Недостаточно кластеров для оценки метрик")
#
#     print("Визуализация кластеров через UMAP...")
#     reducer = umap.UMAP(n_components=2, random_state=42)
#     X_umap = reducer.fit_transform(X_scaled)
#
#     plt.figure(figsize=(10, 7))
#     plt.scatter(X_umap[:, 0], X_umap[:, 1], c=final_labels, cmap='tab10', s=10, alpha=0.6)
#     plt.colorbar(label="Кластер")
#     plt.title("Гибрид: KMeans + HDBSCAN (UMAP)")
#     plt.xlabel("UMAP 1")
#     plt.ylabel("UMAP 2")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()






if __name__ == '__main__':
    df = vaex.open(INPUT_FILE)
    df_pd = df.head(100_000).to_pandas_df()
    batch_process_vaex(INPUT_FILE, OUTPUT_FILE)
    run_clustering_with_metrics(OUTPUT_FILE, k=4, sample_size=100000)
    run_kmeans_greedypp_with_metrics(OUTPUT_FILE, k=4, sample_size=100000)
    run_kmodes_with_metrics(OUTPUT_FILE, k=4, sample_size=100000)
    run_dbscan_with_metrics(OUTPUT_FILE, sample_size=100000, eps=0.5, min_samples=10)
    run_optics_with_metrics(OUTPUT_FILE, sample_size=100000, min_samples=10, xi=0.05)
    run_agglomerative_with_metrics(OUTPUT_FILE, k=4, sample_size=10000, linkage='ward')
    run_hdbscan_with_metrics(OUTPUT_FILE, sample_size=5000000)

    run_kmeans_hdbscan_hybrid_pca(OUTPUT_FILE, sample_size=50000, k=6, hdb_min_cluster_size=10)
    run_kmeans_hdbscan_hybrid_umap(OUTPUT_FILE, sample_size=50000, k=10, hdb_min_cluster_size=40)

