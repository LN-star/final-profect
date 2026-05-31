"""
WeatherInsight — интерактивная визуализация метеоданных.

Функции:
- загрузка данных из weather.db (SQLite);
- выбор города и диапазона дат;
- таблица с фильтрацией;
- производные признаки (категория температуры, уровень осадков, индекс комфорта);
- EDA: гистограмма и boxplot для выбранного показателя;
- сравнение городов по средним значениям;
- временной ряд с простым прогнозом (скользящее среднее).
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px

st.set_page_config(page_title="WeatherInsight", layout="wide")
st.title("🌦 WeatherInsight — визуализация метеоданных")

# -----------------------------
# Загрузка данных из SQLite
# -----------------------------
@st.cache_data
def load_data(db_path: str = "weather.db") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)

    # Автоматически определяем имя таблицы
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
    if tables.empty:
        conn.close()
        raise ValueError("В базе данных нет таблиц.")
    table_name = tables["name"].iloc[0]

    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()

    # Приведение даты
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Приведение булевого признака дождя
    if "is_rainy" in df.columns:
        df["is_rainy"] = df["is_rainy"].astype(bool)

    return df


df = load_data()

# Обработка пропусков
for col in ["avg_temp", "total_precip", "avg_wind"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["avg_temp", "total_precip", "avg_wind"])

# -----------------------------
# Фильтры
# -----------------------------
st.sidebar.header("Фильтры")

# Фильтр по дате
if "date" in df.columns:
    min_d, max_d = df["date"].min(), df["date"].max()
    start_d, end_d = st.sidebar.date_input(
        "Диапазон дат",
        value=(min_d.date(), max_d.date()),
        min_value=min_d.date(),
        max_value=max_d.date(),
    )
    df = df[(df["date"] >= pd.to_datetime(start_d)) & (df["date"] <= pd.to_datetime(end_d))]

# Фильтр по городу
city_col = "city" if "city" in df.columns else None
if city_col:
    cities = sorted(df[city_col].dropna().unique())
    selected_cities = st.sidebar.multiselect("Города", cities, default=cities)
    df = df[df[city_col].isin(selected_cities)]

# -----------------------------
# Производные признаки
# -----------------------------
def temp_category(t):
    if t < 0:
        return "холодно"
    elif t < 20:
        return "умеренно"
    else:
        return "жарко"

def precip_level(p):
    if p == 0:
        return "без осадков"
    elif p < 5:
        return "небольшие"
    else:
        return "сильные"

df["temp_category"] = df["avg_temp"].apply(temp_category)
df["precip_level"] = df["total_precip"].apply(precip_level)
df["comfort_index"] = -abs(df["avg_temp"] - 20) - df["avg_wind"] * 0.5

# -----------------------------
# Таблица данных
# -----------------------------
st.subheader("📄 Исходные данные (с учётом фильтров)")
st.dataframe(df, use_container_width=True)

# -----------------------------
# EDA
# -----------------------------
st.markdown("---")
st.subheader("📊 Разведочный анализ (EDA)")

numeric_cols = ["avg_temp", "total_precip", "avg_wind", "comfort_index"]
eda_col = st.selectbox("Выберите показатель", numeric_cols)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Гистограмма**")
    fig_hist = px.histogram(
        df,
        x=eda_col,
        nbins=30,
        title=f"Распределение {eda_col}",
        color_discrete_sequence=["#ff7f0e"]
    )
    fig_hist.update_layout(xaxis_title=eda_col, yaxis_title="Частота")
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.markdown("**Boxplot**")
    fig_box = px.box(
        df,
        y=eda_col,
        title=f"Boxplot {eda_col}",
        color_discrete_sequence=["#1f77b4"]
    )
    fig_box.update_layout(yaxis_title=eda_col)
    st.plotly_chart(fig_box, use_container_width=True)

# -----------------------------
# Сравнение городов
# -----------------------------
if city_col:
    st.markdown("---")
    st.subheader("🏙 Сравнение показателей между городами")

    metric = st.selectbox("Показатель", numeric_cols, key="city_metric")
    grouped = df.groupby(city_col, as_index=False)[metric].mean()

    fig_bar = px.bar(
        grouped,
        x=city_col,
        y=metric,
        title=f"Средний {metric} по городам",
        color=city_col,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_bar.update_layout(xaxis_title="Город", yaxis_title=metric)
    st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# Временной ряд + прогноз
# -----------------------------
if "date" in df.columns:
    st.markdown("---")
    st.subheader("⏱ Временной ряд и прогноз (скользящее среднее)")

    ts_metric = st.selectbox("Показатель", numeric_cols, key="ts_metric")
    df_ts = df.sort_values("date").set_index("date")

    window = st.slider("Окно сглаживания (дней)", 3, 30, 7)
    df_ts["rolling_mean"] = df_ts[ts_metric].rolling(window=window, min_periods=1).mean()

    fig_ts = px.line(
        df_ts,
        y=[ts_metric, "rolling_mean"],
        title=f"{ts_metric} и скользящее среднее ({window} дней)",
        color_discrete_sequence=["#2ca02c", "#d62728"]
    )
    fig_ts.update_layout(xaxis_title="Дата", yaxis_title=ts_metric)
    st.plotly_chart(fig_ts, use_container_width=True)

# -----------------------------
# Подсказка пользователю
# -----------------------------
st.markdown(
    ":small[Используйте фильтры слева, чтобы исследовать погоду по городам и датам, "
    "анализировать распределения и динамику показателей.]"
)
