"""Подборка вин: демо-приложение к ДЗ-3 (логистическая регрессия).

Запуск: streamlit run app.py. Рядом должны лежать model.joblib, config.json и test.csv.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure  # фигура без pyplot: не копится в памяти между прогонами
from matplotlib.patches import Patch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Пути строим от папки, где лежит app.py, а не от папки, откуда запустили streamlit run
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model.joblib"
CONFIG_PATH = APP_DIR / "config.json"
TEST_PATH = APP_DIR / "test.csv"

# Русские названия признаков и единицы измерения (ключи – имена столбцов в данных)
FEATURES_RU = {
    "fixed_acidity": ("Фиксированная кислотность", "г/л"),
    "volatile_acidity": ("Летучая кислотность", "г/л"),
    "citric_acid": ("Лимонная кислота", "г/л"),
    "residual_sugar": ("Остаточный сахар", "г/л"),
    "chlorides": ("Хлориды", "г/л"),
    "free_sulfur_dioxide": ("Свободный диоксид серы", "мг/л"),
    "total_sulfur_dioxide": ("Общий диоксид серы", "мг/л"),
    "density": ("Плотность", "г/см³"),
    "pH": ("pH", ""),
    "sulphates": ("Сульфаты", "г/л"),
    "alcohol": ("Алкоголь", "% об."),
    "is_red": ("Красное вино", ""),
}

# Цвета графиков
BLUE = "#2a78d6"    # кривые; коэффициенты «за» хорошее вино
RED = "#e34948"     # коэффициенты «против»
ORANGE = "#eb6834"  # точка текущего порога
GRAY = "#898781"    # вспомогательные линии

PROBA_COLUMN = "вероятность_хорошего"
DECISION_COLUMN = "решение"


# ---------- Загрузка файлов и кэш (задание 4.5) ----------
# st.cache_resource хранит ОДИН объект на весь сервер и отдаёт его всем сессиям без
# копирования. Это подходит для «ресурса» – модели: читаем файл один раз, а predict_proba
# модель не меняет. Без декоратора joblib.load читал бы и распаковывал модель с диска
# при каждом перезапуске скрипта (каждый сдвиг ползунка, каждое поле ввода) и держал бы
# отдельную копию в каждой сессии.
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# st.cache_data сохраняет результат в сериализованном виде (pickle) и каждому вызову отдаёт
# новую копию. Это подходит для данных: если код в одной сессии изменит таблицу,
# кэш и другие пользователи этого не увидят.
@st.cache_data
def load_test():
    return pd.read_csv(TEST_PATH)


# config.json – тоже данные (одно число), поэтому cache_data
@st.cache_data
def load_threshold():
    return float(json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["threshold"])


# Вероятности класса 1 («хорошее») на тесте считаются один раз: от порога они не зависят.
# _model с подчёркиванием – Streamlit не хеширует модель, ключ кэша строится по X.
@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


# ROC- и PR-кривые тоже не зависят от порога – считаем один раз по вероятностям
@st.cache_data
def test_curves(y_true, proba):
    fpr, tpr, _ = roc_curve(y_true, proba)
    precision, recall, _ = precision_recall_curve(y_true, proba)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc_score(y_true, proba),
        "precision": precision,
        "recall": recall,
        "ap": average_precision_score(y_true, proba),
    }


# ---------- Вспомогательные функции ----------
def feature_name(column):
    return FEATURES_RU.get(column, (column, ""))[0]


def decimals(column):
    """Сколько знаков после запятой нужно, чтобы точно записать 99% значений (не больше 5)."""
    for d in range(6):
        if np.isclose(column, column.round(d), rtol=0, atol=1e-9).mean() >= 0.99:
            return d
    return 5


def prepare_batch(batch, features):
    """Проверить загруженную таблицу.

    Возвращает (признаки в порядке модели, список ошибок). Если ошибок нет, первое – DataFrame.
    """
    if batch.empty:
        return None, ["В файле нет ни одной строки с данными."]

    missing = [column for column in features if column not in batch.columns]
    if missing:
        message = f"В файле не хватает столбцов: {', '.join(missing)}."
        if len(batch.columns) == 1:  # весь заголовок прочитался одним столбцом
            message += " Похоже, столбцы разделены не запятой."
        return None, [message]

    # Берём только нужные столбцы и в том порядке, в котором модель училась.
    # Текст, который нельзя превратить в число, станет NaN.
    X = batch[features].apply(pd.to_numeric, errors="coerce")

    errors = []
    not_numbers = [
        column for column in features
        if ((X[column].isna() & batch[column].notna()) | np.isinf(X[column])).any()
    ]
    if not_numbers:
        errors.append(f"Нечисловые значения в столбцах: {', '.join(not_numbers)}.")
    gaps = [column for column in features if batch[column].isna().any()]
    if gaps:
        errors.append(f"Пропуски в столбцах: {', '.join(gaps)}.")
    return X, errors


def reset_threshold():
    # Колбэк кнопки: выполняется до перезапуска скрипта, поэтому ползунок
    # в этом прогоне уже нарисуется с порогом из config.json
    st.session_state.threshold = load_threshold()


# ---------- Вкладки ----------
def show_single_wine(model, test, features, threshold):
    st.subheader("Оценка одного вина")
    st.caption(
        "Введите результаты лабораторного анализа. Диапазоны – минимум и максимум "
        "по тестовой выборке, значения по умолчанию – медианы (цвет – самый частый)."
    )

    values = {}
    columns = st.columns(3)
    for i, feature in enumerate(features):  # порядок – как в model.feature_names_in_
        name, unit = FEATURES_RU.get(feature, (feature, ""))
        with columns[i % 3]:
            if feature == "is_red":
                options = [0, 1]
                most_common = int(test[feature].mode()[0])  # белых в данных больше
                values[feature] = st.radio(
                    f"Цвет вина (`{feature}`)",
                    options,
                    index=options.index(most_common),
                    format_func=lambda v: "красное" if v == 1 else "белое",
                    horizontal=True,
                    key=f"input_{feature}",
                )
            else:
                column = test[feature]
                d = decimals(column)
                label = f"{name}, {unit} (`{feature}`)" if unit else f"{name} (`{feature}`)"
                values[feature] = st.number_input(
                    label,
                    min_value=float(column.min()),
                    max_value=float(column.max()),
                    value=round(float(column.median()), d),
                    step=10.0 ** -d,
                    format=f"%.{d}f",
                    key=f"input_{feature}",
                )

    # Модель ждёт таблицу: одна строка, столбцы строго в порядке обучения
    wine = pd.DataFrame([values], columns=features)
    proba = float(model.predict_proba(wine)[0, 1])

    st.divider()
    left, right = st.columns(2)
    with left:
        st.metric("Вероятность «хорошее вино»", f"{proba:.1%}", border=True)
        st.progress(proba)
    with right:
        if proba >= threshold:
            st.success(f"**Берём в подборку**: вероятность {proba:g} ≥ порога {threshold:g}.")
        else:
            st.warning(f"**Не берём**: вероятность {proba:g} < порога {threshold:g}.")


def show_metrics(model, test, features, threshold):
    st.subheader("Модельные метрики на тестовой выборке")

    y_true = test["target"]
    proba = test_proba(model, test[features])  # из кэша: predict_proba не вызывается заново
    y_pred = (proba >= threshold).astype(int)  # от порога зависит только это сравнение

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    selected = tp + fp

    st.markdown(
        f"Порог **{threshold:g}**: в подборку попадает **{selected}** вин из {len(y_true)}, "
        f"из них хороших – {tp}. Хороших вин в тесте {tp + fn}, пропущено {fn}."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Precision", f"{precision:.3f}" if selected else "—", border=True,
        help="Доля хороших вин среди попавших в подборку.",
    )
    c2.metric(
        "Recall", f"{recall:.3f}", border=True,
        help="Доля хороших вин теста, которые попали в подборку.",
    )
    c3.metric("F1", f"{f1:.3f}", border=True, help="Гармоническое среднее precision и recall.")
    if not selected:
        st.caption("При таком пороге в подборку не попало ни одного вина: precision не определён.")

    st.markdown("**Матрица ошибок** (строки – факт, столбцы – прогноз модели)")
    matrix = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Факт: обычное (0)", "Факт: хорошее (1)"],
        columns=["Прогноз: не берём (0)", "Прогноз: берём (1)"],
    )
    st.table(matrix)
    st.caption(
        f"В подборке: {tp} хороших и {fp} посредственных вин (клиент разочаруется). "
        f"За бортом: {fn} хороших вин (подборка беднее) и {tn} посредственных."
    )

    curves = test_curves(y_true, proba)
    fpr_now = fp / (fp + tn)  # точка текущего порога: те же правила, что и выше (p >= t)
    tpr_now = tp / (tp + fn)

    left, right = st.columns(2)
    with left:
        fig = Figure(figsize=(5, 4.5), layout="constrained")
        ax = fig.subplots()
        ax.plot(curves["fpr"], curves["tpr"], color=BLUE, linewidth=2,
                label=f"ROC-кривая (AUC = {curves['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1, label="случайный отбор")
        ax.scatter([fpr_now], [tpr_now], s=80, color=ORANGE, edgecolor="white", zorder=3,
                   label=f"порог {threshold:g}")
        ax.set(xlabel="FPR – доля обычных вин, попавших в подборку", ylabel="TPR (recall)",
               xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), title="ROC-кривая")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        st.pyplot(fig)
    with right:
        fig = Figure(figsize=(5, 4.5), layout="constrained")
        ax = fig.subplots()
        ax.plot(curves["recall"], curves["precision"], color=BLUE, linewidth=2,
                label=f"PR-кривая (AP = {curves['ap']:.3f})")
        ax.axhline(y_true.mean(), color=GRAY, linestyle="--", linewidth=1,
                   label=f"случайный отбор ({y_true.mean():.2f})")
        if selected:  # без вин в подборке precision не определён – точку не ставим
            ax.scatter([tpr_now], [tp / selected], s=80, color=ORANGE, edgecolor="white",
                       zorder=3, label=f"порог {threshold:g}")
        ax.set(xlabel="Recall", ylabel="Precision",
               xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), title="PR-кривая")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
        st.pyplot(fig)

    st.info(
        "**Что будет с подборкой, если двигать порог.** Вино попадает в подборку, "
        "если модель оценивает вероятность «хорошее» не ниже порога.\n\n"
        "- **Порог выше** – в подборке меньше вин, и доля действительно хороших среди них "
        "обычно выше (precision, как правило, растёт; при очень высоком пороге, когда в "
        "подборке остаются единицы вин, может и упасть): клиенты реже разочаровываются. "
        "Зато больше хороших вин остаётся за бортом (recall падает) – подборка скуднее.\n"
        "- **Порог ниже** – подборка шире, хороших вин пропускаем меньше (recall растёт), "
        "но в неё чаще попадают посредственные вина (precision, как правило, падает)."
    )


def show_batch(model, test, features, threshold):
    st.subheader("Пакетное предсказание")
    st.markdown(
        f"Загрузите CSV-файл (разделитель – запятая) со столбцами признаков: "
        f"`{', '.join(features)}`. Столбец `target` не нужен, лишние столбцы не мешают.\n\n"
        f"Порог – из ползунка в боковой панели: сейчас **{threshold:g}**."
    )

    example = test.drop(columns="target").head(5).to_csv(index=False)
    st.download_button(
        "Скачать пример файла", data=example, file_name="wines_example.csv",
        mime="text/csv", on_click="ignore",  # скачивание не перезапускает скрипт
    )

    uploaded = st.file_uploader("CSV-файл с винами", type="csv")  # пока файла нет – None
    if uploaded is None:
        st.info("Файл ещё не загружен.")
        return

    try:
        batch = pd.read_csv(uploaded)
    except Exception as error:  # пустой файл, не CSV, неверная кодировка
        st.error(f"Не удалось прочитать файл как CSV (нужна кодировка UTF-8): {error}")
        return

    X, errors = prepare_batch(batch, features)
    if errors:
        for message in errors:
            st.error(message)
        return

    proba = model.predict_proba(X)[:, 1]
    # Результат – исходная таблица, а впереди вероятность и решение.
    # Старые столбцы с прогнозом (если загрузили наш же файл) убираем, чтобы не было дублей.
    result = batch.drop(columns=[PROBA_COLUMN, DECISION_COLUMN], errors="ignore")
    result.insert(0, PROBA_COLUMN, proba)
    result.insert(1, DECISION_COLUMN, np.where(proba >= threshold, "берём", "не берём"))

    st.metric("Берём в подборку", f"{(proba >= threshold).sum()} из {len(result)}")
    st.dataframe(result, hide_index=True)
    st.download_button(
        "Скачать таблицу с прогнозами",
        data=result.to_csv(index=False).encode("utf-8-sig"),  # utf-8-sig – чтобы Excel понял кириллицу
        file_name="wine_predictions.csv",
        mime="text/csv",
        on_click="ignore",
    )


def show_coefficients(model, features):
    st.subheader("Что выучила модель")

    # Последний шаг пайплайна – LogisticRegression. Масштабирование не меняет набор
    # и порядок признаков, поэтому имена берём из model.feature_names_in_.
    coefs = pd.Series(model[-1].coef_[0], index=features).sort_values()

    fig = Figure(figsize=(8, 5), layout="constrained")
    ax = fig.subplots()
    bars = ax.barh(
        [feature_name(c) for c in coefs.index],
        coefs.values,
        color=[BLUE if v > 0 else RED for v in coefs.values],
    )
    ax.bar_label(bars, fmt="%+.2f", padding=3, fontsize=9)
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.set_xlabel("Коэффициент: изменение логита при росте признака на 1 ст. отклонение")
    ax.margins(x=0.15)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(
        handles=[Patch(color=BLUE, label="за «хорошее»"), Patch(color=RED, label="против")],
        loc="lower right",
    )
    st.pyplot(fig)

    # Подпись строится из самих коэффициентов: топ-3 «за» и топ-3 «против»
    pros = coefs[coefs > 0].sort_values(ascending=False).head(3)
    cons = coefs[coefs < 0].head(3)  # coefs отсортированы по возрастанию: самые отрицательные первые
    left, right = st.columns(2)
    with left:
        st.markdown("**Сильнее всего «голосуют» за хорошее вино:**")
        st.markdown("\n".join(f"- {feature_name(c)} ({v:+.2f})" for c, v in pros.items()) or "нет")
    with right:
        st.markdown("**Сильнее всего «голосуют» против:**")
        st.markdown("\n".join(f"- {feature_name(c)} ({v:+.2f})" for c, v in cons.items()) or "нет")
    st.caption(
        "Перед логистической регрессией признаки стандартизованы (StandardScaler), поэтому "
        "коэффициенты сравнимы между собой: коэффициент – на сколько меняется логит "
        "(логарифм шансов «хорошее») при росте признака на одно стандартное отклонение, "
        "если остальные признаки не меняются. Шансы при этом умножаются на exp(коэффициент). "
        "Знак у сильно связанных признаков (например, плотность, сахар и алкоголь) "
        "читается только вместе с остальными."
    )

    with st.expander("Таблица коэффициентов"):
        # Первый шаг пайплайна – StandardScaler: scale_ – стандартные отклонения на train
        scale = pd.Series(model[0].scale_, index=features)
        table = pd.DataFrame({
            "признак": [feature_name(c) for c in coefs.index],
            "столбец": coefs.index,
            "коэффициент": coefs.values,
            "шансы × exp(коэф.)": np.exp(coefs.values),
            "1 ст. отклонение (в единицах признака)": scale[coefs.index].values,
        })
        st.dataframe(table.iloc[::-1], hide_index=True)  # сверху – самые большие коэффициенты


# ---------- Страница ----------
st.set_page_config(page_title="Подборка вин", page_icon=":material/wine_bar:", layout="wide")

missing_files = [p.name for p in (MODEL_PATH, CONFIG_PATH, TEST_PATH) if not p.exists()]
if missing_files:
    st.error(f"Рядом с app.py нет файлов: {', '.join(missing_files)}. Сохраните их из ноутбука.")
    st.stop()

model = load_model()
features = list(model.feature_names_in_)  # столбцы и их порядок, на которых училась модель
test = load_test()
config_threshold = load_threshold()

st.title("Подборка вин для подписки")
st.caption(
    "Помощник сомелье: по лабораторному анализу логистическая регрессия оценивает "
    "вероятность того, что вино «хорошее» (оценка экспертов не ниже 7). "
    "Если вероятность не ниже порога, вино берём в подборку."
)

# Порог по умолчанию – из config.json. Кладём его в session_state один раз за сессию,
# а ползунок с тем же key берёт значение оттуда и дальше сам его обновляет.
if "threshold" not in st.session_state:
    st.session_state.threshold = config_threshold

with st.sidebar:
    st.header("Порог")
    st.slider(
        "Порог вероятности «хорошее вино»",
        min_value=0.0,
        max_value=1.0,
        step=0.001,
        format="%.3f",
        key="threshold",
        help="Вино берём в подборку, если вероятность «хорошее» не ниже порога.",
    )
    # Значение ползунка по его key: то же, что вернул бы st.slider, доступно во всех вкладках
    threshold = st.session_state.threshold
    st.caption(
        f"Сейчас используется порог: {threshold:g}  \n"
        f"Порог из config.json: {config_threshold}"  # точное значение, без округления
    )
    st.button("Вернуть порог из config.json", on_click=reset_threshold)

tab_single, tab_metrics, tab_batch, tab_coefs = st.tabs(
    ["Оценка одного вина", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"]
)
with tab_single:
    show_single_wine(model, test, features, threshold)
with tab_metrics:
    show_metrics(model, test, features, threshold)
with tab_batch:
    show_batch(model, test, features, threshold)
with tab_coefs:
    show_coefficients(model, features)
