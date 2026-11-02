"""Loss Functions Lab — учебный симулятор функций потерь.

Восьмой проект серии образовательных симуляторов:
    1. One Artificial Neuron   2. Perceptron           3. Mini Neural Network
    4. Gradient Descent        5. Backpropagation      6. Neural Network Training Lab
    7. Activation Functions    8. Loss Functions Lab

Здесь не обучается сеть. Здесь сравнивается, как четыре разные функции потерь
измеряют одну и ту же ошибку предсказания:
    цель y -> предсказание y_hat -> ошибка e -> функция потерь -> L -> dL/dy_hat

Четыре функции: MSE (с множителем 1/2), MAE, Binary Cross-Entropy, Huber.
Для каждой честно показаны область определения, значение, производная по
предсказанию и особые случаи: отсутствие обычной производной у MAE в нулевой
ошибке, взрыв BCE у границ вероятности, переход квадратичной области Huber
в линейную.

Только стандартная библиотека: Python 3, tkinter, tkinter.ttk, math.
Внешние пакеты (NumPy, matplotlib и другие) не используются.

Автор: SteveWorks001

Куда смотреть
-------------
Раздел 1   Области ввода, палитра, обозначения           (строки ~40)
Раздел 2   Форматирование и разбор чисел
Раздел 3   Математика потерь: значение, производная, области применимости
Раздел 4   Сравнение и датасет: таблица свойств, среднее по выборке
Раздел 5   Сэмплирование кривых для графиков
Раздел 6   Готовые эксперименты
Раздел 7   Локализация RU/EN
Раздел 8   Текстовые панели: расчёт, сравнение, эксперимент, справка
Раздел 9   Виджеты: поле числа, холст-график, прокрутка
Раздел 10  Главное окно приложения
Раздел 11  Точка входа
"""

import math
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

# ---------------------------------------------------------------
# 1. Области ввода, палитра, обозначения
# ---------------------------------------------------------------

# Предсказание рисуется и двигается слайдером в этих границах.
# Для MSE/MAE/Huber это обычная числовая прямая, для BCE — вероятность,
# поэтому граница (0, 1) открытая: значения 0 и 1 запрещены самой формулой.
PRED_DOMAIN = {
    "mse": (-3.0, 3.0),
    "mae": (-3.0, 3.0),
    "huber": (-3.0, 3.0),
    "bce": (0.0, 1.0),
}
# Как близко к границам вероятности можно подойти ползунком и полем ввода.
PROB_EDGE = 1e-3
# Внутренний предел отсечения для log() и деления: защищает от log(0) и 1/0.
BCE_EPS = 1e-12
# Допуск на границе ветвей Huber: |e| = δ в двоичной арифметике редко бывает точным.
BRANCH_EPS = 1e-12

TARGET_RANGE = (-3.0, 3.0)
DELTA_RANGE = (0.1, 3.0)
DELTA_DEFAULT = 1.0
TARGET_DEFAULT = 1.0
PRED_DEFAULT = 1.6

# Порог «предсказание совпало с целью»: |e| меньше него считаем нулевой ошибкой.
ZERO_EPS = 1e-9

# Окно оси Y на графике производной: за его пределами значения уходят в бесконечность,
# и рисовать их нельзя — ось сожмётся до нечитаемой.
DERIV_CLIP = 10.0
# То же для графика потерь.
LOSS_CLIP = 12.0

COL_TEXT = "#242424"
COL_BG = "#fbfbf7"
COL_PAPER = "#fdfdf9"
COL_CURVE = "#3f6f96"       # основная кривая (синий)
COL_FWD = "#3f6f96"         # подпись «выше/вверх»
COL_FWD_HEAD = "#31566f"    # заголовки блоков
COL_MARKER = "#b07d10"      # текущая точка (янтарь)
COL_TARGET = "#2f7d4f"      # цель / нулевая ошибка (зелёный)
COL_NEG = "#b3261e"         # ошибка, спад, «слишком» (красный)
COL_SECOND = "#7a5aa0"      # вторая кривая (фиолетовый)
COL_DIM = "#9a9a93"
COL_ACTIVE = "#b07d10"
COL_FRAME = "#c8c8c0"
COL_GRID = "#dcdcd4"
COL_BAD_BG = "#fde8e6"
COL_BAD_FG = "#7a1a13"
COL_ACTIVE_FILL = "#fdf1d8"

LOSS_ORDER = ("mse", "mae", "huber", "bce")
LOSS_COLOR = {"mse": COL_CURVE, "mae": COL_NEG, "huber": COL_SECOND, "bce": COL_TARGET}

MINUS = "-"


# ---------------------------------------------------------------
# 2. Форматирование и разбор чисел
# ---------------------------------------------------------------

def clamp(value, lo, hi):
    return lo if value < lo else hi if value > hi else value


def fmt(value, digits=4):
    """Число для показа: ровно digits знаков, без «-0.0000»."""
    if value is None:
        return "—"
    v = float(value)
    if not is_finite(v):
        return "∞"
    if abs(v) < 0.5 * 10.0 ** (-digits):
        v = 0.0
    text = ("%%.%df" % digits) % v
    if text.startswith("-") and float(text) == 0.0:
        text = text[1:]
    return text


def fmt_smart(value):
    """Короткое число: 4 знака, а крупные — двумя знаками, чтобы влезало в поле."""
    if value is None:
        return "—"
    v = abs(float(value))
    if v >= 100.0:
        return fmt(value, 1)
    if v >= 10.0:
        return fmt(value, 2)
    return fmt(value, 4)


def is_finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def to_text(value):
    """Как показать число во поле ввода: ASCII, без хвостовых нулей."""
    text = ("%.4f" % float(value)).rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def parse_number(raw):
    """Разобрать ввод: точка или запятая, плюс/минус, пустая строка.

    Возвращает None, если разбираемого числа нет — приложение не падает.
    """
    if raw is None:
        return None
    text = str(raw).strip().replace(",", ".").replace(" ", "")
    if text in ("", "-", "+", ".", "-.", "+."):
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not is_finite(value):
        return None
    return value


# ---------------------------------------------------------------
# 3. Математика потерь
# ---------------------------------------------------------------
# Соглашения ровно те, что заявлены в назначении проекта:
#   MSE   L = 1/2 * (y_hat - y)^2   (множитель 1/2 — учебное соглашение)
#   MAE   L = |y_hat - y|
#   BCE   L = -(y ln y_hat + (1-y) ln(1 - y_hat))
#   Huber L = 1/2 e^2            при |e| <= delta
#           delta (|e| - 1/2 delta) иначе

def error(y_hat, y):
    return float(y_hat) - float(y)


def mse_loss(y_hat, y, delta=None):
    e = error(y_hat, y)
    return 0.5 * e * e


def mse_deriv(y_hat, y, delta=None):
    return error(y_hat, y)


def mae_loss(y_hat, y, delta=None):
    return abs(error(y_hat, y))


def mae_deriv(y_hat, y, delta=None):
    """Обычная производная MAE. В нулевой ошибке её не существует — возвращаем None."""
    e = error(y_hat, y)
    if abs(e) < ZERO_EPS:
        return None
    return 1.0 if e > 0.0 else -1.0


def mae_subgradient(y_hat, y, delta=None):
    """Численное соглашение для графика: один подградиент из [-1, +1], в нуле берём 0.

    Это не производная: при предсказании, равном цели, обычная производная
    отсутствует. Подградиент подписывается в интерфейсе отдельно.
    """
    e = error(y_hat, y)
    if abs(e) < ZERO_EPS:
        return 0.0
    return 1.0 if e > 0.0 else -1.0


def bce_applies(y, y_hat):
    """BCE имеет смысл только при 0 <= y <= 1 и 0 < y_hat < 1."""
    return -1e-12 <= float(y) <= 1.0 + 1e-12 and 0.0 < float(y_hat) < 1.0


def bce_loss(y_hat, y, delta=None):
    p = clip_probability(y_hat)
    t = clamp(float(y), 0.0, 1.0)
    return -(t * math.log(p) + (1.0 - t) * math.log(1.0 - p))


def bce_deriv(y_hat, y, delta=None):
    p = clip_probability(y_hat)
    t = clamp(float(y), 0.0, 1.0)
    return -(t / p) + ((1.0 - t) / (1.0 - p))


def clip_probability(value):
    """Обрезать вероятность до (BCE_EPS, 1 - BCE_EPS): защита от log(0) и 1/0."""
    return clamp(float(value), BCE_EPS, 1.0 - BCE_EPS)


def huber_loss(y_hat, y, delta=DELTA_DEFAULT):
    e = error(y_hat, y)
    d = abs(float(delta))
    a = abs(e)
    if a <= d + BRANCH_EPS:
        return 0.5 * e * e
    return d * (a - 0.5 * d)


def huber_deriv(y_hat, y, delta=DELTA_DEFAULT):
    e = error(y_hat, y)
    d = abs(float(delta))
    if abs(e) <= d + BRANCH_EPS:
        return e
    return d if e > 0.0 else -d


def is_huber_linear(y_hat, y, delta=DELTA_DEFAULT):
    return abs(error(y_hat, y)) > abs(float(delta)) + BRANCH_EPS


LOSS_VALUE = {"mse": mse_loss, "mae": mae_loss, "huber": huber_loss, "bce": bce_loss}
LOSS_DERIV = {"mse": mse_deriv, "mae": mae_deriv, "huber": huber_deriv, "bce": bce_deriv}


def loss_value(key, y_hat, y, delta=DELTA_DEFAULT):
    """Значение потерь или None, если функция данным значениям не определена."""
    if key == "bce" and not bce_applies(y, y_hat):
        return None
    value = LOSS_VALUE[key](y_hat, y, delta)
    return value if is_finite(value) else None


def loss_deriv(key, y_hat, y, delta=DELTA_DEFAULT):
    """Производная по предсказанию.

    None означает «обычной производной здесь нет» (MAE в нулевой ошибке).
    Для графика MAE берётся подградиент — он подписан отдельно.
    """
    if key == "bce" and not bce_applies(y, y_hat):
        return None
    value = LOSS_DERIV[key](y_hat, y, delta)
    return value if (value is None or is_finite(value)) else None


def loss_deriv_for_plot(key, y_hat, y, delta=DELTA_DEFAULT):
    if key == "mae":
        return mae_subgradient(y_hat, y, delta)
    return loss_deriv(key, y_hat, y, delta)


def pred_domain_for(key, y_hat, y):
    """Границы предсказания для выбранной функции + домен поля ввода."""
    lo, hi = PRED_DOMAIN[key]
    if key == "bce":
        return (PROB_EDGE, 1.0 - PROB_EDGE)
    return (lo, hi)


def clamp_prediction(key, value, y=None):
    """Привесть предсказание к домену выбранной функции потерь."""
    lo, hi = pred_domain_for(key, value, y)
    return clamp(float(value), lo, hi)


# ---------------------------------------------------------------
# 4. Сравнение и датасет
# ---------------------------------------------------------------

def sample_average(rows, key, delta=DELTA_DEFAULT):
    """Средние потери по выборке пар (target, prediction).

    Возвращает (список потерь по строкам, среднее, количество применимых строк).
    Неприменимые строки (например, BCE вне [0, 1]) дают None и в среднее не входят.
    """
    per_row = [loss_value(key, y_hat, y, delta) for y, y_hat in rows]
    usable = [v for v in per_row if v is not None]
    if not usable:
        return per_row, None, 0
    return per_row, sum(usable) / float(len(usable)), len(usable)


DEFAULT_ROWS = (
    (1.0, 0.9),
    (1.0, 1.4),
    (0.0, 0.2),
    (2.0, 0.5),
    (-1.0, -1.0),
)


# ---------------------------------------------------------------
# 5. Сэмплирование кривых для графиков
# ---------------------------------------------------------------

def nice_ceiling(value):
    """Верхняя граница оси «красивым» числом: 0.05, 0.1, 0.25, 0.5, 1.0 ..."""
    if value <= 0.0:
        return 0.1
    exponent = math.floor(math.log10(value))
    base = 10.0 ** exponent
    for candidate in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if value <= candidate * base:
            return candidate * base
    return 10.0 * base


def nice_step(span, target=6):
    if span <= 0.0:
        return 1.0
    raw = span / float(target)
    mag = 10.0 ** math.floor(math.log10(raw))
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        cand = mult * mag
        if cand >= raw:
            return cand
    return 10.0 * mag


def tick_values(lo, hi, step):
    out = []
    if step <= 0.0:
        return out
    value = math.ceil(lo / step) * step
    while value <= hi + step * 1e-9:
        if value > lo - step * 1e-9:
            out.append(round(value, 10))
        value += step
    return out


def tick_label(value, step):
    if step >= 1.0:
        return "%d" % int(round(value))
    digits = max(1, min(4, int(math.ceil(-math.log10(step)))))
    return ("%%.%df" % digits) % value


def build_curve(key, y, delta, x_lo, x_hi, use_deriv=False, samples=240, gap=None):
    """Точки кривой (x, value) по диапазону предсказания.

    Неприменимые и слишком большие значения дают разрыв: из точек собираются
    отдельные отрезки, чтобы линия не перескакивала через обрыв (MAE) и не
    уходила за пределы окна (BCE у границ вероятности).

    gap = (центр, полуширина) вырезает окрестность точки: так ступенька
    производной MAE остаётся ступенькой, а не вертикальным скачком.
    """
    points = []
    step = (x_hi - x_lo) / float(max(samples - 1, 1))
    limit = DERIV_CLIP if use_deriv else LOSS_CLIP
    for i in range(samples):
        x = x_lo + i * step
        if i == samples - 1:
            x = x_hi
        if gap is not None and abs(x - gap[0]) <= gap[1]:
            points.append(None)
            continue
        if use_deriv:
            value = loss_deriv_for_plot(key, x, y, delta)
        else:
            value = loss_value(key, x, y, delta)
        if value is None or abs(value) > limit:
            points.append(None)
        else:
            points.append((x, value))
    return split_segments(points)


def split_segments(points):
    """[[ (x, y), ... ], ...] — куски кривой без разрывов."""
    segments = []
    current = []
    for point in points:
        if point is None:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(point)
    if current:
        segments.append(current)
    return segments


def curve_limit_exceeded(key, y, delta, x_lo, x_hi, use_deriv=False, samples=240,
                         gap=None):
    """Вышло ли хоть одно значение за предел отсечения на сетке выборок.

    build_curve такие точки отбрасывает, поэтому проверять по уже отсечённым
    точкам бессмысленно: после отсечения значение всегда не больше предела,
    и плашка «кривая уходит за …» не появилась бы никогда. Здесь тот же шаг
    и та же граница, но сравнение с настоящими значениями.
    """
    limit = DERIV_CLIP if use_deriv else LOSS_CLIP
    step = (x_hi - x_lo) / float(max(samples - 1, 1))
    for i in range(samples):
        x = x_hi if i == samples - 1 else x_lo + i * step
        if gap is not None and abs(x - gap[0]) <= gap[1]:
            continue
        if use_deriv:
            value = loss_deriv_for_plot(key, x, y, delta)
        else:
            value = loss_value(key, x, y, delta)
        if value is not None and abs(value) > limit:
            return True
    return False


def bounds_of(segments, include_zero=True):
    """Границы оси Y по кривой; минимум всегда включает 0."""
    values = [p[1] for seg in segments for p in seg]
    if include_zero:
        values.append(0.0)
    if not values:
        return (-1.0, 1.0)
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    return (lo, hi)


# ---------------------------------------------------------------
# 6. Готовые эксперименты
# ---------------------------------------------------------------
# (ключ, функция потерь, target, prediction, delta)
EXPERIMENTS = (
    ("perfect", "mse", 1.0, 1.0, None),
    ("small", "mse", 1.0, 1.15, None),
    ("large", "mse", 1.0, 2.9, None),
    ("under", "mse", 1.0, -0.5, None),
    ("over", "mse", 1.0, 2.5, None),
    ("mse_mae", "mse", 1.0, 2.6, None),
    ("huber_near", "huber", 1.0, 1.9, 1.0),
    ("huber_far", "huber", 1.0, 3.0, 1.0),
    ("bce_right", "bce", 1.0, 0.95, None),
    ("bce_wrong", "bce", 1.0, 0.05, None),
    ("bce_edge", "bce", 0.0, 0.001, None),
)


# ---------------------------------------------------------------
# 7. Локализация RU/EN
# ---------------------------------------------------------------
# Математические записи переводятся плохо и путаются, поэтому формулы живут
# отдельно от TEXTS: они одинаковы в обоих языках.
FORMULAS = {
    "mse": "L = 1/2 * (ŷ - y)^2",
    "mae": "L = |ŷ - y|",
    "bce": "L = -[ y ln(ŷ) + (1 - y) ln(1 - ŷ) ]",
    "huber": "L = 1/2 e^2, |e| <= δ;  δ(|e| - 1/2 δ), |e| > δ",
}
DERIV_FORMULAS = {
    "mse": "dL/dŷ = ŷ - y",
    "mae": "dL/dŷ = +1 (ŷ > y);  -1 (ŷ < y);  undefined (ŷ = y)",
    "bce": "dL/dŷ = -(y / ŷ) + (1 - y) / (1 - ŷ)",
    "huber": "dL/dŷ = e, |e| <= δ;  δ * sign(e), |e| > δ",
}
SUBSTITUTIONS = {
    "mse": "L = 1/2 * ({p} - {t})^2 = 1/2 * {e2}",
    "mae": "L = |{p} - {t}| = |{e}|",
    "bce": "L = -[ {t} ln({p}) + {u} ln(1 - {p}) ]",
    "huber": "L = {branch} = {res}",
}
DERIV_SUBSTITUTIONS = {
    "mse": "dL/dŷ = {p} - {t} = {e}",
    "mae": "dL/dŷ = {sgn}",
    "bce": "dL/dŷ = -({t} / {p}) + ({u} / (1 - {p}))",
    "huber": "dL/dŷ = {branch}",
}


class _Safe(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _safe_format(template, **kw):
    try:
        return template.format_map(_Safe(kw))
    except (ValueError, KeyError, IndexError):
        return template


def T(key, **kw):
    text = TEXTS[LANG].get(key)
    if text is None:
        text = "[" + key + "]"
    if kw:
        return _safe_format(text, **kw)
    return text


LANG = "ru"


def set_language(code):
    global LANG
    if code in TEXTS:
        LANG = code


TEXTS = {
    "ru": {
        # --- заголовок окна -------------------------------------------------
        "app_title": "Loss Functions Lab — функции потерь",
        "subtitle": "Как разные функции потерь измеряют одну и ту же ошибку",
        "lang_ru": "Русский",
        "lang_en": "English",

        # --- левая колонка --------------------------------------------------
        "box_loss": "Функция потерь",
        "box_target": "Цель y",
        "box_pred": "Предсказание ŷ",
        "box_delta": "Порог δ (Huber)",
        "box_actions": "Эксперимент и сброс",
        "lbl_target_hint": "любое число [-3; 3]",
        "lbl_pred_hint": "ползунок или ввод числа",
        "lbl_delta_hint": "|e| = δ — граница",
        "lbl_slider": "Ползунок ŷ",
        "btn_reset": "Сбросить всё",
        "note_pred_domain": "BCE: предсказание — вероятность, доступно 0.001 … 0.999.",
        "combo_preset": "Готовый эксперимент",
        "combo_hint": "значение применится по кнопке",
        "btn_apply": "Применить",
        "hint_step": "шаг кнопок 0.05",
        "btn_pred_up": "+0.05",
        "btn_pred_down": "-0.05",
        "delta_state_quad": "|e| = {abs_e} ≤ δ = {delta}: квадратичная ветвь",
        "delta_state_lin": "|e| = {abs_e} > δ = {delta}: линейная ветвь",

        # --- центральная колонка --------------------------------------------
        "sum_error": "Ошибка e = ŷ − y",
        "sum_abs": "Модуль |e|",
        "sum_loss": "Потери L",
        "sum_deriv": "Производная dL/dŷ",
        "sum_state": "Предсказание",
        "state_under": "ниже цели",
        "state_equal": "ровно цель",
        "state_over": "выше цели",
        "graph_loss_title": "Потери L как функция предсказания ŷ",
        "graph_deriv_title": "Производная dL/dŷ как функция предсказания ŷ",
        "graph_x_pred": "предсказание ŷ",
        "graph_y_loss": "потери L",
        "graph_y_deriv": "dL/dŷ",
        "graph_legend": "{name} | y = {target} | ŷ = {pred} | L = {loss}",
        "graph_deriv_legend": "{name}: dL/dŷ",
        "graph_clip_loss": "кривая уходит за {limit}: видна только её часть",
        "graph_clip_deriv": "|dL/dŷ| > {limit} уходит в бесконечность",
        "graph_target_line": "цель y = {value}",
        "graph_transition": "пунктир: |e| = δ",
        "graph_none": "функция не определена при этих значениях",
        "graph_ndiff": "при ŷ = y производной MAE не существует",

        # --- вкладки --------------------------------------------------------
        "tab_calc": "Расчёт",
        "tab_compare": "Сравнение",
        "tab_dataset": "Датасет",
        "tab_exp": "Эксперименты",
        "tab_help": "Справка",

        # --- панель расчёта -------------------------------------------------
        "calc_intro": "Что именно подставляется в формулу — по шагам.",
        "calc_formula": "Формула",
        "calc_domain": "Область определения",
        "calc_target": "Цель y",
        "calc_pred": "Предсказание ŷ",
        "calc_error": "Ошибка e = ŷ − y",
        "calc_sub": "Подстановка",
        "calc_result": "Результат",
        "calc_dformula": "Формула производной",
        "calc_dsub": "Подстановка в производную",
        "calc_dresult": "Значение производной",
        "calc_meaning": "Что это значит",
        "calc_ndiff": "Обычной производной в этой точке нет: левый предел −1, правый +1.",
        "calc_subgradient": "Для графика взято числовое соглашение — подградиент {value}. Это не производная.",
        "no_value": "не существует",
        "calc_bce_na": "BCE не определена: нужны 0 ≤ y ≤ 1 и 0 < ŷ < 1.",
        "calc_branch_quad": "квадратичная ветвь (|e| ≤ δ)",
        "calc_branch_lin": "линейная ветвь (|e| > δ)",
        "calc_zero_loss": "Предсказание совпало с целью: потери равны нулю.",
        "calc_under": "Предсказание меньше цели: ошибка отрицательная, производная отрицательная — обучение поднимает предсказание.",
        "calc_over": "Предсказание больше цели: ошибка положительная, производная положительная — обучение опускает предсказание.",
        "calc_scale_mse": "Множитель 1/2 — учебное соглашение: с ним производная равна e без двойки. Без 1/2 потери были бы вдвое больше.",
        "calc_prob_note": "Вероятность внутри (0, 1); значения 0 и 1 формулой запрещены (ln 0 и деление на 0).",

        # --- сравнение ------------------------------------------------------
        "cmp_intro": "Одна и та же пара (цель, предсказание) для всех четырёх функций.",
        "cmp_values": "y = {target}, ŷ = {pred}, δ = {delta}",
        "cmp_table_title": "Свойства функций потерь",
        "cmp_graph_title": "Все четыре функции на одном графике",
        "cmp_column_title": "{name}: L = {loss}, dL/dŷ = {deriv}",
        "cmp_bce_na": "BCE не применима к этим числам",
        "cmp_bce_na_detail": "Для BCE нужны 0 ≤ y ≤ 1 и 0 < ŷ < 1; сейчас y = {target}, ŷ = {pred}. Число не показывается, чтобы не вводить в заблуждение.",
        "cmp_mse_mae": "MSE против MAE",
        "cmp_mse_mae_note": "Один и тот же рост ошибки: MSE растёт квадратом, MAE — линейно. Ни одна функция не «лучше вообще» — важна задача.",
        "col_fn": "Функция",
        "col_formula": "Формула",
        "col_domain": "Область",
        "col_loss": "Потери L",
        "col_deriv": "dL/dŷ",
        "col_loss_behav": "Поведение потерь",
        "col_deriv_behav": "Поведение производной",
        "col_adv": "Главное достоинство",
        "col_lim": "Главное ограничение",
        "col_use": "Где применяется",
        "name_mse": "MSE",
        "name_mae": "MAE",
        "name_bce": "BCE",
        "name_huber": "Huber",
        "domain_mse": "любые вещественные y и ŷ",
        "domain_mae": "любые вещественные y и ŷ",
        "domain_bce": "0 ≤ y ≤ 1, 0 < ŷ < 1",
        "domain_huber": "любые y и ŷ, δ > 0",
        "behav_loss_mse": "гладкий параболоид, быстрый рост при большой ошибке",
        "behav_loss_mae": "линейный рост, острый минимум в нуле ошибки",
        "behav_loss_bce": "мягко около правильной вероятности, взрыв у неверных границ",
        "behav_loss_huber": "квадратично при |e| ≤ δ, линейно дальше",
        "behav_deriv_mse": "линейно растёт с ошибкой",
        "behav_deriv_mae": "ступенька +-1, разрыв в нулевой ошибке",
        "behav_deriv_bce": "гипербола, огромна у границ вероятности",
        "behav_deriv_huber": "растёт линейно, затем плато +-delta",
        "adv_mse": "гладкая, дифференцируемая, легко считать и дифференцировать",
        "adv_mae": "устойчива к выбросам, единица измерения понятна",
        "adv_bce": "штрафует уверенность, согласована с вероятностью",
        "adv_huber": "сочетает гладкость MSE и устойчивость MAE",
        "lim_mse": "выброс даёт огромный вклад в потерю",
        "lim_mae": "не дифференцируема в нуле, градиент не исчезает около минимума",
        "lim_bce": "только для вероятностей, неустойчива у 0 и 1",
        "lim_huber": "появляется второй параметр delta, который надо выбирать",
        "use_mse": "регрессия, малый шум, обучение через градиенты",
        "use_mae": "регрессия с выбросами, интерпретируемые метрики",
        "use_bce": "бинарная классификация, вероятность класса",
        "use_huber": "регрессия с выбросами, устойчивая оптимизация",

        # --- датасет --------------------------------------------------------
        "ds_intro": "Несколько пар (цель, предсказание) и средние потери по выборке.",
        "ds_col_target": "Цель y",
        "ds_col_pred": "Предсказание ŷ",
        "ds_col_loss": "Потери по строке",
        "ds_col_error": "Ошибка e",
        "ds_btn_add": "Добавить строку",
        "ds_btn_remove": "Убрать выбранную",
        "ds_btn_restore": "Вернуть пример",
        "ds_avg": "Средние потери по выборке",
        "ds_avg_none": "ни одна строка не применима к этой функции",
        "ds_row": "строка {n}: цель {target}, предсказание {pred}",
        "ds_note": "Потери одной строки и средние по выборке — разные величины. Среднее — это сумма потерь по строкам, делённая на их число.",
        "ds_bce_rows": "BCE считает только строки с 0 ≤ y ≤ 1 и 0 < ŷ < 1; остальные пропущены в среднем.",
        "ds_row_na": "не определена",
        "ds_sum": "сумма потерь = {sum}, применимых строк = {count}",
        "ds_max_rows": "максимум 8 строк: режим учебный, а не система обучения",
        "ds_no_remove": "нужна хотя бы одна строка",
        "ds_caption": "Функция: {name}. Это НЕ обучение: сеть, эпохи и обновление весов здесь не появляются.",

        # --- эксперименты ---------------------------------------------------
        "exp_intro": "Каждый эксперимент меняет состояние приложения и объясняет одно поведение.",
        "exp_apply": "Применить",
        "exp_active": "Применён: {name}",
        "exp_perfect": "Идеальное предсказание",
        "exp_small": "Малая ошибка",
        "exp_large": "Большая ошибка",
        "exp_under": "Недооценка",
        "exp_over": "Переоценка",
        "exp_mse_mae": "MSE против MAE",
        "exp_huber_near": "Huber у границы перехода",
        "exp_huber_far": "Huber с большой ошибкой",
        "exp_bce_right": "BCE: уверенно и верно",
        "exp_bce_wrong": "BCE: уверенно и неверно",
        "exp_bce_edge": "BCE: у границы вероятности",
        "exp_text_perfect": "Цель и предсказание совпали: e = 0, MSE = 0, MAE = 0, Huber = 0. Производные MSE и Huber тоже нулевые. У MAE в этой точке обычная производная отсутствует: левый предел −1, правый +1. BCE при y = 1 и ŷ = 1 не определена — вероятность 1 формулой запрещена.",
        "exp_text_small": "Ошибка 0.15: MSE = 0.01125, MAE = 0.15. При малой ошибке квадратичная функция даёт крошечный штраф, а линейная — заметный. Поэтому у MAE около минимума градиент не уменьшается.",
        "exp_text_large": "Ошибка 1.9: MSE = 1.805, MAE = 1.9. При малой ошибке MSE была меньше MAE, а теперь сравнялась и продолжит расти быстрее: квадрат догоняет линейную зависимость и обгоняет её.",
        "exp_text_under": "Предсказание ниже цели: e отрицательный, производная MSE отрицательна. Обучение сдвигает предсказание вверх — знак производной указывает направление.",
        "exp_text_over": "Предсказание выше цели: e положительный, производная MSE положительна. Направление шага обучения — вниз, к цели.",
        "exp_text_mse_mae": "Сравните: L_MSE = {mse}, L_MAE = {mae}. Увеличивайте предсказание — MAE растёт ровной линией, MSE — параболой. На графике видно, где они пересекаются (|e| = 2 при нашем соглашении 1/2).",
        "exp_text_huber_near": "|e| = 0.9 ≤ δ = 1.0: это ещё квадратичная ветвь, производная равна e и растёт линейно. Подвигайте δ или предсказание и смотрите, как точка пересекает границу |e| = δ.",
        "exp_text_huber_far": "|e| = 2.0 > δ = 1.0: включилась линейная ветвь. Потери растут как δ·|e|, а производная застыла на плато ±δ — выброс не разрывает градиент.",
        "exp_text_bce_right": "y = 1, ŷ = 0.95: штраф мал (L = {loss}), производная сдержанна. Модель уверена в верном классе — обучение почти не вмешивается.",
        "exp_text_bce_wrong": "y = 1, ŷ = 0.05: модель уверена в неверном классе. Штраф L = {loss} — в десятки раз больше, чем при той же по модулю ошибке у MSE. В этом смысл BCE: наказывать уверенную неправоту.",
        "exp_text_bce_edge": "y = 0, ŷ = 0.001: почти ноль, но ln(0.001) конечен, поэтому программа не падает. Границы ползунка 0.001 и 0.999 задают пределы; при ŷ → 0 потери растут логарифмически.",

        # --- справка --------------------------------------------------------
        "help_intro": "16 учебных разделов. Слева — функция, в центре — графики, справа — разбор.",
        "help_toc_title": "Разделы",
        "help_nav_hint": "стрелки ↑↓ работают, когда список разделов в фокусе",
        "h_title_1": "1. Что такое функция потерь",
        "h_text_1": "Функция потерь L — это число, которое говорит, насколько предсказание не совпало с целью. Маленькое L значит «предсказание близко к цели», большое — «далеко». Важно, что одна и та же ошибка может получать очень разные штрафы: MSE при e = 2 даёт 2.0, а при e = 0.2 — 0.02.",
        "h_title_2": "2. Цель и предсказание",
        "h_text_2": "Цель y — правильный ответ. Предсказание ŷ — то, что выдала модель (или что вы задали руками). В этом симуляторе сеть не обучается: вы сами выбираете оба числа и смотрите, что получится.",
        "h_title_3": "3. Ошибка",
        "h_text_3": "Ошибка e = ŷ − y — со знаком. Если e < 0, предсказание ниже цели (недооценка). Если e > 0 — выше цели (переоценка). Модуль |e| отвечает за размер штрафа, а знак — за направление обучения.",
        "h_title_4": "4. Зачем модели функция потерь",
        "h_text_4": "Модель не знает, что «правильно». Она знает только число L. Шаг обучения выбирается так, чтобы L уменьшалось, поэтому выбор функции потерь — это выбор того, какая ошибка считается плохой. Разные функции по-разному ведут один и тот же процесс обучения.",
        "h_title_5": "5. MSE: среднеквадратичные потери",
        "h_text_5": "L = 1/2·(ŷ − y)², производная dL/dŷ = ŷ − y. Множитель 1/2 — учебное соглашение: с ним производная не получает двойку впереди. Без 1/2 (как в некоторых библиотеках) все значения потерь были бы вдвое больше, а форма кривой — та же. График — парабола, гладкая и дифференцируемая везде.",
        "h_title_6": "6. MAE: средние абсолютные потери",
        "h_text_6": "L = |ŷ − y| — штраф ровно в единицах ошибки. Растёт линейно, поэтому большой выброс не наказывается квадратом. Зато график — «воронка» с остриём в нуле, а не гладкая чаша.",
        "h_title_7": "7. Binary Cross-Entropy",
        "h_text_7": "L = −[ y·ln(ŷ) + (1 − y)·ln(1 − ŷ) ], где y — верный класс (0 или 1), а ŷ — предсказанная вероятность. Область: 0 ≤ y ≤ 1 и строгое 0 < ŷ < 1. При ln(0) формула уходит в бесконечность, поэтому симулятор ограничивает вероятности краями 0.001 и 0.999.",
        "h_title_8": "8. Huber Loss",
        "h_text_8": "L = 1/2·e² при |e| ≤ δ и δ(|e| − 1/2·δ) при |e| > δ. Внутри порога — точная копия MSE (с множителем 1/2), снаружи — линейный рост. Производная: e внутри, ±δ снаружи. Переход непрерывен и по значениям, и по производной.",
        "h_title_9": "9. Производная потерь по предсказанию",
        "h_text_9": "dL/dŷ показывает, насколько изменятся потери, если подвинуть предсказание на маленькую величину. Отрицательная — предсказание надо увеличивать, положительная — уменьшать, ноль — точка покоя (минимум или плато).",
        "h_title_10": "10. Почему производные важны в обучении",
        "h_text_10": "Градиентное обучение двигает параметры против градиента. Значит форма кривой производной определяет размер и направление шага: у MSE шаг пропорционален ошибке, у MAE он постоянен по модулю, у BCE огромен у границ вероятности, у Huber ограничен плато ±δ.",
        "h_title_11": "11. MSE против MAE",
        "h_text_11": "До |e| = 2 (при соглашении с множителем 1/2) MSE меньше MAE, после — больше и растёт быстрее. Малые ошибки MSE почти не замечает, большие наказывает сильно; MAE наказывает пропорционально всегда. Поэтому MSE чувствительна к выбросам, а MAE устойчива, но хуже дифференцируется. «Лучше» зависит от задачи, а не от функции.",
        "h_title_12": "12. BCE и уверенная неправота",
        "h_text_12": "BCE сравнивает вероятность и факт. Одинаково неверные предсказания наказаны по-разному: неуверенная ошибка (y = 1, ŷ = 0.4) даёт L = 0.916, уверенная ошибка (y = 1, ŷ = 0.05) — L = 3.0. Именно это делает BCE удобной для классификации: она требует не просто угадать класс, а не быть уверенной в неверном.",
        "h_title_13": "13. Переход Huber",
        "h_text_13": "Граница |e| = δ — смысл Huber. Малые ошибки работают как MSE: градиент пропорционален ошибке, сходимость гладкая. Большие — как MAE: градиент замирает на ±δ, выброс не утаскивает обучение. Меняя δ, вы решаете, какая ошибка считается «обычной», а какая «выбросом».",
        "h_title_14": "14. Отсутствие производной у MAE",
        "h_text_14": "При ŷ = y потери MAE равны нулю, но обычной производной нет: левый предел −1, правый +1, они не совпадают. Численные библиотеки берут подградиент — любое значение из [−1, +1]; популярное соглашение — 0. Симулятор показывает производную как «не существует» и отдельно подписывает использованный подградиент.",
        "h_title_15": "15. Потери одной строки против средних по выборке",
        "h_text_15": "Потери одной пары (цель, предсказание) — sample loss. Средние потери по всем парам — average dataset loss: сумма постранных потерь, делённая на их число. Одна плохая строка поднимает среднее; по ней одной нельзя судить о модели. Вкладка «Датасет» показывает обе величины рядом.",
        "h_title_16": "16. Как работать с симулятором",
        "h_text_16": "1. Выберите функцию потерь слева. 2. Задайте цель y. 3. Двигайте предсказание ŷ ползунком, кнопками ±0.05 или полем ввода. 4. Смотрите, как меняются ошибка, потери и производная в центре и на двух графиках. 5. Откройте «Расчёт», чтобы увидеть подстановку чисел в формулу. 6. «Сравнение» показывает все четыре функции на тех же числах. 7. «Эксперименты» задают готовые наборы значений с объяснением.",

        # --- статус ---------------------------------------------------------
        "status_ready": "Готово: измените предсказание и наблюдайте отклик функций потерь.",
        "status_loss": "Выбраны потери: {name}.",
        "status_preset": "Эксперимент «{name}» применён.",
        "status_reset": "Состояние сброшено: цель 1.0, предсказание 1.6, δ = 1.0.",
        "status_clamped": "Предсказание приведено к домену {lo} … {hi} выбранной функции.",
        "status_target_clamped": "Цель должна быть в [-3; 3]: значение приведено к границе.",
        "status_delta_clamped": "δ должна быть в [0.1; 3.0]: значение приведено к границе.",
        "status_bce_na": "BCE при этих числах не определена — см. вкладку «Расчёт».",
        "status_invalid": "Поле «{field}»: введите число (запятая или точка).",
        "status_lang": "Язык интерфейса переключён, состояние сохранено.",
        "status_ndiff": "Нулевая ошибка: у MAE здесь обычной производной нет.",
        "status_huber_switch": "Huber: |e| = {abs_e}, δ = {delta} — {branch}.",
        "status_ds_added": "Добавлена строка датасета.",
        "status_ds_removed": "Строка датасета удалена.",
        "status_ds_restored": "Датасет возвращён к примеру из 5 строк.",

        # --- общее ----------------------------------------------------------
    },
    "en": {
        "app_title": "Loss Functions Lab",
        "subtitle": "How different losses measure the same prediction error",
        "lang_ru": "Русский",
        "lang_en": "English",

        "box_loss": "Loss function",
        "box_target": "Target y",
        "box_pred": "Prediction ŷ",
        "box_delta": "Huber threshold δ",
        "box_actions": "Experiment and reset",
        "lbl_target_hint": "any number in [-3, 3]",
        "lbl_pred_hint": "slider or typing",
        "lbl_delta_hint": "|e| = δ is the border",
        "lbl_slider": "Prediction slider",
        "btn_reset": "Reset everything",
        "note_pred_domain": "BCE: the prediction is a probability, range 0.001 to 0.999.",
        "combo_preset": "Ready-made experiment",
        "combo_hint": "the value applies on Apply",
        "btn_apply": "Apply",
        "hint_step": "buttons step 0.05",
        "btn_pred_up": "+0.05",
        "btn_pred_down": "-0.05",
        "delta_state_quad": "|e| = {abs_e} ≤ δ = {delta}: quadratic branch",
        "delta_state_lin": "|e| = {abs_e} > δ = {delta}: linear branch",

        "sum_error": "Error e = ŷ − y",
        "sum_abs": "Absolute |e|",
        "sum_loss": "Loss L",
        "sum_deriv": "Derivative dL/dŷ",
        "sum_state": "Prediction",
        "state_under": "below target",
        "state_equal": "exactly the target",
        "state_over": "above target",
        "graph_loss_title": "Loss L as a function of prediction ŷ",
        "graph_deriv_title": "Derivative dL/dŷ as a function of prediction ŷ",
        "graph_x_pred": "prediction ŷ",
        "graph_y_loss": "loss L",
        "graph_y_deriv": "dL/dŷ",
        "graph_legend": "{name} | y = {target} | ŷ = {pred} | L = {loss}",
        "graph_deriv_legend": "{name}: dL/dŷ",
        "graph_clip_loss": "the curve leaves the window at {limit}: only the visible part is drawn",
        "graph_clip_deriv": "values |dL/dŷ| > {limit} run off to infinity",
        "graph_target_line": "target y = {value}",
        "graph_transition": "dotted: |e| = δ",
        "graph_none": "the function is undefined for these values",
        "graph_ndiff": "at ŷ = y the MAE derivative does not exist",

        "tab_calc": "Calculation",
        "tab_compare": "Comparison",
        "tab_dataset": "Dataset",
        "tab_exp": "Experiments",
        "tab_help": "Help",

        "calc_intro": "Exactly what goes into the formula, step by step.",
        "calc_formula": "Formula",
        "calc_domain": "Valid input domain",
        "calc_target": "Target y",
        "calc_pred": "Prediction ŷ",
        "calc_error": "Error e = ŷ − y",
        "calc_sub": "Substitution",
        "calc_result": "Result",
        "calc_dformula": "Derivative formula",
        "calc_dsub": "Derivative substitution",
        "calc_dresult": "Derivative value",
        "calc_meaning": "Meaning",
        "calc_ndiff": "No ordinary derivative exists here: the left limit is -1, the right limit is +1.",
        "calc_subgradient": "The graph uses a numeric convention — the subgradient {value}. That is not a derivative.",
        "no_value": "does not exist",
        "calc_bce_na": "BCE is undefined: it needs 0 <= y <= 1 and 0 < ŷ < 1.",
        "calc_branch_quad": "quadratic branch (|e| <= δ)",
        "calc_branch_lin": "linear branch (|e| > δ)",
        "calc_zero_loss": "The prediction equals the target: loss is zero.",
        "calc_under": "The prediction is below the target: the error is negative, the derivative is negative, training pushes the prediction up.",
        "calc_over": "The prediction is above the target: the error is positive, the derivative is positive, training pulls the prediction down.",
        "calc_scale_mse": "The 1/2 factor is a convention of this simulator: it keeps the derivative equal to e without a leading 2. Without it the loss would be twice as large.",
        "calc_prob_note": "The probability stays inside (0, 1); 0 and 1 are excluded by the formula itself (ln 0 and division by 0).",

        "cmp_intro": "The same target and prediction for all four losses.",
        "cmp_values": "y = {target}, ŷ = {pred}, δ = {delta}",
        "cmp_table_title": "Properties of the loss functions",
        "cmp_graph_title": "All four functions on one graph",
        "cmp_column_title": "{name}: L = {loss}, dL/dŷ = {deriv}",
        "cmp_bce_na": "BCE does not apply to these numbers",
        "cmp_bce_na_detail": "BCE needs 0 <= y <= 1 and 0 < ŷ < 1; now y = {target}, ŷ = {pred}. No number is shown, so it cannot mislead.",
        "cmp_mse_mae": "MSE versus MAE",
        "cmp_mse_mae_note": "Same growth of the error: MSE rises quadratically, MAE linearly. Neither is better in general — the task decides.",
        "col_fn": "Function",
        "col_formula": "Formula",
        "col_domain": "Domain",
        "col_loss": "Loss L",
        "col_deriv": "dL/dŷ",
        "col_loss_behav": "Loss behaviour",
        "col_deriv_behav": "Derivative behaviour",
        "col_adv": "Main advantage",
        "col_lim": "Main limitation",
        "col_use": "Typical use",
        "name_mse": "MSE",
        "name_mae": "MAE",
        "name_bce": "BCE",
        "name_huber": "Huber",
        "domain_mse": "any real y and ŷ",
        "domain_mae": "any real y and ŷ",
        "domain_bce": "0 <= y <= 1, 0 < ŷ < 1",
        "domain_huber": "any real y and ŷ, δ > 0",
        "behav_loss_mse": "smooth paraboloid, grows fast for a large error",
        "behav_loss_mae": "linear growth, sharp minimum at zero error",
        "behav_loss_bce": "mild near a correct probability, explodes at the wrong edge",
        "behav_loss_huber": "quadratic inside |e| <= δ, linear beyond it",
        "behav_deriv_mse": "grows linearly with the error",
        "behav_deriv_mae": "step of +-1, breaks at zero error",
        "behav_deriv_bce": "hyperbola, huge at the probability edges",
        "behav_deriv_huber": "linear rise, then a plateau at +-delta",
        "adv_mse": "smooth, differentiable, easy to compute and differentiate",
        "adv_mae": "robust to outliers, units are easy to read",
        "adv_bce": "penalises confidence, matches probability",
        "adv_huber": "combines the smoothness of MSE with the robustness of MAE",
        "lim_mse": "one outlier dominates the loss",
        "lim_mae": "not differentiable at zero, gradient does not shrink near the minimum",
        "lim_bce": "probabilities only, unstable near 0 and 1",
        "lim_huber": "a second parameter, delta, has to be chosen",
        "use_mse": "regression, light noise, gradient training",
        "use_mae": "regression with outliers, interpretable metrics",
        "use_bce": "binary classification, class probability",
        "use_huber": "regression with outliers, robust optimisation",

        "ds_intro": "A few target and prediction pairs plus the average loss over them.",
        "ds_col_target": "Target y",
        "ds_col_pred": "Prediction ŷ",
        "ds_col_loss": "Loss for the row",
        "ds_col_error": "Error e",
        "ds_btn_add": "Add row",
        "ds_btn_remove": "Remove selected",
        "ds_btn_restore": "Restore example",
        "ds_avg": "Average loss over the dataset",
        "ds_avg_none": "no row applies to this loss function",
        "ds_row": "row {n}: target {target}, prediction {pred}",
        "ds_note": "A single row and the dataset average are different quantities. The average is the sum of the row losses divided by how many rows count.",
        "ds_bce_rows": "BCE only counts rows with 0 <= y <= 1 and 0 < ŷ < 1; the rest are skipped in the average.",
        "ds_row_na": "undefined",
        "ds_sum": "sum of losses = {sum}, counted rows = {count}",
        "ds_max_rows": "up to 8 rows: this is a teaching mode, not a training system",
        "ds_no_remove": "at least one row must stay",
        "ds_caption": "Function: {name}. This is NOT training: no network, epochs or weight updates appear here.",

        "exp_intro": "Each experiment sets the app state and explains one behaviour.",
        "exp_apply": "Apply",
        "exp_active": "Applied: {name}",
        "exp_perfect": "Perfect prediction",
        "exp_small": "Small error",
        "exp_large": "Large error",
        "exp_under": "Underprediction",
        "exp_over": "Overprediction",
        "exp_mse_mae": "MSE versus MAE",
        "exp_huber_near": "Huber near the transition",
        "exp_huber_far": "Huber with a large error",
        "exp_bce_right": "BCE: confident and correct",
        "exp_bce_wrong": "BCE: confident and wrong",
        "exp_bce_edge": "BCE: at the probability edge",
        "exp_text_perfect": "Target and prediction match: e = 0, MSE = 0, MAE = 0, Huber = 0. The MSE and Huber derivatives are zero too. MAE has no ordinary derivative at this point: the left limit is -1, the right limit is +1. BCE at y = 1 and ŷ = 1 is undefined — the probability 1 is not allowed by the formula.",
        "exp_text_small": "Error 0.15: MSE = 0.01125, MAE = 0.15. For a small error the quadratic loss is a tiny penalty while the linear one is visible. That is why the MAE gradient does not shrink near the minimum.",
        "exp_text_large": "Error 1.9: MSE = 1.805, MAE = 1.9. MSE was smaller than MAE at a small error, now they are comparable and MSE keeps growing faster: the square catches the line and passes it.",
        "exp_text_under": "The prediction is below the target: e is negative and the MSE derivative is negative. Training pushes the prediction up — the sign of the derivative gives the direction.",
        "exp_text_over": "The prediction is above the target: e is positive and the MSE derivative is positive. The training step points down, toward the target.",
        "exp_text_mse_mae": "Compare: L_MSE = {mse}, L_MAE = {mae}. Increase the prediction — MAE grows as a straight line, MSE as a parabola. The graph shows where they cross (|e| = 2 under our 1/2 convention).",
        "exp_text_huber_near": "|e| = 0.9 <= delta = 1.0: still the quadratic branch, the derivative equals e and grows linearly. Move delta or the prediction and watch the point cross the |e| = delta boundary.",
        "exp_text_huber_far": "|e| = 2.0 > delta = 1.0: the linear branch took over. Loss grows like delta * |e| while the derivative freezes on the +-delta plateau — the outlier cannot tear the gradient apart.",
        "exp_text_bce_right": "y = 1, ŷ = 0.95: a small penalty (L = {loss}) and a mild derivative. The model is confident about the right class, so training barely interferes.",
        "exp_text_bce_wrong": "y = 1, ŷ = 0.05: the model is confident about the wrong class. The penalty L = {loss} is tens of times larger than MSE gives for the same error. This is the point of BCE: punish confident being wrong.",
        "exp_text_bce_edge": "y = 0, ŷ = 0.001: almost zero, but ln(0.001) is finite, so the program never crashes. The slider stops at 0.001 and 0.999; as the probability goes to 0 the loss grows only logarithmically.",

        "help_intro": "16 teaching sections. Left: the function. Center: the graphs. Right: the breakdown.",
        "help_toc_title": "Sections",
        "help_nav_hint": "the Up and Down arrows work while the section list has focus",
        "h_title_1": "1. What a loss function is",
        "h_text_1": "A loss function L is a number that says how far the prediction is from the target. A small L means the prediction is close, a large L means it is far. The same error can be punished very differently: MSE gives 2.0 at e = 2 but only 0.02 at e = 0.2.",
        "h_title_2": "2. Target versus prediction",
        "h_text_2": "The target y is the right answer. The prediction ŷ is what the model produced, or what you type here. No network is trained in this simulator: you set both numbers and watch the result.",
        "h_title_3": "3. Error",
        "h_text_3": "The error e = ŷ - y carries a sign. e < 0 means the prediction is below the target (underprediction), e > 0 means above it (overprediction). The size |e| drives how big the penalty is, the sign drives which way training moves.",
        "h_title_4": "4. Why a model needs a loss",
        "h_text_4": "A model does not know what is right; it only knows the number L. Training takes steps that reduce L, so choosing a loss function is choosing which errors count as bad. Different losses steer the same training process differently.",
        "h_title_5": "5. MSE: mean squared error",
        "h_text_5": "L = 1/2 * (ŷ - y)^2 with derivative dL/dŷ = ŷ - y. The 1/2 factor is a convention of this simulator: it keeps the derivative free of a leading 2. Without it (as in some libraries) every loss value doubles, the shape stays the same. The curve is a parabola, smooth and differentiable everywhere.",
        "h_title_6": "6. MAE: mean absolute error",
        "h_text_6": "L = |ŷ - y| punishes the error in the units of the error itself. It grows linearly, so an outlier is not squared. The price is a cone with a sharp tip at zero instead of a smooth bowl.",
        "h_title_7": "7. Binary Cross-Entropy",
        "h_text_7": "L = -[ y ln(ŷ) + (1 - y) ln(1 - ŷ) ], where y is the true class (0 or 1) and ŷ the predicted probability. Domain: 0 <= y <= 1 and strictly 0 < ŷ < 1. At ln(0) the formula runs off to infinity, so the simulator limits probabilities to 0.001 and 0.999.",
        "h_title_8": "8. Huber loss",
        "h_text_8": "L = 1/2 e^2 when |e| <= δ and δ(|e| - 1/2 δ) otherwise. Inside the threshold it is exactly the MSE shape, outside it grows linearly. Derivative: e inside, ±δ outside. The transition is continuous in both value and slope.",
        "h_title_9": "9. The derivative of the loss",
        "h_text_9": "dL/dŷ says how much the loss changes if the prediction moves a little. Negative means the prediction should grow, positive means it should shrink, zero is a resting point (a minimum or a plateau).",
        "h_title_10": "10. Why derivatives matter in training",
        "h_text_10": "Gradient training moves parameters against the gradient, so the shape of the derivative curve sets the size and direction of every step: proportional to the error for MSE, constant in magnitude for MAE, enormous at the probability edges for BCE, capped at ±δ for Huber.",
        "h_title_11": "11. MSE versus MAE",
        "h_text_11": "Below |e| = 2 (with the 1/2 factor convention) MSE is smaller than MAE, above it MSE is larger and grows faster. MSE barely notices small errors and punishes big ones hard; MAE always punishes in proportion. That makes MSE sensitive to outliers and MAE robust but harder to differentiate. Which one is better depends on the task, not on the function.",
        "h_title_12": "12. BCE and confident wrongness",
        "h_text_12": "BCE compares a probability with a fact. Two equally wrong predictions are punished differently: an unsure mistake (y = 1, ŷ = 0.4) gives L = 0.916, a confident one (y = 1, ŷ = 0.05) gives L = 3.0. That is why BCE suits classification: it asks not only for the right class but for not being confident about the wrong one.",
        "h_title_13": "13. The Huber transition",
        "h_text_13": "The boundary |e| = delta is the whole idea of Huber. Small errors behave like MSE: the gradient is proportional and convergence is smooth. Large errors behave like MAE: the gradient freezes at +-delta and an outlier cannot drag training away. Changing delta decides which error counts as normal and which as an outlier.",
        "h_title_14": "14. MAE non-differentiability",
        "h_text_14": "At ŷ = y the MAE loss is zero, yet the ordinary derivative does not exist: the left limit is -1, the right limit is +1 and they disagree. Numerical libraries take a subgradient — any value in [-1, +1]; 0 is a common convention. This simulator shows the derivative as 'does not exist' and labels the subgradient used for the graph.",
        "h_title_15": "15. Sample loss versus average dataset loss",
        "h_text_15": "The loss of one target and prediction pair is the sample loss. The average over all pairs is the dataset loss: the sum of row losses divided by the number of rows. One bad row lifts the average, and the row alone says nothing about the model. The Dataset tab shows both side by side.",
        "h_title_16": "16. How to use the simulator",
        "h_text_16": "1. Pick a loss function on the left. 2. Set the target y. 3. Move the prediction with the slider, the +-0.05 buttons or the entry. 4. Watch the error, loss and derivative in the center and on both graphs. 5. Open Calculation to see the numbers substituted into the formula. 6. Comparison shows all four losses on the same numbers. 7. Experiments set ready-made values with an explanation.",

        "status_ready": "Ready: change the prediction and watch how each loss responds.",
        "status_loss": "Selected loss: {name}.",
        "status_preset": "Experiment \"{name}\" applied.",
        "status_reset": "State reset: target 1.0, prediction 1.6, delta 1.0.",
        "status_clamped": "The prediction was moved into the domain {lo} to {hi} of the selected loss.",
        "status_target_clamped": "The target must stay in [-3, 3]: the value was clamped.",
        "status_delta_clamped": "delta must stay in [0.1, 3.0]: the value was clamped.",
        "status_bce_na": "BCE is undefined for these numbers: see the Calculation tab.",
        "status_invalid": "Field \"{field}\": enter a number (comma or dot).",
        "status_lang": "Interface language switched, state preserved.",
        "status_ndiff": "Zero error: MAE has no ordinary derivative here.",
        "status_huber_switch": "Huber: |e| = {abs_e}, delta = {delta} — {branch}.",
        "status_ds_added": "Dataset row added.",
        "status_ds_removed": "Dataset row removed.",
        "status_ds_restored": "Dataset restored to the five-row example.",
    },
}


# ---------------------------------------------------------------
# 8. Текстовые панели: расчёт, сравнение, эксперимент, справка
# ---------------------------------------------------------------
# Панели собирают список пар (тег, строка): 'h' — заголовок, 'k' — ключ
# локализации, 't' — уже готовый текст. Теги нужны tk.Text для отступов.

def calc_lines(key, y, y_hat, delta):
    """Полный разбор: формула -> значения -> подстановка -> результат -> производная."""
    name = T("name_" + key)
    lines = [("h", name), ("t", T("calc_formula") + ": " + FORMULAS[key])]
    lines.append(("t", T("calc_domain") + ": " + T("domain_" + key)))

    if key == "bce" and not bce_applies(y, y_hat):
        lines.append(("t", T("calc_target") + ": y = " + fmt(y)))
        lines.append(("t", T("calc_pred") + ": p = " + fmt(y_hat)))
        lines.append(("b", T("calc_bce_na")))
        lines.append(("t", T("calc_prob_note")))
        return lines

    e = error(y_hat, y)
    loss = loss_value(key, y_hat, y, delta)
    lines.append(("t", "%s: y = %s" % (T("calc_target"), fmt(y))))
    lines.append(("t", "%s: ŷ = %s" % (T("calc_pred"), fmt(y_hat))))
    if key == "bce":
        lines.append(("t", T("calc_prob_note")))
    else:
        lines.append(("t", "%s: e = %s - %s = %s, |e| = %s" % (
            T("calc_error"), fmt(y_hat), fmt(y), fmt(e), fmt(abs(e)))))

    linear = key == "huber" and is_huber_linear(y_hat, y, delta)
    if key == "mse":
        sub = "L = 1/2 * (%s - %s)^2 = 1/2 * %s" % (fmt(y_hat), fmt(y), fmt(e * e, 4))
        dsub = "dL/dŷ = %s - %s = %s" % (fmt(y_hat), fmt(y), fmt(e))
    elif key == "mae":
        sub = "L = |%s - %s| = %s" % (fmt(y_hat), fmt(y), fmt(abs(e)))
        dsub = None
    elif key == "huber":
        branch = T("calc_branch_lin") if linear else T("calc_branch_quad")
        if linear:
            sub = "L = %s * (|%s| - 1/2 * %s) = %s * %s" % (
                fmt(delta), fmt(e), fmt(delta), fmt(delta), fmt(abs(e) - 0.5 * delta))
        else:
            sub = "L = 1/2 * (%s)^2 = 1/2 * %s" % (fmt(e), fmt(e * e, 4))
        dsub = "dL/dŷ = %s" % branch
    else:  # bce
        p = clip_probability(y_hat)
        t = clamp(y, 0.0, 1.0)
        sub = "L = -[ %s * ln(%s) + %s * ln(1 - %s) ]" % (
            fmt(t), fmt(p, 6), fmt(1.0 - t), fmt(p, 6))
        dsub = "dL/dŷ = -(%s / %s) + (%s / (1 - %s))" % (
            fmt(t), fmt(p, 6), fmt(1.0 - t), fmt(p, 6))
    lines.append(("t", T("calc_sub") + ": " + sub))
    lines.append(("k", "%s: L = %s" % (T("calc_result"), fmt_smart(loss))))

    if key == "mae":
        lines.append(("t", T("calc_dformula") + ": " + DERIV_FORMULAS[key]))
        if abs(e) < ZERO_EPS:
            lines.append(("b", T("calc_ndiff")))
            lines.append(("k", "%s: dL/dŷ = %s" % (T("calc_dresult"), T("no_value"))))
            lines.append(("t", T("calc_subgradient", value=fmt(mae_subgradient(y_hat, y)))))
        else:
            d = mae_deriv(y_hat, y)
            side = "+1" if d > 0 else "-1"
            lines.append(("t", "%s: dL/dŷ = %s (|e| = %s)" % (T("calc_dsub"), side, fmt(abs(e)))))
            lines.append(("k", "%s: dL/dŷ = %s" % (T("calc_dresult"), fmt(d))))
    else:
        d = loss_deriv(key, y_hat, y, delta)
        lines.append(("t", T("calc_dformula") + ": " + DERIV_FORMULAS[key]))
        if dsub is not None:
            lines.append(("t", "%s: %s" % (T("calc_dsub"), dsub)))
        lines.append(("k", "%s: dL/dŷ = %s" % (T("calc_dresult"), fmt_smart(d))))

    lines.append(("h", T("calc_meaning")))
    if key == "huber":
        lines.append(("t", T("status_huber_switch", abs_e=fmt(abs(e)),
                             delta=fmt(delta),
                             branch=T("calc_branch_lin") if linear else T("calc_branch_quad"))))
    if abs(e) < ZERO_EPS:
        lines.append(("t", T("calc_zero_loss")))
    elif e < 0.0:
        lines.append(("t", T("calc_under")))
    else:
        lines.append(("t", T("calc_over")))
    if key == "mse":
        lines.append(("t", T("calc_scale_mse")))
    return lines


def comparison_rows(y, y_hat, delta):
    """Строки таблицы сравнения: четыре функции на одних и тех же числах."""
    rows = []
    zero_error = abs(error(y_hat, y)) < ZERO_EPS
    for key in LOSS_ORDER:
        loss = loss_value(key, y_hat, y, delta)
        deriv = loss_deriv(key, y_hat, y, delta)
        applies = loss is not None
        if key == "mae" and zero_error:
            deriv_text = T("no_value")
        else:
            deriv_text = fmt_smart(deriv) if (applies and deriv is not None) else T("ds_row_na")
        rows.append({
            "key": key,
            "name": T("name_" + key),
            "formula": FORMULAS[key],
            "domain": T("domain_" + key),
            "loss": fmt_smart(loss) if applies else T("ds_row_na"),
            "deriv": deriv_text,
            "applies": applies,
        })
    return rows


def comparison_table_rows(y, y_hat, delta):
    """Большая таблица свойств (назначение, раздел 15)."""
    rows = []
    for key in LOSS_ORDER:
        rows.append((
            T("name_" + key),
            FORMULAS[key],
            T("domain_" + key),
            T("behav_loss_" + key),
            T("behav_deriv_" + key),
            T("adv_" + key),
            T("lim_" + key),
            T("use_" + key),
        ))
    return rows


def dataset_lines(rows, key, delta):
    """Потери каждой пары и среднее по выборке: sample loss против average dataset loss."""
    per_row, avg, count = sample_average(rows, key, delta)
    lines = [("t", T("ds_caption", name=T("name_" + key)))]
    total = 0.0
    for index, (y, y_hat) in enumerate(rows):
        value = per_row[index]
        if value is None:
            lines.append(("t", "%s | L = %s" % (
                T("ds_row", n=index + 1, target=fmt(y, 2), pred=fmt(y_hat, 2)),
                T("ds_row_na"))))
            continue
        total += value
        lines.append(("t", "%s | e = %s | L = %s" % (
            T("ds_row", n=index + 1, target=fmt(y, 2), pred=fmt(y_hat, 2)),
            fmt(error(y_hat, y), 3), fmt_smart(value))))
    lines.append(("h", "%s: %s" % (T("ds_avg"), fmt_smart(avg) if avg is not None else T("ds_avg_none"))))
    lines.append(("t", T("ds_sum", sum=fmt_smart(total), count=count)))
    lines.append(("t", T("ds_note")))
    if key == "bce":
        lines.append(("t", T("ds_bce_rows")))
    return lines


def experiment_text(exp_key, y, y_hat, delta):
    """Пояснение эксперимента с уже подставленными числами."""
    if exp_key == "mse_mae":
        return T("exp_text_mse_mae", mse=fmt_smart(mse_loss(y_hat, y)),
                 mae=fmt_smart(mae_loss(y_hat, y)))
    if exp_key in ("bce_right", "bce_wrong", "bce_edge"):
        return T("exp_text_" + exp_key, loss=fmt_smart(bce_loss(y_hat, y)))
    return T("exp_text_" + exp_key)


HELP_SECTIONS = tuple(str(i) for i in range(1, 17))


# ---------------------------------------------------------------
# 9. Виджеты: поле числа, холст-график, прокрутка
# ---------------------------------------------------------------

class NumberField(ttk.Frame):
    """Числовое поле с мягкой валидацией: символ, ввод, подсказка.

    Некорректный ввод не роняет приложение: поле подсвечивается, прежнее
    значение состояния сохраняется. Выход за границу приводит значение к
    ближайшей допустимой и сообщает об этом через callback.
    """

    def __init__(self, master, symbol, lo, hi, on_change, fonts, hint_key=None,
                 entry_width=9):
        super().__init__(master)
        self.lo, self.hi = lo, hi
        self.on_change = on_change
        self.fonts = fonts
        self.hint_key = hint_key
        self._lock = False
        self._bad = False
        self.symbol = ttk.Label(self, text=symbol, width=4, anchor="w")
        self.symbol.pack(side="left")
        self.var = tk.StringVar()
        self.entry = ttk.Entry(self, textvariable=self.var, width=entry_width,
                               justify="right", font=fonts["small"])
        self.entry.pack(side="left")
        if hint_key:
            self.hint = ttk.Label(self, text=T(hint_key), anchor="w",
                                  foreground="#6a6a63", font=fonts["tiny"])
            self._fit_hint()
            self.hint.pack(side="left", padx=(6, 0))
        self.entry.bind("<Return>", self._commit)
        self.entry.bind("<FocusOut>", self._commit)
        self.var.trace_add("write", self._typed)

    def _typed(self, *_args):
        if self._lock:
            return
        self._validate()

    def _validate(self):
        """Подсветка по текущему тексту: число не читается либо вне нынешнего диапазона."""
        raw = self.var.get()
        if raw == "":
            self._mark(False)
            return
        value = parse_number(raw)
        self._mark(value is None or not (self.lo <= value <= self.hi))

    def _commit(self, _event=None):
        if self._lock:
            return
        raw = self.var.get()
        value = parse_number(raw)
        if value is None:
            self._mark(True)
            self.on_change(None, raw, self)
            return
        clamped = False
        if value < self.lo or value > self.hi:
            value = clamp(value, self.lo, self.hi)
            clamped = True
        self._mark(False)
        self.set_value(value)
        self.on_change(value, raw, self, clamped)

    def _mark(self, bad):
        if bad == self._bad:
            return
        self._bad = bad
        self.entry.configure(style="BadEntry.TEntry" if bad else "TEntry")

    def set_value(self, value):
        self._lock = True
        self.var.set(to_text(value))
        self._lock = False
        self._mark(False)

    def set_domain(self, lo, hi):
        self.lo, self.hi = lo, hi
        # Диапазон меняется при выборе другой функции потерь. Прежняя подсветка
        # относится к старому диапазону, поэтому её нельзя сохранять: отметка
        # пересчитывается по текущему тексту и новым границам.
        self._validate()

    def _fit_hint(self):
        unit = self.fonts["tiny"].measure("0") or 1
        need = self.fonts["tiny"].measure(str(self.hint.cget("text"))) + 6
        self.hint.configure(width=max(int(math.ceil(need / float(unit))), 8))

    def retranslate(self):
        if self.hint_key:
            self.hint.configure(text=T(self.hint_key))
            self._fit_hint()


class PlotCanvas(tk.Canvas):
    """Ось X — предсказание, ось Y — потери или производная. Без matplotlib.

    Рисует сетку, деления, нулевые линии, несколько кривых, вертикальные
    ориентиры (цель, границы |e| = delta) и маркер текущего предсказания.
    """

    PAD_L = 62
    PAD_R = 14
    PAD_T = 40
    PAD_B = 46
    SAMPLES = 240

    def __init__(self, master, fonts, **kw):
        super().__init__(master, bg=COL_PAPER, highlightthickness=1,
                         highlightbackground=COL_FRAME, bd=0, **kw)
        self.fonts = fonts
        self._pending = None
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, _event):
        if self._pending:
            self._draw(*self._pending)

    def show(self, title, x_lo, x_hi, series, marker=None, vlines=(), notes=(),
             symmetric=False, y_cap=None):
        """series — [(сегменты, цвет, подпись)], marker — (x, y, подпись)."""
        self._pending = (title, x_lo, x_hi, series, marker, list(vlines),
                         list(notes), symmetric, y_cap)
        if self.winfo_width() > 10:
            self._draw(*self._pending)

    def _draw(self, title, x_lo, x_hi, series, marker, vlines, notes, symmetric, y_cap):
        c = self
        c.delete("all")
        w = max(c.winfo_width(), 320)
        h = max(c.winfo_height(), 170)
        left, right = self.PAD_L, w - self.PAD_R
        top, bottom = self.PAD_T, h - self.PAD_B
        c.create_text(8, 12, text=title, font=self.fonts["midb"], fill=COL_TEXT, anchor="w")
        if right - left < 60 or bottom - top < 40:
            return

        # Границы оси Y по всем кривым и по текущей точке.
        flat = [point[1] for seg, _col, _lab in series for chunk in seg for point in chunk]
        lo, hi = (0.0, 1.0)
        if flat:
            lo, hi = min(flat), max(flat)
        if marker is not None and marker[1] is not None and is_finite(marker[1]):
            lo, hi = min(lo, marker[1]), max(hi, marker[1])
        if symmetric:
            span = max(abs(lo), abs(hi), 0.5)
            lo, hi = -span, span
        else:
            lo = min(lo, 0.0)
            hi = max(hi, lo + 0.1)
        if y_cap is not None and hi > y_cap:
            hi = y_cap
            lo = -y_cap if symmetric else 0.0
        y_hi = nice_ceiling(hi)
        y_lo = -nice_ceiling(-lo) if lo < 0 else 0.0
        if y_hi - y_lo < 1e-9:
            y_hi = y_lo + 1.0

        def px(x):
            span = x_hi - x_lo
            return left + (right - left) * (float(x) - x_lo) / (span if span else 1.0)

        def py(value):
            return bottom - (bottom - top) * (float(value) - y_lo) / (y_hi - y_lo)

        step_y = nice_step(y_hi - y_lo, 5)
        for value in tick_values(y_lo, y_hi, step_y):
            y = py(value)
            c.create_line(left, y, right, y, fill=COL_GRID)
            c.create_text(left - 6, y, text=tick_label(value, step_y),
                          font=self.fonts["tiny"], fill="#5a5a53", anchor="e")
        step_x = nice_step(x_hi - x_lo, 6)
        for value in tick_values(x_lo, x_hi, step_x):
            x = px(value)
            c.create_line(x, bottom, x, bottom + 4, fill=COL_FRAME)
            c.create_text(x, bottom + 14, text=tick_label(value, step_x),
                          font=self.fonts["tiny"], fill="#5a5a53")

        if y_lo < 0 < y_hi:
            c.create_line(left, py(0.0), right, py(0.0), fill="#8d8d86", width=2)
        c.create_line(left, top, left, bottom, fill="#8d8d86")
        c.create_line(left, bottom, right, bottom, fill="#8d8d86")

        for x_value, color, label in vlines:
            if x_value is None or not (x_lo <= x_value <= x_hi):
                continue
            x = px(x_value)
            c.create_line(x, top, x, bottom, fill=color, dash=(4, 3))
            if label:
                c.create_text(x + 3, top + 6, text=label, font=self.fonts["tiny"],
                              fill=color, anchor="w")

        legend_x = left
        for seg, color, label in series:
            for chunk in seg:
                if len(chunk) > 1:
                    coords = [v for point in chunk for v in (px(point[0]), py(point[1]))]
                    c.create_line(*coords, fill=color, width=2)
            if not label:
                continue
            if legend_x < right - 40:
                c.create_line(legend_x, top - 12, legend_x + 14, top - 12, fill=color, width=3)
                c.create_text(legend_x + 18, top - 12, text=label, font=self.fonts["tiny"],
                              fill=COL_TEXT, anchor="w")
                legend_x += 26 + self.fonts["tiny"].measure(label)

        c.create_text((left + right) / 2.0, h - 12, text=T("graph_x_pred"),
                      font=self.fonts["small"], fill="#5a5a53")
        y_axis = T("graph_y_deriv") if symmetric else T("graph_y_loss")
        c.create_text(14, (top + bottom) / 2.0, text=y_axis, font=self.fonts["small"],
                      fill="#5a5a53", angle=90)
        note_y = 12
        for note in notes:
            note_y += 13
            c.create_text(right - 6, note_y, text=note, font=self.fonts["tiny"],
                          fill=COL_DIM, anchor="e")

        # Подпись точки рисуется последней и уходит в первый свободный сдвиг:
        # заголовок, легенда, метки вертикальных линий и плашки уже заняли своё
        # место в верхней полосе, и раньше подписи ложились друг на друга.
        reserved = [c.bbox(i) for i in c.find_all()
                    if c.type(i) == "text" and c.bbox(i)]
        if marker is not None:
            mx, my, text = marker
            label = "" if not text or text in notes else text
            if x_lo <= mx <= x_hi:
                if my is not None and y_lo <= my <= y_hi:
                    x, y = px(mx), py(my)
                    c.create_line(x, y, x, bottom, fill=COL_MARKER, dash=(2, 3))
                    c.create_oval(x - 5, y - 5, x + 5, y + 5, fill=COL_MARKER, outline="")
                    anchor = "e" if x > (left + right) / 2.0 else "w"
                    shift = -9 if anchor == "e" else 9
                    if label:
                        self._place_label(c, x + shift, y, label, anchor,
                                          (-12, 14, 27, -25), reserved)
                else:
                    c.create_line(px(mx), top, px(mx), bottom,
                                  fill=COL_MARKER, dash=(2, 3))
                    if label:
                        self._place_label(c, right - 4, top - 12, label, "e",
                                          (0, 13, 26, 39), reserved)

    def _place_label(self, c, x, y, text, anchor, offsets, reserved):
        """Первый из сдвигов, на котором подпись не задевает уже нарисованный текст."""
        for dy in offsets:
            item = c.create_text(x, y + dy, text=text, font=self.fonts["tiny"],
                                 fill=COL_MARKER, anchor=anchor)
            box = c.bbox(item)
            if box is None or self._free_box(box, reserved):
                if box:
                    reserved.append(box)
                return item
            c.delete(item)
        item = c.create_text(x, y + offsets[-1], text=text, font=self.fonts["tiny"],
                             fill=COL_MARKER, anchor=anchor)
        box = c.bbox(item)
        if box:
            reserved.append(box)
        return item

    @staticmethod
    def _free_box(box, reserved):
        for other in reserved:
            if (min(box[2], other[2]) - max(box[0], other[0]) > 2 and
                    min(box[3], other[3]) - max(box[1], other[1]) > 2):
                return False
        return True
# ---------------------------------------------------------------
# 10. Главное окно приложения
# ---------------------------------------------------------------

class App(object):
    """Три колонки: слева выбор функции и значения, в центре два графика и
    сводка, справа пять вкладок: расчёт, сравнение, датасет, эксперименты, справка."""

    def __init__(self, master):
        self.master = master
        self.fn_key = "mse"
        self.target = TARGET_DEFAULT
        self.pred = PRED_DEFAULT
        self.delta = DELTA_DEFAULT
        self.rows = list(DEFAULT_ROWS)
        self.exp_key = "small"
        self.help_index = 0
        self.ds_selected = 0
        self._text_refs = []
        self._radios = []
        self._tabs = []
        self._tables = {}
        self._table_need = {}
        self._right_floor = None
        self.left_min, self.center_min, self.right_min = 300, 470, 420
        self._slider_echo = None
        self._delta_echo = None
        self._suppress_pred = False
        self._suppress_delta = False

        master.title(T("app_title"))
        master.configure(bg=COL_BG)
        self._make_fonts()
        self._make_styles()
        self._build_header()
        self._build_left()
        self._build_center()
        self._build_right()
        self._build_bottom()
        # Левая колонка — стабильная панель управления, она не должна растягиваться.
        # Освободившееся пространство делят между собой графики и Notebook: при
        # развернутом окне оба растут пропорционально, а не только правая колонка.
        master.columnconfigure(0, minsize=self.left_min, weight=0)
        master.columnconfigure(1, minsize=self.center_min, weight=3)
        master.columnconfigure(2, minsize=self.right_min, weight=2)
        master.rowconfigure(1, weight=1)
        # Правая колонка дополнительно получает нижнюю границу по ширине самой
        # широкой своей таблицы: иначе на обычном рабочем столе 1920x1080
        # формулы в «Сравнении» обрезались бы, и пользователю пришлось бы
        # растягивать окно руками.
        master.bind("<Configure>", self._fit_columns, add="+")
        self._fit_columns()
        self.refresh()
        self.set_status("status_ready")

    # -- оформление --------------------------------------------------------
    def _make_fonts(self):
        base = tkfont.nametofont("TkDefaultFont").actual("family")
        self.fonts = {
            "title": tkfont.Font(family=base, size=16, weight="bold"),
            "sub": tkfont.Font(family=base, size=13),
            "midb": tkfont.Font(family=base, size=12, weight="bold"),
            "small": tkfont.Font(family=base, size=12),
            "smallb": tkfont.Font(family=base, size=12, weight="bold"),
            "tiny": tkfont.Font(family=base, size=10),
        }
        self.option_add("*Font", self.fonts["sub"])

    def _make_styles(self):
        st = ttk.Style(self.master)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=COL_BG, foreground=COL_TEXT)
        st.configure("TFrame", background=COL_BG)
        st.configure("TLabel", background=COL_BG, foreground=COL_TEXT)
        st.configure("TLabelframe", background=COL_BG, bordercolor=COL_FRAME)
        st.configure("TLabelframe.Label", background=COL_BG, foreground="#3a3a35",
                     font=self.fonts["midb"])
        st.configure("TNotebook", background=COL_BG, bordercolor=COL_FRAME)
        st.configure("TNotebook.Tab", padding=(9, 5), font=self.fonts["small"])
        st.configure("TButton", padding=(9, 5), font=self.fonts["small"])
        st.configure("Exp.TButton", padding=(7, 3), font=self.fonts["tiny"], anchor="w")
        st.configure("Go.TButton", padding=(7, 3), font=self.fonts["tiny"], anchor="w",
                     background=COL_PAPER, foreground=COL_ACTIVE,
                     bordercolor=COL_ACTIVE, lightcolor=COL_ACTIVE,
                     darkcolor=COL_ACTIVE)
        st.configure("TRadiobutton", background=COL_BG, font=self.fonts["small"])
        st.configure("Treeview", background=COL_PAPER, fieldbackground=COL_PAPER,
                     font=self.fonts["small"], rowheight=24)
        st.configure("Treeview.Heading", font=self.fonts["smallb"])
        st.configure("TEntry", fieldbackground=COL_PAPER)
        st.configure("BadEntry.TEntry", fieldbackground=COL_BAD_BG, foreground=COL_BAD_FG)

    def option_add(self, *args):
        self.master.option_add(*args)

    def _reg(self, widget, key, attr="text", **kw):
        self._text_refs.append((widget, key, attr, kw))
        return widget

    def set_status(self, key, **kw):
        self.var_status.set(T(key, **kw))

    # -- верхняя строка ----------------------------------------------------
    def _build_header(self):
        head = ttk.Frame(self.master)
        head.grid(row=0, column=0, columnspan=3, sticky="ew", padx=12, pady=(8, 2))
        self._reg(ttk.Label(head, text=T("app_title"), font=self.fonts["title"]),
                  "app_title", font=self.fonts["title"]).pack(side="left")
        self._reg(ttk.Label(head, text=T("subtitle"), font=self.fonts["small"],
                            foreground="#4a4a4a"), "subtitle",
                  font=self.fonts["small"]).pack(side="left", padx=(14, 0))
        lang = ttk.Frame(head)
        lang.pack(side="right")
        self.var_lang = tk.StringVar(value=LANG)
        for code in ("ru", "en"):
            radio = ttk.Radiobutton(lang, text=T("lang_" + code), value=code,
                                    variable=self.var_lang,
                                    command=lambda c=code: self.switch_language(c))
            radio.pack(side="left", padx=(0, 6))
            self._radios.append((radio, "lang_" + code, code))

    # -- левая колонка -----------------------------------------------------
    def _box(self, parent, title_key):
        box = ttk.LabelFrame(parent, text=T(title_key), padding=(8, 5))
        box.pack(fill="x", pady=(0, 6))
        self._text_refs.append((box, title_key, "text", {}))
        return box

    def _build_left(self):
        left = ttk.Frame(self.master)
        left.grid(row=1, column=0, sticky="ns", padx=(12, 2), pady=4)

        box = self._box(left, "box_loss")
        self.var_fn = tk.StringVar(value=self.fn_key)
        self.radio_loss = {}
        for index, key in enumerate(LOSS_ORDER):
            radio = ttk.Radiobutton(box, text=T("name_" + key), value=key,
                                    variable=self.var_fn,
                                    command=lambda k=key: self.select_loss(k))
            radio.grid(row=index // 2, column=index % 2, sticky="w", padx=2)
            self.radio_loss[key] = radio
            self._radios.append((radio, "name_" + key, key))
        self._reg(ttk.Label(box, text=T("note_pred_domain"), font=self.fonts["tiny"],
                            foreground="#6a6a63", wraplength=262, justify="left"),
                  "note_pred_domain", font=self.fonts["tiny"]).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        box = self._box(left, "box_target")
        self.field_target = NumberField(box, "y", TARGET_RANGE[0], TARGET_RANGE[1],
                                        self.on_target, self.fonts, "lbl_target_hint")
        self.field_target.pack(fill="x")
        self.field_target.set_value(self.target)

        box = self._box(left, "box_pred")
        self.field_pred = NumberField(box, "ŷ", *PRED_DOMAIN[self.fn_key],
                                      self.on_pred, self.fonts, "lbl_pred_hint")
        self.field_pred.pack(fill="x")
        self.field_pred.set_value(self.pred)
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(2, 0))
        self._reg(ttk.Label(row, text=T("lbl_slider"), font=self.fonts["tiny"],
                            foreground="#6a6a63"), "lbl_slider",
                  font=self.fonts["tiny"]).pack(side="left")
        self._reg(ttk.Label(row, text=T("hint_step"), font=self.fonts["tiny"],
                            foreground="#6a6a63"), "hint_step",
                  font=self.fonts["tiny"]).pack(side="right")
        self.slider_pred = tk.Scale(box, from_=PRED_DOMAIN[self.fn_key][0],
                                    to=PRED_DOMAIN[self.fn_key][1], resolution=0.01,
                                    orient="horizontal", showvalue=False, length=240,
                                    sliderrelief="flat", sliderlength=24, bd=0,
                                    highlightthickness=1, highlightbackground=COL_FRAME,
                                    font=self.fonts["tiny"], fg=COL_TEXT, bg=COL_BG,
                                    troughcolor=COL_PAPER, command=self._slider_pred)
        self.slider_pred.pack(fill="x", pady=(0, 3))
        self.set_slider_pred(self.pred)
        buttons = ttk.Frame(box)
        buttons.pack(fill="x")
        self.btn_down = self._reg(ttk.Button(buttons, text=T("btn_pred_down"),
                                             command=lambda: self._nudge(-0.05)),
                                  "btn_pred_down")
        self.btn_down.pack(side="left", padx=(0, 6))
        self.btn_up = self._reg(ttk.Button(buttons, text=T("btn_pred_up"),
                                           command=lambda: self._nudge(0.05)), "btn_pred_up")
        self.btn_up.pack(side="left")

        box = self._box(left, "box_delta")
        self.field_delta = NumberField(box, "δ", DELTA_RANGE[0], DELTA_RANGE[1],
                                       self.on_delta, self.fonts, "lbl_delta_hint")
        self.field_delta.pack(fill="x")
        self.field_delta.set_value(self.delta)
        self.slider_delta = tk.Scale(box, from_=DELTA_RANGE[0], to=DELTA_RANGE[1],
                                     resolution=0.05, orient="horizontal", length=240,
                                     showvalue=False,
                                     sliderrelief="flat", sliderlength=22, bd=0,
                                     highlightthickness=1, highlightbackground=COL_FRAME,
                                     font=self.fonts["tiny"], fg=COL_TEXT, bg=COL_BG,
                                     troughcolor=COL_PAPER, command=self._slider_delta)
        self.slider_delta.pack(fill="x", pady=(2, 0))
        self.set_slider_delta(self.delta)
        self.lbl_delta_state = ttk.Label(box, font=self.fonts["tiny"], foreground="#6a6a63",
                                         wraplength=262, justify="left")
        self.lbl_delta_state.pack(anchor="w", pady=(2, 0))

        box = self._box(left, "box_actions")
        self.combo_preset = ttk.Combobox(box, textvariable=tk.StringVar(),
                                         state="readonly", width=24, font=self.fonts["small"])
        self.combo_preset.pack(fill="x")
        self._reg(ttk.Label(box, text=T("combo_hint"), font=self.fonts["tiny"],
                            foreground="#6a6a63"), "combo_hint",
                  font=self.fonts["tiny"]).pack(anchor="w", pady=(2, 0))
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(4, 0))
        self.btn_apply = self._reg(ttk.Button(row, text=T("btn_apply"),
                                              command=self.apply_preset), "btn_apply")
        self.btn_apply.pack(side="left", padx=(0, 6))
        self.btn_reset = self._reg(ttk.Button(row, text=T("btn_reset"),
                                              command=self.reset_all), "btn_reset")
        self.btn_reset.pack(side="left")
        self._fill_preset_combo()
        self.combo_preset.set(T("exp_" + self.exp_key))

    def _fill_preset_combo(self):
        """Список экспериментов в поле слева: подписи берутся из локализации."""
        self.combo_preset.configure(values=[T("exp_" + item[0]) for item in EXPERIMENTS])
        self.combo_preset.current(0)

    # -- центральная колонка -----------------------------------------------
    def _build_center(self):
        center = ttk.Frame(self.master)
        center.grid(row=1, column=1, sticky="nsew", padx=2, pady=4)
        center.columnconfigure(0, weight=1)
        center.rowconfigure(1, weight=3)
        center.rowconfigure(2, weight=2)

        box = ttk.LabelFrame(center, text=T("sum_state"), padding=(8, 5))
        box.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self._text_refs.append((box, "sum_state", "text", {}))
        grid = ttk.Frame(box)
        grid.pack(fill="x")
        for column in range(4):
            grid.columnconfigure(column, weight=1)
        self.var_sum_error = tk.StringVar()
        self.var_sum_abs = tk.StringVar()
        self.var_sum_loss = tk.StringVar()
        self.var_sum_deriv = tk.StringVar()
        self.var_sum_state = tk.StringVar()
        cells = (("sum_error", self.var_sum_error), ("sum_abs", self.var_sum_abs),
                 ("sum_loss", self.var_sum_loss), ("sum_deriv", self.var_sum_deriv))
        for index, (key, var) in enumerate(cells):
            column = ttk.Frame(grid)
            column.grid(row=0, column=index, sticky="w", padx=(0, 6))
            self._reg(ttk.Label(column, text=T(key), font=self.fonts["tiny"],
                                foreground="#6a6a63"), key, font=self.fonts["tiny"]).pack(anchor="w")
            value = ttk.Label(column, textvariable=var, font=self.fonts["midb"])
            value.pack(anchor="w")
            if key == "sum_loss":
                self.lbl_sum_loss = value
        self.lbl_state = ttk.Label(box, textvariable=self.var_sum_state,
                                   font=self.fonts["smallb"])
        self.lbl_state.pack(anchor="w", pady=(4, 0))

        self.plot_loss = PlotCanvas(center, self.fonts)
        self.plot_loss.grid(row=1, column=0, sticky="nsew", pady=(0, 6))
        self.plot_deriv = PlotCanvas(center, self.fonts)
        self.plot_deriv.grid(row=2, column=0, sticky="nsew")

    # -- правая колонка ----------------------------------------------------
    def _build_right(self):
        right = ttk.Frame(self.master)
        right.grid(row=1, column=2, sticky="nsew", padx=(2, 12), pady=4)
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill="both", expand=True)
        self.tab_calc = ttk.Frame(self.nb)
        self.tab_cmp = ttk.Frame(self.nb)
        self.tab_ds = ttk.Frame(self.nb)
        self.tab_exp = ttk.Frame(self.nb)
        self.tab_help = ttk.Frame(self.nb)
        for frame, key in ((self.tab_calc, "tab_calc"), (self.tab_cmp, "tab_compare"),
                           (self.tab_ds, "tab_dataset"), (self.tab_exp, "tab_exp"),
                           (self.tab_help, "tab_help")):
            self.nb.add(frame, text=T(key))
            self._tabs.append((frame, key))

        self.text_calc = self._text_panel(self.tab_calc)

        # Сравнение: текущие числа, таблица свойств, мини-график четырёх функций.
        top = ttk.Frame(self.tab_cmp)
        top.pack(fill="x", pady=(6, 0))
        self.var_cmp_values = tk.StringVar()
        self._reg(ttk.Label(top, text=T("cmp_intro"), font=self.fonts["tiny"],
                            foreground="#6a6a63", wraplength=400, justify="left"),
                  "cmp_intro", font=self.fonts["tiny"], wraplength=400).pack(anchor="w")
        ttk.Label(top, textvariable=self.var_cmp_values, font=self.fonts["smallb"]).pack(anchor="w")
        columns = ("fn", "formula", "domain", "loss", "deriv")
        self.cmp_columns = columns
        # Держатель без propagate: ширина колонок не должна диктовать ширину
        # всей правой колонки, иначе развернутое окно съедает место графиков.
        cmp_holder = ttk.Frame(self.tab_cmp)
        cmp_holder.pack(fill="x", pady=(4, 4))
        cmp_holder.pack_propagate(False)
        self.tree_cmp = ttk.Treeview(cmp_holder, columns=columns, show="headings", height=4)
        for column in columns:
            self.tree_cmp.heading(column, text=T("col_" + column))
        self.tree_cmp.pack(fill="x")
        self._register_table(self.tree_cmp, columns, "col_", self._cmp_cells)
        cmp_holder.configure(width=420, height=self.tree_cmp.winfo_reqheight())
        self.plot_cmp = PlotCanvas(self.tab_cmp, self.fonts, height=190)
        self.plot_cmp.pack(fill="x", pady=(0, 4))
        self.text_cmp = self._text_panel(self.tab_cmp, height=8)

        # Датасет.
        ds_top = ttk.Frame(self.tab_ds)
        ds_top.pack(fill="x", pady=(6, 0))
        self._reg(ttk.Label(ds_top, text=T("ds_intro"), font=self.fonts["tiny"],
                            foreground="#6a6a63", wraplength=400, justify="left"),
                  "ds_intro", font=self.fonts["tiny"], wraplength=400).pack(anchor="w")
        ds_columns = ("target", "pred", "error", "loss")
        self.ds_columns = ds_columns
        ds_holder = ttk.Frame(self.tab_ds)
        ds_holder.pack(fill="x", pady=(4, 2))
        ds_holder.pack_propagate(False)
        self.tree_ds = ttk.Treeview(ds_holder, columns=ds_columns, show="headings", height=5)
        for column in ds_columns:
            self.tree_ds.heading(column, text=T("ds_col_" + column))
        self.tree_ds.pack(fill="x")
        self._register_table(self.tree_ds, ds_columns, "ds_col_")
        ds_holder.configure(width=420, height=self.tree_ds.winfo_reqheight())
        self.tree_ds.bind("<<TreeviewSelect>>", self._ds_select)
        ds_edit = ttk.Frame(self.tab_ds)
        ds_edit.pack(fill="x")
        self.var_ds_target = tk.StringVar()
        self.var_ds_pred = tk.StringVar()
        ttk.Entry(ds_edit, textvariable=self.var_ds_target, width=8,
                  font=self.fonts["small"]).pack(side="left")
        ttk.Entry(ds_edit, textvariable=self.var_ds_pred, width=8,
                  font=self.fonts["small"]).pack(side="left", padx=(4, 0))
        self.btn_ds_set = self._reg(ttk.Button(ds_edit, text=T("btn_apply"),
                                               command=self.ds_apply_edit), "btn_apply")
        self.btn_ds_set.pack(side="left", padx=(4, 0))
        ds_buttons = ttk.Frame(self.tab_ds)
        ds_buttons.pack(fill="x", pady=(3, 3))
        self.btn_ds_add = self._reg(ttk.Button(ds_buttons, text=T("ds_btn_add"),
                                               command=self.ds_add), "ds_btn_add")
        self.btn_ds_add.pack(side="left", padx=(0, 4))
        self.btn_ds_remove = self._reg(ttk.Button(ds_buttons, text=T("ds_btn_remove"),
                                                  command=self.ds_remove), "ds_btn_remove")
        self.btn_ds_remove.pack(side="left", padx=(0, 4))
        self.btn_ds_restore = self._reg(ttk.Button(ds_buttons, text=T("ds_btn_restore"),
                                                   command=self.ds_restore), "ds_btn_restore")
        self.btn_ds_restore.pack(side="left")
        self.var_ds_avg = tk.StringVar()
        ttk.Label(self.tab_ds, textvariable=self.var_ds_avg,
                  font=self.fonts["smallb"]).pack(anchor="w", pady=(2, 0))
        self.text_ds = self._text_panel(self.tab_ds, height=7)

        # Эксперименты.
        exp_head = ttk.Frame(self.tab_exp)
        exp_head.pack(fill="x", pady=(6, 2))
        self._reg(ttk.Label(exp_head, text=T("exp_intro"), font=self.fonts["tiny"],
                            foreground="#6a6a63", wraplength=400, justify="left"),
                  "exp_intro", font=self.fonts["tiny"], wraplength=400).pack(anchor="w")
        self.exp_buttons = {}
        holder = ttk.Frame(self.tab_exp)
        holder.pack(fill="x")
        for index, experiment in enumerate(EXPERIMENTS):
            key = experiment[0]
            button = ttk.Button(holder, text=T("exp_" + key), style="Exp.TButton",
                                command=lambda k=key: self.run_experiment(k))
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=2, pady=1)
            self.exp_buttons[key] = button
            self._radios.append((button, "exp_" + key, key))
        holder.columnconfigure(0, weight=1)
        holder.columnconfigure(1, weight=1)
        self.var_exp_active = tk.StringVar()
        ttk.Label(self.tab_exp, textvariable=self.var_exp_active, font=self.fonts["smallb"],
                  foreground=COL_ACTIVE).pack(anchor="w", pady=(3, 0))
        self.text_exp = self._text_panel(self.tab_exp, height=7)

        # Справка: весь список разделов 1..16 виден без прокрутки навигации.
        self.help_body = ttk.Frame(self.tab_help)
        self.help_list = tk.Listbox(self.tab_help, height=len(HELP_SECTIONS),
                                    width=self._help_nav_width(), exportselection=False,
                                    font=self.fonts["tiny"], bg=COL_PAPER,
                                    highlightthickness=1, highlightbackground=COL_FRAME,
                                    activestyle="none", selectbackground=COL_MARKER,
                                    selectforeground="#ffffff")
        self._help_mode = None
        self.tab_help.bind("<Configure>", self._help_arrange)
        self._help_arrange()
        self.help_list.bind("<<ListboxSelect>>", self._help_select)
        for index in range(len(HELP_SECTIONS)):
            self.help_list.insert("end", "  " + T("h_title_" + HELP_SECTIONS[index]))
        self.help_list.selection_set(self.help_index)
        self.text_help = self._text_panel(self.help_body, height=12)
        self._reg(ttk.Label(self.help_body, text=T("help_nav_hint"), font=self.fonts["tiny"],
                            foreground="#6a6a63", wraplength=222, justify="left"),
                  "help_nav_hint", font=self.fonts["tiny"], wraplength=222).pack(anchor="w", pady=(2, 0))

    def _text_panel(self, parent, height=12):
        # width задан явно: без него tk.Text просит 80 знаков (~930 пикселей),
        # и правая колонка требовала бы окно шире 1300x780.
        # Скроллбар — сосед текста в общем контейнере, а не ребёнок tk.Text:
        # во втором случае он лежит поверх холста и перекрывает правые 15 пикселей
        # перенесённых строк, потому что перенос считается по полной ширине Text.
        box = ttk.Frame(parent)
        box.pack(fill="both", expand=True)
        bar = ttk.Scrollbar(box, orient="vertical")
        bar.pack(side="right", fill="y")
        text = tk.Text(box, wrap="word", height=height, width=24,
                       bg=COL_PAPER, fg=COL_TEXT,
                       relief="flat", borderwidth=1, highlightthickness=1,
                       highlightbackground=COL_FRAME, padx=8, pady=6,
                       font=self.fonts["small"], takefocus=1,
                       yscrollcommand=bar.set)
        text.pack(side="left", fill="both", expand=True)
        bar.configure(command=text.yview)
        text.configure(state="disabled")
        text.tag_configure("h", font=self.fonts["smallb"], foreground=COL_FWD_HEAD,
                           spacing1=6, spacing2=2)
        text.tag_configure("t", foreground=COL_TEXT, spacing2=2)
        text.tag_configure("k", foreground=COL_MARKER, font=self.fonts["smallb"], spacing2=4)
        text.tag_configure("b", foreground=COL_NEG, font=self.fonts["small"], spacing2=4)
        return text

    def _help_nav_px(self):
        """По самой длинной подписи раздела: сколько пикселей нужно навигации."""
        return max(self.fonts["tiny"].measure("  " + T("h_title_" + key))
                   for key in HELP_SECTIONS) + 8

    def _help_nav_width(self):
        """Ширина навигации справки в знаках — пиксельная потребность в единицах нуля."""
        unit = self.fonts["tiny"].measure("0") or 1
        return int(math.ceil(self._help_nav_px() / float(unit)))

    def _help_arrange(self, *_args):
        """Навигация слева, текст справа — если на текст остаётся читаемая ширина.

        Иначе список ложится сверху на всю ширину вкладки: подписи разделов не
        обрезаются, а панель объяснений получает всё остальное место и прокручивается.
        """
        mode = "side" if self.tab_help.winfo_width() >= self._help_nav_px() + 260 else "stack"
        if mode == self._help_mode:
            return
        self._help_mode = mode
        self.help_list.pack_forget()
        self.help_body.pack_forget()
        self.help_list.configure(width=self._help_nav_width())
        if mode == "side":
            self.help_list.pack(side="left", fill="y", pady=(4, 0))
            self.help_body.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=(4, 0))
        else:
            self.help_list.pack(side="top", fill="x", pady=(4, 0))
            self.help_body.pack(side="top", fill="both", expand=True)

    def _cmp_cells(self):
        """Самые длинные значения колонок таблицы сравнения — по текущему языку."""
        rows = comparison_rows(TARGET_DEFAULT, PRED_DEFAULT, DELTA_DEFAULT)
        unknown = [T("ds_row_na"), T("no_value")]
        return {"fn": [row["name"] for row in rows],
                "formula": [row["formula"] for row in rows],
                "domain": [row["domain"] for row in rows] + unknown,
                "loss": [row["loss"] for row in rows] + unknown,
                "deriv": [row["deriv"] for row in rows] + unknown}

    def _fit_columns(self, *_args):
        """Нижняя граница правой колонки — ширина, нужная её самой широкой таблице.

        Граница поднимается только тогда, когда полный размер таблицы помещается,
        не отнимая у графиков их минимум: иначе колонка остаётся в обычном
        пропорциональном делении, и широкие формулы не вынуждают Tk обрезать ни
        графики, ни заголовки. Без этой границы на типовом рабочем столе 1920x1080
        формулы в «Сравнении» обрезались бы, и окно пришлось бы растягивать
        руками.
        """
        need = max([self.right_min] + list(self._table_need.values()))
        width = self.master.winfo_width()
        if width <= 1:
            return
        spare = width - self.left_min - self.center_min - 24
        wanted = need + 10          # закладка Notebook съедает несколько пикселей
        floor = wanted if wanted <= spare else self.right_min
        if floor != self._right_floor:
            self._right_floor = floor
            self.master.columnconfigure(2, minsize=floor)

    def _register_table(self, tree, columns, key_prefix, cells=None):
        self._tables[tree] = (columns, key_prefix, cells)
        tree.bind("<Configure>", lambda _event, t=tree: self._fit_table(t), add="+")
        self._fit_table(tree)

    def _fit_table(self, tree):
        """Колонки раскладываются по фактической ширине таблицы.

        Заголовок обязан читаться целиком (иначе пользователю придётся двигать
        разделитель), значения получают столько, сколько просят, а если всей
        таблицы не хватает по ширине — лишнее срезается пропорционально, но не
        ниже ширины заголовка. Развернутое окно использует всю ширину панели.
        """
        columns, key_prefix, cells = self._tables[tree]
        cells = cells() if callable(cells) else (cells or {})
        head_font, body_font = self.fonts["smallb"], self.fonts["small"]
        mins, needs = {}, {}
        for column in columns:
            head = int(math.ceil(head_font.measure(T(key_prefix + column)))) + 16
            need = head
            for text in cells.get(column, ()):
                need = max(need, int(math.ceil(body_font.measure(text))) + 16)
            mins[column], needs[column] = head, need
        available = tree.winfo_width() - 4
        total_need = sum(needs.values())
        self._table_need[tree] = total_need
        total_min = sum(mins.values())
        if available <= 1:
            widths = needs
        elif total_need <= available:
            factor = available / float(total_need)
            widths = {c: int(needs[c] * factor) for c in columns}
        else:
            spare = max(0, available - total_min)
            extra = float(total_need - total_min) or 1.0
            widths = {c: mins[c] + int(spare * (needs[c] - mins[c]) / extra)
                      for c in columns}
        for column in columns:
            tree.column(column, width=widths[column], minwidth=mins[column],
                        stretch=False, anchor="w")
        self._fit_columns()
        return sum(widths.values())

    def _fill_text(self, widget, lines):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        for tag, text in lines:
            widget.insert("end", text + "\n", tag)
        widget.configure(state="disabled")
        widget.see("1.0")

    # -- нижняя строка -----------------------------------------------------
    def _build_bottom(self):
        bar = ttk.Frame(self.master)
        bar.grid(row=2, column=0, columnspan=3, sticky="ew", padx=12, pady=(0, 6))
        # Имя автора постоянно, внизу справа, не переводится и не является ключом.
        ttk.Label(bar, text="SteveWorks001", font=self.fonts["smallb"],
                  foreground="#8a8a82").pack(side="right")
        self.var_status = tk.StringVar()
        self.lbl_status = ttk.Label(bar, textvariable=self.var_status, font=self.fonts["small"])
        self.lbl_status.pack(side="left", anchor="w")

    # -- ввод пользователя --------------------------------------------------
    def select_loss(self, key):
        if key == self.fn_key:
            return
        self.fn_key = key
        lo, hi = pred_domain_for(key, self.pred, self.target)
        if not (lo <= self.pred <= hi):
            clamped = clamp(self.pred, lo, hi)
            self.pred = clamped
            self.field_pred.set_value(clamped)
            self.set_slider_pred(clamped)
            self.set_status("status_clamped", lo=fmt(lo, 3), hi=fmt(hi, 3))
        else:
            self.set_status("status_loss", name=T("name_" + key))
        self.field_pred.set_domain(lo, hi)
        self._set_slider_range(lo, hi)
        self.refresh()

    def on_target(self, value, raw, field, clamped=False):
        if value is None:
            self.set_status("status_invalid", field=T("box_target"))
            return
        self.target = value
        if clamped:
            self.set_status("status_target_clamped")
        self.refresh()

    def on_pred(self, value, raw, field, clamped=False):
        if value is None:
            self.set_status("status_invalid", field=T("box_pred"))
            return
        self.pred = value
        if clamped:
            lo, hi = pred_domain_for(self.fn_key, value, self.target)
            self.set_status("status_clamped", lo=fmt(lo, 3), hi=fmt(hi, 3))
        self.set_slider_pred(value)
        self.refresh()

    def on_delta(self, value, raw, field, clamped=False):
        if value is None:
            self.set_status("status_invalid", field=T("box_delta"))
            return
        self.delta = value
        if clamped:
            self.set_status("status_delta_clamped")
        self.set_slider_delta(value)
        self.refresh()

    def _nudge(self, amount):
        value = clamp(round(self.pred + amount, 4), *pred_domain_for(self.fn_key, self.pred,
                                                                     self.target))
        self.pred = value
        self.field_pred.set_value(value)
        self.set_slider_pred(value)
        self.refresh()

    def _set_slider_range(self, lo, hi):
        self._suppress_pred = True
        self.slider_pred.configure(from_=lo, to=hi,
                                   resolution=0.001 if self.fn_key == "bce" else 0.01)
        self.slider_pred.set(self.pred)
        self._suppress_pred = False
        self._slider_echo = self.pred

    def set_slider_pred(self, value):
        self._suppress_pred = True
        self.slider_pred.set(value)
        self._suppress_pred = False
        self._slider_echo = value

    def set_slider_delta(self, value):
        self._suppress_delta = True
        self.slider_delta.set(value)
        self._suppress_delta = False
        self._delta_echo = value

    def _slider_pred(self, raw):
        if self._suppress_pred:
            return
        value = parse_number(raw)
        if value is None:
            return
        resolution = float(self.slider_pred.cget("resolution"))
        if self._slider_echo is not None and abs(value - self._slider_echo) <= 0.5 * resolution:
            return
        self.pred = value
        self._slider_echo = value
        self.field_pred.set_value(value)
        self.refresh()

    def _slider_delta(self, raw):
        if self._suppress_delta:
            return
        value = parse_number(raw)
        if value is None:
            return
        if self._delta_echo is not None and abs(value - self._delta_echo) <= 0.026:
            return
        self.delta = value
        self._delta_echo = value
        self.field_delta.set_value(value)
        self.refresh()

    # -- эксперименты -------------------------------------------------------
    def apply_preset(self):
        index = self.combo_preset.current()
        if index < 0:
            index = 0
        self.run_experiment(EXPERIMENTS[index][0])

    def run_experiment(self, key):
        for experiment in EXPERIMENTS:
            if experiment[0] != key:
                continue
            _name, fn_key, y, y_hat, delta = experiment
            self.fn_key = fn_key
            self.var_fn.set(fn_key)
            self.target = y
            self.delta = DELTA_DEFAULT if delta is None else delta
            lo, hi = pred_domain_for(fn_key, y_hat, y)
            self.pred = clamp(y_hat, lo, hi)
            self.field_target.set_value(self.target)
            self.field_delta.set_value(self.delta)
            self.set_slider_delta(self.delta)
            self.field_pred.set_domain(lo, hi)
            self._set_slider_range(lo, hi)
            self.field_pred.set_value(self.pred)
            self.exp_key = key
            self.combo_preset.set(T("exp_" + key))
            self.set_status("status_preset", name=T("exp_" + key))
            self.nb.select(self.tab_exp)
            self.refresh()
            return
        self.refresh()

    def reset_all(self):
        self.fn_key = "mse"
        self.var_fn.set("mse")
        self.target = TARGET_DEFAULT
        self.pred = PRED_DEFAULT
        self.delta = DELTA_DEFAULT
        self.rows = list(DEFAULT_ROWS)
        self.exp_key = "small"
        self.help_index = 0
        self.field_target.set_value(self.target)
        self.field_pred.set_domain(*PRED_DOMAIN["mse"])
        self._set_slider_range(*PRED_DOMAIN["mse"])
        self.field_pred.set_value(self.pred)
        self.field_delta.set_value(self.delta)
        self.set_slider_delta(self.delta)
        self.combo_preset.set(T("exp_small"))
        self.help_list.selection_clear(0, "end")
        self.help_list.selection_set(self.help_index)
        self.set_status("status_reset")
        self.refresh()

    # -- датасет ------------------------------------------------------------
    def _ds_select(self, _event=None):
        selected = self.tree_ds.selection()
        if selected:
            index = self.tree_ds.index(selected[0])
            self.ds_selected = index
            self.var_ds_target.set(to_text(self.rows[index][0]))
            self.var_ds_pred.set(to_text(self.rows[index][1]))

    def ds_apply_edit(self):
        y = parse_number(self.var_ds_target.get())
        y_hat = parse_number(self.var_ds_pred.get())
        if y is None or y_hat is None:
            self.set_status("status_invalid", field=T("tab_dataset"))
            return
        index = clamp(self.ds_selected, 0, len(self.rows) - 1)
        self.rows[index] = (clamp(y, TARGET_RANGE[0], TARGET_RANGE[1]),
                            clamp(y_hat, -3.0, 3.0))
        self.refresh()

    def ds_add(self):
        if len(self.rows) >= 8:
            self.set_status("ds_max_rows")
            return
        self.rows.append((0.0, 0.5))
        self.ds_selected = len(self.rows) - 1
        self.set_status("status_ds_added")
        self.refresh()

    def ds_remove(self):
        if len(self.rows) <= 1:
            self.set_status("ds_no_remove")
            return
        del self.rows[clamp(self.ds_selected, 0, len(self.rows) - 1)]
        self.ds_selected = clamp(self.ds_selected, 0, len(self.rows) - 1)
        self.set_status("status_ds_removed")
        self.refresh()

    def ds_restore(self):
        self.rows = list(DEFAULT_ROWS)
        self.ds_selected = 0
        self.set_status("status_ds_restored")
        self.refresh()

    def _help_select(self, _event=None):
        active = self.help_list.curselection()
        if active:
            self.help_index = active[0]
            self._refresh_help()

    # -- язык ---------------------------------------------------------------
    def switch_language(self, code):
        if code == LANG:
            return
        set_language(code)
        # Переключатель языка отражает текущий язык и при программном вызове,
        # а не только когда по нему ударили мышью.
        self.var_lang.set(LANG)
        self.master.title(T("app_title"))
        for widget, key, attr, kw in self._text_refs:
            try:
                widget.configure(**{attr: T(key, **kw)} if kw else {attr: T(key)})
            except tk.TclError:
                pass
        for widget, key, _value in self._radios:
            try:
                widget.configure(text=T(key))
            except tk.TclError:
                pass
        for frame, key in self._tabs:
            try:
                self.nb.tab(frame, text=T(key))
            except tk.TclError:
                pass
        for column in self.cmp_columns:
            self.tree_cmp.heading(column, text=T("col_" + column))
        self._fit_table(self.tree_cmp)
        for column in self.ds_columns:
            self.tree_ds.heading(column, text=T("ds_col_" + column))
        self._fit_table(self.tree_ds)
        for field in (self.field_target, self.field_pred, self.field_delta):
            field.retranslate()
        self.help_list.delete(0, "end")
        for index in range(len(HELP_SECTIONS)):
            self.help_list.insert("end", "  " + T("h_title_" + HELP_SECTIONS[index]))
        self.help_list.selection_clear(0, "end")
        self.help_list.selection_set(self.help_index)
        # Подписи разделов поменялись в длине, поэтому навигации нужен новый
        # размер и, возможно, другой способ компоновки вкладки.
        self._help_mode = None
        self._help_arrange()
        self._fill_preset_combo()
        self.combo_preset.set(T("exp_" + self.exp_key))
        self.set_status("status_lang")
        self.refresh()

    # -- обновление ----------------------------------------------------------
    def refresh(self):
        e = error(self.pred, self.target)
        loss = loss_value(self.fn_key, self.pred, self.target, self.delta)
        deriv = loss_deriv(self.fn_key, self.pred, self.target, self.delta)
        self.var_sum_error.set("e = " + fmt_smart(e))
        self.var_sum_abs.set("|e| = " + fmt_smart(abs(e)))
        self.var_sum_loss.set("L = " + (fmt_smart(loss) if loss is not None else T("ds_row_na")))
        if self.fn_key == "mae" and deriv is None:
            self.var_sum_deriv.set("dL/dŷ = " + T("no_value"))
        else:
            self.var_sum_deriv.set("dL/dŷ = " + (fmt_smart(deriv) if deriv is not None
                                                 else T("ds_row_na")))
        if abs(e) < ZERO_EPS:
            state_key, color = "state_equal", COL_TARGET
        elif e < 0.0:
            state_key, color = "state_under", COL_NEG
        else:
            state_key, color = "state_over", COL_FWD
        self.var_sum_state.set("%s: %s" % (T("sum_state"), T(state_key)))
        self.lbl_state.configure(foreground=color)
        self.lbl_sum_loss.configure(foreground=color)
        if loss is None:
            self.set_status("status_bce_na")
        elif self.fn_key == "mae" and deriv is None:
            self.set_status("status_ndiff")

        self._draw_main_plots(loss, deriv)
        self._refresh_calc(loss)
        self._refresh_compare()
        self._refresh_dataset()
        self._refresh_experiment()
        self._refresh_help()
        self._refresh_delta_state()

    def _refresh_delta_state(self):
        key = "delta_state_lin" if abs(error(self.pred, self.target)) > self.delta \
            else "delta_state_quad"
        self.lbl_delta_state.configure(text=T(key, abs_e=fmt(abs(error(self.pred, self.target)), 3),
                                              delta=fmt(self.delta, 2)))

    def _main_series(self, key, use_deriv=False):
        lo, hi = pred_domain_for(key, self.pred, self.target)
        gap = (self.target, 0.06) if (key == "mae" and use_deriv) else None
        segments = build_curve(key, self.target, self.delta, lo, hi,
                               use_deriv=use_deriv, samples=PlotCanvas.SAMPLES, gap=gap)
        return segments, (lo, hi)

    def _draw_main_plots(self, loss, deriv):
        key = self.fn_key
        segments, (lo, hi) = self._main_series(key)
        vlines = [(self.target, COL_TARGET, T("graph_target_line", value=fmt(self.target, 2)))]
        notes = []
        if key == "huber":
            for offset in (-self.delta, self.delta):
                x = self.target + offset
                if lo <= x <= hi:
                    vlines.append((x, COL_SECOND, ""))
            notes.append(T("graph_transition"))
        if not segments:
            notes.append(T("graph_none"))
        elif curve_limit_exceeded(key, self.target, self.delta, lo, hi):
            notes.append(T("graph_clip_loss", limit=fmt(LOSS_CLIP, 0)))
        loss_marker = (self.pred, loss, "L = " + fmt_smart(loss)) if loss is not None \
            else (self.pred, None, T("graph_none"))
        self.plot_loss.show(
            T("graph_loss_title"), lo, hi,
            [(segments, LOSS_COLOR[key], T("name_" + key))],
            marker=loss_marker, vlines=vlines, notes=notes, y_cap=LOSS_CLIP)

        dsegments, (dlo, dhi) = self._main_series(key, use_deriv=True)
        dnotes = []
        if key == "mae":
            dnotes.append(T("graph_ndiff"))
        if not dsegments:
            dnotes.append(T("graph_none"))
        elif curve_limit_exceeded(key, self.target, self.delta, dlo, dhi,
                                  use_deriv=True,
                                  gap=(self.target, 0.06) if key == "mae" else None):
            dnotes.append(T("graph_clip_deriv", limit=fmt(DERIV_CLIP, 0)))
        dvalue = deriv if deriv is not None else (
            mae_subgradient(self.pred, self.target) if key == "mae" else None)
        deriv_marker = (self.pred, dvalue,
                        "dL/dŷ = " + (fmt_smart(deriv) if deriv is not None else T("no_value")))
        self.plot_deriv.show(
            T("graph_deriv_title"), dlo, dhi,
            [(dsegments, LOSS_COLOR[key], T("graph_deriv_legend", name=T("name_" + key)))],
            marker=deriv_marker, vlines=vlines, notes=dnotes,
            symmetric=True, y_cap=DERIV_CLIP)

    def _refresh_calc(self, loss):
        self._fill_text(self.text_calc, calc_lines(self.fn_key, self.target, self.pred,
                                                   self.delta))

    def _refresh_compare(self):
        self.var_cmp_values.set(T("cmp_values", target=fmt(self.target),
                                  pred=fmt(self.pred), delta=fmt(self.delta)))
        rows = comparison_rows(self.target, self.pred, self.delta)
        self.tree_cmp.delete(*self.tree_cmp.get_children())
        for row in rows:
            self.tree_cmp.insert("", "end", values=(row["name"], row["formula"],
                                                    row["domain"], row["loss"], row["deriv"]),
                                 tags=("na",) if not row["applies"] else ())
        self.tree_cmp.tag_configure("na", foreground=COL_DIM)
        segments = []
        for key in LOSS_ORDER:
            _segments, (key_lo, key_hi) = self._main_series(key)
            if key == "bce":
                # Вероятности живут на (0, 1), остальное — на [-3, 3]:
                # переносим ось так, чтобы 0.5 попал в центр общего диапазона.
                scale = (PRED_DOMAIN["mse"][1] - PRED_DOMAIN["mse"][0]) / 2.0
                center = (PRED_DOMAIN["mse"][1] + PRED_DOMAIN["mse"][0]) / 2.0
                _segments = [[(center + (x - 0.5) * scale, value)
                              for x, value in chunk] for chunk in _segments]
                key_lo, key_hi = center - 0.5 * scale, center + 0.5 * scale
            segments.append((key, _segments, key_lo, key_hi))
        lo, hi = PRED_DOMAIN["mse"][0], PRED_DOMAIN["mse"][1]
        series = [(seg, LOSS_COLOR[key], T("name_" + key)) for key, seg, _l, _h in segments]
        notes = []
        if not bce_applies(self.target, clamp(self.pred, PROB_EDGE, 1.0 - PROB_EDGE)):
            notes.append(T("cmp_bce_na"))
        marker = loss_value(self.fn_key, self.pred, self.target, self.delta)
        self.plot_cmp.show(T("cmp_graph_title"), lo, hi, series,
                           marker=(self.pred, marker, "ŷ") if marker is not None else None,
                           vlines=[(self.target, COL_TARGET, "")], notes=notes,
                           y_cap=LOSS_CLIP)
        lines = [("h", T("cmp_table_title"))]
        for key in LOSS_ORDER:
            lines.append(("h", T("name_" + key) + " — " + FORMULAS[key]))
            for field in ("domain", "behav_loss", "behav_deriv", "adv", "lim", "use"):
                label_key = {"domain": "col_domain", "behav_loss": "col_loss_behav",
                             "behav_deriv": "col_deriv_behav", "adv": "col_adv",
                             "lim": "col_lim", "use": "col_use"}[field]
                lines.append(("t", "%s: %s" % (T(label_key), T(field + "_" + key))))
        lines.append(("b", T("cmp_mse_mae_note")))
        lines.append(("b", T("cmp_bce_na_detail", target=fmt(self.target, 2),
                             pred=fmt(self.pred, 2))))
        self._fill_text(self.text_cmp, lines)

    def _refresh_dataset(self):
        per_row, avg, count = sample_average(self.rows, self.fn_key, self.delta)
        self.tree_ds.delete(*self.tree_ds.get_children())
        for index, (y, y_hat) in enumerate(self.rows):
            value = per_row[index]
            self.tree_ds.insert("", "end", values=(to_text(y), to_text(y_hat),
                                                   fmt(error(y_hat, y), 3),
                                                   fmt_smart(value) if value is not None
                                                   else T("ds_row_na")),
                                tags=("na",) if value is None else ())
        self.tree_ds.tag_configure("na", foreground=COL_DIM)
        self.var_ds_avg.set("%s: %s" % (T("ds_avg"),
                                         fmt_smart(avg) if avg is not None else T("ds_avg_none")))
        if self.rows and not self.tree_ds.selection():
            children = self.tree_ds.get_children()
            if children:
                self.tree_ds.selection_set(children[clamp(self.ds_selected, 0, len(children) - 1)])
        self._fill_text(self.text_ds, dataset_lines(self.rows, self.fn_key, self.delta))

    def _refresh_experiment(self):
        self.var_exp_active.set(T("exp_active", name=T("exp_" + self.exp_key)))
        for key, button in self.exp_buttons.items():
            button.configure(style="Go.TButton" if key == self.exp_key else "Exp.TButton")
        lines = [("h", T("exp_" + self.exp_key)),
                 ("t", experiment_text(self.exp_key, self.target, self.pred, self.delta))]
        for experiment in EXPERIMENTS:
            if experiment[0] == self.exp_key:
                lines.append(("k", "%s: y = %s, ŷ = %s, δ = %s" % (
                    T("exp_apply"), fmt(experiment[2]), fmt(experiment[3]),
                    fmt(DELTA_DEFAULT if experiment[4] is None else experiment[4]))))
                break
        lines.append(("t", T("graph_legend", name=T("name_" + self.fn_key),
                             target=fmt(self.target, 2), pred=fmt(self.pred, 2),
                             loss=fmt_smart(loss_value(self.fn_key, self.pred, self.target,
                                                      self.delta)))))
        self._fill_text(self.text_exp, lines)

    def _refresh_help(self):
        index = HELP_SECTIONS[self.help_index]
        # Учебный текст объясняет предмет, а не автора: подпись разработчика
        # живёт внизу окна и в README, здесь её быть не должно.
        lines = [("h", T("h_title_" + index)), ("t", T("h_text_" + index))]
        self._fill_text(self.text_help, lines)

    def current_state(self):
        """Состояние для проверки: язык не должен ничего в нём менять."""
        return {"fn": self.fn_key, "target": self.target, "pred": self.pred,
                "delta": self.delta, "rows": list(self.rows), "exp": self.exp_key,
                "help": self.help_index, "loss": loss_value(self.fn_key, self.pred,
                                                            self.target, self.delta)}


# ---------------------------------------------------------------
# 11. Точка входа
# ---------------------------------------------------------------

def main():
    root = tk.Tk()
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    # Минимум = наименьший из поддерживаемых размеров, иначе окно 1366x768
    # молча поднималось бы до 780 и честная проверка размера была бы невозможна.
    root.minsize(min(1300, screen_w - 40), min(768, screen_h - 60))
    # Основной сценарий — развернутое окно. Крупную геометрию ставим заранее:
    # часть оконных менеджеров отказывает в программной максимизации, и тогда
    # приложению всё равно есть куда расти без ручного раздвигания панелей.
    root.geometry("%dx%d+%d+%d" % (
        max(1300, int(screen_w * 0.92)), max(780, int(screen_h * 0.90)),
        max(0, int(screen_w * 0.04)), max(0, int(screen_h * 0.05))))
    try:
        root.state("zoomed")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

