import re
import math
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="UAS - MIE 2026", layout="wide")

# ==============================================================================
# CONFIG
# ==============================================================================
DATA_EMP_PATH = "data/df_modulo-mie_sitc_20260413.xlsx"

COL_VISITA_ID = "IDENTIFICADOR DE VISITA"
COL_HOGAR = "CODIGO HOGAR"
COL_UT = "UNIDAD TERRITORIAL"
COL_DEP = "DEPARTAMENTO"
COL_PROV = "PROVINCIA"
COL_DIST = "DISTRITO"
COL_CP = "CENTRO POBLADO"
COL_DNI = "DNI_TITULAR"
COL_TITULAR = "TITULAR"
COL_ESTADO_VISITA = "ESTADO DE LA VISITA"
COL_FECHA_VISITA = "FECHA VISITA"

COL_Q_EMPR = "¿Usted o algún miembro de su hogar realiza un emprendimiento o actividad independiente para generar ingresos? (Ej.: venta, servicios, crianza, producción, negocio propio)"
COL_Q_TIPO_ACT = "¿Qué tipo de actividad realiza?"
COL_Q_QUIEN_EMPR = "¿Quién realiza principalmente esta actividad en el hogar?"
COL_Q_INTERES_CAP_EMPR = "¿Le gustaría recibir capacitación para mejorar su emprendimiento  o iniciar uno propio?"
COL_Q_TEMA_EMPR = "¿En qué tema le gustaría capacitarse para emprender?"
COL_Q_INTERES_EMPLEAB = "¿Usted o algún miembro de su hogar desearia capacitarse en algun tema para encontrar trabajo?"
COL_Q_QUIEN_EMPLEAB = "¿Quién del hogar tiene ese interés?"
COL_Q_TEMA_EMPLEAB = "¿En qué área o tema le gustaría que se le capacite a usted o algún miembro de su hogar para encontrar trabajo ?"
COL_Q_INGRESOS = "¿El hogar genera ingresos monetarios propios actualmente?"
COL_Q_ING_MONTO = "Aproximadamente, ¿cuál es el ingreso monetario mensual del hogar? (monto total)"
COL_Q_FUENTE_ING = "¿Cuál es la principal fuente de ingresos del hogar?"

# ==============================================================================
# PALETA / ESTILO
# ==============================================================================
P_DARK = "#1F5A53"
P_MID = "#4F7F7A"
P_LIGHT = "#A9C7C1"
P_PALE = "#D8E5E2"
P_TEXT = "#0F172A"
P_MUTED = "#5B6775"
P_GRID = "#E6EEEC"
P_BORDER = "#E5E7EB"

BLUE_DARK = "#1E3A8A"
BLUE_LIGHT = "#93C5FD"
BLUE_MID = "#3B82F6"

GREEN_DARK = "#166534"
GREEN_LIGHT = "#86EFAC"

AMBER_DARK = "#B45309"
AMBER_LIGHT = "#FCD34D"

st.markdown(
    f"""
    <style>
      .block-container {{
        padding-top: 1.0rem;
        padding-bottom: 1.2rem;
        max-width: 1500px;
      }}

      section[data-testid="stSidebar"] {{
        background: {P_PALE};
        padding-top: 0.6rem;
        border-right: 1px solid {P_BORDER};
      }}

      h1 {{
        font-size: 2.05rem;
        letter-spacing: -0.2px;
        margin-bottom: 0.2rem;
        color: {P_TEXT};
      }}

      .subtitle {{
        color: {P_MUTED};
        font-size: 0.95rem;
        margin-top: -0.2rem;
        margin-bottom: 0.7rem;
      }}

      .chart-title {{
        font-size: 1.02rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
        color: {P_TEXT};
      }}

      .chart-subtitle {{
        font-size: 0.84rem;
        color: {P_MUTED};
        margin-bottom: 0.45rem;
      }}

      .kpi-divider {{
        border: 0;
        height: 1px;
        background: {P_BORDER};
        margin: 0.9rem 0 1rem 0;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

ALT_THEME = {
    "config": {
        "view": {"stroke": "transparent"},
        "axis": {
            "labelFontSize": 11,
            "titleFontSize": 11,
            "labelColor": P_TEXT,
            "titleColor": P_TEXT,
            "gridColor": P_GRID,
            "tickColor": P_GRID,
        },
        "legend": {
            "labelFontSize": 11,
            "titleFontSize": 11,
            "labelColor": P_TEXT,
            "titleColor": P_TEXT,
        },
        "mark": {"tooltip": True},
    }
}
alt.themes.register("mie_theme", lambda: ALT_THEME)
alt.themes.enable("mie_theme")

# ==============================================================================
# HELPERS
# ==============================================================================
LOWER_WORDS = {
    "de", "del", "la", "las", "el", "los", "y", "e", "o", "u",
    "en", "para", "por", "a", "al", "con", "sin", "un", "una"
}
ACRONYMS = {"UT", "ONG", "MIDIS", "MIMP", "MINSA", "RENIEC", "SUNAT", "PNP", "UPN", "HW/NJ", "HW", "NJ", "UAS", "CSE"}

def _smart_title_case(s):
    if pd.isna(s):
        return pd.NA
    s = str(s).strip()
    if not s:
        return pd.NA
    s = re.sub(r"_+", " ", s)
    s = re.sub(r"\s+", " ", s)
    parts = re.split(r"(\s+|/|-)", s)
    out = []
    first_word_done = False
    for p in parts:
        if p.isspace() or p in {"/", "-"}:
            out.append(p)
            continue
        raw = p.strip()
        if not raw:
            continue
        if raw.upper() in ACRONYMS:
            out.append(raw.upper())
            first_word_done = True
            continue
        low = raw.lower()
        if low in LOWER_WORDS and first_word_done:
            out.append(low)
        else:
            out.append(low.capitalize())
            first_word_done = True
    return "".join(out)

def _clean_key(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "NAN": pd.NA,
            "<NA>": pd.NA,
            "None": pd.NA,
        })
    )

def safe_pct(num, den):
    return (num / den * 100) if den and den > 0 else 0.0

def norm_bin(x):
    if pd.isna(x):
        return pd.NA
    x = str(x).strip().lower()
    if x.startswith("si"):
        return 1
    if x.startswith("no"):
        return 0
    return pd.NA

def card_title(txt: str, subtitle: str | None = None):
    st.markdown(f'<div class="chart-title">{txt}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="chart-subtitle">{subtitle}</div>', unsafe_allow_html=True)

def add_bar_labels(chart_df: pd.DataFrame, y_col: str, x_col: str, fmt: str = ",", dx: int = 4, sort_val="-x"):
    return (
        alt.Chart(chart_df)
        .mark_text(align="left", dx=dx, fontSize=11, color=P_TEXT)
        .encode(
            y=alt.Y(f"{y_col}:N", sort=sort_val),
            x=alt.X(f"{x_col}:Q"),
            text=alt.Text(f"{x_col}:Q", format=fmt),
        )
    )

def make_horizontal_bar(df_plot, y_col, x_col, color, title=None, subtitle=None, height=320,
                        label_limit=260, sort_val="-x", x_domain=None, fmt=".1f",
                        tooltip_title_y=None, tooltip_title_x=None):
    if title:
        card_title(title, subtitle)

    x_enc = alt.X(f"{x_col}:Q", title=None)
    if x_domain is not None:
        x_enc = alt.X(f"{x_col}:Q", title=None, scale=alt.Scale(domain=x_domain))

    tooltip_y = alt.Tooltip(f"{y_col}:N", title=tooltip_title_y or y_col)

    if tooltip_title_x and "%" in tooltip_title_x:
        tooltip_x = alt.Tooltip(f"{x_col}:Q", title=tooltip_title_x, format=fmt)
    else:
        tooltip_x = alt.Tooltip(f"{x_col}:Q", title=tooltip_title_x or x_col)

    bars = (
        alt.Chart(df_plot)
        .mark_bar(color=color)
        .encode(
            y=alt.Y(f"{y_col}:N", title=None, sort=sort_val, axis=alt.Axis(labelLimit=label_limit, labelOverlap=False)),
            x=x_enc,
            tooltip=[tooltip_y, tooltip_x],
        )
        .properties(height=height)
    )

    labels = (
        alt.Chart(df_plot)
        .mark_text(align="left", dx=4, fontSize=11, color=P_TEXT)
        .encode(
            y=alt.Y(f"{y_col}:N", sort=sort_val),
            x=x_enc,
            text=alt.Text(f"{x_col}:Q", format=fmt),
        )
    )

    return bars + labels

# ==============================================================================
# CARGA
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_emp(path: str) -> pd.DataFrame:
    if path.lower().endswith(".xls"):
        return pd.read_excel(path, engine="xlrd")
    return pd.read_excel(path)

df_emp_raw = load_emp(DATA_EMP_PATH).copy()

# ==============================================================================
# BASE 2026 PRINCIPAL
# ==============================================================================
@st.cache_data(show_spinner=False)
def build_df_diag_2026(df_emp: pd.DataFrame) -> pd.DataFrame:
    dfx = df_emp.copy()
    dfx.columns = dfx.columns.str.strip()

    dfx[COL_HOGAR] = _clean_key(dfx[COL_HOGAR])
    dfx[COL_UT] = _clean_key(dfx[COL_UT]).str.upper()
    dfx[COL_DEP] = _clean_key(dfx[COL_DEP]).map(_smart_title_case)
    dfx[COL_PROV] = _clean_key(dfx[COL_PROV]).map(_smart_title_case)
    dfx[COL_DIST] = _clean_key(dfx[COL_DIST]).map(_smart_title_case)
    dfx[COL_CP] = _clean_key(dfx[COL_CP]).map(_smart_title_case)
    dfx[COL_DNI] = _clean_key(dfx[COL_DNI])
    dfx[COL_TITULAR] = _clean_key(dfx[COL_TITULAR]).map(_smart_title_case)
    dfx[COL_ESTADO_VISITA] = _clean_key(dfx[COL_ESTADO_VISITA]).str.upper()
    dfx[COL_FECHA_VISITA] = pd.to_datetime(dfx[COL_FECHA_VISITA], errors="coerce", dayfirst=True)

    dfx["tiene_emprendimiento"] = dfx[COL_Q_EMPR].apply(norm_bin)
    dfx["interes_cap_empr"] = dfx[COL_Q_INTERES_CAP_EMPR].apply(norm_bin)
    dfx["interes_empleabilidad"] = dfx[COL_Q_INTERES_EMPLEAB].apply(norm_bin)
    dfx["genera_ingresos"] = dfx[COL_Q_INGRESOS].apply(norm_bin)

    dfx["tipo_actividad"] = _clean_key(dfx[COL_Q_TIPO_ACT]).map(_smart_title_case)
    dfx["quien_realiza_empr"] = _clean_key(dfx[COL_Q_QUIEN_EMPR]).map(_smart_title_case)
    dfx["tema_cap_empr"] = _clean_key(dfx[COL_Q_TEMA_EMPR]).map(_smart_title_case)
    dfx["quien_interes_empleab"] = _clean_key(dfx[COL_Q_QUIEN_EMPLEAB]).map(_smart_title_case)
    dfx["tema_empleabilidad"] = _clean_key(dfx[COL_Q_TEMA_EMPLEAB]).map(_smart_title_case)
    dfx["fuente_ingresos"] = _clean_key(dfx[COL_Q_FUENTE_ING]).map(_smart_title_case)
    dfx["ingreso_mensual"] = pd.to_numeric(dfx[COL_Q_ING_MONTO], errors="coerce")

    dfx = dfx[dfx[COL_ESTADO_VISITA] == "EFECTIVA"].copy()
    dfx = dfx.sort_values([COL_FECHA_VISITA, COL_VISITA_ID], ascending=[False, False])
    dfx = dfx.drop_duplicates(subset=[COL_HOGAR], keep="first").copy()

    dfx["ut_label"] = dfx[COL_UT].map(_smart_title_case)
    dfx["hogar_id"] = dfx[COL_HOGAR]

    return dfx

df_diag_2026 = build_df_diag_2026(df_emp_raw)

# ==============================================================================
# FILTROS JERÁRQUICOS
# ==============================================================================
ut_options = ["Todos"] + sorted(df_diag_2026["ut_label"].dropna().unique().tolist())

with st.sidebar:
    st.markdown("### Filtros")

    ut_choice = st.selectbox("UT", ut_options, index=0)

    if ut_choice == "Todos":
        prov_options = ["Todos"]
        prov_choice = st.selectbox("Provincia", prov_options, index=0, disabled=True)

        dist_options = ["Todos"]
        dist_choice = st.selectbox("Distrito", dist_options, index=0, disabled=True)

        cp_options = ["Todos"]
        cp_choice = st.selectbox("Centro poblado", cp_options, index=0, disabled=True)

    else:
        df_ut = df_diag_2026[df_diag_2026["ut_label"] == ut_choice].copy()

        prov_options = ["Todos"] + sorted(df_ut[COL_PROV].dropna().unique().tolist())
        prov_choice = st.selectbox("Provincia", prov_options, index=0)

        if prov_choice == "Todos":
            df_ut_prov = df_ut.copy()
        else:
            df_ut_prov = df_ut[df_ut[COL_PROV] == prov_choice].copy()

        dist_options = ["Todos"] + sorted(df_ut_prov[COL_DIST].dropna().unique().tolist())
        dist_choice = st.selectbox("Distrito", dist_options, index=0)

        if dist_choice == "Todos":
            df_ut_prov_dist = df_ut_prov.copy()
        else:
            df_ut_prov_dist = df_ut_prov[df_ut_prov[COL_DIST] == dist_choice].copy()

        cp_options = ["Todos"] + sorted(df_ut_prov_dist[COL_CP].dropna().unique().tolist())
        cp_choice = st.selectbox("Centro poblado", cp_options, index=0)

def apply_filters_2026(dfx: pd.DataFrame) -> pd.DataFrame:
    out = dfx.copy()

    if ut_choice != "Todos":
        out = out[out["ut_label"] == ut_choice]

    if prov_choice != "Todos":
        out = out[out[COL_PROV] == prov_choice]

    if dist_choice != "Todos":
        out = out[out[COL_DIST] == dist_choice]

    if cp_choice != "Todos":
        out = out[out[COL_CP] == cp_choice]

    return out

df_2026 = apply_filters_2026(df_diag_2026)

# ==============================================================================
# HEADER
# ==============================================================================
tcol1, tcol2 = st.columns([6, 1], vertical_alignment="center")

with tcol1:
    st.title("Mi Independencia Económica 2026")
    st.markdown(
        '<div class="subtitle">Monitoreo de identificación de emprendimientos y demanda territorial</div>',
        unsafe_allow_html=True
    )

with tcol2:
    st.markdown("<div style='height: 0.65rem;'></div>", unsafe_allow_html=True)
    st.link_button(
        "⬇️ Descargar Padrones por UT",
        "https://drive.google.com/drive/folders/17YP__o3u63BIHN01ljX8CcNJkjgofPhC",
        use_container_width=True
    )

# ==============================================================================
# KPIs 2026
# ==============================================================================
hogares_visitados = int(df_2026["hogar_id"].nunique())
hogares_empr = int(df_2026[df_2026["tiene_emprendimiento"] == 1]["hogar_id"].nunique())
pct_hogares_empr = safe_pct(hogares_empr, hogares_visitados)

hogares_interes_cap = int(df_2026[df_2026["interes_cap_empr"] == 1]["hogar_id"].nunique())
pct_hogares_interes_cap = safe_pct(hogares_interes_cap, hogares_empr)

hogares_interes_empleab = int(df_2026[df_2026["interes_empleabilidad"] == 1]["hogar_id"].nunique())
pct_hogares_interes_empleab = safe_pct(hogares_interes_empleab, hogares_visitados)

hogares_ingresos = int(df_2026[df_2026["genera_ingresos"] == 1]["hogar_id"].nunique())
pct_hogares_ingresos = safe_pct(hogares_ingresos, hogares_visitados)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Hogares visitados", f"{hogares_visitados:,}")
c2.metric("Hogares con emprendimiento", f"{hogares_empr:,}", f"{pct_hogares_empr:.1f}%")
c3.metric("Interés en capacitación", f"{hogares_interes_cap:,}", f"{pct_hogares_interes_cap:.1f}%")
c4.metric("Interés en empleabilidad", f"{hogares_interes_empleab:,}", f"{pct_hogares_interes_empleab:.1f}%")
c5.metric("Hogares con ingresos propios", f"{hogares_ingresos:,}", f"{pct_hogares_ingresos:.1f}%")

st.markdown('<hr class="kpi-divider"/>', unsafe_allow_html=True)

# ==============================================================================
# TERRITORIAL DINÁMICO CONTINUO
# ==============================================================================
show_geo_chart = cp_choice == "Todos"

if show_geo_chart:
    if ut_choice == "Todos":
        geo_group = "ut_label"
        subtitle = "Por UT"
        geo_label = "UT"
    elif prov_choice == "Todos":
        geo_group = COL_PROV
        subtitle = "Por provincia"
        geo_label = "Provincia"
    elif dist_choice == "Todos":
        geo_group = COL_DIST
        subtitle = "Por distrito"
        geo_label = "Distrito"
    else:
        geo_group = COL_CP
        subtitle = "Por centro poblado"
        geo_label = "Centro poblado"

    geo_res = (
        df_2026.groupby(geo_group)
        .agg(
            hogares=("hogar_id", "nunique"),
            hogares_empr=("tiene_emprendimiento", "sum"),
        )
        .reset_index()
        .dropna(subset=[geo_group])
    )
    geo_res["pct_empr"] = np.where(geo_res["hogares"] > 0, geo_res["hogares_empr"] / geo_res["hogares"] * 100, 0)
    geo_res = geo_res.sort_values("pct_empr", ascending=False).reset_index(drop=True)

    n_geo = len(geo_res)
    max_pct = float(geo_res["pct_empr"].max()) if n_geo > 0 else 0.0
    x_domain = [0, max(5.0, max_pct * 1.08)]
    threshold_single = 12

    if n_geo <= threshold_single:
        card_title("% hogares con emprendimiento", subtitle)
        h_geo = max(280, 26 * n_geo)
        chart = make_horizontal_bar(
            geo_res,
            y_col=geo_group,
            x_col="pct_empr",
            color=BLUE_DARK,
            height=h_geo,
            label_limit=260,
            sort_val="-x",
            x_domain=x_domain,
            fmt=".1f",
            tooltip_title_y=geo_label,
            tooltip_title_x="% emprendimiento",
        )
        st.altair_chart(chart, use_container_width=True)

    else:
        split_idx = math.ceil(n_geo / 2)
        geo_left = geo_res.iloc[:split_idx].copy()
        geo_right = geo_res.iloc[split_idx:].copy()

        row_geo = st.columns(2, gap="large")

        with row_geo[0]:
            card_title("% hogares con emprendimiento", subtitle)
            h_left = max(280, 26 * len(geo_left))
            chart = make_horizontal_bar(
                geo_left,
                y_col=geo_group,
                x_col="pct_empr",
                color=BLUE_DARK,
                height=h_left,
                label_limit=260,
                sort_val="-x",
                x_domain=x_domain,
                fmt=".1f",
                tooltip_title_y=geo_label,
                tooltip_title_x="% emprendimiento",
            )
            st.altair_chart(chart, use_container_width=True)

        with row_geo[1]:
            st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
            h_right = max(280, 26 * len(geo_right))
            chart = make_horizontal_bar(
                geo_right,
                y_col=geo_group,
                x_col="pct_empr",
                color=BLUE_DARK,
                height=h_right,
                label_limit=260,
                sort_val="-x",
                x_domain=x_domain,
                fmt=".1f",
                tooltip_title_y=geo_label,
                tooltip_title_x="% emprendimiento",
            )
            st.altair_chart(chart, use_container_width=True)

    st.markdown('<hr class="kpi-divider"/>', unsafe_allow_html=True)

# ==============================================================================
# CARACTERIZACIÓN Y DEMANDA
# ==============================================================================
row2 = st.columns(3, gap="large")

df_empr = df_2026[df_2026["tiene_emprendimiento"] == 1].copy()

tipo_emp = (
    df_empr.groupby("tipo_actividad")["hogar_id"]
    .nunique()
    .reset_index(name="hogares")
    .dropna()
    .sort_values("hogares", ascending=False)
    .head(10)
)

with row2[0]:
    card_title("Tipos de actividad de emprendimiento", "Top 10 actividades económicas detectadas")
    chart = make_horizontal_bar(
        tipo_emp,
        y_col="tipo_actividad",
        x_col="hogares",
        color=BLUE_MID,
        height=320,
        label_limit=240,
        sort_val="-x",
        fmt=",",
        tooltip_title_y="Tipo de actividad",
        tooltip_title_x="Hogares",
    )
    st.altair_chart(chart, use_container_width=True)

tema_empr = (
    df_2026[df_2026["interes_cap_empr"] == 1]
    .groupby("tema_cap_empr")["hogar_id"]
    .nunique()
    .reset_index(name="hogares")
    .dropna()
    .sort_values("hogares", ascending=False)
    .head(10)
)

with row2[1]:
    card_title("Temas más solicitados para emprender", "Top 10 temas demandados para mejorar o iniciar emprendimientos")
    chart = make_horizontal_bar(
        tema_empr,
        y_col="tema_cap_empr",
        x_col="hogares",
        color=GREEN_DARK,
        height=320,
        label_limit=240,
        sort_val="-x",
        fmt=",",
        tooltip_title_y="Tema",
        tooltip_title_x="Hogares",
    )
    st.altair_chart(chart, use_container_width=True)

tema_empleab = (
    df_2026[df_2026["interes_empleabilidad"] == 1]
    .groupby("tema_empleabilidad")["hogar_id"]
    .nunique()
    .reset_index(name="hogares")
    .dropna()
    .sort_values("hogares", ascending=False)
    .head(10)
)

with row2[2]:
    card_title("Temas más solicitados para empleabilidad", "Top 10 áreas demandadas para inserción laboral")
    chart = make_horizontal_bar(
        tema_empleab,
        y_col="tema_empleabilidad",
        x_col="hogares",
        color=GREEN_DARK,
        height=320,
        label_limit=240,
        sort_val="-x",
        fmt=",",
        tooltip_title_y="Tema",
        tooltip_title_x="Hogares",
    )
    st.altair_chart(chart, use_container_width=True)

st.markdown('<hr class="kpi-divider"/>', unsafe_allow_html=True)

# ==============================================================================
# INGRESOS Y FUENTE EN UNA SOLA FILA
# ==============================================================================
row3 = st.columns(2, gap="large")

ingreso_bins = df_2026.copy()
ingreso_bins["tramo_ingreso"] = np.select(
    [
        ingreso_bins["ingreso_mensual"].eq(0),
        ingreso_bins["ingreso_mensual"].between(1, 300, inclusive="both"),
        ingreso_bins["ingreso_mensual"].between(301, 600, inclusive="both"),
        ingreso_bins["ingreso_mensual"].between(601, 1000, inclusive="both"),
        ingreso_bins["ingreso_mensual"].between(1001, 2000, inclusive="both"),
        ingreso_bins["ingreso_mensual"] > 2000,
    ],
    [
        "0",
        "1–300",
        "301–600",
        "601–1000",
        "1001–2000",
        "2000+",
    ],
    default="Prefiere no responder",
)

orden_tramos = ["0", "1–300", "301–600", "601–1000", "1001–2000", "2000+", "Prefiere no responder"]

dist_ing = (
    ingreso_bins.groupby("tramo_ingreso")["hogar_id"]
    .nunique()
    .reset_index(name="hogares")
)

with row3[0]:
    card_title("Distribución de ingreso monetario mensual", "Universo: todos los hogares efectivos deduplicados filtrados")
    bars = (
        alt.Chart(dist_ing)
        .mark_bar(color=AMBER_DARK)
        .encode(
            x=alt.X("tramo_ingreso:N", title=None, sort=orden_tramos, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("hogares:Q", title=None),
            tooltip=[
                alt.Tooltip("tramo_ingreso:N", title="Tramo"),
                alt.Tooltip("hogares:Q", title="Hogares"),
            ],
        )
        .properties(height=320)
    )
    labels = (
        alt.Chart(dist_ing)
        .mark_text(dy=-6, fontSize=11, color=P_TEXT)
        .encode(
            x=alt.X("tramo_ingreso:N", sort=orden_tramos),
            y=alt.Y("hogares:Q"),
            text=alt.Text("hogares:Q", format=","),
        )
    )
    st.altair_chart(bars + labels, use_container_width=True)

fuente_ing = (
    df_2026.groupby("fuente_ingresos")["hogar_id"]
    .nunique()
    .reset_index(name="hogares")
    .dropna()
    .sort_values("hogares", ascending=False)
    .head(12)
)

with row3[1]:
    card_title("Principal fuente de ingresos del hogar", "Universo: todos los hogares efectivos deduplicados filtrados")
    chart = make_horizontal_bar(
        fuente_ing,
        y_col="fuente_ingresos",
        x_col="hogares",
        color=AMBER_DARK,
        height=320,
        label_limit=260,
        sort_val="-x",
        fmt=",",
        tooltip_title_y="Fuente",
        tooltip_title_x="Hogares",
    )
    st.altair_chart(chart, use_container_width=True)