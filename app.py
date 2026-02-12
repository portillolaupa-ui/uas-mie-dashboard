import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt
import re
from io import BytesIO

st.set_page_config(page_title="UAS - Mi Independencia Económica", layout="wide")

#----------------------------------------------------------------------------------
DATA_PATH = "data/actividades_db_mieuas_v5.1.0_20260902.parquet"
USE_PARQUET = DATA_PATH.lower().endswith(".parquet")
#----------------------------------------------------------------------------------

COL_DNI = "DNI"
COL_SEXO = "SEXO"
COL_EDAD = "EDAD"
COL_INTERV = "INTERVENCION"
COL_LAT = "NU_LATITUD_PGH"
COL_LON = "NU_LONGITUD_PGH"

# NUEVAS CONSTANTES (no alteran tus nombres)
COL_UT = "UT"
COL_CSE = "CSE"
COL_TIPO_ACT = "TIPO_DE_ACTIVIDAD"
COL_ENTIDAD = "ENTIDAD_QUE_APOYA"

INTERV_ORDER = [
    "CAPACITACION EMPLEABILIDAD",
    "CAPACITACION GESTION DE NEGOCIOS O EMPRENDIMIENTO",
    "CAPACITACION SERVICIOS FINANCIEROS",
    "INTERCAMBIO COMERCIAL",
]

#----------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_excel(path)
    return df

df = load_data(DATA_PATH).copy()

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Normalizaciones mínimas
df[COL_DNI] = df[COL_DNI].astype("string")
df[COL_SEXO] = df[COL_SEXO].astype("string").str.strip().str.upper()
df[COL_EDAD] = pd.to_numeric(df[COL_EDAD], errors="coerce")
df[COL_INTERV] = df[COL_INTERV].astype("string").str.strip().str.upper()

#==================================================================================================
# PALETA (basada en la imagen): verde petróleo + verdes suaves + grises verdosos
#==================================================================================================
P_DARK   = "#1F5A53"   # verde petróleo (principal)
P_MID    = "#4F7F7A"   # verde medio
P_LIGHT  = "#A9C7C1"   # verde claro
P_PALE   = "#D8E5E2"   # fondo suave / acentos
P_TEXT   = "#0F172A"   # texto oscuro
P_MUTED  = "#5B6775"   # texto secundario
P_GRID   = "#E6EEEC"   # grillas
P_BORDER = "#E5E7EB"   # líneas / divisores

# sexo (sobrio)
COLOR_MASC = P_DARK
COLOR_FEM  = P_LIGHT

#==================================================================================================
# ESTILO (tipografías, tamaños, jerarquía, separadores + sidebar acorde a paleta)
#==================================================================================================
st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 1.2rem; padding-bottom: 1.2rem; }}

      section[data-testid="stSidebar"] {{
        background: {P_PALE};
        padding-top: 0.6rem;
        border-right: 1px solid {P_BORDER};
      }}

      h1 {{
        font-size: 2.0rem;
        letter-spacing: -0.2px;
        margin-bottom: 0.2rem;
        color: {P_TEXT};
      }}

      .subtitle {{
        color: {P_MUTED};
        font-size: 0.95rem;
        margin-top: -0.3rem;
        margin-bottom: 0.8rem;
      }}

      .chart-title {{
        font-size: 1.02rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
        color: {P_TEXT};
      }}

      .chart-title .note {{
        font-size: 0.85rem;
        font-weight: 400;
        color: {P_MUTED};
        margin-left: 0.35rem;
      }}

      .kpi-divider {{
        border: 0;
        height: 1px;
        background: {P_BORDER};
        margin: 0.9rem 0 1.0rem 0;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

#==================================================================================================
# MAPEO SEMÁNTICO (presentación)
#==================================================================================================
CSE_MAP = {
    "PNE": "Pobre",
    "PE": "Pobre extremo",
    "NP": "No pobre",
}

INTERV_MAP = {
    "CAPACITACION EMPLEABILIDAD": "Empleabilidad",
    "CAPACITACION GESTION DE NEGOCIOS O EMPRENDIMIENTO": "Gestión de negocios",
    "CAPACITACION SERVICIOS FINANCIEROS": "Servicios financieros",
    "INTERCAMBIO COMERCIAL": "Intercambio comercial",
}

SEXO_MAP = {
    "MASCULINO": "Masculino",
    "FEMENINO": "Femenino",
}

# Stopwords para Title Case institucional (español)
LOWER_WORDS = {
    "de", "del", "la", "las", "el", "los", "y", "e", "o", "u", "en", "para", "por", "a", "al", "con", "sin", "un", "una"
}

# Siglas que deben permanecer en mayúsculas (ajusta si quieres)
ACRONYMS = {
    "UT", "ONG", "MIDIS", "MIMP", "MINSA", "RENIEC", "SUNAT", "PNP", "UPN",
    "HW/NJ", "HW", "NJ", "UAS", "CSE"
}

def _smart_title_case(s: str) -> str:
    if s is None:
        return s
    s = str(s).strip()
    if not s:
        return s

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

        if "/" in raw:
            subs = raw.split("/")
            subs2 = []
            for sub in subs:
                if sub.upper() in ACRONYMS:
                    subs2.append(sub.upper())
                else:
                    subs2.append(sub.lower().capitalize())
            out.append("/".join(subs2))
            first_word_done = True
            continue

        low = raw.lower()
        if (low in LOWER_WORDS) and first_word_done:
            out.append(low)
        else:
            out.append(low.capitalize())
            first_word_done = True

    return "".join(out)

def _normalize_institutional(s: str) -> str:
    s0 = _smart_title_case(s)
    FIX = {
        "Banco De La Nacion": "Banco de la Nación",
        "Gobierno Local": "Gobierno local",
        "Gobierno Local ...": "Gobierno local",
        "Foncodes": "Foncodes",
        "Foncodes Mem": "Foncodes MEM",
        "Foncodes Hw/Nj": "Foncodes HW/NJ",
    }
    return FIX.get(s0, s0)

#==================================================================================================
# HELPERS: opciones con labels (UI) pero filtro con valores reales
#==================================================================================================
def _sorted_unique(series: pd.Series) -> list:
    vals = series.dropna().astype("string").str.strip()
    vals = vals[vals != ""].unique().tolist()
    return sorted(vals)

#==================================================================================================
# FILTROS GLOBALES (SIDEBAR): UT, EDAD, INTERVENCIÓN
# - SIN checkbox "Seleccionar todas las UT"
# - UT en Ley de escritura
# - Intervención con etiquetas de INTERV_MAP
#==================================================================================================
ut_options_raw = _sorted_unique(df[COL_UT]) if COL_UT in df.columns else []
interv_options_raw = _sorted_unique(df[COL_INTERV]) if COL_INTERV in df.columns else []

# UT raw -> label
ut_label_map = {u: _smart_title_case(u) for u in ut_options_raw}
ut_label_to_raw = {v: k for k, v in ut_label_map.items()}
ut_options_label = sorted(ut_label_to_raw.keys())

# Intervención raw -> label
interv_label_map = {r: INTERV_MAP.get(str(r).strip().upper(), _smart_title_case(r)) for r in interv_options_raw}
interv_label_to_raw = {v: k for k, v in interv_label_map.items()}

# Mantener el orden esperado (y agregar extras al final)
INTERV_LABEL_ORDER = [INTERV_MAP.get(x, x) for x in INTERV_ORDER]
interv_options_label = [lbl for lbl in INTERV_LABEL_ORDER if lbl in interv_label_to_raw]
extras = sorted([lbl for lbl in interv_label_to_raw.keys() if lbl not in interv_options_label])
interv_options_label = interv_options_label + extras

edad_min_data = int(pd.to_numeric(df[COL_EDAD], errors="coerce").min()) if df[COL_EDAD].notna().any() else 0
edad_max_data = int(pd.to_numeric(df[COL_EDAD], errors="coerce").max()) if df[COL_EDAD].notna().any() else 100
edad_min_data = max(0, edad_min_data)
edad_max_data = max(edad_min_data, edad_max_data)

with st.sidebar:
    st.markdown("### Filtros")

    ut_selected_label = st.multiselect(
        "UT",
        options=ut_options_label,
        default=ut_options_label,   # por defecto: todas seleccionadas
    )

    edad_range = st.slider(
        "Edad (rango)",
        min_value=edad_min_data,
        max_value=edad_max_data,
        value=(edad_min_data, edad_max_data),
        step=1,
    )

    interv_selected_label = st.multiselect(
        "Intervención",
        options=interv_options_label,
        default=interv_options_label,   # por defecto: todas seleccionadas
    )


# Labels -> valores reales
ut_selected_raw = [ut_label_to_raw[x] for x in ut_selected_label] if ut_selected_label else []
interv_selected_raw = [interv_label_to_raw[x] for x in interv_selected_label] if interv_selected_label else []

def apply_filters(df_in: pd.DataFrame) -> pd.DataFrame:
    dfx = df_in.copy()

    if COL_UT in dfx.columns and ut_selected_raw:
        dfx = dfx[dfx[COL_UT].astype("string").str.strip().isin(ut_selected_raw)]

    if COL_EDAD in dfx.columns:
        dfx = dfx[dfx[COL_EDAD].between(edad_range[0], edad_range[1], inclusive="both")]

    if COL_INTERV in dfx.columns and interv_selected_raw:
        dfx = dfx[dfx[COL_INTERV].astype("string").str.strip().isin(interv_selected_raw)]

    return dfx

df_f = apply_filters(df)

#==================================================================================================
# EXPORTACIÓN A EXCEL (Streamlit)
#==================================================================================================
@st.cache_data(show_spinner=False)
def to_excel_bytes(df_in: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_in.to_excel(writer, index=False, sheet_name="Data_filtrada")
    return output.getvalue()

excel_bytes = to_excel_bytes(df_f)

#==================================================================================================
# Capa de presentación (labels) – no altera el original
#==================================================================================================
df_f["SEXO_label"] = (
    df_f[COL_SEXO].astype("string").str.strip().str.upper()
    .map(SEXO_MAP)
    .fillna(df_f[COL_SEXO].astype("string"))
)

df_f["CSE_label"] = (
    df_f[COL_CSE].astype("string").str.strip().str.upper()
    .map(CSE_MAP)
    .fillna(df_f[COL_CSE].astype("string").str.strip())
)

df_f["INTERV_label"] = (
    df_f[COL_INTERV].astype("string").str.strip().str.upper()
    .map(INTERV_MAP)
    .fillna(df_f[COL_INTERV].astype("string").map(_smart_title_case))
)

df_f["TIPO_ACT_label"] = df_f[COL_TIPO_ACT].astype("string").map(_normalize_institutional)
df_f["ENTIDAD_label"] = df_f[COL_ENTIDAD].astype("string").map(_normalize_institutional)

order_map = {k: i for i, k in enumerate(INTERV_LABEL_ORDER)}

#==================================================================================================
# KPIs (recalculados con filtros)
#==================================================================================================
dni_total = df_f[COL_DNI].dropna().nunique()

df_18_29 = df_f[df_f[COL_EDAD].between(18, 29, inclusive="both") & df_f[COL_DNI].notna()]
dni_18_29 = df_18_29[COL_DNI].nunique()

dni_18_29_sexo = (
    df_f[
        df_f[COL_EDAD].between(18, 29, inclusive="both")
        & df_f[COL_DNI].notna()
        & df_f["SEXO_label"].notna()
    ]
    .groupby("SEXO_label")[COL_DNI].nunique()
)

dni_18_29_hombres = int(dni_18_29_sexo.get("Masculino", 0))
dni_18_29_mujeres = int(dni_18_29_sexo.get("Femenino", 0))

#==================================================================================================
# Header
#==================================================================================================
tcol1, tcol2 = st.columns([6, 1], vertical_alignment="center")

with tcol1:
    st.title("Mi Independencia Económica")
    st.markdown(
        '<div class="subtitle">Unidad de Acompañamiento y Servicios Complementarios</div>',
        unsafe_allow_html=True
    )

with tcol2:
    st.markdown("<div style='height: 0.65rem;'></div>", unsafe_allow_html=True)
    st.download_button(
        label="Descargar reporte (Excel)",
        data=excel_bytes,
        file_name="reporte_mi_independencia_economica.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# Fila 1: tarjetas principales (NO TOCAR)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("N° participantes", f"{dni_total:,}")
c2.metric("N° actividades registradas", f"{len(df_f):,}")
c3.metric("N° 18–29 años", f"{dni_18_29:,}")
c4.metric("18–29 años | ♂️ hombres", f"{dni_18_29_hombres}")
c5.metric("18–29 años | ♀️ mujeres", f"{dni_18_29_mujeres}")

st.markdown('<hr class="kpi-divider"/>', unsafe_allow_html=True)

#==================================================================================================
# HELPERS + Altair theme
#==================================================================================================
def card_title(txt: str):
    st.markdown(f'<div class="chart-title">{txt}</div>', unsafe_allow_html=True)

def card_caption(txt: str):
    # No imprime nada (se eliminan notas bajo gráficos)
    return

def _sex_color_scale():
    return alt.Scale(domain=["Masculino", "Femenino"], range=[COLOR_MASC, COLOR_FEM])

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
alt.themes.register("minimal_uas", lambda: ALT_THEME)
alt.themes.enable("minimal_uas")

#==================================================================================================
# FILA 1: Perfil de participantes y cobertura (4)
#==================================================================================================
row1 = st.columns(4, gap="large")

# (1) Participantes por sexo + etiqueta
sex_unique = (
    df_f.dropna(subset=[COL_DNI, "SEXO_label"])
    .groupby("SEXO_label")[COL_DNI].nunique()
    .reset_index(name="participantes")
)

with row1[0]:
    card_title("N° Participantes por sexo")
    base = (
        alt.Chart(sex_unique)
        .mark_bar()
        .encode(
            x=alt.X("SEXO_label:N", title=None, sort=["Masculino", "Femenino"], axis=alt.Axis(labelAngle=0)),
            y=alt.Y("participantes:Q", title=None),
            color=alt.Color("SEXO_label:N", scale=_sex_color_scale(), legend=None),
            tooltip=[alt.Tooltip("SEXO_label:N", title="Sexo"), alt.Tooltip("participantes:Q", title="DNIs únicos")],
        )
        .properties(height=260)
    )
    labels = (
        alt.Chart(sex_unique)
        .mark_text(dy=-6, fontSize=11, color=P_TEXT)
        .encode(
            x=alt.X("SEXO_label:N", sort=["Masculino", "Femenino"]),
            y=alt.Y("participantes:Q"),
            text=alt.Text("participantes:Q", format=","),
        )
    )
    st.altair_chart(base + labels, use_container_width=True)
    card_caption("")

# (2) CSE por sexo (apilado) + total encima
cse_sexo = (
    df_f.dropna(subset=[COL_DNI, "CSE_label", "SEXO_label"])
    .groupby(["CSE_label", "SEXO_label"])[COL_DNI].nunique()
    .reset_index(name="participantes")
)

with row1[1]:
    card_title("N° Participantes por CSE")
    stacked = (
        alt.Chart(cse_sexo)
        .mark_bar()
        .encode(
            x=alt.X("CSE_label:N", title=None, sort="-y", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("participantes:Q", title=None),
            color=alt.Color("SEXO_label:N", scale=_sex_color_scale(), legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[
                alt.Tooltip("CSE_label:N", title="CSE"),
                alt.Tooltip("SEXO_label:N", title="Sexo"),
                alt.Tooltip("participantes:Q", title="DNIs únicos"),
            ],
        )
        .properties(height=260)
    )
    totals = (
        alt.Chart(cse_sexo)
        .transform_aggregate(total="sum(participantes)", groupby=["CSE_label"])
        .mark_text(dy=-6, fontSize=11, color=P_TEXT)
        .encode(
            x=alt.X("CSE_label:N", sort="-y"),
            y=alt.Y("total:Q"),
            text=alt.Text("total:Q", format=","),
        )
    )
    st.altair_chart(stacked + totals, use_container_width=True)
    card_caption("")

# (3) Grupo de edad por sexo (apilado) + total encima
bins = [11, 17, 29, 44, 200]
labels_age = ["12–17", "18–29", "30–44", "45+"]

df_age = df_f.dropna(subset=[COL_DNI, COL_EDAD, "SEXO_label"]).copy()
df_age["Grupo_edad"] = pd.cut(df_age[COL_EDAD], bins=bins, labels=labels_age)

age_sexo = (
    df_age.dropna(subset=["Grupo_edad"])
    .groupby(["Grupo_edad", "SEXO_label"])[COL_DNI].nunique()
    .reset_index(name="participantes")
)

with row1[2]:
    card_title("N° Participantes por grupo de edad")
    stacked = (
        alt.Chart(age_sexo)
        .mark_bar()
        .encode(
            x=alt.X("Grupo_edad:N", title=None, sort=labels_age, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("participantes:Q", title=None),
            color=alt.Color("SEXO_label:N", scale=_sex_color_scale(), legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[
                alt.Tooltip("Grupo_edad:N", title="Grupo de edad"),
                alt.Tooltip("SEXO_label:N", title="Sexo"),
                alt.Tooltip("participantes:Q", title="DNIs únicos"),
            ],
        )
        .properties(height=260)
    )
    totals = (
        alt.Chart(age_sexo)
        .transform_aggregate(total="sum(participantes)", groupby=["Grupo_edad"])
        .mark_text(dy=-6, fontSize=11, color=P_TEXT)
        .encode(
            x=alt.X("Grupo_edad:N", sort=labels_age),
            y=alt.Y("total:Q"),
            text=alt.Text("total:Q", format=","),
        )
    )
    st.altair_chart(stacked + totals, use_container_width=True)
    card_caption("")

# (4) Intervención por sexo (horizontal apilado) + total al final
interv_sexo = (
    df_f.dropna(subset=[COL_DNI, "INTERV_label", "SEXO_label"])
    .groupby(["INTERV_label", "SEXO_label"])[COL_DNI].nunique()
    .reset_index(name="participantes")
)
interv_sexo["ord"] = interv_sexo["INTERV_label"].map(order_map).fillna(999).astype(int)
interv_sexo = interv_sexo.sort_values(["ord", "INTERV_label"])

with row1[3]:
    card_title("N° Participantes por intervención")
    stacked = (
        alt.Chart(interv_sexo)
        .mark_bar()
        .encode(
            y=alt.Y("INTERV_label:N", title=None, sort=None),
            x=alt.X("participantes:Q", title=None),
            color=alt.Color("SEXO_label:N", scale=_sex_color_scale(), legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[
                alt.Tooltip("INTERV_label:N", title="Intervención"),
                alt.Tooltip("SEXO_label:N", title="Sexo"),
                alt.Tooltip("participantes:Q", title="DNIs únicos"),
            ],
        )
        .properties(height=260)
    )
    totals = (
        alt.Chart(interv_sexo)
        .transform_aggregate(total="sum(participantes)", groupby=["INTERV_label"])
        .mark_text(align="left", dx=4, fontSize=11, color=P_TEXT)
        .encode(
            y=alt.Y("INTERV_label:N", sort=None),
            x=alt.X("total:Q"),
            text=alt.Text("total:Q", format=","),
        )
    )
    st.altair_chart(stacked + totals, use_container_width=True)
    card_caption("")

#==================================================================================================
# FILA 2: Actividades (Opción A)
#==================================================================================================
row2 = st.columns(4, gap="large")

# (5) Nº intervenciones por participante + sexo (apilado) + total encima
dni_interv_n_sexo = (
    df_f.dropna(subset=[COL_DNI, "SEXO_label", "INTERV_label"])
    .groupby([COL_DNI, "SEXO_label"])["INTERV_label"].nunique()
    .reset_index(name="n_intervenciones")
)
dni_interv_n_sexo = dni_interv_n_sexo[dni_interv_n_sexo["n_intervenciones"].between(1, 4)]

dni_interv_dist_sexo = (
    dni_interv_n_sexo.groupby(["n_intervenciones", "SEXO_label"])[COL_DNI]
    .nunique()
    .reset_index(name="participantes")
)

with row2[0]:
    card_title("N° Participantes según número de intervenciones")
    stacked = (
        alt.Chart(dni_interv_dist_sexo)
        .mark_bar()
        .encode(
            x=alt.X("n_intervenciones:O", title=None, sort=["1", "2", "3", "4"], axis=alt.Axis(labelAngle=0)),
            y=alt.Y("participantes:Q", title=None),
            color=alt.Color("SEXO_label:N", scale=_sex_color_scale(), legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[
                alt.Tooltip("n_intervenciones:O", title="Nº intervenciones"),
                alt.Tooltip("SEXO_label:N", title="Sexo"),
                alt.Tooltip("participantes:Q", title="DNIs únicos"),
            ],
        )
        .properties(height=260)
    )
    totals = (
        alt.Chart(dni_interv_dist_sexo)
        .transform_aggregate(total="sum(participantes)", groupby=["n_intervenciones"])
        .mark_text(dy=-6, fontSize=11, color=P_TEXT)
        .encode(
            x=alt.X("n_intervenciones:O", sort=["1", "2", "3", "4"]),
            y=alt.Y("total:Q"),
            text=alt.Text("total:Q", format=","),
        )
    )
    st.altair_chart(stacked + totals, use_container_width=True)
    card_caption("")

# (6) Actividades por intervención (horizontal) + etiqueta al final
interv_acts = (
    df_f.dropna(subset=["INTERV_label"])
    .groupby("INTERV_label")
    .size()
    .reset_index(name="actividades")
)
interv_acts["ord"] = interv_acts["INTERV_label"].map(order_map).fillna(999).astype(int)
interv_acts = interv_acts.sort_values(["ord", "actividades"], ascending=[True, False])

with row2[1]:
    card_title("N° actividades por intervención")
    bars = (
        alt.Chart(interv_acts)
        .mark_bar(color=P_DARK)
        .encode(
            y=alt.Y("INTERV_label:N", title=None, sort=None),
            x=alt.X("actividades:Q", title=None),
            tooltip=[
                alt.Tooltip("INTERV_label:N", title="Intervención"),
                alt.Tooltip("actividades:Q", title="Registros"),
            ],
        )
        .properties(height=260)
    )
    lab = (
        alt.Chart(interv_acts)
        .mark_text(align="left", dx=4, fontSize=11, color=P_TEXT)
        .encode(
            y=alt.Y("INTERV_label:N", sort=None),
            x=alt.X("actividades:Q"),
            text=alt.Text("actividades:Q", format=","),
        )
    )
    st.altair_chart(bars + lab, use_container_width=True)
    card_caption("")

# (7) Actividades por tipo (Top 10) + etiqueta al final
tipo_acts = (
    df_f.dropna(subset=["TIPO_ACT_label"])
    .assign(TIPO_ACT_label=lambda d: d["TIPO_ACT_label"].astype("string").str.strip())
)
tipo_acts = tipo_acts[tipo_acts["TIPO_ACT_label"] != ""]
tipo_acts = (
    tipo_acts.groupby("TIPO_ACT_label")
    .size()
    .reset_index(name="actividades")
    .sort_values("actividades", ascending=False)
    .head(10)
)

with row2[2]:
    card_title("Top 10: Tipos de actividad más frecuentes")
    bars = (
        alt.Chart(tipo_acts)
        .mark_bar(color=P_DARK)
        .encode(
            y=alt.Y("TIPO_ACT_label:N", title=None, sort="-x"),
            x=alt.X("actividades:Q", title=None),
            tooltip=[
                alt.Tooltip("TIPO_ACT_label:N", title="Tipo"),
                alt.Tooltip("actividades:Q", title="Registros"),
            ],
        )
        .properties(height=260)
    )
    lab = (
        alt.Chart(tipo_acts)
        .mark_text(align="left", dx=4, fontSize=11, color=P_TEXT)
        .encode(
            y=alt.Y("TIPO_ACT_label:N", sort="-x"),
            x=alt.X("actividades:Q"),
            text=alt.Text("actividades:Q", format=","),
        )
    )
    st.altair_chart(bars + lab, use_container_width=True)
    card_caption("")

# (8) Actividades por entidad (Top 10) + etiqueta al final
ent_acts = (
    df_f.dropna(subset=["ENTIDAD_label"])
    .assign(ENTIDAD_label=lambda d: d["ENTIDAD_label"].astype("string").str.strip())
)
ent_acts = ent_acts[ent_acts["ENTIDAD_label"] != ""]
ent_acts = (
    ent_acts.groupby("ENTIDAD_label")
    .size()
    .reset_index(name="actividades")
    .sort_values("actividades", ascending=False)
    .head(10)
)

with row2[3]:
    card_title("Top 10: Entidades por volumen de actividad")
    bars = (
        alt.Chart(ent_acts)
        .mark_bar(color=P_DARK)
        .encode(
            y=alt.Y("ENTIDAD_label:N", title=None, sort="-x"),
            x=alt.X("actividades:Q", title=None),
            tooltip=[
                alt.Tooltip("ENTIDAD_label:N", title="Entidad"),
                alt.Tooltip("actividades:Q", title="Registros"),
            ],
        )
        .properties(height=260)
    )
    lab = (
        alt.Chart(ent_acts)
        .mark_text(align="left", dx=4, fontSize=11, color=P_TEXT)
        .encode(
            y=alt.Y("ENTIDAD_label:N", sort="-x"),
            x=alt.X("actividades:Q"),
            text=alt.Text("actividades:Q", format=","),
        )
    )
    st.altair_chart(bars + lab, use_container_width=True)
    card_caption("")
