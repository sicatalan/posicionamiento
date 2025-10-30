import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


# -------------------------
# Configuración de la app
# -------------------------
st.set_page_config(page_title="Análisis de Posicionamiento", layout="wide")


# -------------------------
# Utilitarios
# -------------------------
def normalize_text(s):
    if s is None:
        return s
    import unicodedata, re
    s = str(s)
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = s.replace("%", "pct")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s


def to_numeric_safe(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "biufc":
        return series
    return pd.to_numeric(
        series.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def _cached_key_for_file(uploaded_or_path):
    # Devuelve una clave hashable para cache: bytes si viene de uploader, (path, mtime) si es ruta.
    if uploaded_or_path is None:
        return None
    if hasattr(uploaded_or_path, "getvalue"):
        return uploaded_or_path.getvalue()
    p = Path(uploaded_or_path)
    if p.exists():
        try:
            return (str(p.resolve()), p.stat().st_mtime_ns)
        except Exception:
            return str(p.resolve())
    return str(p)


@st.cache_data(show_spinner=False)
def load_excel_first_sheet(uploaded_or_path, preferred_sheet: Optional[str] = None):
    key = _cached_key_for_file(uploaded_or_path)
    if key is None:
        return None
    if hasattr(uploaded_or_path, "read"):
        uploaded_or_path.seek(0)
        return pd.read_excel(uploaded_or_path, sheet_name=0)
    # Ruta local
    try:
        return pd.read_excel(uploaded_or_path, sheet_name=0)
    except Exception:
        if preferred_sheet:
            return pd.read_excel(uploaded_or_path, sheet_name=preferred_sheet)
        raise


def prepare_base(df: pd.DataFrame, canal_interno: str, usar_ultima_fecha: bool) -> pd.DataFrame:
    df = df.copy()
    # Normalización de columnas
    df.columns = [normalize_text(c) for c in df.columns]
    # Tipos numéricos
    for col in ["precio", "precio_unitario", "peso_neto_umv"]:
        if col in df.columns:
            df[col] = to_numeric_safe(df[col])
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    if "canal" in df.columns:
        df["canal"] = df["canal"].astype(str).str.strip().str.lower()

    # Precio de análisis
    if "precio_unitario" in df.columns and df["precio_unitario"].notna().any():
        precio_analisis = df["precio_unitario"]
    else:
        if "precio" in df.columns and "peso_neto_umv" in df.columns:
            precio_analisis = df["precio"] / df["peso_neto_umv"].replace(0, np.nan)
        else:
            precio_analisis = df["precio"] if "precio" in df.columns else np.nan
    df["precio_analisis"] = to_numeric_safe(precio_analisis)

    # Selección última fila por SKU-Canal
    df["sku"] = df["sku"].astype(str).str.strip()
    if usar_ultima_fecha and "fecha" in df.columns:
        df = (
            df.sort_values(["sku", "canal", "fecha"]).drop_duplicates(
                subset=["sku", "canal"], keep="last"
            )
        )
    return df


def prepare_maestro(maestro: pd.DataFrame) -> pd.DataFrame:
    m = maestro.copy()
    m.columns = [normalize_text(c) for c in m.columns]
    keep = [c for c in ["sku", "categoria", "familia"] if c in m.columns]
    m = m[keep].drop_duplicates()
    m["sku"] = m["sku"].astype(str).str.strip()
    return m


@st.cache_data(show_spinner=False)
def build_model(base_df: pd.DataFrame, maestro_df: pd.DataFrame, canal_interno: str):
    data = base_df.copy()
    maestro = maestro_df.copy()

    # Merge maestro
    data = data.merge(maestro, on="sku", how="left")

    # Diagnóstico básico
    diag = {
        "filas": len(data),
        "rango_fechas": (
            (data["fecha"].min(), data["fecha"].max()) if "fecha" in data.columns else (None, None)
        ),
        "filas_por_canal": data["canal"].value_counts(dropna=False).rename("filas_por_canal").to_frame(),
        "stats_por_canal": data.groupby("canal")["precio_analisis"].agg(["count", "mean", "median", "std"]).sort_values("mean", ascending=False),
    }

    # Interno y externo por SKU
    interno = (
        data.query("canal == @canal_interno")[["sku", "precio_analisis"]]
        .groupby("sku", as_index=False)["precio_analisis"].median()
        .rename(columns={"precio_analisis": "precio_interno"})
    )

    externo = (
        data.query("canal != @canal_interno")[["sku", "precio_analisis", "canal"]]
        .groupby("sku", as_index=False)
        .agg(precio_externo=("precio_analisis", "mean"), canales_externos=("canal", "nunique"))
    )

    sku_comp = (
        maestro[["sku", "categoria", "familia"]]
        .merge(interno, on="sku", how="left")
        .merge(externo, on="sku", how="left")
    )
    if "descripcion" in data.columns:
        sku_comp = sku_comp.merge(
            data[["sku", "descripcion"]].drop_duplicates("sku"), on="sku", how="left"
        )

    sku_comp["gap_abs"] = sku_comp["precio_externo"] - sku_comp["precio_interno"]
    sku_comp["gap_pct"] = np.where(
        sku_comp["precio_interno"].fillna(0) > 0,
        sku_comp["precio_externo"] / sku_comp["precio_interno"] - 1,
        np.nan,
    )

    # Top desviaciones por SKU
    sku_ok = sku_comp.dropna(subset=["precio_interno", "precio_externo"]).copy()
    # Clipping p99 por signo para mejor escala
    if len(sku_ok):
        p_pos = np.nanpercentile(sku_ok["gap_abs"].clip(lower=0), 99) if (sku_ok["gap_abs"] > 0).any() else None
        p_neg = np.nanpercentile((-sku_ok["gap_abs"]).clip(lower=0), 99) if (sku_ok["gap_abs"] < 0).any() else None
    else:
        p_pos = p_neg = None

    def clip_gap_abs(v):
        if pd.isna(v):
            return v
        if v > 0 and p_pos is not None:
            return min(v, p_pos)
        if v < 0 and p_neg is not None:
            return max(v, -p_neg)
        return v

    sku_ok["gap_abs_clip"] = sku_ok["gap_abs"].apply(clip_gap_abs)

    top_pos = (
        sku_ok.sort_values("gap_abs", ascending=False).head(20)[
            ["sku", "descripcion", "categoria", "familia", "precio_interno", "precio_externo", "gap_abs", "gap_pct", "canales_externos", "gap_abs_clip"]
        ]
    )
    top_neg = (
        sku_ok.sort_values("gap_abs", ascending=True).head(20)[
            ["sku", "descripcion", "categoria", "familia", "precio_interno", "precio_externo", "gap_abs", "gap_pct", "canales_externos", "gap_abs_clip"]
        ]
    )

    # Agregado Cat | Fam (box y top 15)
    df_box = sku_ok.copy()
    df_box["cat_fam"] = df_box["categoria"].astype(str) + " | " + df_box["familia"].astype(str)

    agg_cf = (
        df_box.groupby(["categoria", "familia"], dropna=False)["gap_pct"]
        .mean()
        .rename("gap_pct_prom")
        .reset_index()
        .sort_values("gap_pct_prom", ascending=False)
    )
    top15_cf = agg_cf.head(15)

    # Heatmap ratio por Familia y Canal vs interno
    base_ratios = data.merge(
        interno.rename(columns={"precio_interno": "precio_interno_ref"}), on="sku", how="left"
    )
    base_ratios = base_ratios[
        base_ratios["precio_interno_ref"].notna() & (base_ratios["precio_interno_ref"] > 0)
    ].copy()
    base_ratios["ratio_vs_interno"] = base_ratios["precio_analisis"] / base_ratios["precio_interno_ref"]
    base_ratios = base_ratios.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio_vs_interno"])
    base_ratios["cat_fam"] = base_ratios["categoria"].astype(str) + " | " + base_ratios["familia"].astype(str)
    fam_canal = (
        base_ratios[base_ratios["canal"] != canal_interno]
        .groupby(["cat_fam", "canal"])["ratio_vs_interno"]
        .mean()
        .unstack("canal")
    )
    fam_canal_counts = (
        base_ratios[base_ratios["canal"] != canal_interno]
        .groupby(["cat_fam", "canal"])["ratio_vs_interno"]
        .size()
        .unstack("canal")
    )
    # Ordenar por categoría y familia para mejor legibilidad
    if {"categoria", "familia"}.issubset(data.columns):
        row_meta_df = (
            base_ratios[["cat_fam", "categoria", "familia"]]
            .drop_duplicates()
            .sort_values(["categoria", "familia"])  # natural
            .set_index("cat_fam")
        )
        orden_cf = row_meta_df.index
        fam_canal = fam_canal.reindex(orden_cf)
        fam_canal_counts = fam_canal_counts.reindex(orden_cf)

    # Resumen por canal: promedio de ratio vs interno
    res_canal = (
        base_ratios[base_ratios["canal"] != canal_interno]
        .groupby("canal")["ratio_vs_interno"]
        .mean()
        .rename("ratio_promedio_vs_interno")
        .sort_values(ascending=False)
        .to_frame()
    )

    # Detalle por SKU x Fuente
    df_src = data.copy()
    col_fuente = "fuente_comercial" if "fuente_comercial" in df_src.columns else ("fuente" if "fuente" in df_src.columns else None)
    if col_fuente is None:
        df_src["fuente_limpia"] = np.where(df_src["canal"] == canal_interno, "interno", df_src["canal"].astype(str))
    else:
        df_src["fuente_limpia"] = np.where(
            df_src["canal"] == canal_interno,
            "interno",
            df_src[col_fuente].astype(str).str.strip().str.lower(),
        )
    df_src = df_src.dropna(subset=["sku", "precio_analisis"]).copy()
    if "fecha" in df_src.columns:
        df_src = (
            df_src.sort_values(["sku", "fuente_limpia", "fecha"]).drop_duplicates(
                subset=["sku", "fuente_limpia"], keep="last"
            )
        )
    pvt = (
        df_src.pivot_table(index=["sku"], columns="fuente_limpia", values="precio_analisis", aggfunc="first")
        .reset_index()
    )
    meta_cols = [c for c in ["sku", "descripcion", "categoria", "familia"] if c in data.columns]
    if meta_cols:
        pvt = data.drop_duplicates("sku")[meta_cols].merge(pvt, on="sku", how="left")
    cols_fuentes = [c for c in pvt.columns if c not in ["sku", "descripcion", "categoria", "familia", "interno"]]
    if "interno" not in pvt.columns:
        pvt["interno"] = np.nan
    pvt["min_ext"], pvt["max_ext"] = pvt[cols_fuentes].min(axis=1, skipna=True), pvt[cols_fuentes].max(axis=1, skipna=True)
    pvt["fuente_min"], pvt["fuente_max"] = pvt[cols_fuentes].idxmin(axis=1), pvt[cols_fuentes].idxmax(axis=1)
    pvt["gap_min_pct"] = np.where(pvt["interno"].notna() & pvt["min_ext"].notna(), pvt["min_ext"]/pvt["interno"] - 1, np.nan)
    pvt["gap_max_pct"] = np.where(pvt["interno"].notna() & pvt["max_ext"].notna(), pvt["max_ext"]/pvt["interno"] - 1, np.nan)

    return {
        "data": data,
        "diag": diag,
        "interno": interno,
        "externo": externo,
        "sku_comp": sku_comp,
        "sku_ok": sku_ok,
        "top_pos": top_pos,
        "top_neg": top_neg,
        "df_box": df_box,
        "top15_cf": top15_cf,
        "fam_canal": fam_canal,
        "fam_canal_counts": fam_canal_counts,
        "row_meta": row_meta_df if 'row_meta_df' in locals() else None,
        "res_canal": res_canal,
        "pvt": pvt,
        "cols_fuentes": cols_fuentes,
    }


def filter_by_sku_text(df: pd.DataFrame, text: str, cols=("sku", "descripcion")) -> pd.DataFrame:
    if not text:
        return df
    mask = None
    for c in cols:
        if c in df.columns:
            m = df[c].astype(str).str.contains(text, case=False, na=False)
            mask = m if mask is None else (mask | m)
    return df[mask] if mask is not None else df


# -------------------------
# Sidebar / Parámetros
# -------------------------
st.sidebar.header("Parámetros")

default_base_path = Path("base.xlsx")
default_maestro_path = Path("maestro_skus.xlsx")

uploaded_base = st.sidebar.file_uploader("Base de precios (Excel)", type=["xlsx", "xls"], key="base")
uploaded_maestro = st.sidebar.file_uploader("Maestro de SKUs (Excel)", type=["xlsx", "xls"], key="maestro")

canal_interno = st.sidebar.text_input("Nombre canal interno", value="interno")
usar_ultima_fecha = st.sidebar.checkbox("Usar último precio por SKU-Canal", value=True)

# Lectura de archivos (subidos o por defecto en disco)
base_df = None
maestro_df = None

try:
    if uploaded_base is not None:
        base_df = load_excel_first_sheet(uploaded_base, preferred_sheet="base")
    elif default_base_path.exists():
        base_df = load_excel_first_sheet(str(default_base_path), preferred_sheet="base")
    else:
        st.warning("Sube el archivo base.xlsx en la barra lateral.")
except Exception as e:
    st.error(f"Error leyendo base: {e}")

try:
    if uploaded_maestro is not None:
        maestro_df = load_excel_first_sheet(uploaded_maestro)
    elif default_maestro_path.exists():
        maestro_df = load_excel_first_sheet(str(default_maestro_path))
    else:
        st.warning("Sube el archivo maestro_skus.xlsx en la barra lateral.")
except Exception as e:
    st.error(f"Error leyendo maestro: {e}")


ready = (base_df is not None) and (maestro_df is not None)

if ready:
    base_proc = prepare_base(base_df, canal_interno=canal_interno, usar_ultima_fecha=usar_ultima_fecha)
    maestro_proc = prepare_maestro(maestro_df)
    model = build_model(base_proc, maestro_proc, canal_interno)
else:
    st.stop()


# -------------------------
# Tabs
# -------------------------
tabs = st.tabs([
    "Resumen General",
    "Canal Interno vs Resto (SKU)",
    "Top Desviaciones SKU",
    "Cat | Fam (Top 15 y Box)",
    "Comparacion por Familia",
    "Detalle SKU x Fuente",
    "Exportar",
])


# 1) Resumen General
with tabs[0]:
    st.subheader("Carga y diagnóstico")
    st.caption("Notas: 'precio_análisis' = 'precio_unitario' si existe, de lo contrario 'precio' / 'peso_neto_umv'. Las estadísticas por canal se calculan sobre 'precio_análisis'.")
    st.info(
        "Como se calculan los datos principales:\n"
        "- precio_analisis = 'precio_unitario' (si existe); de lo contrario 'precio' / 'peso_neto_umv'\n"
        "- Las estadisticas por canal usan 'precio_analisis'."
    )
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.metric("Filas", f"{model['diag']['filas']:,}")
        fmin, fmax = model["diag"]["rango_fechas"]
        if fmin is not None and pd.notna(fmin):
            st.write(f"Rango de fechas: {fmin.date()} → {fmax.date() if pd.notna(fmax) else 'N/D'}")
    with c2:
        st.write("Filas por canal")
        st.dataframe(model["diag"]["filas_por_canal"], use_container_width=True)
    with c3:
        st.write("Estadísticas por canal (precio_análisis)")
        st.dataframe(model["diag"]["stats_por_canal"], use_container_width=True)

    st.write("Datos base (post-limpieza)")
    txt = st.text_input("Filtrar SKU o descripción (contiene)", key="filtro_resumen")
    df_show = filter_by_sku_text(model["data"], txt)
    st.dataframe(df_show, use_container_width=True, height=420)


# 2) Comparación canal interno vs resto (SKU)
with tabs[1]:
    st.subheader("Comparación canal interno vs resto (nivel SKU)")
    st.caption("Definiciones: 'precio_interno' = mediana de 'precio_análisis' del canal interno por SKU; 'precio_externo' = promedio en canales ≠ interno; gap_abs = precio_externo - precio_interno; gap_% = (precio_externo / precio_interno) - 1. La barra muestra el promedio de (precio_canal / precio_interno) por canal.")
    st.info(
        "Definiciones clave:\n"
        "- precio_interno = mediana de 'precio_analisis' del canal interno por SKU\n"
        "- precio_externo = promedio de 'precio_analisis' en canales != interno\n"
        "- gap_abs = precio_externo - precio_interno\n"
        "- gap_% = (precio_externo / precio_interno) - 1\n"
        "- Grafico: promedio de (precio_canal / precio_interno) por canal"
    )
    # Filtro adicional por familia (afecta tabla y gráfico)
    familias_opts = []
    if "familia" in model["data"].columns:
        familias_opts = sorted([str(x) for x in model["data"]["familia"].dropna().unique()])
    fam_sel = st.multiselect(
        "Filtrar por familia (afecta tabla y gráfico)", options=familias_opts, key="filtro_familia_canal"
    )
    df = model["sku_comp"].copy()
    if fam_sel:
        df = df[df["familia"].astype(str).isin(fam_sel)]
    txt = st.text_input("Filtrar SKU o descripción (contiene)", key="filtro_sku_comp")
    if txt:
        df = filter_by_sku_text(df, txt)
    st.dataframe(df, use_container_width=True, height=480)

    st.markdown("Promedio de (precio canal / precio interno) por canal")
    # Si hay filtro de familia, recalcular el resumen por canal sobre el subconjunto
    if fam_sel and len(fam_sel) and {"familia", "sku", "canal", "precio_analisis"}.issubset(model["data"].columns):
        data_use = model["data"].copy()
        data_use = data_use[data_use["familia"].astype(str).isin(fam_sel)]
        interno_ref = (
            data_use.query("canal == @canal_interno")[["sku", "precio_analisis"]]
            .groupby("sku", as_index=False)["precio_analisis"].median()
            .rename(columns={"precio_analisis": "precio_interno_ref"})
        )
        base_ratios_f = data_use.merge(interno_ref, on="sku", how="left")
        base_ratios_f = base_ratios_f[
            base_ratios_f["precio_interno_ref"].notna() & (base_ratios_f["precio_interno_ref"] > 0)
        ].copy()
        base_ratios_f["ratio_vs_interno"] = base_ratios_f["precio_analisis"] / base_ratios_f["precio_interno_ref"]
        base_ratios_f = base_ratios_f.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio_vs_interno"])
        rc = (
            base_ratios_f[base_ratios_f["canal"] != canal_interno]
            .groupby("canal")["ratio_vs_interno"].mean()
            .rename("ratio_promedio_vs_interno")
            .sort_values(ascending=False)
            .reset_index()
        )
    else:
        rc = model["res_canal"].reset_index()
    fig = px.bar(rc, x="canal", y="ratio_promedio_vs_interno", text=rc["ratio_promedio_vs_interno"].map(lambda x: f"{x:.2f}"))
    fig.update_layout(xaxis_title="Canal", yaxis_title="Ratio vs interno", height=420)
    st.plotly_chart(fig, use_container_width=True)


# 3) Top desviaciones por SKU
with tabs[2]:
    st.subheader("Top desviaciones por SKU")
    # Gráficos de barras horizontales (categoría por color)
    st.caption("Fórmulas: gap_abs = precio_externo - precio_interno; gap_% = (precio_externo / precio_interno) - 1.")
    st.info(
        "Formulas utilizadas:\n"
        "- gap_abs = precio_externo - precio_interno\n"
        "- gap_% = (precio_externo / precio_interno) - 1"
    )
    # Filtro por familia
    fam_opts_td = []
    if "familia" in model["data"].columns:
        fam_opts_td = sorted([str(x) for x in model["data"]["familia"].dropna().unique()])
    fam_sel_td = st.multiselect("Filtrar por familia (afecta gráficos y tablas)", options=fam_opts_td, key="filtro_familia_top")

    # Preparar data filtrada
    tp_all = model["top_pos"].copy()
    tn_all = model["top_neg"].copy()
    if fam_sel_td:
        tp_all = tp_all[tp_all["familia"].astype(str).isin(fam_sel_td)]
        tn_all = tn_all[tn_all["familia"].astype(str).isin(fam_sel_td)]

    gp1, gp2 = st.columns(2)
    with gp1:
        st.write("Top 20 gap positivo (externo > interno)")
        tp = tp_all.copy()
        if len(tp):
            tp["sku_label"] = (
                tp["sku"].astype(str)
                + " - "
                + tp.get("descripcion", pd.Series([""] * len(tp))).astype(str)
            )
            figp = px.bar(
                tp.sort_values("gap_abs_clip"),
                x="gap_abs_clip",
                y="sku_label",
                color="categoria",
                orientation="h",
                hover_data=["sku", "familia", "precio_interno", "precio_externo", "gap_abs", "gap_pct"],
            )
            figp.update_layout(
                height=max(420, 22 * len(tp) + 120),
                xaxis_title="Gap absoluto (externo - interno)",
                yaxis_title="SKU",
                legend_title_text="Categoría",
                margin=dict(l=200),
            )
            figp.add_vline(x=0, line_dash="dot", line_color="gray")
            st.plotly_chart(figp, use_container_width=True)
        else:
            st.info("No hay datos para graficar top positivo.")
    with gp2:
        st.write("Top 20 gap negativo (interno > externo)")
        tn = tn_all.copy()
        if len(tn):
            tn["sku_label"] = (
                tn["sku"].astype(str)
                + " - "
                + tn.get("descripcion", pd.Series([""] * len(tn))).astype(str)
            )
            fign = px.bar(
                tn.sort_values("gap_abs_clip"),
                x="gap_abs_clip",
                y="sku_label",
                color="categoria",
                orientation="h",
                hover_data=["sku", "familia", "precio_interno", "precio_externo", "gap_abs", "gap_pct"],
            )
            fign.update_layout(
                height=max(420, 22 * len(tn) + 120),
                xaxis_title="Gap absoluto (externo - interno)",
                yaxis_title="SKU",
                legend_title_text="Categoría",
                margin=dict(l=200),
            )
            fign.add_vline(x=0, line_dash="dot", line_color="gray")
            st.plotly_chart(fign, use_container_width=True)
        else:
            st.info("No hay datos para graficar top negativo.")

    # Tablas con detalle de top positivos y negativos
    c1, c2 = st.columns(2)
    with c1:
        st.write("Tabla: Top 20 gap positivo (externo > interno)")
        st.dataframe(tp_all, use_container_width=True, height=520)
    with c2:
        st.write("Tabla: Top 20 gap negativo (interno > externo)")
        st.dataframe(tn_all, use_container_width=True, height=520)


# 4) Categoría | Familia: Top 15 y Box
with tabs[3]:
    st.subheader("Agregado: Categoría | Familia")
    st.caption("Se usa gap_% por SKU. 'Top 15' muestra el promedio de gap_% por Categoría | Familia; el boxplot muestra su distribución.")
    st.info(
        "Agregacion por Categoria | Familia:\n"
        "- 'Top 15' muestra el promedio de gap_% por grupo\n"
        "- El boxplot muestra la distribucion de gap_% en cada grupo"
    )
    # Filtro por familia
    fam_opts_cf = []
    if "familia" in model["data"].columns:
        fam_opts_cf = sorted([str(x) for x in model["data"]["familia"].dropna().unique()])
    fam_sel_cf = st.multiselect("Filtrar por familia (afecta Top 15 y Box)", options=fam_opts_cf, key="filtro_familia_catfam")

    # Data filtrada
    top15_cf = model["top15_cf"].copy()
    df_box_all = model["df_box"].copy()
    if fam_sel_cf:
        if "familia" in top15_cf.columns:
            top15_cf = top15_cf[top15_cf["familia"].astype(str).isin(fam_sel_cf)]
        if "familia" in df_box_all.columns:
            df_box_all = df_box_all[df_box_all["familia"].astype(str).isin(fam_sel_cf)]

    c1, c2 = st.columns([1, 1])
    with c1:
        st.write("Top 15 familias por gap % promedio (externo vs interno)")
        st.dataframe(top15_cf, use_container_width=True, height=520)
    with c2:
        st.write("Distribución gap % por Categoría | Familia")
        df_box = df_box_all
        if len(df_box):
            fig_box = px.box(df_box, y="cat_fam", x="gap_pct", points="outliers")
            fig_box.update_layout(
                yaxis={"automargin": True},
                xaxis_title="Gap %",
                yaxis_title="Categoría | Familia",
                height=max(800, 22 * df_box["cat_fam"].nunique() + 220),
                margin=dict(l=420),
            )
            fig_box.add_vline(x=0, line_dash="dot", line_color="gray")
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Sin datos suficientes para graficar")


# 5) Heatmap: ratio por Familia y Canal (vs interno)
with tabs[4]:
    st.subheader("Comparacion por Familia: precios interno vs canales")
    st.caption("Compara precios absolutos por familia entre el canal interno y otros canales. Filtra por categoria, familia y canales.")
    st.info(
        "Lectura:\n"
        "- Cada palito y marcador representa el precio agregado por canal (promedio por defecto, o mediana).\n"
        "- El canal interno aparece destacado y puede incluirse/excluirse en los filtros."
    )
    # Nota: Esta pestaña muestra ahora precios absolutos por familia y canal
    st.caption("Vista actual: comparacion de precios por familia (interno vs canales) usando grafico tipo lollipop.")
    fam_canal = model["fam_canal"]
    fam_counts = model.get("fam_canal_counts")
    # Vista alternativa: barras facetadas por familia + distribucin por canal
    if fam_canal is not None and len(fam_canal):
        # Tomar canales desde los datos crudos para incluir tambien el interno
        try:
            channels_new = sorted(model["data"]["canal"].astype(str).dropna().unique().tolist())
        except Exception:
            channels_new = list(fam_canal.columns)
        sel_channels_new = st.multiselect("Canales a mostrar", options=channels_new, default=channels_new, key="hm_channels_new")
        # Filtros de categor y familia
        cat_opts_new = sorted([str(x) for x in model["data"]["categoria"].dropna().unique()]) if "categoria" in model["data"].columns else []
        sel_cats_new = st.multiselect("Filtrar categor", options=cat_opts_new, key="hm_cats_new")
        # Familias desde indice "Cat | Fam"
        try:
            fam_opts_new = sorted(pd.Series(fam_canal.index).astype(str).str.split(" | ").str[1].dropna().unique())
        except Exception:
            fam_opts_new = []
        sel_fams_new = st.multiselect("Filtrar familia", options=fam_opts_new, key="hm_fams_new")
        # Controles especificos de la vista de precios
        st.subheader("Precios por familia vs canales")
        st.caption("Muestra precio interno y de canales por familia (lollipop)")
        agg_opt = st.selectbox("Agregacion de precio", options=["mediana", "promedio"], index=1, key="fam_agg")
        show_internal_always = st.checkbox("Mostrar canal interno destacado", value=True, key="fam_show_int")

        # Construccion de tabla de precios por familia x canal (absoluto)
        data_all = model["data"].copy()
        if sel_cats_new and "categoria" in data_all.columns:
            data_all = data_all[data_all["categoria"].astype(str).isin(sel_cats_new)]
        if sel_fams_new and "familia" in data_all.columns:
            data_all = data_all[data_all["familia"].astype(str).isin(sel_fams_new)]

        # Precios por SKU
        interno_sku = (
            data_all.query("canal == @canal_interno")[["sku", "familia", "categoria", "precio_analisis"]]
            .groupby(["sku", "familia", "categoria"], as_index=False)["precio_analisis"].median()
            .rename(columns={"precio_analisis": "precio_interno"})
        )
        externo_sku = (
            data_all.query("canal != @canal_interno")[["sku", "familia", "categoria", "canal", "precio_analisis"]]
            .groupby(["sku", "familia", "categoria", "canal"], as_index=False)["precio_analisis"].median()
            .rename(columns={"precio_analisis": "precio_canal"})
        )

        agg_func = "median" if agg_opt == "mediana" else "mean"
        intern_fam = getattr(interno_sku.groupby(["categoria", "familia"])["precio_interno"], agg_func)().reset_index()
        ext_fam = getattr(externo_sku.groupby(["categoria", "familia", "canal"])["precio_canal"], agg_func)().reset_index()

        df_int_long = intern_fam.assign(canal=canal_interno, precio=intern_fam["precio_interno"])[["categoria", "familia", "canal", "precio"]]
        df_ext_long = ext_fam.rename(columns={"precio_canal": "precio"})[["categoria", "familia", "canal", "precio"]]
        df_prices = pd.concat([df_int_long, df_ext_long], ignore_index=True)

        # Aplicar canales seleccionados
        sel_can = sel_channels_new if sel_channels_new else list(df_prices["canal"].astype(str).unique())
        df_prices = df_prices[df_prices["canal"].astype(str).isin(sel_can)]

        if len(df_prices) == 0:
            st.info("Sin datos para graficar con los filtros actuales.")
        else:
            familias = sorted(df_prices["familia"].astype(str).unique())
            # Colores por canal
            palette = px.colors.qualitative.Set2 + px.colors.qualitative.Set1 + px.colors.qualitative.Dark24
            color_map = {c: palette[i % len(palette)] for i, c in enumerate(sel_can)}
            if canal_interno in color_map:
                color_map[canal_interno] = "#FFD54F"

            # posiciones numericas por familia para separar canales
            pos_map = {fam: i for i, fam in enumerate(familias)}
            # Asegurar incluir interno destacado si se solicita
            draw_canales = list(sel_can)
            if show_internal_always and canal_interno not in draw_canales:
                draw_canales = [canal_interno] + draw_canales
            m = max(1, len(draw_canales))
            try:
                offsets = list(np.linspace(-0.22, 0.22, m))
            except Exception:
                offsets = [0] * m

            fig = go.Figure()
            for j, canal in enumerate(draw_canales):
                # Asegurar único valor por familia/canal para evitar reindex con duplicados
                ser = (
                    df_prices[df_prices["canal"].astype(str) == str(canal)]
                    .groupby("familia", as_index=True)["precio"]
                    .agg("median" if agg_opt == "mediana" else "mean")
                    .reindex(familias)
                )
                # Tallos (lollipop)
                xs, ys = [], []
                for fam, val in ser.items():
                    if pd.notna(val):
                        x0 = pos_map[fam] + offsets[j]
                        xs += [x0, x0, None]
                        ys += [0, val, None]
                if xs:
                    fig.add_trace(
                        go.Scatter(
                            x=xs,
                            y=ys,
                            mode="lines",
                            line=dict(color=color_map.get(canal, "#888"), width=(4 if str(canal)==str(canal_interno) else 2)),
                            name=f"{canal} (tallo)",
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
                # Marcadores
                fig.add_trace(
                    go.Scatter(
                        x=[pos_map[f] + offsets[j] for f in familias],
                        y=ser.values,
                        mode="markers",
                        name=str(canal),
                        marker=dict(
                            color=color_map.get(canal, "#888"),
                            size=13 if str(canal) == str(canal_interno) else 9,
                            symbol="diamond" if str(canal) == str(canal_interno) else "circle",
                        ),
                        customdata=[[fam, str(canal)] for fam in familias],
                        hovertemplate="familia=%{customdata[0]}<br>canal=%{customdata[1]}<br>precio=%{y:,.0f}<extra></extra>",
                    )
                )

            fig.update_xaxes(tickmode="array", tickvals=list(range(len(familias))), ticktext=familias)
            fig.update_layout(
                xaxis_title="Familia",
                yaxis_title="Precio",
                margin=dict(l=200, r=40, t=40, b=80),
                height=max(440, 48 * max(1, len(familias)) + 160),
            )
            # Bandas alternadas para separar familias visualmente
            for i in range(len(familias)):
                if i % 2 == 0:
                    fig.add_vrect(x0=i-0.5, x1=i+0.5, fillcolor="rgba(255,255,255,0.03)", line_width=0, layer="below")
            st.plotly_chart(fig, use_container_width=True)
    if False and fam_canal is not None and len(fam_canal):
        # Controles
        channels = list(fam_canal.columns)
        sel_channels = st.multiselect("Canales a mostrar", options=channels, default=channels)
        # Filtro por familia (desde 'Cat | Fam')
        try:
            fam_opts_hm = sorted(pd.Series(fam_canal.index).astype(str).str.split(" | ").str[1].dropna().unique())
        except Exception:
            fam_opts_hm = []
        fam_sel_hm = st.multiselect("Filtrar familia (heatmap)", options=fam_opts_hm, key="filtro_familia_heatmap")
        filtro_texto = st.text_input("Filtrar Categoría | Familia (contiene)", key="filtro_heatmap")
        sort_by_dev = st.checkbox("Ordenar por desviación promedio (|ratio-1|)", value=True)
        show_numbers = st.checkbox("Mostrar números en celdas", value=False)
        add_separators = st.checkbox("Agregar separadores entre categorías", value=True)
        rango = st.slider("Rango de color (± alrededor de 1)", min_value=0.05, max_value=1.0, value=0.3, step=0.05)

        fc = fam_canal.copy()
        # Aplicar filtro de familia primero
        if 'fam_sel_hm' in locals() and fam_sel_hm:
            try:
                fam_part = pd.Series(fc.index, index=fc.index).astype(str).str.split(" | ").str[1]
                fc = fc[fam_part.isin(fam_sel_hm)]
            except Exception:
                pass
        if sel_channels:
            fc = fc[sel_channels]
        if filtro_texto:
            fc = fc[fc.index.str.contains(filtro_texto, case=False, na=False)]
        if sort_by_dev and len(fc.columns):
            order = (fc - 1.0).abs().mean(axis=1).sort_values(ascending=False).index
            fc = fc.loc[order]

        # Counts alineados
        counts = None
        if fam_counts is not None:
            counts = fam_counts.reindex(fc.index)
            if sel_channels:
                counts = counts[sel_channels]

        # Separadores por categoría (filas NaN entre grupos)
        if add_separators:
            # Derivar categoría por fila
            try:
                row_meta = model.get("row_meta")
                meta_ser = row_meta.reindex(fc.index)["categoria"].astype(str)
            except Exception:
                meta_ser = pd.Series(fc.index, index=fc.index).astype(str).str.split(" \| ").str[0]

            fc_blocks = []
            ct_blocks = [] if counts is not None else None
            last_cat = None
            sep_count = 0
            for idx, row in fc.iterrows():
                cat = meta_ser.loc[idx]
                if last_cat is not None and cat != last_cat:
                    # insertar fila separador
                    sep_label = " " * (1 + (sep_count % 3))  # etiquetas en blanco únicas
                    fc_blocks.append(pd.DataFrame([np.nan] * len(fc.columns), index=fc.columns).T.set_index(pd.Index([sep_label])))
                    if ct_blocks is not None:
                        ct_blocks.append(pd.DataFrame([np.nan] * len(counts.columns), index=counts.columns).T.set_index(pd.Index([sep_label])))
                    sep_count += 1
                fc_blocks.append(row.to_frame().T)
                if ct_blocks is not None:
                    ct_blocks.append(counts.loc[[idx]] if idx in counts.index else pd.DataFrame([np.nan] * len(counts.columns), index=counts.columns).T.set_index(pd.Index([idx])))
                last_cat = cat

            fc = pd.concat(fc_blocks, axis=0)
            if counts is not None:
                counts = pd.concat(ct_blocks, axis=0)

        z = fc.values
        x = list(fc.columns)
        y = list(fc.index)
        custom = counts.values if counts is not None else np.full_like(z, np.nan, dtype=float)

        heatmap = go.Heatmap(
            z=z,
            x=x,
            y=y,
            customdata=custom,
            colorscale="RdBu",
            reversescale=True,
            zmin=1 - rango,
            zmax=1 + rango,
            zmid=1.0,
            hovertemplate="cat_fam=%{y}<br>canal=%{x}<br>ratio=%{z:.2f}<br>n=%{customdata}<extra></extra>",
            colorbar=dict(title="ratio", len=0.8),
            showscale=True,
        )

        fig = go.Figure(data=[heatmap])
        fig.update_layout(
            xaxis_title="Canal",
            yaxis_title="Categoría | Familia",
            title="Ratio promedio (precio canal / interno)",
            margin=dict(l=340, r=60, t=60, b=40),
            height=max(500, min(1000, 24 * max(1, len(y)) + 140)),
        )
        if show_numbers:
            heatmap.text = np.round(z, 2)
            heatmap.texttemplate = "%{text}"
            heatmap.textfont = dict(size=10)

        st.plotly_chart(fig, use_container_width=True)
    else:
        pass


# 6) Detalle por SKU x Fuente
with tabs[5]:
    st.subheader("Detalle por SKU x Fuente (precios comparados)")
    st.caption("Por SKU: min_ext = mínimo externo; max_ext = máximo externo; gap_min_% = (min_ext / interno) - 1; gap_max_% = (max_ext / interno) - 1.")
    st.info(
        "Detalle por SKU x Fuente:\n"
        "- min_ext = minimo precio externo\n"
        "- max_ext = maximo precio externo\n"
        "- gap_min_% = (min_ext / interno) - 1\n"
        "- gap_max_% = (max_ext / interno) - 1"
    )
    # Filtro por familia
    fam_opts_pvt = []
    if "familia" in model["pvt"].columns:
        fam_opts_pvt = sorted([str(x) for x in model["pvt"]["familia"].dropna().unique()])
    fam_sel_pvt = st.multiselect("Filtrar por familia (afecta la tabla)", options=fam_opts_pvt, key="filtro_familia_pvt")

    pvt = model["pvt"].copy()
    if fam_sel_pvt and "familia" in pvt.columns:
        pvt = pvt[pvt["familia"].astype(str).isin(fam_sel_pvt)]
    txt = st.text_input("Filtrar SKU o descripción (contiene)", key="filtro_pvt")
    if txt:
        pvt = filter_by_sku_text(pvt, txt)
    # Columnas de fuentes externas para resaltar mínimos y máximos
    cols_fuentes = model.get("cols_fuentes")
    if not cols_fuentes:
        cols_fuentes = [c for c in pvt.columns if c not in ["sku", "descripcion", "categoria", "familia", "interno", "min_ext", "max_ext", "fuente_min", "fuente_max", "gap_min_pct", "gap_max_pct"]]

    # Estilo: colores con buen contraste para modo oscuro
    def _style_row(s: pd.Series):
        styles = [''] * len(s)
        idx = {c: i for i, c in enumerate(s.index)}
        if 'interno' in idx and pd.notna(s.get('interno')):
            styles[idx['interno']] = 'background-color:#4a3b00;color:#ffd54f'
        ext_vals = s[cols_fuentes]
        ext_vals = ext_vals[ext_vals.notna()]
        if len(ext_vals):
            cmin = ext_vals.idxmin(); cmax = ext_vals.idxmax()
            if cmin in idx:
                styles[idx[cmin]] = 'background-color:#0b3d2e;color:#d6ffe6'
            if cmax in idx:
                styles[idx[cmax]] = 'background-color:#5b0f0f;color:#ffd6d6'
        return styles

    # Formatos
    num_cols = [c for c in pvt.columns if pvt[c].dtype.kind in "biufc" and c not in ["gap_min_pct", "gap_max_pct"]]
    fmt = {c: "{:,.0f}" for c in num_cols}
    fmt.update({"gap_min_pct": "{:+.1%}", "gap_max_pct": "{:+.1%}"})

    try:
        styled = pvt.style.apply(_style_row, axis=1, subset=pvt.columns).format(fmt)
        # Mostrar una sola tabla, interactiva y con altura ampliada
        st.dataframe(styled, use_container_width=True, height=760)
    except Exception:
        st.dataframe(pvt, use_container_width=True, height=760)


# 7) Exportar
with tabs[6]:
    st.subheader("Exportar resultados")
    st.caption("Las tablas exportadas usan las definiciones anteriores de precios, gaps y ratios.")
    st.info(
        "Exportacion:\n"
        "- Las tablas respetan las definiciones de precios, gaps y ratios"
    )
    st.write("Descarga las tablas clave en Excel y CSV.")

    out_tables = {
        "sku_comp": model["sku_comp"],
        "canal_vs_interno": model["res_canal"],
        "precios_por_fuente": model["pvt"],
    }

    # Excel único con hojas
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as wb:
        for name, df in out_tables.items():
            # Limitar tamaño de hoja si es muy grande por performance del viewer
            df.to_excel(wb, sheet_name=name[:31], index=False)
    st.download_button(
        label="Descargar Excel (analisis_posicionamiento.xlsx)",
        data=buffer.getvalue(),
        file_name="analisis_posicionamiento.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # CSV adicional de precios por fuente
    csv_bytes = model["pvt"].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Descargar CSV (precios_por_fuente.csv)",
        data=csv_bytes,
        file_name="precios_por_fuente.csv",
        mime="text/csv",
    )
