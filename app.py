import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google import genai
import json
import matplotlib.pyplot as plt
import pymssql
import io

#import warnings
#from plotly.subplots import make_subplots


# ==========================================
# 1. CONFIGURACIÓN INICIAL DE LA APP
# ==========================================
st.set_page_config(
    page_title="Mantenimiento Green Gold", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CLAVE DE LA API PARA ANÁLISIS AI
API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

# ESTRUCTURA DE ICONO Y TITULO
col_logo, col_titulo = st.columns([1, 8])

with col_logo:
    st.image("GREEN GOLD.png", width=80)

with col_titulo:
    st.title("Dashboard Mantenimiento Green Gold Organic")

 
# --- 2. FUNCIÓN PARA GENERAR EL HTML DE LA CARD (INFORMACIÓN SUPERIOR DEL DASHBOARD) ---
def create_card(icon, title, value):
    return f"""
        <div class="metric-card">
            <div class="metric-icon"><i class="fa-solid {icon}"></i></div>
            <div>
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
            </div>
        </div>
    """


# --- 2. CARGA DE DATOS / CONEXIÓN SQL (CON CACHÉ) ---
@st.cache_data(ttl=600)
def cargar_datos():
    server = st.secrets["DB_SERVER"]
    user = st.secrets["DB_USER"]
    password = st.secrets["DB_PASSWORD"]
    database = st.secrets["DB_NAME"]

    try:
        conn = pymssql.connect(server, user, password, database)
        query = """
        SELECT
            sm.id_solicitud,
            sm.folio_solicitud,
            sm.fecha_registro,
            sm.hora_registro,
            u.nombre As Ubicación,
            (e.nombre + ' ' + e.apellido_paterno) As Reporta,
            (c.clasificacion + ' ' + c.nombre_categoria) As Categoría,
            (m.no_economico + ' ' + m.especificacion) As Maquinaria,
            sm.horometro,
            sm.fecha_servicio,
            sm.fecha_entrega,
            sm.grado_urgencia,
            (resp.nombre + ' ' + resp.apellido_paterno) As Responsable,
            sm.costo_reparacion,
            sm.tipo_servicio,
            (rep.nombre + ' ' + rep.apellido_paterno) As Reparación,
            sm.proveedor_reparacion,
            sm.descripcion_problema,
            sm.estatus,
            (usuario.nombre + ' ' + usuario.apellido_paterno) AS Usuario,
            i.nombre As Instalación,
            sm.Asignado
        FROM SOLICITUD_MTTO sm
        INNER JOIN UBICACION u ON sm.id_ubicacion = u.id_ubicacion
        INNER JOIN EMPLEADO e ON sm.id_empleado = e.id_empleado
        INNER JOIN CATEGORIA c ON sm.id_categoria = c.id_categoria
        LEFT JOIN MAQUINARIA m ON sm.id_maquinaria = m.id_maquinaria
        INNER JOIN EMPLEADO resp ON sm.id_responsable = resp.id_empleado
        LEFT JOIN EMPLEADO rep ON sm.id_respReparacion = rep.id_empleado
        INNER JOIN USUARIO us ON sm.id_usuario = us.id_usuario
        INNER JOIN EMPLEADO usuario ON us.id_empleado = usuario.id_empleado
        LEFT JOIN INSTALACION i ON sm.id_instalacion = i.id_instalacion
        """
        df = pd.read_sql(query, conn)
        conn.close()

        # --- LIMPIEZA INICIAL ---
        df.columns = [col.lower().strip() for col in df.columns]
        df['fecha_registro'] = pd.to_datetime(df['fecha_registro'], dayfirst=True, errors='coerce')
        df['fecha_servicio'] = pd.to_datetime(df['fecha_servicio'], dayfirst=True, errors='coerce')
        df['fecha_entrega'] = pd.to_datetime(df['fecha_entrega'], dayfirst=True, errors='coerce')
        df['costo_reparacion'] = pd.to_numeric(df['costo_reparacion'], errors='coerce').fillna(0)

        # Cálculo MTTR
        df['dias_reparacion'] = (df['fecha_entrega'] - df['fecha_servicio']).dt.days
        df.loc[df['dias_reparacion'] < 0, 'dias_reparacion'] = 0

        # Columnas para filtros
        df['año'] = df['fecha_registro'].dt.year
        df['mes_nombre'] = df['fecha_registro'].dt.strftime('%b-%y')
        df['mes_sort'] = df['fecha_registro'].dt.to_period('M')

        # Mapeo de Clasificación
        def mapear_clasificacion(cat):
            cat = str(cat).upper()
            if 'AGRICOLA' in cat or 'AGRÍCOLA' in cat:
                if 'MAQUINARIA' in cat: return 'Maquinaria Agrícola'
                if 'HERRAMIENTA' in cat: return 'Herramienta Agrícola'
            if 'PESADA' in cat: return 'Maquinaria Pesada'
            if 'INSTALACION' in cat or 'INSTALACIÓN' in cat: return 'Instalaciones'
            return 'Otros'

        df['clasificacion_filtro'] = df['categoría'].apply(mapear_clasificacion)
        return df
    except Exception as connection_error:
        st.error(f"⚠️ Error al conectar con SQL Server: {connection_error}")
        return pd.DataFrame()


# -------------- INICIO ANALISIS DE LAS FALLAS MÁS RECURRENTES CON IA --------------------
# ==========================================
# 2. FUNCIONES DE PROCESAMIENTO E IA
# ==========================================
def analizar_fallas_con_ia(df_filtrado):
    if df_filtrado.empty:
        return None

    # Preparamos los datos (últimos 30 reportes para no saturar)
    reportes = ""
    for idx, row in df_filtrado.tail(30).iterrows():
        reportes += f"ID:{idx} - {row['descripcion_problema']} | "

    prompt = f"""
    Clasifica estos reportes en 5 categorías técnicas.
    Reportes: {reportes}
    
    Responde ÚNICAMENTE un JSON con esta estructura:
    {{
      "Categorías": {{"Motor": 5, "Hidráulico": 2}},
      "Asignaciones": {{"Motor": [10, 15, 20], "Hidráulico": [12, 14]}}
    }}
    Donde los números en 'Asignaciones' son los IDs que te envié.
    """

    try:
        # Usamos el modelo que confirmamos en tu lista
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite", 
            contents=prompt
        )
        
        # Limpiamos la respuesta para asegurar que sea un JSON válido
        texto_limpio = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(texto_limpio)
    except Exception as e:
        st.error(f"Error en el análisis de IA: {e}")
        return None


# -------------- FIN ANALISIS DE LAS FALLAS MÁS RECURRENTES CON IA --------------------

# --- 1. ESTILO CSS PERSONALIZADO (Añadir al inicio del bloque de métricas) ---
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <style>
    [data-testid="stTabs"] {
        margin-top: 30px !important;
    }

    /* 🟢 CORRECCIÓN DE OPACIDAD: Forzamos la inyección visual en los botones primarios */
    button[data-testid="baseButton-primary"] {
        background-color: rgba(46, 125, 50, 0.20) !important; /* Fondo verde ultra tenue al 20% */
        color: #A9DFBF !important; /* Texto en verde pastel claro de alta legibilidad */
        border: 1px solid rgba(46, 125, 50, 0.4) !important; /* Borde suave traslúcido */
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.2s ease-in-out !important;
    }

    /* Hover sobre los botones tenues seleccionados */
    button[data-testid="baseButton-primary"]:hover {
        background-color: rgba(46, 125, 50, 0.40) !important; /* Se ilumina un poco más */
        color: #FFFFFF !important;
        border: 1px solid #2E7D32 !important;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.3) !important;
    }

    /* Evitar que Streamlit altere el color interno del texto del botón primario */
    button[data-testid="baseButton-primary"] p, 
    button[data-testid="baseButton-primary"] div,
    button[data-testid="baseButton-primary"] span {
        color: #A9DFBF !important;
    }
    button[data-testid="baseButton-primary"]:hover p {
        color: #FFFFFF !important;
    }

    .metric-card {
        background-color: #121212;
        border-left: 5px solid #2E7D32;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 2px 4px 15px rgba(0,0,0,0.5);
        transition: all 0.3s ease-in-out;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 20px;
        height: 120px;
    }
    .metric-card:hover {
        transform: translateY(-8px);
        border-left: 5px solid #FFD700;
        box-shadow: 0 10px 25px rgba(46, 125, 50, 0.3);
    }
    .metric-icon {
        font-size: 35px;
        color: #2E7D32;
        transition: color 0.3s ease;
    }
    .metric-card:hover .metric-icon {
        color: #FFD700;
    }
    .metric-title {
        color: #C0C0C0;
        font-size: 14px;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 28px;
        font-weight: bold;
    }

    /* Cambia el tamaño y estilo de las pestañas en su estado normal */
    button[data-baseweb="tab"] p {
        font-size: 20px !important; /* Ajusta este número (ej: 22px, 24px) a tu gusto */
        font-weight: bold !important;
    }

    /* Opcional: Cambia el color del texto de la pestaña activa para que resalte más */
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #FFD700 !important; /* Dorado corporativo para la pestaña seleccionada */
    }
    
</style>
""", unsafe_allow_html=True)


df_raw = cargar_datos()

# --- 3. BARRA LATERAL CON FILTROS DINÁMICOS ---
st.sidebar.title("🎯 Filtros de Datos")

años_disp = sorted(df_raw['año'].dropna().unique().astype(int), reverse=True)
f_año = st.sidebar.multiselect("Año:", años_disp, default=años_disp[:1])

df_temp_mes = df_raw[df_raw['año'].isin(f_año)]
meses_disp = df_temp_mes.sort_values('mes_sort')['mes_nombre'].unique()
f_mes = st.sidebar.multiselect("Mes:", meses_disp, default=meses_disp)

clases_disp = ["Maquinaria Agrícola", "Maquinaria Pesada", "Herramienta Agrícola", "Instalaciones"]
f_clas = st.sidebar.multiselect("Clasificación:", clases_disp, default=clases_disp)

df_temp_ubi = df_raw[
    (df_raw['año'].isin(f_año)) & 
    (df_raw['mes_nombre'].isin(f_mes)) & 
    (df_raw['clasificacion_filtro'].isin(f_clas))
]
ubis_disp = sorted(df_temp_ubi['ubicación'].unique())
f_ubi = st.sidebar.multiselect("📍 Ubicación:", ubis_disp, default=ubis_disp)

# --- APLICACIÓN DE FILTROS ---
df_res = df_raw[
    (df_raw['año'].isin(f_año)) & 
    (df_raw['mes_nombre'].isin(f_mes)) & 
    (df_raw['clasificacion_filtro'].isin(f_clas)) &
    (df_raw['ubicación'].isin(f_ubi))
].copy()

# --- LIMPIEZA DE URGENCIA Y PROVEEDORES ---
df_res['urg_final'] = df_res['grado_urgencia'].astype(str).str.strip().str.upper()
mapeo_urg = {'URGENTE': 'Urgente', 'ALTA': 'Urgente', 'MEDIO': 'Media', 'BAJO': 'Baja'}
df_res['urg_final'] = df_res['urg_final'].map(mapeo_urg).fillna('S/E')

# Limpieza de proveedores
mapeo_duplicados = { 
    'georgina elisabeth': 'Georgina Elizabeth', 'georgiana elisabeth': 'Georgina Elizabeth', 
    'georgiana elizabeth': 'Georgina Elizabeth', 'gerogina elizabet': 'Georgina Elizabeth', 
    'georgina elizabet': 'Georgina Elizabeth', 'georgina elizabhet': 'Georgina Elizabeth', 
    'georgina elizabth': 'Georgina Elizabeth', 'jeorgina elisabeth': 'Georgina Elizabeth', 
    'juaquin inojosa': 'Joaquín Hinojosa', 'juaquin hinojosa': 'Joaquín Hinojosa',
    'juaquin inojoza': 'Joaquín Hinojosa', 'juaquin hinojoza': 'Joaquín Hinojosa', 
    'joaquin inojoza': 'Joaquín Hinojosa', 'joaquin inojosa': 'Joaquín Hinojosa', 
    'joaquin hinojoza': 'Joaquín Hinojosa', 'joaquin hinojosa': 'Joaquín Hinojosa',
    'none': 'Serv. Internos', 'nan': 'Serv. Internos'
}
df_res['prov_limpio'] = df_res['proveedor_reparacion'].astype(str).str.strip().str.lower()
df_res['prov_limpio'] = df_res['prov_limpio'].map(mapeo_duplicados).fillna(df_res['prov_limpio'].str.title())


# ===================================================
# 💾 BOTÓN DE EXPORTACIÓN A EXCEL (REPORTES DINÁMICOS)
# ===================================================

st.sidebar.write("---")
st.sidebar.markdown("### 📥 Reportes Ejecutivos")

if not df_res.empty:
    # 1. Creamos un buffer en memoria para simular el archivo de Excel sin guardarlo en el disco del servidor.
    buffer = io.BytesIO()

    # 1. Diccionario de mapeo: 'nombre_columna_sql' : 'Nombre Limpio para Excel'
    # Ajusta los nombres del lado derecho según cómo quieras que se lean en tu reporte
    diccionario_columnas = {
        'id_solicitud': 'ID Solicitud',
        'folio_solicitud': 'Folio',
        'ubicación': 'Ubicación',
        'reporta': 'Empleado Reporta',
        'categoría': 'Categoría',
        'maquinaria': 'No.Económico',
        'horometro': 'Horometro',
        'fecha_servicio': 'Fecha Reparación',
        'fecha_entrega': 'Fecha Entrega',
        'grado_urgencia': 'Grado de Urgencia',
        'responsable': 'Responsable',
        'costo_reparacion': 'Costo Aproximado',
        'tipo_servicio': 'Tipo de Servicio',
        'proveedor_reparacion': 'Reparado Por',
        'descripcion_problema': 'Descripción del Problema',
        'estatus': 'Estatus de la Solicitud'
    } 

    # 2. Copiamos y preparamos el DataFrame para la exportación.
    columnas_deseadas = [
        'id_solicitud',
        'folio_solicitud',
        'ubicación',
        'reporta',
        'categoría',
        'maquinaria',
        'horometro',
        'fecha_servicio',
        'fecha_entrega',
        'grado_urgencia',
        'responsable',
        'costo_reparacion',
        'tipo_servicio',
        'proveedor_reparacion',
        'descripcion_problema',
        'estatus'
    ]

    df_export = df_res[columnas_deseadas].copy()

    for col_fecha in ['fecha_servicio', 'fecha_entrega']:
        if col_fecha in df_export.columns:
            df_export[col_fecha] = pd.to_datetime(df_export[col_fecha], errors='coerce').dt.strftime('%d-%m-%Y')

    df_export = df_export.rename(columns=diccionario_columnas)


    # 3. Inicializamos el escritor de Excel de Pandas con XlsxWriter
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_export.to_excel(writer, sheet_name='Mantenimiento_Filtrado', index=False)

        # --- APERTURA DEL MOTOR PARA ESTILOS ---
        workbook = writer.book
        worksheet = writer.sheets['Mantenimiento_Filtrado']

        # Definimos el rango total de la tabla (desde la celda A1 hasta la última columna y fila)
        max_row, max_col = df_export.shape
        column_settings = [{'header': column} for column in df_export.columns]

        # Aplicar formato (estilos) de tabla oficial
        worksheet.add_table(0, 0, max_row, max_col - 1, {
            'columns': column_settings,
            'style': 'TableStyleMedium2', # Formato de tabla integrado de Excel
            'banded_rows': True
        })

        # --- CONFIGURACIÓN DE ALINEACIONES Y FORMATOS ---
        # 1. Creamos los estilos de alineación
        formato_centro = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
        formato_derecha = workbook.add_format({'align': 'right', 'valign': 'vcenter'})
        formato_izquierda = workbook.add_format({'align': 'left', 'valign':'vcenter'})

        formato_fecha = workbook.add_format({'num_format': 'dd-mm-yyyy', 'align': 'center', 'valign': 'vcenter'})

        for i, col in enumerate(df_export.columns): 
            # 🟢 CORRECCIÓN DE ANCHO SEGURA: Filtramos nulos antes de medir cadenas
            valores_validos = df_export[col].dropna().astype(str)
            
            if not valores_validos.empty:
                max_len = max(valores_validos.map(len).max(), len(str(col))) + 3
            else:
                max_len = len(str(col)) + 5 # Si está vacía la columna, hereda el tamaño del título

            if max_len > 40: max_len = 40

            col_original = columnas_deseadas[i]

            if col_original in ['fecha_servicio', 'fecha_entrega']:
                worksheet.set_column(i, i, max_len, formato_fecha)
            
            elif col_original in ['id_solicitud', 'folio_solicitud', 'ubicación', 'maquinaria', 'horometro', 'grado_urgencia', 'costo_reparacion', 'tipo_servicio', 'estatus']:
                worksheet.set_column(i, i, max_len, formato_centro)
            
            elif col_original in ['reporta', 'categoría', 'responsable', 'proveedor_reparacion', 'descripcion_problema']:
                worksheet.set_column(i, i, max_len, formato_izquierda)
            else:
                worksheet.set_column(i, i, max_len, formato_izquierda)

    st.sidebar.download_button(
        label="🟢 Descargar Reporte en Excel",
        data=buffer.getvalue(),
        file_name=f"Reporte_Mantenimiento_{df_res['año'].iloc[0] if 'año' in df_res.columns else '2026'}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
else:
    st.sidebar.warning("⚠️ No hay datos seleccionados para exportar.")


# --- MÉTRICAS PRINCIPALES (CARDS) ---
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

total_solicitudes = len(df_res)
gasto_total = df_res['costo_reparacion'].sum()
serv_urgentes = len(df_res[df_res['urg_final'] == 'Urgente'])
porcentaje_urgencia = (serv_urgentes / total_solicitudes * 100) if total_solicitudes > 0 else 0
mttr = df_res['dias_reparacion'].mean() if total_solicitudes > 0 else 0 

with col_m1:
    #st.metric(label="Total Solicitudes", value=f"{total_solicitudes}")
    st.markdown(create_card("fa-clipboard-list", "Total Solicitudes", f"{total_solicitudes}"), unsafe_allow_html=True)
with col_m2:
    #st.metric(label="Gasto Total", value=f"${gasto_total:,.0f}")
    st.markdown(create_card("fa-sack-dollar", "Gasto Total", f"${gasto_total:,.0f}"), unsafe_allow_html=True)
with col_m3:
    #st.metric(label="% Urgencias", value=f"{porcentaje_urgencia:.1f}%", delta=f"{serv_urgentes} críticos", delta_color="inverse")
    st.markdown(create_card("fa-triangle-exclamation", "% Servicios Urgentes", f"{porcentaje_urgencia:.1f}%"), unsafe_allow_html=True)
with col_m4:
    #st.metric(label="MTTR (Días)", value=f"{mttr:.1f} días", help="Tiempo promedio desde el reporte hasta la entrega.")
    st.markdown(create_card("fa-stopwatch", "Promedio Resolución", f"{mttr:.1f} días"), unsafe_allow_html=True)

#st.divider()


# --- 4. DISEÑO DEL DASHBOARD (PLOTLY) ---
config_plotly = {'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True}

# --- CONSTRUCCIÓN DE LAS 6 FIGURAS (Solo asignación de variables, sin st.plotly_chart) ---

# 1. Usuarios
user_data = df_res.groupby('usuario').agg({'id_solicitud': 'count', 'costo_reparacion': 'sum'}).nlargest(10, 'id_solicitud').sort_values('id_solicitud')
fig_usuarios = go.Figure(go.Bar(
    y=user_data.index, 
    x=user_data['id_solicitud'], 
    orientation='h',
    marker_color='#1D3557', 
    text=[f"${c:,.0f}" for c in user_data['costo_reparacion']], 
    textposition='outside',
    customdata=user_data['costo_reparacion'].values,

    hovertemplate="👤 <b>Usuario:</b> %{y}<br>" +
                  "🛠️ <b>No. Servicios:</b> %{x}<br>" +
                  "💰 <b>Gasto Total:</b> %{customdata:$,.0f}<extra></extra>"
))
fig_usuarios.update_layout(title="📊 Top 10 Usuarios (Demanda y Gasto)",template="plotly_dark", margin=dict(l=10, r=10, t=50, b=10))

# 2. Maquinaria
from plotly.subplots import make_subplots 
maq_data = df_res.groupby('maquinaria').agg({'id_solicitud': 'count', 'costo_reparacion': 'sum'}).nlargest(10, 'id_solicitud')
fig_maquinaria = make_subplots(specs=[[{"secondary_y": True}]])
fig_maquinaria.add_trace(go.Bar(x=maq_data.index, y=maq_data['id_solicitud'], name="Servicios", marker_color='#A8DADC'), secondary_y=False)
fig_maquinaria.add_trace(go.Scatter(x=maq_data.index, y=maq_data['costo_reparacion'], name="Gasto", line=dict(color='#E63946', width=3)), secondary_y=True)
fig_maquinaria.update_layout(title="🚜 Maquinaria: Volumen vs Gasto Total",template="plotly_dark", hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10))

# 3. Histórico
mes_data_plot = df_res.groupby(['mes_sort', 'mes_nombre']).agg({'id_solicitud': 'count', 'costo_reparacion': 'sum'}).reset_index()
fig_historico = make_subplots(specs=[[{"secondary_y": True}]])
fig_historico.add_trace(go.Bar(x=mes_data_plot['mes_nombre'], y=mes_data_plot['id_solicitud'], name="Servicios", marker_color='#A8DADC'), secondary_y=False)
fig_historico.add_trace(go.Scatter(x=mes_data_plot['mes_nombre'], y=mes_data_plot['costo_reparacion'], name="Gasto", line=dict(color='#E63946', width=3)), secondary_y=True)
fig_historico.update_layout(title="📅 Histórico Mensual: Cantidad y Costos",template="plotly_dark", hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10))

# 4. Ubicaciones
ubi_data = df_res.groupby('ubicación').agg({'id_solicitud': 'count', 'costo_reparacion': 'sum'}).nlargest(10, 'id_solicitud').sort_values('id_solicitud')
fig_ubicaciones = go.Figure()
fig_ubicaciones.add_trace(go.Bar(
    y=ubi_data.index, 
    x=ubi_data['id_solicitud'], 
    orientation='h', 
    marker_color='#457B9D',
    text=[f"${c:,.0f}" for c in ubi_data['costo_reparacion']], 
    textposition='outside', 
    cliponaxis=False, 
    customdata=ubi_data['costo_reparacion'].values,
    
    hovertemplate="👤 <b>Ubicación:</b> %{y}<br>" +
                  "🛠️ <b>No. Servicios:</b> %{x}<br>" +
                  "💰 <b>Gasto Total:</b> %{customdata:$,.0f}<extra></extra>"
))
fig_ubicaciones.update_layout(title="📍 Ubicaciones: Servicios y Gasto Total",
    template="plotly_dark", margin=dict(l=20, r=100, t=50, b=20),
    xaxis=dict(title="Cantidad de Servicios", range=[0, ubi_data['id_solicitud'].max() * 1.25]), yaxis=dict(title="")
)

# 5. Urgencia
urg_data_plot = df_res.groupby('urg_final').agg({'id_solicitud': 'count', 'costo_reparacion': 'sum'})
orden_plot = ['Baja', 'Media', 'Urgente', 'S/E']
urg_plot = urg_data_plot.reindex([o for o in orden_plot if o in urg_data_plot.index]).fillna(0)
colores_urg = {'Baja': '#2F9946', 'Media': '#AFB030', 'Urgente': '#E63030', 'S/E': '#A8A8A8'}
fig_urgencia = go.Figure(go.Bar(
    x=urg_plot.index, 
    y=urg_plot['id_solicitud'], 
    marker_color=[colores_urg.get(x, '#A8A8A8') for x in urg_plot.index],
    text=[f"${c:,.0f}" for c in urg_plot['costo_reparacion']], 
    textposition='outside',
    customdata=urg_data_plot['costo_reparacion'].values,
    
    hovertemplate="👤 <b>Grado:</b> %{x}<br>" +
                  "🛠️ <b>No. Servicios:</b> %{y}<br>" +
                  "💰 <b>Gasto Total:</b> %{customdata:$,.0f}<extra></extra>"
))
fig_urgencia.update_layout(title="🚨 Servicios por Grado de Urgencia",template="plotly_dark",margin=dict(l=20, r=100, t=50, b=20))

# 6. Proveedores
df_prov = df_res.copy()
df_prov['prov_limpio'] = df_prov['proveedor_reparacion'].astype(str).str.replace(',', '').str.strip().str.lower()
df_prov['prov_limpio'] = df_prov['prov_limpio'].map(mapeo_duplicados).fillna(df_prov['prov_limpio'].str.title())
excluir = ['S/E', 'Nan', 'None', '', 'Serv. Internos', 'Serv.internos', '0', 'Servicio Interno' ]
prov_filtrado = df_prov[~df_prov['prov_limpio'].isin(excluir)]
prov_data = prov_filtrado.groupby('prov_limpio').agg({'id_solicitud': 'count', 'costo_reparacion': 'sum'}).nlargest(10, 'id_solicitud').sort_values('id_solicitud')

fig_proveedores = go.Figure(go.Bar(
    y=prov_data.index, 
    x=prov_data['id_solicitud'], 
    orientation='h', 
    marker_color='#E63946', 
    text=[f"{int(n)} serv. (${c:,.0f})" for n, c in zip(prov_data['id_solicitud'], prov_data['costo_reparacion'])],
    textposition='outside', 
    cliponaxis=False,
    customdata=prov_data['costo_reparacion'].values,

    hovertemplate="👤 <b>Proveedor:</b> %{y}<br>" +
                  "🛠️ <b>No. Servicios:</b> %{x}<br>" +
                  "💰 <b>Gasto Total:</b> %{customdata:$,.0f}<extra></extra>"
))
fig_proveedores.update_layout(title="🤝 Top 10 Proveedores Externos",template="plotly_dark", margin=dict(l=10, r=100, t=50, b=10), xaxis=dict(title="Cantidad de Servicios"))


# ==========================================
# 6. ESTRUCTURA DE PESTAÑAS (TABS)
# ==========================================
tab_operativo, tab_ia = st.tabs(["📊 Panel Interactivo", "🧠 Diagnóstico IA"])

# ------------------------------------------
# TAB 1: LIENZO DINÁMICO (TIPO POWERPOINT)
# ------------------------------------------
with tab_operativo:
    if 'graficas_activas' not in st.session_state:
        st.session_state['graficas_activas'] = []

    # Dividimos la pantalla: 80% para el Lienzo Central, 20% Barra Lateral Derecha
    col_lienzo, col_miniaturas = st.columns([4, 1])

    # BARRA LATERAL DERECHA (Panel de Miniaturas)
    with col_miniaturas:
        st.markdown("<h5 style='text-align: center; color: #FFFFF;'>📊 Selecciona una Gráfica</h5>", unsafe_allow_html=True)
        
        # BOTÓN: TOP USUARIOS
        btn_usuarios_tipo = "primary" if "usuarios" in st.session_state['graficas_activas'] else "secondary"
        if st.button("👥 Top Usuarios", use_container_width=True, type=btn_usuarios_tipo):
            if "usuarios" in st.session_state['graficas_activas']: st.session_state['graficas_activas'].remove("usuarios")
            else: st.session_state['graficas_activas'].append("usuarios")
            st.rerun()

        # BOTÓN: HISTÓRICO MENSUAL
        btn_maq_tipo = "primary" if "maquinaria" in st.session_state['graficas_activas'] else "secondary"
        if st.button("🚜 Maquinaria vs Gasto", use_container_width=True, type=btn_maq_tipo):
            if "maquinaria" in st.session_state['graficas_activas']: st.session_state['graficas_activas'].remove("maquinaria")
            else: st.session_state['graficas_activas'].append("maquinaria")
            st.rerun()

        # BOTÓN: HISTÓRICO MENSUAL
        btn_hist_tipo = "primary" if "historico" in st.session_state['graficas_activas'] else "secondary"
        if st.button("📅 Histórico Mensual", use_container_width=True, type=btn_hist_tipo):
            if "historico" in st.session_state['graficas_activas']: st.session_state['graficas_activas'].remove("historico")
            else: st.session_state['graficas_activas'].append("historico")
            st.rerun()

        # BOTÓN: UBICACIONES
        btn_ubi_tipo = "primary" if "ubicaciones" in st.session_state['graficas_activas'] else "secondary"
        if st.button("📍 Costo por Ubicación", use_container_width=True, type=btn_ubi_tipo):
            if "ubicaciones" in st.session_state['graficas_activas']: st.session_state['graficas_activas'].remove("ubicaciones")
            else: st.session_state['graficas_activas'].append("ubicaciones")
            st.rerun()

        # BOTÓN: URGENCIA
        btn_urg_tipo = "primary" if "urgencia" in st.session_state['graficas_activas'] else "secondary"
        if st.button("🚨 Costos por Urgencia", use_container_width=True, type=btn_urg_tipo):
            if "urgencia" in st.session_state['graficas_activas']: st.session_state['graficas_activas'].remove("urgencia")
            else: st.session_state['graficas_activas'].append("urgencia")
            st.rerun()

        # BOTÓN: PROVEEDORES
        btn_prov_tipo = "primary" if "proveedores" in st.session_state['graficas_activas'] else "secondary"
        if st.button("🤝 Top Proveedores", use_container_width=True, type=btn_prov_tipo):
            if "proveedores" in st.session_state['graficas_activas']: st.session_state['graficas_activas'].remove("proveedores")
            else: st.session_state['graficas_activas'].append("proveedores")
            st.rerun()

        st.write("---")
        if st.button("🗑️ Limpiar Pantalla", type="primary", use_container_width=True):
            st.session_state['graficas_activas'] = []
            st.rerun()

    # LIENZO PRINCIPAL (Lógica Dinámica de Cuadrícula)
    with col_lienzo:
        lista_visibles = st.session_state['graficas_activas']
        
        if not lista_visibles:
            st.info("⬅️ Selecciona una o más diapositivas del panel derecho para comenzar a analizarlas en este espacio.")
            
        elif len(lista_visibles) == 1:
            g = lista_visibles[0]
            if g == "usuarios": st.plotly_chart(fig_usuarios, use_container_width=True, config=config_plotly)
            elif g == "maquinaria": st.plotly_chart(fig_maquinaria, use_container_width=True, config=config_plotly)
            elif g == "historico": st.plotly_chart(fig_historico, use_container_width=True, config=config_plotly)
            elif g == "ubicaciones": st.plotly_chart(fig_ubicaciones, use_container_width=True, config=config_plotly)
            elif g == "urgencia": st.plotly_chart(fig_urgencia, use_container_width=True, config=config_plotly)
            elif g == "proveedores": st.plotly_chart(fig_proveedores, use_container_width=True, config=config_plotly)

        else:
            columnas_render = st.columns(2)
            for i, g in enumerate(lista_visibles):
                with columnas_render[i % 2]:
                    if g == "usuarios": st.plotly_chart(fig_usuarios, use_container_width=True, config=config_plotly)
                    elif g == "maquinaria": st.plotly_chart(fig_maquinaria, use_container_width=True, config=config_plotly)
                    elif g == "historico": st.plotly_chart(fig_historico, use_container_width=True, config=config_plotly)
                    elif g == "ubicaciones": st.plotly_chart(fig_ubicaciones, use_container_width=True, config=config_plotly)
                    elif g == "urgencia": st.plotly_chart(fig_urgencia, use_container_width=True, config=config_plotly)
                    elif g == "proveedores": st.plotly_chart(fig_proveedores, use_container_width=True, config=config_plotly)


# ------------------------------------------
# TAB 2: EL CEREBRO DE LA IA
# ------------------------------------------
with tab_ia:
    st.header("Análisis Principales Problemas Reportados")
    
    # --- FILA 1: BOTÓN DE ACCIÓN (Ancho completo o controlado) ---
    col_btn_1, col_btn_2 = st.columns([1, 2])
    with col_btn_1:
        st.info("La IA lee todas las problemáticas de las solicitudes de mantenimiento que realizan los usuarios para agrupar/categorizar las fallas reportadas por contexto real.")
        if st.button("🚀 Iniciar Procesamiento Inteligente", use_container_width=True):
            if 'df_res' in locals() and not df_res.empty:
                with st.spinner("Gemini está analizando los problemas..."):
                    resultado = analizar_fallas_con_ia(df_res)
                    if resultado:
                        st.session_state['res_ia'] = resultado
                        st.rerun()
            else:
                st.warning("No hay datos cargados o los filtros aplicados están vacíos.")
                
    st.write("---") # Separador estético para delimitar la zona de control de la de resultados

    # --- FILA 2: CONTENEDOR DE RESULTADOS (ABAJO DEL BOTÓN) ---
    if 'res_ia' in st.session_state:
        res = st.session_state['res_ia']
        datos_grafica = res.get('Categorías', {})
        
        df_ia = pd.DataFrame(list(datos_grafica.items()), columns=['Tipo de Falla', 'Casos']).sort_values('Casos')
        
        fig_ia = px.bar(
            df_ia, x='Casos', y='Tipo de Falla', orientation='h',
            title="Principales Causas Raíz Detectadas",
            color_discrete_sequence=['#FFD700']
        )
        fig_ia.update_layout(
            template="plotly_dark", 
            clickmode='event+select',
            margin=dict(l=10, r=10, t=30, b=10)
        )

        # Aquí creamos el contenedor que se despliega abajo del botón ocupando todo el ancho de la pantalla
        contenedor_resultados = st.container()
        
        with contenedor_resultados:
            # Inicializamos la variable de la categoría en el session_state si no existe
            if 'categoria_seleccionada' not in st.session_state:
                st.session_state['categoria_seleccionada'] = None

            # Si ya se hizo clic en una barra, dividimos TODA la pantalla a la mitad (50% gráfica, 50% tabla)
            if st.session_state['categoria_seleccionada']:
                col_grafica, col_tabla = st.columns(2)
            else:
                # Si no hay clic, la gráfica aprovecha el 100% del ancho inferior de la pantalla
                col_grafica = st.container()
                col_tabla = None

            # Dibujamos la gráfica en su respectivo contenedor
            with col_grafica:
                evento_seleccion = st.plotly_chart(fig_ia, use_container_width=True, on_select="rerun")
            
            # Capturamos el clic del usuario
            if evento_seleccion and len(evento_seleccion["selection"]["points"]) > 0:
                st.session_state['categoria_seleccionada'] = evento_seleccion["selection"]["points"][0]["y"]
            else:
                st.session_state['categoria_seleccionada'] = None

            # Si el usuario seleccionó una barra, desplegamos la tabla inmediatamente al lado
            if st.session_state['categoria_seleccionada'] and col_tabla is not None:
                categoria_sel = st.session_state['categoria_seleccionada']
                ids_exactos = res.get('Asignaciones', {}).get(categoria_sel, [])
                
                with col_tabla:
                    st.markdown(f"##### 📂 Detalles: {categoria_sel}")
                    
                    if ids_exactos and 'df_res' in locals():
                        df_detalle = df_res.loc[ids_exactos]
                        
                        st.dataframe(
                            df_detalle[['maquinaria', 'descripcion_problema', 'fecha_registro']], 
                            use_container_width=True, 
                            hide_index=True,
                            height=330 # Le damos un poco más de altura para que luzca imponente
                        )
                    else:
                        st.warning("No se encontraron registros indexados.")
        
