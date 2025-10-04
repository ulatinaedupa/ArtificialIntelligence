"""
Streamlit Financial Agent - Asesor Financiero de Estados de Cuenta Bancarios
Full Streamlit UI with API key input, file upload, chat interface, export, and visualizations
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import json
import sqlite3
import chromadb
from chromadb.config import Settings
import hashlib
from pathlib import Path
import io
import base64

try:
    import pdfplumber
except:
    pdfplumber = None

# ============= PAGE CONFIG =============
st.set_page_config(
    page_title="Financial Agent - Asesor Financiero",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= CONFIGURACIÓN =============
DATABASE_CONFIG = {
    "type": "sqlite",
    "path": "estados_cuenta.db"
}

CHROMA_CONFIG = {
    "persist_directory": "./financial_db",
    "collection_name": "bank_statements"
}

CHECKPOINT_PATH = "checkpoints.db"

# ============= SESSION STATE =============
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'thread_id' not in st.session_state:
    st.session_state.thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

if 'agent' not in st.session_state:
    st.session_state.agent = None

if 'api_key' not in st.session_state:
    st.session_state.api_key = None

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# ============= CHROMADB FUNCTIONS =============
def init_chromadb():
    """Inicializa ChromaDB para búsquedas semánticas"""
    persist_dir = CHROMA_CONFIG["persist_directory"]
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    return client

# ============= TOOLS =============
@tool
def load_real_bank_statement(file_path: str) -> str:
    """
    Carga un estado de cuenta REAL desde Excel, CSV o PDF.
    Auto-detecta las columnas necesarias y construye la base de datos.

    Args:
        file_path: Ruta al archivo (Excel, CSV o PDF)

    Returns:
        JSON con resumen de la carga
    """
    try:
        if not Path(file_path).exists():
            return json.dumps({"error": f"❌ Archivo no encontrado: {file_path}"})

        ext = Path(file_path).suffix.lower()

        if ext in ['.xlsx', '.xls']:
            # Intentar leer el archivo buscando la fila de encabezados
            df_raw = pd.read_excel(file_path, header=None)

            # Buscar la fila que contiene los encabezados (palabras clave: Fecha, Descripción, Cargos, Pagos)
            header_row = None
            for idx, row in df_raw.iterrows():
                row_str = ' '.join([str(x).lower() for x in row if pd.notna(x)])
                if ('fecha' in row_str or 'date' in row_str) and ('descripci' in row_str or 'description' in row_str):
                    header_row = idx
                    break

            if header_row is not None:
                # Leer nuevamente usando la fila de encabezado encontrada
                df = pd.read_excel(file_path, header=header_row)
                # Filtrar columnas vacías
                df = df.loc[:, df.columns.notna()]
                # Filtrar filas completamente vacías
                df = df.dropna(how='all')
                # Filtrar columnas "Unnamed" que estén completamente vacías
                df = df.loc[:, ~(df.columns.str.contains('Unnamed', na=False) & df.isna().all())]
                print(f"DEBUG - Encabezados encontrados en fila {header_row}")
            else:
                # Si no se encuentra, leer normalmente
                df = pd.read_excel(file_path)

        elif ext == '.csv':
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except:
                    continue
        elif ext == '.pdf':
            if not pdfplumber:
                return json.dumps({"error": "❌ Instala pdfplumber: pip install pdfplumber"})

            # Extraer tablas del PDF
            with pdfplumber.open(file_path) as pdf:
                all_tables = []
                all_text = []

                for page in pdf.pages:
                    # Intentar extraer tablas
                    tables = page.extract_tables()
                    if tables:
                        all_tables.extend(tables)

                    # Extraer texto como respaldo
                    text = page.extract_text()
                    if text:
                        all_text.append(text)

                # Método 1: Intentar con tablas estructuradas
                df = None
                if all_tables:
                    for table in all_tables:
                        if len(table) > 1:
                            # Filtrar filas vacías
                            table_filtered = [row for row in table if row and any(cell for cell in row)]
                            if len(table_filtered) > 1:
                                try:
                                    df = pd.DataFrame(table_filtered[1:], columns=table_filtered[0])
                                    df = df.fillna('')
                                    # Verificar que tenga al menos 3 columnas
                                    if len(df.columns) >= 3:
                                        break
                                except:
                                    continue

                # Método 2: Si no hay tablas, intentar extraer del texto
                if df is None or df.empty:
                    if not all_text:
                        return json.dumps({"error": "⚠️ No se encontró información estructurada en el PDF. Intenta con Excel o CSV."})

                    # Intentar parsear texto línea por línea
                    import re
                    transactions = []

                    # DEBUG: Guardar texto extraído
                    debug_text = "\n".join(all_text[:2])  # Primeras 2 páginas
                    print(f"DEBUG - Texto extraído del PDF:\n{debug_text[:1000]}")

                    for text in all_text:
                        lines = text.split('\n')

                        for i, line in enumerate(lines):
                            line = line.strip()
                            if not line:
                                continue

                            # Patrón 1: Fecha al inicio, monto al final (puede ser negativo o con $)
                            # Ejemplo: "15/01/2024 Compra en supermercado -1,234.56" o "15/01/2024 Depósito $5,000.00"
                            match1 = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+(.+?)\s+([-+]?[\$]?\s*[\d,]+\.?\d*)\s*$', line)

                            # Patrón 2: Fecha, descripción extensa, monto separado por múltiples espacios
                            # Ejemplo: "01/02/2024  TRANSFERENCIA SPEI RECIBIDA  10,000.00"
                            match2 = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s{2,}(.+?)\s{2,}([-+]?[\$]?\s*[\d,]+\.?\d*)\s*$', line)

                            # Patrón 3: Fecha - Descripción - Monto en líneas separadas o concatenadas
                            # Buscar fecha primero
                            fecha_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', line)
                            if fecha_match and not (match1 or match2):
                                # Buscar monto en la misma línea o siguiente
                                monto_match = re.search(r'([-+]?[\$]?\s*[\d,]+\.\d{2})\s*$', line)
                                if monto_match:
                                    # Extraer descripción entre fecha y monto
                                    fecha = fecha_match.group(1)
                                    monto = monto_match.group(1)
                                    desc_start = fecha_match.end()
                                    desc_end = monto_match.start()
                                    desc = line[desc_start:desc_end].strip()

                                    if desc:  # Solo si hay descripción
                                        match3 = True
                                        fecha, desc, monto = fecha, desc, monto
                                    else:
                                        match3 = None
                                else:
                                    match3 = None
                            else:
                                match3 = None

                            match = match1 or match2 or match3

                            if match:
                                if match1:
                                    fecha, desc, monto = match1.groups()
                                elif match2:
                                    fecha, desc, monto = match2.groups()
                                # match3 ya asignó las variables

                                # Limpiar monto: remover $, espacios, comas
                                monto_clean = monto.replace('$', '').replace(',', '').replace(' ', '').strip()

                                # Verificar que el monto sea válido
                                try:
                                    float(monto_clean)
                                    transactions.append({
                                        'Fecha': fecha.strip(),
                                        'Descripción': desc.strip(),
                                        'Monto': monto_clean
                                    })
                                    print(f"DEBUG - Transacción encontrada: {fecha.strip()} | {desc.strip()} | {monto_clean}")
                                except ValueError:
                                    continue

                    print(f"DEBUG - Total transacciones extraídas: {len(transactions)}")

                    if transactions:
                        df = pd.DataFrame(transactions)
                    else:
                        return json.dumps({
                            "error": "⚠️ No se pudo extraer transacciones del PDF",
                            "sugerencia": "El PDF no tiene un formato reconocible. Usa Excel (.xlsx) o CSV para mejores resultados.",
                            "texto_extraido_muestra": debug_text[:800] if 'debug_text' in locals() else "",
                            "total_paginas": len(all_text),
                            "debug": "Revisa la consola para ver el texto extraído"
                        })
        else:
            return json.dumps({"error": f"❌ Formato no soportado: {ext}"})

        col_mapping = {}

        # Limpiar nombres de columnas eliminando tabs y espacios múltiples
        df.columns = [str(col).replace('\t', ' ').strip() for col in df.columns]

        # Detectar fecha - priorizar "fecha transacción" o "fecha transaccion"
        fecha_patterns = ['fecha transacc', 'fecha transac', 'fecha', 'date', 'día', 'dia', 'period', 'transaction date']
        for col in df.columns:
            col_clean = str(col).lower().strip()
            if any(pattern in col_clean for pattern in fecha_patterns):
                col_mapping['fecha'] = col
                break

        # Detectar descripción
        desc_patterns = ['descripcion', 'description', 'descripci', 'concepto', 'detalle', 'detail', 'reference']
        for col in df.columns:
            col_clean = str(col).lower().strip()
            if any(pattern in col_clean for pattern in desc_patterns):
                col_mapping['descripcion'] = col
                break

        # Detectar monto general (si existe una columna única de monto)
        monto_patterns = ['monto', 'amount', 'importe', 'valor', 'value', 'total']
        for col in df.columns:
            col_clean = str(col).lower().strip()
            if any(pattern in col_clean for pattern in monto_patterns):
                col_mapping['monto'] = col
                break

        # Detectar ingresos/créditos - priorizar "pagos (cr)" o "cr"
        ingreso_patterns = ['pago', 'cr)', 'cr', 'credit', 'ingreso', 'deposit', 'abono', 'deposito']
        cargo_patterns = ['cargo', 'db)', 'db', 'debit', 'retiro', 'withdrawal', 'gasto']

        for col in df.columns:
            col_clean = str(col).lower().strip()
            # Detectar ingresos solo si contiene 'cr' o 'pago' (evitar confusión con 'proceso')
            if any(pattern in col_clean for pattern in ingreso_patterns):
                if 'cr' in col_clean or 'pago' in col_clean or 'credit' in col_clean or 'ingreso' in col_clean:
                    col_mapping['ingresos'] = col
            # Detectar cargos
            if any(pattern in col_clean for pattern in cargo_patterns):
                if 'db' in col_clean or 'cargo' in col_clean or 'debit' in col_clean or 'gasto' in col_clean:
                    col_mapping['cargos'] = col

        # DEBUG: Mostrar columnas detectadas
        print(f"DEBUG - Columnas en el archivo: {list(df.columns)}")
        print(f"DEBUG - Mapeo detectado: {col_mapping}")

        if 'fecha' not in col_mapping or 'descripcion' not in col_mapping:
            return json.dumps({
                "error": "❌ No se detectaron columnas de fecha o descripción",
                "columnas_encontradas": list(df.columns),
                "mapeo_detectado": col_mapping,
                "sugerencia": "El archivo debe tener columnas de: Fecha transacción, Descripción, Cargos (Db), Pagos (Cr)"
            }, ensure_ascii=False)

        datos_procesados = []

        for _, row in df.iterrows():
            try:
                fecha = row[col_mapping['fecha']]
                descripcion = str(row[col_mapping['descripcion']]).strip()

                # Saltar filas sin descripción válida
                if not descripcion or descripcion.lower() in ['nan', 'none', '']:
                    continue

                if 'monto' in col_mapping:
                    monto_val = row[col_mapping['monto']]
                    # Limpiar y convertir monto
                    if isinstance(monto_val, str):
                        monto_val = monto_val.replace('$', '').replace(',', '').strip()
                    try:
                        monto = float(monto_val)
                        tipo = 'ingreso' if monto >= 0 else 'gasto'
                        monto = abs(monto)
                    except:
                        continue
                elif 'ingresos' in col_mapping and 'cargos' in col_mapping:
                    ing = row[col_mapping['ingresos']]
                    car = row[col_mapping['cargos']]

                    # Limpiar valores si son strings
                    if isinstance(ing, str):
                        ing = ing.replace('$', '').replace(',', '').strip()
                        ing = None if ing == '' else ing
                    if isinstance(car, str):
                        car = car.replace('$', '').replace(',', '').strip()
                        car = None if car == '' else car

                    if pd.notna(ing) and ing not in [0, '0', None, '']:
                        try:
                            tipo = 'ingreso'
                            monto = abs(float(ing))
                        except:
                            continue
                    elif pd.notna(car) and car not in [0, '0', None, '']:
                        try:
                            tipo = 'gasto'
                            monto = abs(float(car))
                        except:
                            continue
                    else:
                        continue
                else:
                    return json.dumps({"error": "❌ No se pudo determinar estructura de montos", "mapeo": col_mapping}, ensure_ascii=False)

                # Verificar que el monto sea mayor que 0
                if monto <= 0:
                    continue

                categoria = auto_categorize(descripcion)

                # Formatear fecha a formato estándar
                fecha_str = str(fecha)
                # Intentar parsear fecha en diferentes formatos
                try:
                    if '/' in fecha_str:
                        # Formato DD/MM/YYYY
                        partes = fecha_str.split('/')
                        if len(partes) == 3:
                            dia, mes, anio = partes[0], partes[1], partes[2][:4]
                            fecha_formatted = f"{anio}-{mes.zfill(2)}-{dia.zfill(2)}"
                        else:
                            fecha_formatted = fecha_str[:10]
                    else:
                        fecha_formatted = fecha_str[:10]
                except:
                    fecha_formatted = fecha_str[:10]

                datos_procesados.append({
                    'fecha': fecha_formatted,
                    'descripcion': descripcion,
                    'categoria': categoria,
                    'monto': round(abs(monto), 2),
                    'tipo': tipo
                })
            except Exception as e:
                print(f"DEBUG - Error procesando fila: {e}")
                continue

        conn = sqlite3.connect(DATABASE_CONFIG['path'])
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transacciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha DATE NOT NULL,
            descripcion TEXT NOT NULL,
            categoria VARCHAR(50) NOT NULL,
            monto DECIMAL(10,2) NOT NULL,
            tipo VARCHAR(20) NOT NULL
        )
        ''')

        insertados = 0
        duplicados = 0

        for dato in datos_procesados:
            cursor.execute("""
                SELECT COUNT(*) FROM transacciones
                WHERE fecha = ? AND descripcion = ? AND monto = ?
            """, (dato['fecha'], dato['descripcion'], dato['monto']))

            if cursor.fetchone()[0] > 0:
                duplicados += 1
                continue

            cursor.execute("""
                INSERT INTO transacciones (fecha, descripcion, categoria, monto, tipo)
                VALUES (?, ?, ?, ?, ?)
            """, (dato['fecha'], dato['descripcion'], dato['categoria'], dato['monto'], dato['tipo']))
            insertados += 1

        conn.commit()

        client = init_chromadb()
        try:
            collection = client.get_or_create_collection(
                name=CHROMA_CONFIG["collection_name"],
                metadata={"description": "Bank statements for semantic search"}
            )

            for i, dato in enumerate(datos_procesados):
                doc_id = hashlib.md5(f"{dato['fecha']}{dato['descripcion']}{dato['monto']}".encode()).hexdigest()
                doc_text = f"Fecha: {dato['fecha']} | {dato['descripcion']} | ${dato['monto']} | {dato['categoria']}"

                collection.upsert(
                    ids=[doc_id],
                    documents=[doc_text],
                    metadatas=[dato]
                )
        except Exception as e:
            print(f"⚠️ Error indexando en ChromaDB: {e}")

        conn.close()

        # Calcular resumen de categorías
        categorias_resumen = {}
        total_ingresos = 0
        total_gastos = 0

        for dato in datos_procesados:
            cat = dato['categoria']
            if cat not in categorias_resumen:
                categorias_resumen[cat] = {'count': 0, 'monto': 0}
            categorias_resumen[cat]['count'] += 1
            categorias_resumen[cat]['monto'] += dato['monto']

            if dato['tipo'] == 'ingreso':
                total_ingresos += dato['monto']
            else:
                total_gastos += dato['monto']

        return json.dumps({
            "success": True,
            "archivo": Path(file_path).name,
            "columnas_detectadas": col_mapping,
            "transacciones_insertadas": insertados,
            "duplicados_omitidos": duplicados,
            "total_procesado": len(datos_procesados),
            "resumen": {
                "total_ingresos": round(total_ingresos, 2),
                "total_gastos": round(total_gastos, 2),
                "balance": round(total_ingresos - total_gastos, 2),
                "categorias": categorias_resumen
            },
            "mensaje": f"✅ Estado de cuenta cargado exitosamente!\n\n📊 Resumen:\n- {insertados} transacciones nuevas\n- {duplicados} duplicados omitidos\n- Total ingresos: ${round(total_ingresos, 2)}\n- Total gastos: ${round(total_gastos, 2)}\n- Balance: ${round(total_ingresos - total_gastos, 2)}"
        }, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"❌ Error procesando archivo: {str(e)}"})

def auto_categorize(descripcion: str) -> str:
    """Auto-categoriza transacciones basándose en palabras clave"""
    desc_lower = descripcion.lower()

    categorias = {
        'pagos': ['gracias por su pago', 'pago', 'payment', 'abono'],
        'seguros': ['seguro', 'insurance', 'desgravamen', 'fraude'],
        'financiamiento': ['financiamiento', 'financing', 'interes', 'interest'],
        'impuestos': ['itbms', 'iva', 'tax', 'impuesto'],
        'vivienda': ['renta', 'alquiler', 'predial', 'hipoteca', 'mantenimiento'],
        'servicios': ['cfe', 'luz', 'agua', 'gas', 'telmex', 'internet', 'telefono', 'telcel'],
        'alimentacion': ['walmart', 'soriana', 'chedraui', 'super', 'mercado', 'oxxo', '7-eleven'],
        'transporte': ['uber', 'gasolina', 'pemex', 'metro', 'taxi', 'didi'],
        'salud': ['farmacia', 'hospital', 'doctor', 'clinica', 'medicamento'],
        'restaurantes': ['restaurante', 'comida', 'pizza', 'tacos', 'starbucks', 'mcdonalds'],
        'entretenimiento': ['netflix', 'spotify', 'cine', 'cinepolis', 'teatro'],
        'suscripciones': ['netflix', 'spotify', 'prime', 'disney', 'hbo', 'gym'],
        'compras': ['amazon', 'mercado libre', 'liverpool', 'palacio', 'zara'],
        'delivery': ['rappi', 'uber eats', 'didi food'],
        'educacion': ['colegiatura', 'universidad', 'curso', 'libro']
    }

    for categoria, keywords in categorias.items():
        if any(keyword in desc_lower for keyword in keywords):
            return categoria

    return 'otros'

@tool
def semantic_search_transactions(query: str, n_results: int = 5) -> str:
    """
    Busca transacciones usando búsqueda semántica en ChromaDB.

    Args:
        query: Texto de búsqueda en lenguaje natural
        n_results: Número de resultados (default: 5)

    Returns:
        JSON con transacciones relevantes encontradas
    """
    try:
        client = init_chromadb()
        collection = client.get_collection(name=CHROMA_CONFIG["collection_name"])

        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )

        transacciones = []
        if results['documents'] and results['documents'][0]:
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                transacciones.append({
                    "relevancia": i + 1,
                    "fecha": meta.get('fecha'),
                    "descripcion": meta.get('descripcion'),
                    "categoria": meta.get('categoria'),
                    "monto": meta.get('monto'),
                    "tipo": meta.get('tipo')
                })

        return json.dumps({
            "query": query,
            "resultados_encontrados": len(transacciones),
            "transacciones": transacciones
        }, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Error en búsqueda semántica: {str(e)}"})

@tool
def analyze_bank_statement(database_path: str = "estados_cuenta.db", months: int = 3) -> str:
    """
    Analiza el estado de cuenta bancario completo.

    Args:
        database_path: Ruta a la base de datos
        months: Número de meses a analizar (default: 3)

    Returns:
        JSON con análisis completo
    """
    try:
        conn = sqlite3.connect(database_path)

        count_check = pd.read_sql_query("SELECT COUNT(*) as total FROM transacciones", conn)
        if count_check['total'].iloc[0] == 0:
            conn.close()
            return json.dumps({
                "error": "⚠️ No hay datos en la base de datos",
                "mensaje": "Usa la herramienta 'load_real_bank_statement' para cargar tu estado de cuenta"
            })

        fecha_inicio = (datetime.now() - timedelta(days=months*30)).strftime('%Y-%m-%d')

        resumen = pd.read_sql_query(f"""
            SELECT
                COUNT(*) as total_movimientos,
                SUM(CASE WHEN tipo = 'ingreso' THEN monto ELSE 0 END) as ingresos_totales,
                SUM(CASE WHEN tipo = 'gasto' THEN monto ELSE 0 END) as gastos_totales,
                SUM(CASE WHEN tipo = 'ingreso' THEN monto ELSE -monto END) as balance
            FROM transacciones
            WHERE fecha >= '{fecha_inicio}'
        """, conn)

        gastos_cat = pd.read_sql_query(f"""
            SELECT categoria, SUM(monto) as total, COUNT(*) as num_transacciones
            FROM transacciones
            WHERE tipo = 'gasto' AND fecha >= '{fecha_inicio}'
            GROUP BY categoria
            ORDER BY total DESC
        """, conn)

        grandes_gastos = pd.read_sql_query(f"""
            SELECT fecha, descripcion, categoria, monto
            FROM transacciones
            WHERE tipo = 'gasto' AND fecha >= '{fecha_inicio}'
            ORDER BY monto DESC
            LIMIT 10
        """, conn)

        recurrentes = pd.read_sql_query(f"""
            SELECT descripcion, categoria, COUNT(*) as frecuencia, AVG(monto) as monto_promedio
            FROM transacciones
            WHERE tipo = 'gasto' AND fecha >= '{fecha_inicio}'
            GROUP BY descripcion, categoria
            HAVING frecuencia >= 2
            ORDER BY monto_promedio DESC
            LIMIT 10
        """, conn)

        conn.close()

        ingresos = float(resumen['ingresos_totales'].iloc[0])
        gastos = float(resumen['gastos_totales'].iloc[0])
        balance = float(resumen['balance'].iloc[0])

        resultado = {
            "periodo_analizado": f"Últimos {months} meses",
            "resumen": {
                "total_movimientos": int(resumen['total_movimientos'].iloc[0]),
                "ingresos_totales": round(ingresos, 2),
                "gastos_totales": round(gastos, 2),
                "balance_neto": round(balance, 2),
                "tasa_ahorro": round((balance / ingresos * 100) if ingresos > 0 else 0, 2),
                "promedio_diario_gasto": round(gastos / (months * 30), 2)
            },
            "gastos_por_categoria": [
                {
                    "categoria": row['categoria'],
                    "total": round(float(row['total']), 2),
                    "porcentaje": round((row['total'] / gastos * 100) if gastos > 0 else 0, 2),
                    "num_transacciones": int(row['num_transacciones'])
                }
                for _, row in gastos_cat.iterrows()
            ],
            "gastos_mas_grandes": grandes_gastos.to_dict('records'),
            "pagos_recurrentes_detectados": [
                {
                    "concepto": row['descripcion'],
                    "categoria": row['categoria'],
                    "frecuencia": int(row['frecuencia']),
                    "monto_promedio": round(float(row['monto_promedio']), 2)
                }
                for _, row in recurrentes.iterrows()
            ]
        }

        return json.dumps(resultado, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Error analizando: {str(e)}"})

@tool
def identify_savings_opportunities(database_path: str = "estados_cuenta.db") -> str:
    """
    Identifica oportunidades de ahorro en el estado de cuenta.

    Args:
        database_path: Ruta a la base de datos

    Returns:
        JSON con oportunidades de ahorro
    """
    try:
        conn = sqlite3.connect(database_path)

        categorias_discrecionales = ['entretenimiento', 'restaurantes', 'compras', 'suscripciones', 'delivery']
        categorias_esenciales = ['vivienda', 'alimentacion', 'transporte', 'salud', 'servicios']

        placeholders = ','.join(['?'] * len(categorias_discrecionales))
        query_disc = f"""
            SELECT categoria, SUM(monto) as total, COUNT(*) as frecuencia
            FROM transacciones
            WHERE tipo = 'gasto' AND categoria IN ({placeholders})
            GROUP BY categoria
            ORDER BY total DESC
        """

        discrecionales = pd.read_sql_query(query_disc, conn, params=categorias_discrecionales)

        placeholders_esen = ','.join(['?'] * len(categorias_esenciales))
        query_esen = f"""
            SELECT categoria, SUM(monto) as total
            FROM transacciones
            WHERE tipo = 'gasto' AND categoria IN ({placeholders_esen})
            GROUP BY categoria
        """

        esenciales = pd.read_sql_query(query_esen, conn, params=categorias_esenciales)

        query_subs = """
            SELECT descripcion, SUM(monto) as total_anual, COUNT(*) as meses_activo
            FROM transacciones
            WHERE tipo = 'gasto' AND categoria = 'suscripciones'
            GROUP BY descripcion
            ORDER BY total_anual DESC
        """

        suscripciones = pd.read_sql_query(query_subs, conn)
        conn.close()

        total_discrecional = float(discrecionales['total'].sum()) if not discrecionales.empty else 0
        total_esencial = float(esenciales['total'].sum()) if not esenciales.empty else 0

        oportunidades = []

        for _, row in discrecionales.iterrows():
            categoria = row['categoria']
            total = float(row['total'])
            frecuencia = int(row['frecuencia'])

            ahorro_20 = total * 0.20
            ahorro_30 = total * 0.30

            oportunidades.append({
                "categoria": categoria,
                "gasto_actual": round(total, 2),
                "frecuencia": frecuencia,
                "ahorro_potencial_20": round(ahorro_20, 2),
                "ahorro_potencial_30": round(ahorro_30, 2),
                "ahorro_anual_20": round(ahorro_20 * 12, 2),
                "sugerencia": f"Reduce {categoria} en 20-30%"
            })

        resultado = {
            "gastos_discrecionales_totales": round(total_discrecional, 2),
            "gastos_esenciales_totales": round(total_esencial, 2),
            "ratio_discrecional_esencial": round((total_discrecional / total_esencial * 100) if total_esencial > 0 else 0, 2),
            "oportunidades_ahorro": oportunidades,
            "suscripciones_activas": [
                {
                    "servicio": row['descripcion'],
                    "costo_mensual": round(float(row['total_anual']) / max(row['meses_activo'], 1), 2),
                    "costo_anual": round(float(row['total_anual']), 2),
                    "meses_activo": int(row['meses_activo'])
                }
                for _, row in suscripciones.iterrows()
            ] if not suscripciones.empty else [],
            "recomendaciones": [
                "Revisa suscripciones - cancela las no usadas",
                "Reduce comidas fuera - cocina más en casa",
                "Compara precios antes de comprar",
                "Usa regla de 24h para compras grandes"
            ]
        }

        return json.dumps(resultado, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Error: {str(e)}"})

@tool
def create_statement_visualization(viz_type: str, data_json: str, title: str = "Análisis") -> str:
    """
    Crea visualizaciones del estado de cuenta.

    Args:
        viz_type: category_pie, monthly_trend, spending_bars
        data_json: Datos en JSON
        title: Título del gráfico

    Returns:
        Confirmación con ruta del archivo
    """
    try:
        data = json.loads(data_json)

        fig, ax = plt.subplots(figsize=(12, 6))
        plt.style.use('seaborn-v0_8-whitegrid')

        if viz_type == "category_pie":
            if 'gastos_por_categoria' not in data:
                return "❌ Error: datos no contienen gastos_por_categoria"

            categorias = [item['categoria'] for item in data['gastos_por_categoria']]
            valores = [item['total'] for item in data['gastos_por_categoria']]
            colors = plt.cm.Set3(np.linspace(0, 1, len(categorias)))

            wedges, texts, autotexts = ax.pie(
                valores,
                labels=categorias,
                autopct=lambda pct: f'{pct:.1f}%\n(${pct*sum(valores)/100:,.0f})',
                colors=colors,
                startangle=90,
                textprops={'fontsize': 10, 'fontweight': 'bold'}
            )

            for autotext in autotexts:
                autotext.set_color('white')

            ax.axis('equal')

        elif viz_type == "spending_bars":
            categorias = [item['categoria'] for item in data['gastos_por_categoria']]
            valores = [item['total'] for item in data['gastos_por_categoria']]
            colors = plt.cm.viridis(np.linspace(0, 1, len(categorias)))

            bars = ax.barh(categorias, valores, color=colors, edgecolor='black', linewidth=1.2)
            ax.set_xlabel('Monto ($)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Categoría', fontsize=12, fontweight='bold')

            for i, (bar, valor) in enumerate(zip(bars, valores)):
                ax.text(valor + max(valores)*0.01, i, f'${valor:,.0f}',
                       va='center', fontsize=10, fontweight='bold')

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()

        # Save to session state
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)

        if 'visualizations' not in st.session_state:
            st.session_state.visualizations = []

        st.session_state.visualizations.append({
            'image': img_buffer,
            'title': title,
            'type': viz_type
        })

        plt.close()

        return f"✅ Visualización creada: {title}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============= AGENT SETUP =============
def initialize_agent(api_key: str, model_name: str = "gemini-1.5-pro"):
    """Inicializa el agente con la API key y modelo proporcionados"""
    tools = [
        load_real_bank_statement,
        semantic_search_transactions,
        analyze_bank_statement,
        identify_savings_opportunities,
        create_statement_visualization
    ]

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.3
    )

    ADVISOR_PROMPT = """Eres un Asesor Financiero Personal experto en análisis de estados de cuenta bancarios.

🎯 ESPECIALIDAD: Analizar estados de cuenta REALES de usuarios

🛠️ HERRAMIENTAS:
1. load_real_bank_statement: Carga estados de cuenta reales (Excel/CSV/PDF)
2. semantic_search_transactions: Búsqueda semántica en transacciones
3. analyze_bank_statement: Análisis completo (usa SIEMPRE primero para verificar datos)
4. identify_savings_opportunities: Encuentra ahorros
5. create_statement_visualization: Crea gráficos

📋 PROCESO OBLIGATORIO:
1. Para CUALQUIER pregunta del usuario, PRIMERO ejecuta analyze_bank_statement() para verificar si hay datos
2. Si analyze_bank_statement() devuelve datos válidos: Úsalos para responder
3. Si analyze_bank_statement() devuelve error de "no hay datos": Pide al usuario que cargue archivo
4. Para búsquedas específicas (ej: "gastos en restaurantes"): Usa semantic_search_transactions con la query del usuario
5. Para consejos de ahorro: Usa identify_savings_opportunities
6. Para visualizaciones: Usa create_statement_visualization

💡 ESTILO:
- Empático y profesional
- Usa números específicos del estado de cuenta REAL
- Da recomendaciones accionables
- Explica patrones encontrados
- Usa emojis relevantes

⚠️ IMPORTANTE:
- NUNCA inventes datos - solo usa información real cargada
- SIEMPRE ejecuta analyze_bank_statement() PRIMERO antes de decir que no hay datos
- La base de datos SQLite es persistente - los datos siguen ahí aunque reinicies la app
- Usa búsqueda semántica para preguntas específicas sobre categorías o conceptos
- Siempre contextualiza con datos previos de la conversación

Responde en español, de forma clara y profesional."""

    conn = sqlite3.connect(CHECKPOINT_PATH, check_same_thread=False)
    memory = SqliteSaver(conn)

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=ADVISOR_PROMPT,
        checkpointer=memory,
        debug=False,
        version="v2"
    )

    return agent

# ============= HELPER FUNCTIONS =============
def export_conversation():
    """Exporta la conversación a JSON"""
    conversation_data = {
        "thread_id": st.session_state.thread_id,
        "timestamp": datetime.now().isoformat(),
        "messages": [
            {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg.get("timestamp", "")
            }
            for msg in st.session_state.messages
        ]
    }

    return json.dumps(conversation_data, indent=2, ensure_ascii=False)

def clear_conversation():
    """Limpia la conversación actual"""
    st.session_state.messages = []
    st.session_state.thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if 'visualizations' in st.session_state:
        st.session_state.visualizations = []

# ============= STREAMLIT UI =============
def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
    }
    .predefined-btn {
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<div class="main-header">💰 Financial Agent - Asesor Financiero</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")

        # API Key Input
        api_key = st.text_input(
            "Google API Key",
            type="password",
            value=st.session_state.get('api_key', ''),
            help="Ingresa tu API key de Google Gemini"
        )

        # Model Selector
        model_options = {
            "Gemini 2.5 Pro (Mejor rendimiento)": "gemini-2.5-pro",
            "Gemini 2.5 Flash (Precio-rendimiento)": "gemini-2.5-flash",
            "Gemini 2.5 Flash-Lite (Más rápido)": "gemini-2.5-flash-lite",
            "Gemini 2.0 Flash": "gemini-2.0-flash",
            "Gemini 2.0 Flash-Lite": "gemini-2.0-flash-lite"
        }

        selected_model_display = st.selectbox(
            "Modelo Gemini",
            options=list(model_options.keys()),
            index=0,
            help="Selecciona el modelo de Gemini a utilizar"
        )

        selected_model = model_options[selected_model_display]

        if api_key and (api_key != st.session_state.get('api_key') or selected_model != st.session_state.get('selected_model')):
            st.session_state.api_key = api_key
            st.session_state.selected_model = selected_model
            try:
                st.session_state.agent = initialize_agent(api_key, selected_model)
                st.success(f"✅ Agente inicializado con {selected_model}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

        st.markdown("---")

        # File Upload
        st.header("📂 Subir Estado de Cuenta")
        uploaded_file = st.file_uploader(
            "Sube tu archivo (Excel/CSV/PDF)",
            type=['xlsx', 'xls', 'csv', 'pdf'],
            help="Formatos soportados: Excel, CSV, PDF"
        )

        if uploaded_file:
            # Save uploaded file
            temp_path = Path(f"./temp_{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if st.button("📥 Cargar Archivo"):
                if not st.session_state.agent:
                    st.error("⚠️ Primero configura tu API Key")
                else:
                    with st.spinner("Cargando archivo..."):
                        # Add to messages
                        st.session_state.messages.append({
                            "role": "user",
                            "content": f"Carga mi estado de cuenta desde {temp_path}",
                            "timestamp": datetime.now().isoformat()
                        })

                        # Process
                        config = {"configurable": {"thread_id": st.session_state.thread_id}}
                        result = st.session_state.agent.invoke(
                            {"messages": [HumanMessage(content=f"Carga mi estado de cuenta desde {temp_path}")]},
                            config=config
                        )

                        # Add response
                        response_content = result["messages"][-1].content
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_content,
                            "timestamp": datetime.now().isoformat()
                        })

                        # Check if data was loaded successfully
                        if "success" in response_content.lower() or "cargado" in response_content.lower():
                            st.session_state.data_loaded = True

                        st.rerun()

        st.markdown("---")

        # Conversation Management
        st.header("💬 Conversación")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑️ Limpiar", use_container_width=True):
                clear_conversation()
                st.rerun()

        with col2:
            if st.session_state.messages:
                conversation_json = export_conversation()
                st.download_button(
                    label="📥 Exportar",
                    data=conversation_json,
                    file_name=f"conversation_{st.session_state.thread_id}.json",
                    mime="application/json",
                    use_container_width=True
                )

        st.markdown("---")

        # Session Info
        st.header("ℹ️ Información")

        # Check if database has data
        try:
            conn = sqlite3.connect(DATABASE_CONFIG['path'])
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM transacciones")
            count = cursor.fetchone()[0]
            conn.close()
            if count > 0:
                st.session_state.data_loaded = True
                st.success(f"✅ Datos cargados: {count} transacciones")
            else:
                st.session_state.data_loaded = False
                st.info("📂 No hay datos cargados")
        except:
            st.session_state.data_loaded = False
            st.info("📂 No hay datos cargados")

        st.caption(f"🆔 Thread ID: {st.session_state.thread_id[:15]}...")
        st.caption(f"💬 Mensajes: {len(st.session_state.messages)}")

    # Main Content
    col1, col2 = st.columns([3, 1])

    with col1:
        st.header("💬 Chat")

        # Display messages
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

                    # Check if response contains table data
                    if msg["role"] == "assistant":
                        try:
                            # Try to parse JSON from response
                            if "gastos_por_categoria" in msg["content"]:
                                # Extract and display as table
                                json_match = msg["content"]
                                if "{" in json_match and "}" in json_match:
                                    data = json.loads(json_match[json_match.find("{"):json_match.rfind("}")+1])
                                    if "gastos_por_categoria" in data:
                                        df = pd.DataFrame(data["gastos_por_categoria"])
                                        st.dataframe(df, use_container_width=True)
                        except:
                            pass

        # Display visualizations
        if 'visualizations' in st.session_state and st.session_state.visualizations:
            st.markdown("---")
            st.subheader("📊 Visualizaciones")
            for viz in st.session_state.visualizations[-3:]:  # Show last 3
                st.image(viz['image'], caption=viz['title'], use_container_width=True)

        # Chat input
        user_input = st.chat_input("Escribe tu pregunta...")

        if user_input:
            if not st.session_state.agent:
                st.error("⚠️ Primero configura tu API Key en la barra lateral")
            else:
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().isoformat()
                })

                # Get response
                with st.spinner("Pensando..."):
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    result = st.session_state.agent.invoke(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=config
                    )

                    response_content = result["messages"][-1].content
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_content,
                        "timestamp": datetime.now().isoformat()
                    })

                st.rerun()

    with col2:
        st.header("🎯 Preguntas Rápidas")

        predefined_questions = [
            "¿Cómo están mis finanzas?",
            "Muestra mis gastos por categoría",
            "¿Dónde puedo ahorrar?",
            "¿Cuánto gasté en restaurantes?",
            "Identifica pagos recurrentes",
            "Crea un gráfico de gastos",
            "Busca gastos en entretenimiento",
            "¿Cuál es mi balance actual?"
        ]

        for question in predefined_questions:
            if st.button(question, key=f"btn_{question}", use_container_width=True):
                if not st.session_state.agent:
                    st.error("⚠️ Primero configura tu API Key")
                else:
                    # Add user message
                    st.session_state.messages.append({
                        "role": "user",
                        "content": question,
                        "timestamp": datetime.now().isoformat()
                    })

                    # Get response
                    with st.spinner("Procesando..."):
                        config = {"configurable": {"thread_id": st.session_state.thread_id}}
                        result = st.session_state.agent.invoke(
                            {"messages": [HumanMessage(content=question)]},
                            config=config
                        )

                        response_content = result["messages"][-1].content
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_content,
                            "timestamp": datetime.now().isoformat()
                        })

                    st.rerun()

if __name__ == "__main__":
    main()
