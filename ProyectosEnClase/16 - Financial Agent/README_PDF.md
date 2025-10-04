# Soporte para Archivos PDF

## 📋 Características

El Financial Agent ahora soporta la carga y análisis de estados de cuenta en formato PDF.

## 🚀 Cómo usar archivos PDF

### 1. Formatos Soportados
- **Excel** (.xlsx, .xls)
- **CSV** (.csv)
- **PDF** (.pdf) ✨ NUEVO

### 2. Cargar un PDF

1. En la barra lateral, ve a "📂 Subir Estado de Cuenta"
2. Haz clic en "Browse files" y selecciona tu archivo PDF
3. Verás una **vista previa** del contenido del PDF
4. Haz clic en "📥 Cargar Archivo" para procesarlo

### 3. Vista Previa del PDF

Antes de procesar el archivo, puedes ver:
- 📄 Número total de páginas
- 📊 Tablas extraídas (si las hay)
- 📝 Texto de la primera página

### 4. Extracción de Datos

El sistema intentará:

**Opción 1: Extraer Tablas**
- Usa `pdfplumber` para detectar y extraer tablas automáticamente
- Ideal para PDFs con formato estructurado

**Opción 2: Extracción de Texto**
- Si no hay tablas, extrae el texto y lo parsea
- Busca patrones de: fecha, descripción, monto

### 5. Columnas Detectadas

El sistema auto-detecta columnas usando palabras clave:

**Fecha:**
- fecha, date, día, dia, period, transaction date

**Descripción:**
- descripcion, description, concepto, detalle, detail, reference

**Monto:**
- monto, amount, importe, valor, value, total

**Ingresos/Cargos:**
- ingreso, deposit, credit, abono, deposito
- cargo, debit, retiro, withdrawal, gasto

## ⚠️ Requisitos

Asegúrate de tener instalado `pdfplumber`:

```bash
pip install pdfplumber
```

## 💡 Consejos

1. **PDFs con tablas estructuradas** funcionan mejor
2. **Evita PDFs escaneados** (imágenes) - usa PDFs con texto seleccionable
3. Si tienes problemas, prueba convertir el PDF a Excel o CSV primero
4. La calidad de extracción depende del formato del PDF original

## 🔧 Solución de Problemas

### "No se pudieron extraer datos del PDF"
- El PDF podría ser una imagen escaneada
- Intenta usar OCR o convertir a otro formato

### "Error extrayendo PDF"
- Verifica que el archivo no esté corrupto
- Prueba abrirlo manualmente primero

### Columnas no detectadas
- El PDF podría tener un formato no estándar
- Revisa la vista previa para ver qué datos se extrajeron
- Considera convertir a Excel/CSV para mayor control

## 📊 Ejemplo de Uso

1. Carga tu estado de cuenta PDF
2. El agente procesará automáticamente
3. Pregunta cosas como:
   - "¿Cómo están mis finanzas?"
   - "Muestra mis gastos por categoría"
   - "¿Dónde puedo ahorrar?"

## 🎯 Próximos Pasos

Una vez cargado tu PDF, el agente puede:
- ✅ Analizar tus gastos e ingresos
- ✅ Categorizar transacciones automáticamente
- ✅ Crear visualizaciones (gráficos de pastel, barras)
- ✅ Identificar oportunidades de ahorro
- ✅ Buscar transacciones específicas
