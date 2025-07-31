from typing import Literal, Optional
from langchain_core.tools import tool
import sqlite3
from datetime import datetime
from dateutil.relativedelta import relativedelta


@tool
def get_transactions(
    tipo_transaccion: Literal["debito", "credito"],
    n: int,
    recent: Literal["primera", "ultima"] = "ultima",
    fecha_inicio: Optional[str] = None,
    fecha_fin: Optional[str] = None,
) -> str:
    """Devuelve las primeras o últimas transacciones de cierto tipo en un rango de fechas opcional."""
    connection = sqlite3.connect('excercise.db')
    cursor = connection.cursor()

    order = "DESC" if recent == "ultima" else "ASC"
    where_clause = f"{tipo_transaccion} > 0"

    if fecha_inicio:
        where_clause += f" AND fecha >= '{fecha_inicio}'"
    if fecha_fin:
        where_clause += f" AND fecha <= '{fecha_fin}'"

    query = f"""
        SELECT fecha, descripcion, {tipo_transaccion}
        FROM Transacciones
        WHERE {where_clause}
        ORDER BY fecha {order}
        LIMIT {n}
    """
    cursor.execute(query)
    result = cursor.fetchall()
    connection.close()

    return '\n'.join([f"{r[0]} | {r[1]} | {r[2]}" for r in result])


@tool
def get_total_amount(
    tipo_transaccion: Literal["debito", "credito"],
    fecha_inicio: Optional[str] = None,
    fecha_fin: Optional[str] = None,
) -> str:
    """
    Devuelve el monto total de transacciones de cierto tipo en un rango de fechas.

    Args:
        tipo_transaccion: 'debito' o 'credito'
        fecha_inicio: Fecha inicial (formato 'dd-MMM-YYYY')
        fecha_fin: Fecha final (formato 'dd-MMM-YYYY')

    Returns:
        Monto total como string
    """
    connection = sqlite3.connect('excercise.db')
    cursor = connection.cursor()

    where_clause = f"{tipo_transaccion} > 0"
    if fecha_inicio:
        where_clause += f" AND fecha >= '{fecha_inicio}'"
    if fecha_fin:
        where_clause += f" AND fecha <= '{fecha_fin}'"

    query = f"""
        SELECT SUM({tipo_transaccion})
        FROM Transacciones
        WHERE {where_clause}
    """
    cursor.execute(query)
    total = cursor.fetchone()[0]
    connection.close()

    return f"Monto total de {tipo_transaccion}: {total:.2f}" if total else f"No hay transacciones de tipo {tipo_transaccion}."


@tool
def get_max_transaction(tipo_transaccion: Literal["debito", "credito"]) -> str:
    """Devuelve la transacción con mayor monto del tipo especificado."""
    connection = sqlite3.connect('excercise.db')
    cursor = connection.cursor()

    cursor.execute(f"""
        SELECT fecha, descripcion, {tipo_transaccion}
        FROM Transacciones
        WHERE {tipo_transaccion} > 0
        ORDER BY {tipo_transaccion} DESC
        LIMIT 1
    """)
    result = cursor.fetchone()
    connection.close()

    return f"{result[0]} | {result[1]} | {result[2]}" if result else "No se encontraron transacciones."


@tool
def get_most_frequent_transaction(tipo_transaccion: Literal["debito", "credito"]) -> str:
    """Devuelve la transacción más frecuente por descripción del tipo especificado."""
    connection = sqlite3.connect('excercise.db')
    cursor = connection.cursor()

    cursor.execute(f"""
        SELECT descripcion, COUNT(*) as frecuencia
        FROM Transacciones
        WHERE {tipo_transaccion} > 0
        GROUP BY descripcion
        ORDER BY frecuencia DESC
        LIMIT 1
    """)
    result = cursor.fetchone()
    connection.close()

    return f"Descripción: {result[0]}, Frecuencia: {result[1]}" if result else "No se encontraron transacciones."


@tool
def get_curr_date() -> str:
    """Devuelve la fecha actual en formato 'dd-MMM-YYYY'."""
    return datetime.now().strftime("%d-%b-%Y")


@tool
def get_adjusted_date(date_str: str, days: int) -> str:
    """Devuelve la fecha ajustada restando días a partir de una fecha dada."""
    adjusted = datetime.strptime(date_str, "%d-%b-%Y") - relativedelta(days=days)
    return adjusted.strftime("%d-%b-%Y")



# Lista generalizada de tools
state_account_tools = [
    get_transactions,
    get_max_transaction,
    get_most_frequent_transaction,
    get_curr_date,
    get_adjusted_date,
]
