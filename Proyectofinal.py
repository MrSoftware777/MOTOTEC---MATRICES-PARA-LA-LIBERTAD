import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Proyecto Álgebra - MOTOTEC", layout="centered")

ventas = {
    "Tipo 1": [172, 185, 202, 225, 252, 286, 316, 342, 371, 402],
    "Tipo 2": [89, 116, 155, 188, 200, 199, 240, 245, 280, 302],
    "Tipo 3": [18, 49, 98, 96, 148, 173, 204, 235, 266, 297],
    "Tipo 4": [28, 33, 49, 44, 59, 72, 70, 96, 140, 250]
}

t = np.arange(10) 
A = np.vstack([np.ones(len(t)), t]).T
t_futuro = np.array([10, 11, 12])  
resultados = {}




for tipo, y in ventas.items():
    y = np.array(y)
    x, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = x
    y_fit = A @ x
    ecm = np.mean((y - y_fit) ** 2)
    y_pred = a + b * t_futuro
    resultados[tipo] = {
        "a": a,
        "b": b,
        "ajustado": y_fit,
        "2023": y_pred[0],
        "2024": y_pred[1],
        "2025": y_pred[2],

        "ECM": ecm,
        "real": y
    }


st.markdown(
    "<h1 style='text-align: center;'>Proyecto Álgebra Lineal – MOTOTEC</h1>",
    unsafe_allow_html=True
)

st.markdown("Predicción de ventas y demanda de componentes usando mínimos cuadrados")
st.markdown("---")


seccion = st.selectbox("Ir a sección:", ["Proyecciones", "Resumen de predicciones", "Demanda de componentes"])

#SECCIONES:

# Sección → Proyecciones

if seccion == "Proyecciones":
    st.subheader("Proyecciones por tipo de moto")

    tipo_seleccionado = st.selectbox("Selecciona un tipo de moto:", list(ventas.keys()))
    datos = resultados[tipo_seleccionado]

    fig, ax = plt.subplots()
    ax.plot(t + 2013, datos["real"], 'o-', label='Datos reales')
    ax.plot(t + 2013, datos["ajustado"], '--', label='Modelo ajustado')
    ax.plot([2023, 2024, 2025], [datos["2023"], datos["2024"], datos["2025"]], 'rx', label='Predicciones 2023–2025')

    ax.set_xlabel("Año")
    ax.set_ylabel("Ventas")
    ax.set_title(f"{tipo_seleccionado} - ECM: {datos['ECM']:.2f}")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    

#Sección → Predicciones

if seccion == "Resumen de predicciones":
    st.subheader("Resumen de modelos y predicciones")

    tabla = {

    "Tipo": [],
    "2023": [],
    "2024": [],
    "2025": [],
    "ECM": []


    }

    for tipo, d in resultados.items():
        tabla["Tipo"].append(tipo)
        tabla["2023"].append(round(d["2023"]))
        tabla["2024"].append(round(d["2024"]))
        tabla["2025"].append(round(d["2025"]))
        tabla["ECM"].append(round(d["ECM"], 2))

    st.dataframe(tabla, use_container_width=True)


# Sección → Componentes


if seccion == "Demanda de componentes":
    st.subheader("Demanda proyectada de componentes")

    componentes = np.array([
        [1, 1, 1, 0],
        [2, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [3, 2, 0, 0],
        [1, 4, 0, 0],
        [5, 2, 0, 1],
        [1, 1, 2, 0],
        [1, 1, 0, 0]
    ])
    
    v_2023 = np.round([resultados[t]["2023"] for t in ventas.keys()]).astype(int)
    v_2024 = np.round([resultados[t]["2024"] for t in ventas.keys()]).astype(int)
    v_2025 = np.round([resultados[t]["2025"] for t in ventas.keys()]).astype(int)
    
    
    
    d_2023 = componentes @ v_2023
    d_2024 = componentes @ v_2024
    d_2025 = componentes @ v_2025

    fig2, ax2 = plt.subplots()
    labels = [f"C{i+1}" for i in range(10)]
    x = np.arange(len(labels))
    
    width = 0.25
    ax2.bar(x - width, d_2023, width, label='2023')
    ax2.bar(x, d_2024, width, label='2024')
    ax2.bar(x + width, d_2025, width, label='2025')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Unidades")
    ax2.set_title("Demanda total por componente (2023–2025)")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)


st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>"
    "<b>Hecho por:</b> Luis Fernando Zapata Vanegas, José David Zapata Franco – Tecnología en Desarrollo de Software"
    "</div>",
    unsafe_allow_html=True
) 