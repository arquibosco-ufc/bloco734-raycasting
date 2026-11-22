import streamlit as st
import numpy as np
from motor_ct import RenderizadorCT

st.set_page_config(page_title="Bloco 734 CT + Escultura Cinética + Cáusticas", layout="wide")
st.title("Visualizador 3D & Ray Casting — Bloco 734 (CT), Escultura Cinética e Fonte")

@st.cache_resource
def carregar_motor():
    return RenderizadorCT(largura=500, altura=500)

renderer = carregar_motor()

# --- CONTROLE GLOBAL DIA / NOITE NA BARRA LATERAL ---
st.sidebar.header("Iluminação Global")
ciclo_noite = st.sidebar.toggle("Modo Noturno (Postes + Spots Submersos + Cáusticas)", value=False)
st.sidebar.caption("À noite, a luz solar é desligada e os spots da fonte projetam cáusticas na fachada do Bloco 734.")

st.sidebar.divider()
st.sidebar.header("Controles de Câmera")
modo = st.sidebar.radio("Modo de Câmera:", ["Presets da Cena", "Órbita Abóbada (0° a 90°)", "Câmera Manual"])

if modo == "Presets da Cena":
    opcoes = {
        "2 Pontos de Fuga (Principal)": "2_pontos_fuga",
        "1 Ponto de Fuga (Frontal Fonte e Fachada)": "1_ponto_fuga",
        "3 Pontos de Fuga (Aérea)": "3_pontos_fuga",
        "Zoom In (Escultura 5 Losangos + Fonte)": "zoom_in",
        "Zoom Out (Campo Ampliado)": "zoom_out",
        "Projeção Ortográfica": "ortografica",
        "Projeção Oblíqua": "obliqua",
    }
    escolha = st.sidebar.selectbox("Escolha a vista:", list(opcoes.keys()))
    cfg = renderer.presets[opcoes[escolha]].copy()

elif modo == "Órbita Abóbada (0° a 90°)":
    azim = st.sidebar.slider("Azimute (°):", -180, 180, -125)
    elev = st.sidebar.slider("Elevação (°) [Travada 0° piso a 90° topo]:", 0, 90, 12)
    el, az = np.deg2rad(elev), np.deg2rad(azim)
    c_domo, r_domo = renderer.centro_domo, renderer.raio_domo
    eye_domo = [
        float(c_domo[0] + r_domo * np.cos(el) * np.cos(az)),
        float(max(0.8, c_domo[1] + r_domo * np.sin(el))),
        float(c_domo[2] + r_domo * np.cos(el) * np.sin(az))
    ]
    cfg = {
        'titulo': f'Órbita Abóbada (Azimute={azim}°, Elevação={elev}°)',
        'eye': eye_domo, 'at': c_domo.tolist(), 'up': [0.0, 1.0, 0.0],
        'd': 1.5, 'win': (-0.98, 0.98, -0.98, 0.98), 'proj': 'perspectiva'
    }

else:
    col1, col2, col3 = st.sidebar.columns(3)
    ex = col1.number_input("Eye X", value=6.5)
    ey = col2.number_input("Eye Y", value=3.4)
    ez = col3.number_input("Eye Z", value=5.5)
    ax = col1.number_input("At X", value=28.0)
    ay = col2.number_input("At Y", value=2.8)
    az_at = col3.number_input("At Z", value=28.0)
    d_val = st.sidebar.number_input("Distância Focal (d)", value=1.5)
    cfg = {
        'titulo': f'Câmera Manual Eye=({ex}, {ey}, {ez})',
        'eye': [ex, ey, ez], 'at': [ax, ay, az_at], 'up': [0.0, 1.0, 0.0],
        'd': d_val, 'win': (-1.02, 1.02, -1.02, 1.02), 'proj': 'perspectiva'
    }

# Dispara nova renderização ao clicar no botão ou ao alternar Dia/Noite
chave_estado = (str(cfg), ciclo_noite)
if st.sidebar.button("RENDERIZAR RAY CASTING", type="primary") or st.session_state.get('ultima_chave') != chave_estado:
    with st.spinner("Calculando Ray Casting (AABB, Texturas, Sombras e Cáusticas)..."):
        img, titulo = renderer.renderizar_config(cfg, modo_noite=ciclo_noite)
        st.session_state['ultima_img'] = img
        st.session_state['ultimo_titulo'] = titulo
        st.session_state['ultima_chave'] = chave_estado

st.subheader(st.session_state['ultimo_titulo'])
st.image(st.session_state['ultima_img'], use_container_width=False, width=650)
