import streamlit as st
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from motor_ct import RenderizadorCT

st.set_page_config(page_title="Bloco 734 CT + Escultura Cinética + Cáusticas", layout="wide")
st.title("Visualizador 3D & Ray Casting — Bloco 734 (CT), Escultura Cinética e Fonte")

@st.cache_resource
def carregar_motor():
    return RenderizadorCT(largura=500, altura=500)

renderer = carregar_motor()

# --- CONTROLE GLOBAL DIA / NOITE ---
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
    posicoes_camera = {
        "Personalizado (digitar Eye X, Y, Z)": None,
        "Origem do Mundo (0.0, 0.0, 0.0)": (0.0, 0.0, 0.0),
        "Origem Nível do Olhar (0.0, 1.7, 0.0)": (0.0, 1.7, 0.0),
        "Padrão Frontal-Esquerda (6.5, 3.4, 5.5)": (6.5, 3.4, 5.5),
        "Frente ao Espelho d'Água (20.0, 3.2, 11.0)": (20.0, 3.2, 11.0),
        "Vista Aérea Elevada (7.5, 18.0, 6.0)": (7.5, 18.0, 6.0),
    }
    alvos_rapidos = {
        "Personalizado (digitar At X, Y, Z)": None,
        "Centro do Conjunto (28.0, 2.8, 28.0)": (28.0, 2.8, 28.0),
        "Bloco 734 (Fachada Curva / Placa)": (22.0, 3.5, 24.0),
        "Escultura Cinética (5 Losangos)": (15.0, 2.2, 22.0),
        "Espelho d'Água (Fonte e Cáusticas)": (27.0, 0.5, 19.8),
        "Abrigo Metálico (Malha)": (46.5, 1.5, 18.8),
        "Cones de Sinalização": (35.5, 0.6, 16.0),
    }

    pos_escolhida = st.sidebar.selectbox("Atalho de Posição (Eye):", list(posicoes_camera.keys()))
    alvo_escolhido = st.sidebar.selectbox("Atalho de Alvo (At):", list(alvos_rapidos.keys()))

    coords_eye = posicoes_camera[pos_escolhida]
    coords_alvo = alvos_rapidos[alvo_escolhido]

    col1, col2, col3 = st.sidebar.columns(3)
    ex = col1.number_input("Eye X", value=coords_eye[0] if coords_eye else 6.5)
    ey = col2.number_input("Eye Y", value=coords_eye[1] if coords_eye else 3.4)
    ez = col3.number_input("Eye Z", value=coords_eye[2] if coords_eye else 5.5)
    ax = col1.number_input("At X", value=coords_alvo[0] if coords_alvo else 28.0)
    ay = col2.number_input("At Y", value=coords_alvo[1] if coords_alvo else 2.8)
    az_at = col3.number_input("At Z", value=coords_alvo[2] if coords_alvo else 28.0)
    d_val = st.sidebar.number_input("Distância Focal (d)", value=1.5)

    cfg = {
        'titulo': f'Câmera Manual Eye=({ex}, {ey}, {ez}) -> At=({ax}, {ay}, {az_at})',
        'eye': [ex, ey, ez], 'at': [ax, ay, az_at], 'up': [0.0, 1.0, 0.0],
        'd': d_val, 'win': (-1.02, 1.02, -1.02, 1.02), 'proj': 'perspectiva'
    }

# Renderiza apenas quando mudar a câmera ou o botão Dia/Noite (mantém o buffer de picking salvo)
chave_estado = (str(cfg), ciclo_noite)
if st.sidebar.button("RENDERIZAR RAY CASTING", type="primary") or st.session_state.get('ultima_chave') != chave_estado:
    with st.spinner("Calculando Ray Casting (AABB, Texturas, Sombras e Cáusticas)..."):
        img, titulo = renderer.renderizar_config(cfg, modo_noite=ciclo_noite)
        st.session_state['ultima_img'] = img
        st.session_state['ultimo_titulo'] = titulo
        st.session_state['ultima_chave'] = chave_estado
        st.session_state['pick_buffer'] = renderer.ultimo_buffer_pick

st.subheader(st.session_state['ultimo_titulo'])
st.caption("Clique com o mouse ou toque com o dedo sobre qualquer elemento da imagem para executar o **Picking** (Objeto, Material e Coordenada 3D).")

# Converte o array float [0..1] para imagem PIL clicável/tocável
img_uint8 = (np.clip(st.session_state['ultima_img'], 0.0, 1.0) * 255).astype(np.uint8)
pil_img = Image.fromarray(img_uint8)

# Exibe a imagem interativa que captura clique do mouse e toque na tela
largura_exibicao = 580
coords = streamlit_image_coordinates(pil_img, width=largura_exibicao, key="viewport_picking")

# Processa o Picking quando o usuário clica ou toca na imagem
buf = st.session_state.get('pick_buffer')
if coords is not None and buf is not None:
    escala_x = renderer.W / float(largura_exibicao)
    escala_y = renderer.H / float(largura_exibicao)
    px = int(np.clip(coords["x"] * escala_x, 0, renderer.W - 1))
    py = int(np.clip(coords["y"] * escala_y, 0, renderer.H - 1))

    oid = buf['obj_idx'][py, px]
    if oid < 0:
        st.info(f"**PICK ({px}, {py}):** Céu / Fundo")
    else:
        prim = renderer.cena[oid]
        pw = buf['P_world'][py, px]
        if prim.tipo == 'bloco_ct_arqueado':
            mat_nome = renderer.materiais_bloco[buf['sub_mat'][py, px]].nome
        else:
            mat_nome = prim.mat.nome

        st.success(
            f"**PICK ({px}, {py})** — **Objeto:** `{prim.nome}` | "
            f"**Material:** `{mat_nome}` | "
            f"**Mundo (X, Y, Z):** `({pw[0]:.2f} m, {pw[1]:.2f} m, {pw[2]:.2f} m)`"
        )
