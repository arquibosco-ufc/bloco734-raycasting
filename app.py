import streamlit as st
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from motor_ct import RenderizadorCT

st.set_page_config(page_title="Bloco 734 CT + Escultura Cinética + Cáusticas", layout="wide")
st.title("Visualizador 3D & Ray Casting — Bloco 734 (CT), Escultura Cinética e Fonte")

@st.cache_resource
def carregar_motor(resolucao=360):
    return RenderizadorCT(largura=resolucao, altura=resolucao)

# --- 1. CONTROLE GLOBAL DE MODO (TEMPO REAL vs RAY CASTING) ---
st.sidebar.header("Modo de Visualização")
modo_motor = st.sidebar.radio(
    "Motor Gráfico:",
    ["Preview 3D em Tempo Real (Instantâneo)", "Ray Casting Completo (Sombras + Cáusticas)"],
    index=0
)

ciclo_noite = st.sidebar.toggle("Modo Noturno (Postes + Spots + Cáusticas)", value=False)
estilo_3d = st.sidebar.selectbox("Estilo do Preview 3D:", ["Materiais (Realista)", "Sólido (Shaded)", "Linhas (Wireframe)"])

st.sidebar.divider()
st.sidebar.header("Controles de Câmera")
modo_cam = st.sidebar.radio("Seleção de Câmera:", ["Presets da Cena", "Órbita Abóbada (0° a 90°)", "Câmera Manual (Coordenadas)"])

if modo_cam == "Presets da Cena":
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
    motor_tmp = carregar_motor(360)
    cfg = motor_tmp.presets[opcoes[escolha]].copy()

elif modo_cam == "Órbita Abóbada (0° a 90°)":
    motor_tmp = carregar_motor(360)
    azim = st.sidebar.slider("Azimute (°):", -180, 180, -125)
    elev = st.sidebar.slider("Elevação (°) [0° piso a 90° topo]:", 0, 90, 12)
    el, az = np.deg2rad(elev), np.deg2rad(azim)
    c_domo, r_domo = motor_tmp.centro_domo, motor_tmp.raio_domo
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

# ==============================================================================
# CONSTRUTOR DE CENA WEBGL LEVE (PLOTLY 3D - RESPOSTA INSTANTÂNEA)
# ==============================================================================
def gerar_figura_webgl_leve(renderer, cfg_cam, modo_noite, estilo):
    fig = go.Figure()

    wire = (estilo == "Linhas (Wireframe)")
    shaded = (estilo == "Sólido (Shaded)")

    if modo_noite:
        bg_color = "#0B1120"
        c_bloco, c_teto, c_vidro = ("#334155", "#1E293B", "#090D16") if not shaded else ("#475569", "#334155", "#1E293B")
        c_asfalto, c_calcada, c_agua, c_aco = "#111827", "#334155", "#06B6D4", "#E0F2FE"
    else:
        bg_color = "#F8FAFC"
        c_bloco, c_teto, c_vidro = ("#F1F5F9", "#E2E8F0", "#1E293B") if not shaded else ("#CBD5E1", "#94A3B8", "#475569")
        c_asfalto, c_calcada, c_agua, c_aco = "#334155", "#CBD5E1", "#38BDF8", "#FFFFFF"

    def add_box(xmin, xmax, ymin, ymax, zmin, zmax, color, name, mat_name, opacity=1.0):
        # Coordenadas Plotly: X = Mundo X, Y = Mundo Z (profundidade), Z = Mundo Y (altura)
        vx = [xmin, xmax, xmax, xmin, xmin, xmax, xmax, xmin]
        vy = [zmin, zmin, zmax, zmax, zmin, zmin, zmax, zmax]
        vz = [ymin, ymin, ymin, ymin, ymax, ymax, ymax, ymax]
        if wire:
            edges_x = [xmin,xmax,xmax,xmin,xmin, None, xmin,xmax,xmax,xmin,xmin, None, xmin,xmin, None, xmax,xmax, None, xmax,xmax, None, xmin,xmin]
            edges_y = [zmin,zmin,zmax,zmax,zmin, None, zmin,zmin,zmax,zmax,zmin, None, zmin,zmin, None, zmin,zmin, None, zmax,zmax, None, zmax,zmax]
            edges_z = [ymin,ymin,ymin,ymin,ymin, None, ymax,ymax,ymax,ymax,ymax, None, ymin,ymax, None, ymin,ymax, None, ymin,ymax, None, ymin,ymax]
            fig.add_trace(go.Scatter3d(
                x=edges_x, y=edges_y, z=edges_z, mode='lines',
                line=dict(color='#38BDF8' if modo_noite else '#0F172A', width=2),
                name=name, hovertext=f"{name} | {mat_name}", hoverinfo="text+x+y+z"
            ))
        else:
            i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
            j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
            k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
            fig.add_trace(go.Mesh3d(
                x=vx, y=vy, z=vz, i=i, j=j, k=k,
                color=color, opacity=opacity, flatshading=True,
                name=name, hovertext=f"PICK: {name} | Material: {mat_name}", hoverinfo="text+x+y+z"
            ))

    # 1. Via e Calçada
    add_box(2.0, 68.0, 0.0, 0.1, 2.0, 16.0, c_asfalto, "Via Asfáltica CT", "Asfalto CBUQ")
    add_box(2.0, 68.0, 0.0, 0.2, 16.0, 60.0, c_calcada, "Calçada Bloco 734", "Concreto")

    # 2. Bloco 734 (Geometria Exata com Arco Tangencial)
    cx, cz, R, H = 37.0, 42.0, 22.0, 6.6
    thetas = np.linspace(np.pi, 1.5 * np.pi, 24)
    arco_x = cx + R * np.cos(thetas)
    arco_z = cz + R * np.sin(thetas)
    pts_x = np.concatenate([[15.0], arco_x, [59.0, 59.0, 15.0]])
    pts_z = np.concatenate([[53.0], arco_z, [20.0, 53.0, 53.0]])

    # Contorno das paredes e laje do Bloco 734
    wx, wy, wz = [], [], []
    for k in range(len(pts_x) - 1):
        wx.extend([pts_x[k], pts_x[k+1], pts_x[k+1], pts_x[k], pts_x[k], None])
        wy.extend([pts_z[k], pts_z[k+1], pts_z[k+1], pts_z[k], pts_z[k], None])
        wz.extend([0.2, 0.2, H, H, 0.2, None])

    fig.add_trace(go.Scatter3d(
        x=wx, y=wy, z=wz, mode='lines',
        line=dict(color='#38BDF8' if (wire and modo_noite) else ('#0F172A' if wire else '#475569'), width=3),
        name="Bloco 734 (CT)", hovertext="PICK: Bloco 734 (CT/Asfalto) | Alvenaria Branca", hoverinfo="text+x+y+z"
    ))

    if not wire:
        # Malha sólida das paredes do Bloco 734
        n_pts = len(pts_x)
        vx_b = np.concatenate([pts_x, pts_x])
        vy_b = np.concatenate([pts_z, pts_z])
        vz_b = np.concatenate([np.full(n_pts, 0.2), np.full(n_pts, H)])
        ib, jb, kb = [], [], []
        for k in range(n_pts - 1):
            ib.extend([k, k])
            jb.extend([k + 1, k + 1 + n_pts])
            kb.extend([k + 1 + n_pts, k + n_pts])
        fig.add_trace(go.Mesh3d(
            x=vx_b, y=vy_b, z=vz_b, i=ib, j=jb, k=kb,
            color=c_bloco, flatshading=True, name="Bloco 734 Paredes",
            hovertext="PICK: Bloco 734 (CT/Asfalto) | Material: Alvenaria Branca", hoverinfo="text+x+y+z"
        ))
        # Laje de cobertura do Bloco 734
        fig.add_trace(go.Mesh3d(
            x=pts_x[:-1], y=pts_z[:-1], z=np.full(n_pts - 1, H),
            delaunayaxis='z', color=c_teto, flatshading=True, name="Cobertura Bloco 734",
            hovertext="PICK: Cobertura Bloco 734 | Laje Impermeabilizada", hoverinfo="text+x+y+z"
        ))

    # Janelas e Porta na Fachada Curva
    for j_idx in range(9):
        s0 = 0.24 + (j_idx + 0.18) * (0.72 / 9.0)
        s1 = 0.24 + (j_idx + 0.82) * (0.72 / 9.0)
        a0, a1 = np.pi + s0 * (np.pi / 2.0), np.pi + s1 * (np.pi / 2.0)
        jx = [cx + (R+0.12)*np.cos(a0), cx + (R+0.12)*np.cos(a1), cx + (R+0.12)*np.cos(a1), cx + (R+0.12)*np.cos(a0)]
        jz = [cz + (R+0.12)*np.sin(a0), cz + (R+0.12)*np.sin(a1), cz + (R+0.12)*np.sin(a1), cz + (R+0.12)*np.sin(a0)]
        jy = [1.35, 1.35, 3.25, 3.25]
        fig.add_trace(go.Mesh3d(
            x=jx, y=jz, z=jy, i=[0, 0], j=[1, 2], k=[2, 3],
            color=c_vidro, name=f"Janela {j_idx+1}",
            hovertext=f"PICK: Janela Fachada Curva ({j_idx+1}) | Vidro Fumê", hoverinfo="text+x+y+z"
        ))

    # 3. Espelho d'Água (Fonte) + Spots Submersos + Feixes de Cáustica
    f_inf = renderer.extras_3d['fonte']
    add_box(f_inf['x_min'], f_inf['x_max'], 0.20, 0.52, f_inf['z_min'], f_inf['z_max'], "#1E293B", "Borda da Fonte", "Granito Escuro")
    add_box(f_inf['x_min']+0.2, f_inf['x_max']-0.2, 0.40, 0.44, f_inf['z_min']+0.2, f_inf['z_max']-0.2, c_agua, "Espelho d'Água (Fonte)", "Água (n=1.333)", opacity=0.85)

    for idx_s, (sp_pos, sp_dir) in enumerate(f_inf['spots_submersos'], start=1):
        fig.add_trace(go.Scatter3d(
            x=[sp_pos[0]], y=[sp_pos[2]], z=[sp_pos[1]], mode='markers',
            marker=dict(size=6, color='#67E8F9'), name=f"Spot Submerso {idx_s}",
            hovertext=f"PICK: Spot Submerso Fonte {idx_s} | LED Subaquático", hoverinfo="text+x+y+z"
        ))
        if modo_noite:
            p_fim = sp_pos + sp_dir * 10.5
            fig.add_trace(go.Scatter3d(
                x=[sp_pos[0], p_fim[0]], y=[sp_pos[2], p_fim[2]], z=[sp_pos[1], p_fim[1]],
                mode='lines', line=dict(color='#67E8F9', width=3, dash='dash'),
                name=f"Feixe Cáustica {idx_s}", hovertext=f"Refração Cáustica Spot {idx_s}", hoverinfo="text"
            ))

    # 4. Escultura Cinética em Aço (5 Losangos Rotacionados a 45°)
    add_box(13.4, 16.6, 0.20, 0.48, 20.3, 23.7, "#64748B", "Base Concreto Escultura", "Concreto Aparente")
    for q in renderer.dados_escultura['quadros']:
        vw = q['vertices_world']
        xs = np.append(vw[:, 0], vw[0, 0])
        zs = np.append(vw[:, 2], vw[0, 2])
        ys = np.append(vw[:, 1], vw[0, 1])
        fig.add_trace(go.Scatter3d(
            x=xs, y=zs, z=ys, mode='lines',
            line=dict(color=c_aco if not wire else ('#38BDF8' if modo_noite else '#0F172A'), width=7),
            name=f"Losango {q['indice']}",
            hovertext=f"PICK: Escultura Cinética - Losango {q['indice']} | Aço Pintura Branca", hoverinfo="text+x+y+z"
        ))

    # 5. Árvores, Postes, Abrigo Metálico e Cones
    for arv in renderer.extras_3d['arvores']:
        fig.add_trace(go.Scatter3d(
            x=[arv['x'], arv['x']], y=[arv['z'], arv['z']], z=[0.2, arv['alt'] + 0.2],
            mode='lines', line=dict(color='#5C3A21', width=8), name="Tronco Árvore",
            hovertext="PICK: Tronco Árvore | Casca", hoverinfo="text+x+y+z"
        ))
        cx_a = [e[0] for e in arv['esferas']]
        cz_a = [e[2] for e in arv['esferas']]
        cy_a = [e[1] for e in arv['esferas']]
        fig.add_trace(go.Scatter3d(
            x=cx_a, y=cz_a, z=cy_a, mode='markers',
            marker=dict(size=22, color='#064E3B' if modo_noite else '#16A34A', opacity=0.9),
            name="Copa Árvore", hovertext="PICK: Copa de Árvore | Folhagem", hoverinfo="text+x+y+z"
        ))

    for idx_p, (px_pos, pz_pos, off_lx) in enumerate(renderer.extras_3d['postes'], start=1):
        fig.add_trace(go.Scatter3d(
            x=[px_pos, px_pos, px_pos + off_lx],
            y=[pz_pos, pz_pos, pz_pos + 0.4],
            z=[0.2, 6.4, 6.3],
            mode='lines+markers',
            line=dict(color='#475569', width=5),
            marker=dict(size=[2, 2, 9 if modo_noite else 5], color='#FDE047' if modo_noite else '#94A3B8'),
            name=f"Poste {idx_p}", hovertext=f"PICK: Poste Iluminação {idx_p}", hoverinfo="text+x+y+z"
        ))

    add_box(44.5, 49.0, 2.3, 2.6, 17.2, 20.4, "#94A3B8", "Abrigo Metálico", "Aço Galvanizado")
    for idx_c, (cx_c, cz_c) in enumerate(renderer.extras_3d['cones'], start=1):
        fig.add_trace(go.Scatter3d(
            x=[cx_c], y=[cz_c], z=[0.65], mode='markers',
            marker=dict(size=7, symbol='diamond', color='#EA580C'),
            name=f"Cone {idx_c}", hovertext=f"PICK: Cone Sinalização {idx_c}", hoverinfo="text+x+y+z"
        ))

    # Conversão exata de Eye e At da câmera para o Preview WebGL
    eye = np.array(cfg_cam['eye'], dtype=np.float64)
    at = np.array(cfg_cam['at'], dtype=np.float64)
    centro_cena = np.array([35.0, 13.0, 31.0])
    escala_norm = 33.0
    cam_eye_plotly = dict(
        x=float((eye[0] - centro_cena[0]) / escala_norm),
        y=float((eye[2] - centro_cena[2]) / escala_norm),
        z=float(max(0.02, (eye[1] - centro_cena[1]) / escala_norm))
    )
    cam_center_plotly = dict(
        x=float((at[0] - centro_cena[0]) / escala_norm),
        y=float((at[2] - centro_cena[2]) / escala_norm),
        z=float((at[1] - centro_cena[1]) / escala_norm)
    )
    proj_type = "orthographic" if cfg_cam.get('proj') == 'ortografica' else "perspective"

    fig.update_layout(
        height=620,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor=bg_color,
        showlegend=False,
        scene=dict(
            bgcolor=bg_color,
            xaxis=dict(title="X (m)", range=[0, 70], showbackground=False, color="#94A3B8"),
            yaxis=dict(title="Z (m)", range=[0, 62], showbackground=False, color="#94A3B8"),
            zaxis=dict(title="Y Altura (m)", range=[0, 26], showbackground=False, color="#94A3B8"),
            aspectmode="manual",
            aspectratio=dict(x=1.15, y=1.0, z=0.45),
            camera=dict(
                eye=cam_eye_plotly,
                center=cam_center_plotly,
                up=dict(x=0, y=0, z=1),
                projection=dict(type=proj_type)
            )
        )
    )
    return fig


# ==============================================================================
# EXIBIÇÃO PRINCIPAL: PREVIEW INSTANTÂNEO (PADRÃO) OU RAY CASTING SOB DEMANDA
# ==============================================================================
if modo_motor == "Preview 3D em Tempo Real (Instantâneo)":
    renderer_leve = carregar_motor(360)
    turno_lbl = "NOITE (Spots + Cáusticas + Postes)" if ciclo_noite else "DIA (Luz Solar)"
    st.subheader(f"Preview 3D Interativo — {cfg['titulo']} [{turno_lbl}]")
    st.caption("Modo leve em tempo real: gire com o dedo/mouse livremente, passe o cursor ou toque em qualquer elemento para **Picking 3D**, ou mude para **Ray Casting Completo** na barra lateral para gerar sombras e cáusticas físicas.")

    fig_3d = gerar_figura_webgl_leve(renderer_leve, cfg, ciclo_noite, estilo_3d)
    st.plotly_chart(fig_3d, use_container_width=True)

else:
    qualidade = st.sidebar.select_slider(
        "Resolução do Ray Casting:",
        options=["Rápido (280x280)", "Equilibrado (380x380)", "Alta (500x500)"],
        value="Equilibrado (380x380)"
    )
    res_map = {"Rápido (280x280)": 280, "Equilibrado (380x380)": 380, "Alta (500x500)": 500}
    res_px = res_map[qualidade]
    renderer_rc = carregar_motor(res_px)

    chave_estado = (str(cfg), ciclo_noite, res_px)
    if st.sidebar.button("RENDERIZAR RAY CASTING AGORA", type="primary") or st.session_state.get('ultima_chave') != chave_estado:
        with st.spinner(f"Calculando Ray Casting {res_px}x{res_px} (AABB Culling, Texturas, Sombras e Cáusticas)..."):
            img, titulo = renderer_rc.renderizar_config(cfg, modo_noite=ciclo_noite)
            st.session_state['ultima_img'] = img
            st.session_state['ultimo_titulo'] = titulo
            st.session_state['ultima_chave'] = chave_estado
            st.session_state['pick_buffer'] = renderer_rc.ultimo_buffer_pick

    st.subheader(st.session_state['ultimo_titulo'])
    st.caption("Clique com o mouse ou toque com o dedo sobre a imagem renderizada para executar o **Picking** (Objeto, Material e Coordenada 3D).")

    img_uint8 = (np.clip(st.session_state['ultima_img'], 0.0, 1.0) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)

    largura_exibicao = 580
    coords = streamlit_image_coordinates(pil_img, width=largura_exibicao, key="viewport_picking")

    buf = st.session_state.get('pick_buffer')
    if coords is not None and buf is not None:
        escala_x = renderer_rc.W / float(largura_exibicao)
        escala_y = renderer_rc.H / float(largura_exibicao)
        px = int(np.clip(coords["x"] * escala_x, 0, renderer_rc.W - 1))
        py = int(np.clip(coords["y"] * escala_y, 0, renderer_rc.H - 1))

        oid = buf['obj_idx'][py, px]
        if oid < 0:
            st.info(f"**PICK ({px}, {py}):** Céu / Fundo")
        else:
            prim = renderer_rc.cena[oid]
            pw = buf['P_world'][py, px]
            if prim.tipo == 'bloco_ct_arqueado':
                mat_nome = renderer_rc.materiais_bloco[buf['sub_mat'][py, px]].nome
            else:
                mat_nome = prim.mat.nome

            st.success(
                f"**PICK ({px}, {py})** — **Objeto:** `{prim.nome}` | "
                f"**Material:** `{mat_nome}` | "
                f"**Mundo (X, Y, Z):** `({pw[0]:.2f} m, {pw[1]:.2f} m, {pw[2]:.2f} m)`"
            )
