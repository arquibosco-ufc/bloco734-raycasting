from motor_ct import RenderizadorCT
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Bloco 734 CT + Escultura Cinética + Cáusticas", layout="wide"
)
st.title(
    "Visualizador 3D & Ray Casting — Bloco 734 (CT), Escultura Cinética e Fonte"
)


@st.cache_resource
def carregar_motor(resolucao=320):
  return RenderizadorCT(largura=resolucao, altura=resolucao)


def gerar_mesh_box(xmin, xmax, ymin, ymax, zmin, zmax, cor, nome):
  vx = [xmin, xmax, xmax, xmin, xmin, xmax, xmax, xmin]
  vy = [zmin, zmin, zmax, zmax, zmin, zmin, zmax, zmax]
  vz = [ymin, ymin, ymin, ymin, ymax, ymax, ymax, ymax]
  i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
  j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
  k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]
  return go.Mesh3d(
      x=vx,
      y=vy,
      z=vz,
      i=i,
      j=j,
      k=k,
      color=cor,
      opacity=1.0,
      flatshading=True,
      name=nome,
      hoverinfo="name+x+y+z",
  )


def construir_figura_3d_interativa(
    renderer, estilo="materiais", modo_noite=False, cfg_cam=None
):
  fig = go.Figure()

  if estilo == "wireframe":
    c_bloco, c_teto, c_vidro, c_734 = (
        "#38BDF8" if modo_noite else "#0F172A",
        "#38BDF8",
        "#38BDF8",
        "#1D4ED8",
    )
    c_asfalto, c_calcada, c_esc_aco, c_agua = (
        "#475569",
        "#64748B",
        "#38BDF8" if modo_noite else "#0F172A",
        "#0284C7",
    )
  elif estilo == "shaded":
    c_bloco, c_teto, c_vidro, c_734 = (
        ("#334155", "#1E293B", "#0F172A", "#1E3A8A")
        if modo_noite
        else ("#CBD5E1", "#94A3B8", "#64748B", "#475569")
    )
    c_asfalto, c_calcada, c_esc_aco, c_agua = (
        "#1E293B",
        "#475569",
        "#E0F2FE",
        "#0284C7",
    )
  else:
    if modo_noite:
      c_bloco, c_teto, c_vidro, c_734 = (
          "#3B4D6B",
          "#1E293B",
          "#090D16",
          "#1D4ED8",
      )
      c_asfalto, c_calcada, c_esc_aco, c_agua = (
          "#111827",
          "#334155",
          "#E0F2FE",
          "#06B6D4",
      )
    else:
      c_bloco, c_teto, c_vidro, c_734 = (
          "#F8FAFC",
          "#E2E8F0",
          "#1E293B",
          "#1E3A8A",
      )
      c_asfalto, c_calcada, c_esc_aco, c_agua = (
          "#334155",
          "#CBD5E1",
          "#F8FAFC",
          "#38BDF8",
      )

  # 1. Piso
  if estilo != "wireframe":
    fig.add_trace(
        gerar_mesh_box(2, 68, 0.0, 0.10, 2, 16, c_asfalto, "Via Asfáltica CT")
    )
    fig.add_trace(
        gerar_mesh_box(2, 68, 0.0, 0.20, 16, 60, c_calcada, "Calçada Bloco 734")
    )

  # 2. Bloco 734
  cx, cz, R, H = 37.0, 42.0, 22.0, 6.6
  thetas = np.linspace(np.pi, 1.5 * np.pi, 24)
  arco_x = cx + R * np.cos(thetas)
  arco_z = cz + R * np.sin(thetas)
  pts_x = np.concatenate([[15.0], arco_x, [59.0, 59.0, 15.0]])
  pts_z = np.concatenate([[53.0], arco_z, [20.0, 53.0, 53.0]])

  if estilo == "wireframe":
    fig.add_trace(
        go.Scatter3d(
            x=pts_x,
            y=pts_z,
            z=np.full_like(pts_x, H),
            mode="lines",
            line=dict(color=c_bloco, width=3),
            name="Bloco 734 (Teto)",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=pts_x,
            y=pts_z,
            z=np.full_like(pts_x, 0.2),
            mode="lines",
            line=dict(color=c_bloco, width=2),
            name="Bloco 734 (Base)",
        )
    )
  else:
    n_p = len(pts_x) - 1
    vx = np.concatenate([pts_x[:n_p], pts_x[:n_p]])
    vy = np.concatenate([pts_z[:n_p], pts_z[:n_p]])
    vz = np.concatenate([np.full(n_p, 0.2), np.full(n_p, H)])
    ii, jj, kk = [], [], []
    for m in range(n_p):
      m_next = (m + 1) % n_p
      ii.extend([m, m])
      jj.extend([m_next, m_next + n_p])
      kk.extend([m_next + n_p, m + n_p])
    for m in range(1, n_p - 1):
      ii.append(n_p)
      jj.append(n_p + m)
      kk.append(n_p + m + 1)
    fig.add_trace(
        go.Mesh3d(
            x=vx,
            y=vy,
            z=vz,
            i=ii,
            j=jj,
            k=kk,
            color=c_bloco,
            flatshading=True,
            name="Bloco 734 (Fachada)",
        )
    )

  # 3. Fonte / Espelho d'Água
  f_inf = renderer.extras_3d["fonte"]
  fx0, fx1, fz0, fz1, ya = (
      f_inf["x_min"],
      f_inf["x_max"],
      f_inf["z_min"],
      f_inf["z_max"],
      f_inf["y_agua"],
  )
  fig.add_trace(
      gerar_mesh_box(
          fx0, fx1, 0.20, ya, fz0, fz1, c_agua, "Espelho d'Água (Fonte)"
      )
  )

  # 4. Escultura Cinética (5 Losangos)
  esc = renderer.dados_escultura
  bp = esc["base_pts_world"]
  fig.add_trace(
      gerar_mesh_box(
          bp[:, 0].min(),
          bp[:, 0].max(),
          0.20,
          0.48,
          bp[:, 2].min(),
          bp[:, 2].max(),
          "#475569",
          "Base Escultura",
      )
  )
  for q in esc["quadros"]:
    vw = q["vertices_world"]
    xs = np.append(vw[:, 0], vw[0, 0])
    zs = np.append(vw[:, 2], vw[0, 2])
    ys = np.append(vw[:, 1], vw[0, 1])
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=zs,
            z=ys,
            mode="lines",
            line=dict(color=c_esc_aco, width=7),
            name=f"Losango {q['indice']}",
        )
    )

  # 5. Árvores e Postes
  for arv in renderer.extras_3d["arvores"]:
    fig.add_trace(
        go.Scatter3d(
            x=[arv["x"], arv["x"]],
            y=[arv["z"], arv["z"]],
            z=[0.2, arv["alt"] + 0.2],
            mode="lines",
            line=dict(color="#5C3A21", width=8),
            showlegend=False,
        )
    )
    for px_e, py_e, pz_e, r_e, cor_e in arv["esferas"]:
      fig.add_trace(
          go.Scatter3d(
              x=[px_e],
              y=[pz_e],
              z=[py_e],
              mode="markers",
              marker=dict(
                  size=18, color="#064E3B" if modo_noite else cor_e, opacity=0.9
              ),
              showlegend=False,
          )
      )

  for idx_p, (px_pos, pz_pos, off_lx) in enumerate(
      renderer.extras_3d["postes"], start=1
  ):
    fig.add_trace(
        go.Scatter3d(
            x=[px_pos, px_pos, px_pos + off_lx],
            y=[pz_pos, pz_pos, pz_pos + 0.4],
            z=[0.2, 6.4, 6.3],
            mode="lines+markers",
            line=dict(color="#334155", width=4),
            marker=dict(
                size=6,
                color="#FDE047" if modo_noite else "#94A3B8",
            ),
            name=f"Poste {idx_p}",
        )
    )

  # 6. Malhas (.OBJ e Abrigo Metálico)
  cores_malhas = {
      "Abrigo Metálico (Malha)": "#94A3B8",
      "SUV na Via CT (OBJ)": "#CBD5E1",
      "Carro Esportivo Via CT (OBJ)": "#DC2626",
      "Estudante Casual (OBJ)": "#2563EB",
      "Estudante Vestido (OBJ)": "#D97706",
      "Palmeira Praça CT (OBJ)": "#15803D",
  }
  for prim in renderer.cena:
    if prim.tipo == "malha" and prim.nome in cores_malhas:
      vw = prim.dados.get("vertices_world", np.zeros((0, 3)))
      fc = prim.dados.get("faces", np.zeros((0, 3), dtype=int))
      if len(vw) == 0 or len(fc) == 0:
        continue
      fig.add_trace(
          go.Mesh3d(
              x=vw[:, 0],
              y=vw[:, 2],
              z=vw[:, 1],
              i=fc[:, 0],
              j=fc[:, 1],
              k=fc[:, 2],
              color=cores_malhas[prim.nome],
              flatshading=True,
              name=prim.nome,
          )
      )

  bg_color = "#0B1120" if modo_noite else "#F8FAFC"
  txt_color = "#E2E8F0" if modo_noite else "#0F172A"
  proj_type = (
      "orthographic"
      if (cfg_cam and cfg_cam.get("proj") == "ortografica")
      else "perspective"
  )

  fig.update_layout(
      height=620,
      margin=dict(l=0, r=0, t=30, b=0),
      paper_bgcolor=bg_color,
      font=dict(color=txt_color),
      showlegend=False,
      scene=dict(
          bgcolor=bg_color,
          aspectmode="data",
          camera=dict(
              eye=dict(x=-1.4, y=-1.4, z=0.65),
              projection=dict(type=proj_type),
          ),
          xaxis=dict(title="X (m)", range=[2, 68]),
          yaxis=dict(title="Z (m)", range=[2, 60]),
          zaxis=dict(title="Y (m)", range=[0, 22]),
      ),
  )
  return fig


# ==============================================================================
# BARRA LATERAL E CONTROLES
# ==============================================================================
st.sidebar.header("Controles da Cena")
resolucao = st.sidebar.select_slider(
    "Resolução do Ray Casting", options=[200, 280, 360], value=280
)
renderer = carregar_motor(resolucao)

modo_noite = st.sidebar.toggle(
    "Modo Noturno (Postes + Cáusticas da Fonte)", value=False
)
estilo_preview = st.sidebar.radio(
    "Estilo do Preview 3D",
    options=["materiais", "shaded", "wireframe"],
    format_func=lambda x: {
        "materiais": "Materiais",
        "shaded": "Planos (Shaded)",
        "wireframe": "Linhas (Wireframe)",
    }[x],
    horizontal=True,
)

preset_nomes = {
    "2_pontos_fuga": "2 Pontos de Fuga (Padrão)",
    "1_ponto_fuga": "1 Ponto de Fuga (Frontal)",
    "3_pontos_fuga": "3 Pontos de Fuga (Aérea)",
    "zoom_in": "Zoom In (Escultura + Fonte)",
    "zoom_out": "Zoom Out (Campo Ampliado)",
    "ortografica": "Projeção Ortográfica",
    "obliqua": "Projeção Oblíqua",
}
preset_escolhido = st.sidebar.selectbox(
    "Câmera Predefinida",
    options=list(preset_nomes.keys()),
    format_func=lambda k: preset_nomes[k],
)
cfg_atual = renderer.presets[preset_escolhido].copy()

with st.sidebar.expander("Câmera Manual (Eye / At)"):
  usar_manual = st.checkbox("Ativar coordenadas manuais", value=False)
  col_e1, col_e2, col_e3 = st.columns(3)
  ex = col_e1.number_input("Eye X", value=float(cfg_atual["eye"][0]))
  ey = col_e2.number_input("Eye Y", value=float(cfg_atual["eye"][1]))
  ez = col_e3.number_input("Eye Z", value=float(cfg_atual["eye"][2]))

  col_a1, col_a2, col_a3 = st.columns(3)
  ax_v = col_a1.number_input("At X", value=float(cfg_atual["at"][0]))
  ay_v = col_a2.number_input("At Y", value=float(cfg_atual["at"][1]))
  az_v = col_a3.number_input("At Z", value=float(cfg_atual["at"][2]))

  if usar_manual:
    cfg_atual["eye"] = [ex, ey, ez]
    cfg_atual["at"] = [ax_v, ay_v, az_v]
    cfg_atual["titulo"] = f"Câmera Manual Eye=({ex:.1f}, {ey:.1f}, {ez:.1f})"

aba_3d, aba_rc = st.tabs(
    ["1. Preview 3D Interativo", "2. Render Final (Ray Casting + Cáusticas)"]
)

with aba_3d:
  fig_3d = construir_figura_3d_interativa(
      renderer,
      estilo=estilo_preview,
      modo_noite=modo_noite,
      cfg_cam=cfg_atual,
  )
  st.plotly_chart(fig_3d, width="stretch")

with aba_rc:
  if st.button(
      "RENDERIZAR RAY CASTING AGORA", type="primary", width="stretch"
  ):
    with st.spinner("Calculando interseções, sombras, reflexões e cáusticas..."):
      img, titulo_render = renderer.renderizar_config(
          cfg_atual, modo_noite=modo_noite
      )
      st.session_state["img_render"] = img
      st.session_state["tit_render"] = titulo_render

  if "img_render" in st.session_state:
    st.subheader(st.session_state["tit_render"])
    st.image(st.session_state["img_render"], width="stretch")
  else:
    st.info(
        "Clique no botão acima para gerar a imagem renderizada por Ray Casting."
    )
