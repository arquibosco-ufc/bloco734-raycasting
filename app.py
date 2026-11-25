from motor_ct import RenderizadorCT
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Bloco 734 CT — Trabalho Final Computação Gráfica I",
    layout="wide",
)

# Cabeçalho Institucional e Acadêmico
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #0F172A 0%, #1E3A8A 100%); padding: 14px 20px; border-radius: 10px; border: 1px solid #38BDF8; margin-bottom: 14px;">
      <div style="font-size: 20px; font-weight: 800; color: #F8FAFC;">Trabalho Final — Computação Gráfica I (Bloco 734 / CT)</div>
      <div style="font-size: 14px; color: #BAE6FD; margin-top: 4px;">
        <b>Professor:</b> Prof. Creto Vidal &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Estudante:</b> João Bosco Pinheiro Dantas Filho
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def carregar_motor(resolucao=420):
  return RenderizadorCT(largura=resolucao, altura=resolucao)


CATALOGO_OBJETOS_PICKING = {
    "Bloco 734 (Alvenaria)": (
        "Edificação Principal do Centro de Tecnologia (33m x 44m, H=6.6m) com fachada curva tangencial de raio 22m."
    ),
    "Bloco 734 (Platibanda)": (
        "Platibanda superior do Bloco 734 com textura procedural."
    ),
    "Platibanda Nº 734": (
        "Identificação numérica 734 mapeada na fachada arqueada."
    ),
    "Porta Principal Bloco 734": (
        "Acesso principal em vidro fumê refletivo (R=0.55) e esquadria escura."
    ),
    "Escada Acesso Telhado": (
        "Escada metálica lateral (X=59.35m, Z=24.0m) para acesso à cobertura do Bloco 734."
    ),
    "Espelho d'Água (Fonte)": (
        "Lâmina d'água frontal (15m x 6m) com índice de refração n=1.333 e projeção de cáusticas."
    ),
    "Borda Fonte": "Mureta de contenção em granito escuro.",
    "Base Escultura Cinética": (
        "Pedestal em concreto aparente (3.2m x 3.4m) rotacionado em -24° no eixo Y."
    ),
    "Via Asfáltica CT": (
        "Pavimento asfáltico CBUQ com textura procedural granular e sinalização horizontal."
    ),
    "Calçada Bloco 734": (
        "Passeio em placas de concreto rejuntadas a cada 1.5m."
    ),
    "Faixa Viária": "Sinalização horizontal amarela da via do CT.",
    "Abrigo Metálico (Malha)": (
        "Cobertura metálica construída por malha poligonal com transformação de cisalhamento XY."
    ),
    "SUV na Via CT (OBJ)": (
        "Malha externa .OBJ normalizada e sombreada com normais suaves por vértice."
    ),
    "Carro Esportivo Via CT (OBJ)": (
        "Malha externa .OBJ em pintura vermelha metálica."
    ),
    "Estudante Casual (OBJ)": (
        "Modelo humano .OBJ posicionado na calçada frontal."
    ),
    "Estudante Vestido (OBJ)": (
        "Modelo humano .OBJ posicionado próximo à fonte."
    ),
    "Palmeira Praça CT (OBJ)": (
        "Vegetação tropical .OBJ posicionada na praça lateral."
    ),
}


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


def converter_camera_para_plotly(cfg_cam, azimute_deg=None, elevacao_deg=None):
  if azimute_deg is not None and elevacao_deg is not None:
    az = np.deg2rad(azimute_deg)
    el = np.deg2rad(np.clip(elevacao_deg, 2.0, 88.0))
    r = 1.65
    ex = r * np.cos(el) * np.cos(az)
    ey = r * np.cos(el) * np.sin(az)
    ez = max(0.06, r * np.sin(el))
    proj_type = (
        "orthographic"
        if cfg_cam and cfg_cam.get("proj") in ("ortografica", "obliqua")
        else "perspective"
    )
    return dict(
        eye=dict(x=float(ex), y=float(ey), z=float(ez)),
        center=dict(x=0.0, y=0.0, z=-0.12),
        up=dict(x=0, y=0, z=1),
        projection=dict(type=proj_type),
    )

  eye = np.array(cfg_cam["eye"], dtype=float)
  at = np.array(cfg_cam["at"], dtype=float)
  centro_cena = np.array([35.0, 31.0, 3.5])
  escala_unif = 36.0

  eye_pl = np.array([eye[0], eye[2], max(eye[1], 1.2)])
  at_pl = np.array([at[0], at[2], max(at[1], 1.0)])

  cam_eye = (eye_pl - centro_cena) / escala_unif
  cam_center = (at_pl - centro_cena) / escala_unif
  cam_eye[2] = max(0.06, cam_eye[2])

  proj_type = (
      "orthographic"
      if cfg_cam.get("proj") in ("ortografica", "obliqua")
      else "perspective"
  )
  return dict(
      eye=dict(x=float(cam_eye[0]), y=float(cam_eye[1]), z=float(cam_eye[2])),
      center=dict(
          x=float(cam_center[0]), y=float(cam_center[1]), z=float(cam_center[2])
      ),
      up=dict(x=0, y=0, z=1),
      projection=dict(type=proj_type),
  )


def construir_figura_3d_interativa(
    renderer,
    estilo="materiais",
    modo_noite=False,
    cfg_cam=None,
    azimute_deg=None,
    elevacao_deg=None,
):
  fig = go.Figure()

  if estilo == "wireframe":
    c_bloco = "#38BDF8" if modo_noite else "#0F172A"
    c_vidro = "#38BDF8"
    c_734 = "#1D4ED8"
    c_asfalto, c_calcada, c_esc_aco, c_agua = (
        "#475569",
        "#64748B",
        "#38BDF8" if modo_noite else "#0F172A",
        "#0284C7",
    )
  elif estilo == "shaded":
    c_bloco = "#334155" if modo_noite else "#CBD5E1"
    c_vidro = "#0F172A" if modo_noite else "#64748B"
    c_734 = "#1E3A8A" if modo_noite else "#475569"
    c_asfalto, c_calcada, c_esc_aco, c_agua = (
        "#1E293B",
        "#475569",
        "#E0F2FE",
        "#0284C7",
    )
  else:
    if modo_noite:
      c_bloco, c_vidro, c_734 = "#3B4D6B", "#090D16", "#1D4ED8"
      c_asfalto, c_calcada, c_esc_aco, c_agua = (
          "#111827",
          "#334155",
          "#E0F2FE",
          "#06B6D4",
      )
    else:
      c_bloco, c_vidro, c_734 = "#F8FAFC", "#1E293B", "#1E3A8A"
      c_asfalto, c_calcada, c_esc_aco, c_agua = (
          "#334155",
          "#CBD5E1",
          "#F8FAFC",
          "#38BDF8",
      )

  if estilo != "wireframe":
    fig.add_trace(
        gerar_mesh_box(2, 68, 0.0, 0.10, 2, 16, c_asfalto, "Via Asfáltica CT")
    )
    fig.add_trace(
        gerar_mesh_box(2, 68, 0.0, 0.20, 16, 60, c_calcada, "Calçada Bloco 734")
    )
    for fx in range(4, 66, 6):
      fig.add_trace(
          gerar_mesh_box(
              fx, fx + 3, 0.10, 0.12, 11.6, 11.8, "#EAB308", "Faixa Viária"
          )
      )

  cx, cz, R, H = 37.0, 42.0, 22.0, 6.6
  thetas = np.linspace(np.pi, 1.5 * np.pi, 32)
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
            name="Bloco 734 (Platibanda)",
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
    for k_v in range(0, len(pts_x) - 1, 3):
      fig.add_trace(
          go.Scatter3d(
              x=[pts_x[k_v], pts_x[k_v]],
              y=[pts_z[k_v], pts_z[k_v]],
              z=[0.2, H],
              mode="lines",
              line=dict(color=c_bloco, width=2),
              showlegend=False,
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
            name="Bloco 734 (Alvenaria)",
        )
    )

  for idx_j in range(9):
    s0 = 0.24 + (idx_j + 0.16) * (0.72 / 9.0)
    s1 = 0.24 + (idx_j + 0.84) * (0.72 / 9.0)
    ang0, ang1 = s0 * (np.pi / 2.0), s1 * (np.pi / 2.0)
    jx = [
        cx - (R + 0.06) * np.cos(ang0),
        cx - (R + 0.06) * np.cos(ang1),
        cx - (R + 0.06) * np.cos(ang1),
        cx - (R + 0.06) * np.cos(ang0),
    ]
    jz = [
        cz - (R + 0.06) * np.sin(ang0),
        cz - (R + 0.06) * np.sin(ang1),
        cz - (R + 0.06) * np.sin(ang1),
        cz - (R + 0.06) * np.sin(ang0),
    ]
    fig.add_trace(
        go.Mesh3d(
            x=jx,
            y=jz,
            z=[1.35, 1.35, 3.25, 3.25],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color=c_vidro,
            name=f"Janela Vidro Fumê {idx_j+1}",
        )
    )

  ang_p0, ang_p1 = 0.06 * (np.pi / 2.0), 0.20 * (np.pi / 2.0)
  px_p = [
      cx - (R + 0.06) * np.cos(ang_p0),
      cx - (R + 0.06) * np.cos(ang_p1),
      cx - (R + 0.06) * np.cos(ang_p1),
      cx - (R + 0.06) * np.cos(ang_p0),
  ]
  pz_p = [
      cz - (R + 0.06) * np.sin(ang_p0),
      cz - (R + 0.06) * np.sin(ang_p1),
      cz - (R + 0.06) * np.sin(ang_p1),
      cz - (R + 0.06) * np.sin(ang_p0),
  ]
  fig.add_trace(
      go.Mesh3d(
          x=px_p,
          y=pz_p,
          z=[0.2, 0.2, 2.75, 2.75],
          i=[0, 0],
          j=[1, 2],
          k=[2, 3],
          color=c_vidro,
          name="Porta Principal Bloco 734",
      )
  )

  fig.add_trace(
      gerar_mesh_box(
          59.05, 59.65, 0.20, 6.80, 23.4, 24.6, "#F59E0B", "Escada Acesso Telhado"
      )
  )

  ang_734 = 0.14 * (np.pi / 2.0)
  fig.add_trace(
      go.Scatter3d(
          x=[cx - (R + 0.15) * np.cos(ang_734)],
          y=[cz - (R + 0.15) * np.sin(ang_734)],
          z=[5.7],
          mode="text",
          text=["<b>734</b>"],
          textfont=dict(size=16, color=c_734),
          name="Platibanda Nº 734",
      )
  )

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
          fx0 - 0.15, fx1 + 0.15, 0.20, 0.54, fz0 - 0.15, fz0 + 0.15, "#1E293B", "Borda Fonte"
      )
  )
  fig.add_trace(
      gerar_mesh_box(
          fx0 - 0.15, fx1 + 0.15, 0.20, 0.54, fz1 - 0.15, fz1 + 0.15, "#1E293B", "Borda Fonte"
      )
  )
  fig.add_trace(
      gerar_mesh_box(
          fx0, fx1, 0.20, ya, fz0, fz1, c_agua, "Espelho d'Água (Fonte)"
      )
  )

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
          "Base Escultura Cinética",
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
            line=dict(color=c_esc_aco, width=8),
            name=f"Escultura Cinética - Losango {q['indice']}",
        )
    )

  for idx_a, arv in enumerate(renderer.extras_3d["arvores"], start=1):
    fig.add_trace(
        go.Scatter3d(
            x=[arv["x"], arv["x"]],
            y=[arv["z"], arv["z"]],
            z=[0.2, arv["alt"] + 0.2],
            mode="lines",
            line=dict(color="#5C3A21", width=9),
            name=f"Tronco Árvore {idx_a}",
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
                  size=18, color="#064E3B" if modo_noite else cor_e, opacity=0.92
              ),
              name=f"Copa Árvore {idx_a}",
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
            line=dict(color="#334155", width=5),
            marker=dict(
                size=7,
                color="#FDE047" if modo_noite else "#94A3B8",
            ),
            name=f"Poste Iluminação {idx_p}",
        )
    )

  for idx_c, (cx_c, cz_c) in enumerate(renderer.extras_3d["cones"], start=1):
    fig.add_trace(
        go.Scatter3d(
            x=[cx_c, cx_c],
            y=[cz_c, cz_c],
            z=[0.2, 1.15],
            mode="lines+markers",
            line=dict(color="#EA580C", width=8),
            marker=dict(size=5, color="#F97316"),
            name=f"Cone Sinalização {idx_c}",
        )
    )

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
              opacity=1.0,
              flatshading=False,
              lighting=dict(ambient=0.55, diffuse=0.8, specular=0.35),
              name=prim.nome,
              hoverinfo="name+x+y+z",
          )
      )

  bg_color = "#0B1120" if modo_noite else "#F8FAFC"
  txt_color = "#E2E8F0" if modo_noite else "#0F172A"
  camera_plotly = converter_camera_para_plotly(
      cfg_cam, azimute_deg=azimute_deg, elevacao_deg=elevacao_deg
  )

  fig.update_layout(
      height=600,
      margin=dict(l=0, r=0, t=20, b=0),
      paper_bgcolor=bg_color,
      font=dict(color=txt_color),
      showlegend=False,
      clickmode="event+select",
      scene=dict(
          bgcolor=bg_color,
          aspectmode="data",
          camera=camera_plotly,
          xaxis=dict(title="X (m)", range=[2, 68]),
          yaxis=dict(title="Z (m)", range=[2, 60]),
          zaxis=dict(title="Altura Y (m)", range=[0, 12]),
      ),
  )
  return fig


def renderizar_inspecao_webgl_roblox(modo_noite=False):
  """Visualizador estilo Roblox (WebGL sem Raycast) exclusivo para Inspeção 3D e Picking interativo."""
  noite_js = "true" if modo_noite else "false"
  html_code = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8" />
<style>
  body {{ margin: 0; overflow: hidden; background: #0B1120; font-family: system-ui, -apple-system, sans-serif; user-select: none; }}
  #hud-insp {{
    position: absolute; top: 12px; left: 12px; z-index: 10;
    background: rgba(15, 23, 42, 0.88); color: #F8FAFC;
    padding: 11px 15px; border-radius: 10px; border: 1px solid rgba(56, 189, 248, 0.45);
    font-size: 12.5px; line-height: 1.45;
  }}
  #pick-popup {{
    display: none; position: absolute; top: 12px; right: 12px; z-index: 20;
    background: rgba(15, 23, 42, 0.94); color: #F8FAFC;
    padding: 14px 18px; border-radius: 12px; border: 2px solid #38BDF8;
    width: 310px; box-shadow: 0 10px 25px rgba(0,0,0,0.65); font-size: 13px; line-height: 1.5;
  }}
  #pick-popup h4 {{ margin: 0 0 6px 0; color: #38BDF8; font-size: 15px; }}
  .tag-info {{ display: inline-block; background: #1E3A8A; color: #93C5FD; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; margin-bottom: 6px; }}
</style>
</head>
<body>
<div id="hud-insp">
  <div style="font-weight:800; color:#38BDF8; font-size:14px;">Inspeção 3D — Modo WebGL (Sem Raycast)</div>
  <div>• <b>Clique Esquerdo em qualquer objeto</b>: Picking completo</div>
  <div>• <b>Arrastar Mouse</b>: Orbitar (0°–360° Azimute | 2°–88° Elevação)</div>
  <div>• <b>Scroll do Mouse</b>: Zoom In / Out</div>
</div>

<div id="pick-popup">
  <span class="tag-info" id="pk-cat">PRIMITIVA 3D</span>
  <h4 id="pk-nome">Objeto</h4>
  <div><b>Material:</b> <span id="pk-mat">-</span></div>
  <div><b>Coordenadas Clique:</b> <span id="pk-coord">-</span></div>
  <div><b>Transformação / Detalhe:</b> <span id="pk-desc">-</span></div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
  const MODO_NOITE = {noite_js};
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(MODO_NOITE ? 0x070d19 : 0x93c5fd);

  const camera = new THREE.PerspectiveCamera(52, window.innerWidth / window.innerHeight, 0.1, 300);
  const renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = true;
  document.body.appendChild(renderer.domElement);

  scene.add(new THREE.HemisphereLight(MODO_NOITE ? 0x1e293b : 0xffffff, MODO_NOITE ? 0x0f172a : 0x64748b, MODO_NOITE ? 0.6 : 0.9));
  const dirLight = new THREE.DirectionalLight(MODO_NOITE ? 0x3b82f6 : 0xfffbeb, MODO_NOITE ? 0.35 : 1.15);
  dirLight.position.set(12, 34, 10);
  dirLight.castShadow = true;
  scene.add(dirLight);

  const pickables = [];
  function registrarPick(mesh, nome, cat, mat, desc) {{
    mesh.userData = {{ nome, cat, mat, desc }};
    pickables.push(mesh);
  }}

  const via = new THREE.Mesh(new THREE.BoxGeometry(66, 0.1, 14), new THREE.MeshStandardMaterial({{ color: 0x1e293b }}));
  via.position.set(35, 0.05, 9); scene.add(via);
  registrarPick(via, "Via Asfáltica CT", "Cubo + Textura Procedural", "Asfalto CBUQ", "Escala (66m, 0.1m, 14m) com faixas viárias.");

  const calcada = new THREE.Mesh(new THREE.BoxGeometry(66, 0.2, 44), new THREE.MeshStandardMaterial({{ color: MODO_NOITE ? 0x334155 : 0xcbd5e1 }}));
  calcada.position.set(35, 0.1, 38); scene.add(calcada);
  registrarPick(calcada, "Calçada Bloco 734", "Cubo + Textura Procedural", "Concreto Rejuntado", "Passeio frontal com juntas a cada 1.5m.");

  // Bloco 734
  const shapeBloco = new THREE.Shape();
  const cx = 37, cz = 42, R = 22, H = 6.6;
  shapeBloco.moveTo(15, -53); shapeBloco.lineTo(15, -42);
  shapeBloco.absarc(cx, -cz, R, Math.PI, 0.5 * Math.PI, true);
  shapeBloco.lineTo(59, -20); shapeBloco.lineTo(59, -53); shapeBloco.closePath();
  const extrudeGeo = new THREE.ExtrudeGeometry(shapeBloco, {{ depth: H, bevelEnabled: false, curveSegments: 32 }});
  extrudeGeo.rotateX(-Math.PI / 2);
  const meshBloco = new THREE.Mesh(extrudeGeo, new THREE.MeshStandardMaterial({{ color: MODO_NOITE ? 0x475569 : 0xf8fafc }}));
  meshBloco.position.y = 0.2; scene.add(meshBloco);
  registrarPick(meshBloco, "Bloco 734 (CT/Asfalto)", "Sólido Extrudado + Arco R=22m", "Alvenaria Branca + Vidro Fumê", "Planta 33m x 44m, altura 6.6m, platibanda nº 734.");

  // Escada
  const escada = new THREE.Mesh(new THREE.BoxGeometry(0.55, 6.6, 1.1), new THREE.MeshStandardMaterial({{ color: 0xf59e0b }}));
  escada.position.set(59.35, 3.4, 24.0); scene.add(escada);
  registrarPick(escada, "Escada Acesso Telhado", "Cubo Transformado", "Aço Pintado Amarelo", "Acesso vertical à laje de cobertura (H=6.6m).");

  // Fonte
  const fonte = new THREE.Mesh(new THREE.BoxGeometry(15.0, 0.34, 6.0), new THREE.MeshStandardMaterial({{ color: 0x06b6d4, roughness: 0.15 }}));
  fonte.position.set(27.0, 0.37, 19.8); scene.add(fonte);
  registrarPick(fonte, "Espelho d'Água (Fonte Cáustica)", "Cubo + Spots Submersos", "Água (n=1.333, R=0.42)", "Gera cáusticas por refração na fachada curva.");

  // Escultura Cinética
  const escBase = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.28, 3.4), new THREE.MeshStandardMaterial({{ color: 0x64748b }}));
  escBase.position.set(15.0, 0.34, 22.0); escBase.rotation.y = THREE.MathUtils.degToRad(-24); scene.add(escBase);
  registrarPick(escBase, "Base Escultura Cinética", "Cubo Rotacionado (-24° Y)", "Concreto Aparente", "Sustenta 5 quadros metálicos rotacionados a 45°.");

  for (let k = 0; k < 5; k++) {{
    const fc = k - 2; const L = 2.35 * (1 - 0.055 * fc);
    const losango = new THREE.Mesh(new THREE.BoxGeometry(L, L, 0.16), new THREE.MeshStandardMaterial({{ color: 0xf8fafc, wireframe: false }}));
    losango.position.set(15.0 + fc * 0.22, 0.48 + L / Math.SQRT2, 22.0 + fc * 0.58);
    losango.rotation.set(0, THREE.MathUtils.degToRad(-24), Math.PI / 4);
    scene.add(losango);
    registrarPick(losango, `Escultura Cinética - Losango ${{k+1}}`, "Perfil Rotacionado 45° + Translação", "Aço Pintura Branca (R=0.14)", `Quadro paramétrico #${{k+1}} (L=${{L.toFixed(2)}}m).`);
  }}

  // Árvores e Postes
  [[10.5, 28.5, 1.15], [11.5, 38.0, 1.25], [37.5, 17.5, 0.9]].forEach(([ax, az, esc], idx) => {{
    const copa = new THREE.Mesh(new THREE.DodecahedronGeometry(1.9 * esc, 1), new THREE.MeshStandardMaterial({{ color: 0x16a34a }}));
    copa.position.set(ax, 3.8 * esc, az); scene.add(copa);
    registrarPick(copa, `Árvore do Entorno #${{idx+1}}`, "Esferas + Rotação por Quatérnio", "Folhagem Verde", `Escala uniforme ${{esc}}x na posição (${{ax}}, ${{az}}).`);
  }});

  [[9.5, 21.0], [18.5, 16.2], [41.0, 16.5], [54.5, 17.5]].forEach(([px, pz], idx) => {{
    const poste = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.14, 6.2, 10), new THREE.MeshStandardMaterial({{ color: 0x334155 }}));
    poste.position.set(px, 3.3, pz); scene.add(poste);
    registrarPick(poste, `Poste Iluminação #${{idx+1}}`, "Cilindro + Fonte Pontual", "Alumínio Escuro + LED", "Iluminação pública com atenuação quadrática.");
  }});

  // Abrigo Metálico
  const abrigo = new THREE.Mesh(new THREE.BoxGeometry(4.5, 0.25, 3.2), new THREE.MeshStandardMaterial({{ color: 0x94a3b8 }}));
  abrigo.position.set(46.7, 2.5, 18.8); abrigo.rotation.z = -0.15; scene.add(abrigo);
  registrarPick(abrigo, "Abrigo Metálico (Malha)", "Malha Poligonal + Cisalhamento XY", "Aço Galvanizado (R=0.25)", "Cobertura inclinada por cisalhamento Sh_xy=0.15.");

  // Órbita travada acima do solo (Azimute 0..360, Elevação 2..88)
  let az = 2.35, el = 0.42, dist = 46.0;
  const target = new THREE.Vector3(32.0, 2.5, 28.0);
  let dragging = false, prev = {{ x: 0, y: 0 }}, downPos = {{ x: 0, y: 0 }};

  renderer.domElement.addEventListener('mousedown', (e) => {{
    dragging = true;
    prev = {{ x: e.clientX, y: e.clientY }};
    downPos = {{ x: e.clientX, y: e.clientY }};
  }});
  window.addEventListener('mouseup', (e) => {{
    dragging = false;
    if (Math.hypot(e.clientX - downPos.x, e.clientY - downPos.y) < 5) {{
      fazerPicking(e);
    }}
  }});
  window.addEventListener('mousemove', (e) => {{
    if (!dragging) return;
    az -= (e.clientX - prev.x) * 0.007;
    el = Math.max(0.04, Math.min(1.50, el + (e.clientY - prev.y) * 0.006));
    prev = {{ x: e.clientX, y: e.clientY }};
  }});
  renderer.domElement.addEventListener('wheel', (e) => {{
    dist = Math.max(10.0, Math.min(85.0, dist + e.deltaY * 0.03));
  }});

  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  let lastHighlight = null;

  function fazerPicking(e) {{
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(pickables, false);
    if (hits.length > 0) {{
      const h = hits[0];
      const info = h.object.userData;
      if (lastHighlight && lastHighlight.material.emissive) {{
        lastHighlight.material.emissive.setHex(0x000000);
      }}
      if (h.object.material.emissive) {{
        h.object.material.emissive.setHex(0x0284c7);
        lastHighlight = h.object;
      }}
      document.getElementById('pk-cat').textContent = info.cat;
      document.getElementById('pk-nome').textContent = info.nome;
      document.getElementById('pk-mat').textContent = info.mat;
      document.getElementById('pk-coord').textContent = `X=${{h.point.x.toFixed(2)}}m, Y=${{h.point.y.toFixed(2)}}m, Z=${{h.point.z.toFixed(2)}}m`;
      document.getElementById('pk-desc').textContent = info.desc;
      document.getElementById('pick-popup').style.display = 'block';
    }}
  }}

  function animate() {{
    requestAnimationFrame(animate);
    camera.position.set(
      target.x + dist * Math.cos(el) * Math.cos(az),
      Math.max(0.8, target.y + dist * Math.sin(el)),
      target.z + dist * Math.cos(el) * Math.sin(az)
    );
    camera.lookAt(target);
    renderer.render(scene, camera);
  }}
  animate();
</script>
</body>
</html>
"""
  components.html(html_code, height=600)


def renderizar_walkthrough_webgl(modo_noite=False):
  noite_js = "true" if modo_noite else "false"
  html_code = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8" />
<style>
  body {{ margin: 0; overflow: hidden; background: #0B1120; font-family: system-ui, -apple-system, sans-serif; user-select: none; }}
  #hud {{
    position: absolute; top: 10px; left: 10px; z-index: 10;
    background: rgba(15, 23, 42, 0.88); color: #F8FAFC;
    padding: 10px 14px; border-radius: 10px; border: 1px solid rgba(148, 163, 184, 0.35);
    max-width: 375px; font-size: 12px; line-height: 1.42;
  }}
  #hud b {{ color: #38BDF8; }}
  .btn-cam {{
    margin-top: 6px; margin-right: 4px; padding: 5px 8px;
    background: #1E293B; color: #F8FAFC; border: 1px solid #475569;
    border-radius: 6px; cursor: pointer; font-size: 11px; font-weight: 700;
  }}
  .btn-cam:hover {{ background: #0284C7; border-color: #38BDF8; }}

  #stations-panel {{
    position: absolute; top: 10px; right: 10px; z-index: 10;
    background: rgba(15, 23, 42, 0.88); color: #F8FAFC;
    padding: 10px 14px; border-radius: 10px; border: 1px solid rgba(148, 163, 184, 0.35);
    width: 285px; font-size: 11.5px; line-height: 1.45;
  }}
  .st-title {{ font-weight: 800; color: #FACC15; font-size: 12.5px; margin-bottom: 4px; }}
  .st-sub {{ font-size: 10.5px; color: #FCA5A5; margin-bottom: 6px; }}
  .st-item {{ display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.08); padding: 2px 0; }}
  .st-done {{ color: #4ADE80; font-weight: 700; }}
  .st-todo {{ color: #94A3B8; }}

  #prompt-box {{
    display: none; position: absolute; bottom: 50px; left: 50%; transform: translateX(-50%); z-index: 15;
    background: rgba(2, 132, 199, 0.94); color: #FFFFFF; padding: 8px 18px;
    border-radius: 999px; font-weight: 700; font-size: 13px; border: 1px solid #7DD3FC;
  }}

  #toxic-bar-wrap {{
    display: none; position: absolute; bottom: 92px; left: 50%; transform: translateX(-50%); z-index: 15;
    background: rgba(15,23,42,0.92); padding: 8px 14px; border-radius: 8px; border: 1px solid #22D3EE; width: 240px; text-align: center;
  }}
  #toxic-bar {{ height: 10px; background: #06B6D4; width: 0%; border-radius: 4px; }}

  #coords {{
    position: absolute; bottom: 10px; right: 10px; z-index: 10;
    background: rgba(15, 23, 42, 0.85); color: #E2E8F0;
    padding: 5px 11px; border-radius: 8px; font-family: monospace; font-size: 11.5px;
  }}

  /* JUMPSCARE OVERLAY (MÁSCARA PÂNICO) */
  #jumpscare-overlay {{
    display: none; position: absolute; inset: 0; z-index: 40;
    background: #000000; flex-direction: column; align-items: center; justify-content: center;
    animation: shake 0.08s infinite;
  }}
  @keyframes shake {{
    0% {{ transform: translate(4px, 3px) scale(1.03); }}
    50% {{ transform: translate(-4px, -3px) scale(1.06); }}
    100% {{ transform: translate(3px, -4px) scale(1.03); }}
  }}

  #wasted-overlay {{
    display: none; position: absolute; inset: 0; z-index: 30;
    background: radial-gradient(circle, rgba(127,29,29,0.55) 0%, rgba(15,23,42,0.92) 100%);
    flex-direction: column; align-items: center; justify-content: center; text-align: center;
  }}
  #wasted-title {{ font-size: 48px; font-weight: 900; color: #EF4444; text-shadow: 0 4px 20px rgba(0,0,0,0.9); }}
  #wasted-sub {{ font-size: 18px; color: #FCA5A5; margin-top: 8px; font-weight: 600; max-width: 520px; }}
  #wasted-warn {{ font-size: 14px; color: #FDE047; margin-top: 6px; font-weight: 700; }}
  #btn-respawn {{
    margin-top: 20px; padding: 11px 24px; font-size: 14px; font-weight: 800;
    background: #EF4444; color: white; border: none; border-radius: 8px; cursor: pointer;
  }}

  #victory-banner {{
    display: none; position: absolute; top: 90px; left: 50%; transform: translateX(-50%); z-index: 25;
    background: linear-gradient(135deg, #16A34A, #0284C7); color: white;
    padding: 16px 26px; border-radius: 12px; font-weight: 800; font-size: 16px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.65); text-align: center; border: 2px solid #86EFAC;
  }}
  #flash-foto {{
    display: none; position: absolute; inset: 0; z-index: 28; background: white; opacity: 0.85; pointer-events: none;
  }}
</style>
</head>
<body>
<div id="hud">
  <div style="font-size:14px; font-weight:800; margin-bottom:3px;">Walkthrough & Parkour — Bloco 734</div>
  <div><b>WASD</b>: Andar | <b>Shift</b>: Correr | <b>Espaço</b>: Pular | <b>E</b>: Escalar | <b>F</b>: Foto</div>
  <div>
    <button class="btn-cam" onclick="setCameraMode('third')">3ª Pessoa</button>
    <button class="btn-cam" onclick="setCameraMode('iso')">Isométrica</button>
    <button class="btn-cam" onclick="resetTotalMissoes()">Resetar Missões</button>
  </div>
</div>

<div id="stations-panel">
  <div class="st-title">ESTAÇÕES PARKOUR (0/10)</div>
  <div class="st-sub">Regra: Se morrer, TODAS as missões zeram!</div>
  <div class="st-item"><span>1. Foto da Escultura (F)</span><span id="st-foto_esc" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>2. Foto da Fonte (F)</span><span id="st-foto_fonte" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>3. Foto Carro Passando (F)</span><span id="st-foto_carro" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>4. Foto do Bloco 734 (F)</span><span id="st-foto_bloco" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>5. Foto da Árvore (F)</span><span id="st-foto_arvore" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>6. Fonte Rápida (&lt;3.5s)</span><span id="st-fonte" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>7. Escalar Poste (E)</span><span id="st-poste" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>8. Escalar Árvore 1 (E)</span><span id="st-arvore" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>9. Atravessar Rua & Voltar</span><span id="st-rua" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>10. Telhado pela Escada (E)</span><span id="st-telhado" class="st-todo">PENDENTE</span></div>
</div>

<div id="prompt-box">Ação disponível</div>
<div id="toxic-bar-wrap">
  <div style="color:#67E8F9; font-size:11px; font-weight:700; margin-bottom:4px;">ÁGUA CONTAMINADA! SAIA JÁ!</div>
  <div id="toxic-bar"></div>
</div>

<div id="victory-banner">
  <div>PARABÉNS! VOCÊ COMPLETOU AS 10 MISSÕES SEM MORRER!</div>
  <button class="btn-cam" style="margin-top:10px; background:#0F172A;" onclick="resetTotalMissoes()">JOGAR NOVAMENTE (ZERAR CHECKS)</button>
</div>
<div id="flash-foto"></div>

<!-- Tela de Jumpscare Estilo Pânico -->
<div id="jumpscare-overlay">
  <svg width="310" height="410" viewBox="0 0 200 260">
    <path d="M100,10 C45,10 25,65 32,125 C38,175 68,245 100,252 C132,245 162,175 168,125 C175,65 155,10 100,10 Z" fill="#F8FAFC" stroke="#000" stroke-width="5"/>
    <path d="M52,92 C42,115 62,135 82,122 C92,112 75,85 52,92 Z" fill="#090D16"/>
    <path d="M148,92 C158,115 138,135 118,122 C108,112 125,85 148,92 Z" fill="#090D16"/>
    <path d="M92,142 L100,128 L108,142 Z" fill="#090D16"/>
    <path d="M76,168 C74,205 84,235 100,235 C116,235 126,205 124,168 C112,182 88,182 76,168 Z" fill="#090D16"/>
  </svg>
  <div style="color:#EF4444; font-size:34px; font-weight:900; letter-spacing:3px;">BUUUU!!</div>
</div>

<div id="wasted-overlay">
  <div id="wasted-title">DUMB WAYS TO DIE</div>
  <div id="wasted-sub">Você encontrou um fim trágico!</div>
  <div id="wasted-warn">Todas as missões concluídas foram ZERADAS! Tente completar sem morrer.</div>
  <button id="btn-respawn" onclick="resetPlayer()">RENASCER E RECOMEÇAR (Tecla R)</button>
</div>

<div id="coords">Mortes: 0 | X=14.0 | Y=0.2 | Z=14.0</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
  const MODO_NOITE = {noite_js};
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(MODO_NOITE ? 0x070d19 : 0x93c5fd);
  scene.fog = new THREE.FogExp2(MODO_NOITE ? 0x070d19 : 0x93c5fd, 0.008);

  const camera = new THREE.PerspectiveCamera(58, window.innerWidth / window.innerHeight, 0.1, 300);
  const renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = true;
  document.body.appendChild(renderer.domElement);

  scene.add(new THREE.HemisphereLight(MODO_NOITE ? 0x1e293b : 0xffffff, MODO_NOITE ? 0x0f172a : 0x64748b, MODO_NOITE ? 0.55 : 0.85));
  const dirLight = new THREE.DirectionalLight(MODO_NOITE ? 0x3b82f6 : 0xfffbeb, MODO_NOITE ? 0.25 : 1.15);
  dirLight.position.set(12, 32, 10); dirLight.castShadow = true; scene.add(dirLight);

  // Via e Calçada
  const via = new THREE.Mesh(new THREE.BoxGeometry(66, 0.1, 14), new THREE.MeshStandardMaterial({{ color: 0x1e293b }}));
  via.position.set(35, 0.05, 9); scene.add(via);
  for (let fx = 4; fx < 66; fx += 4.5) {{
    const faixa = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.02, 0.18), new THREE.MeshBasicMaterial({{ color: 0xfacc15 }}));
    faixa.position.set(fx, 0.11, 11.7); scene.add(faixa);
  }}
  const calcada = new THREE.Mesh(new THREE.BoxGeometry(66, 0.2, 44), new THREE.MeshStandardMaterial({{ color: MODO_NOITE ? 0x334155 : 0xcbd5e1 }}));
  calcada.position.set(35, 0.1, 38); scene.add(calcada);

  // Plataforma objetivo do outro lado da rua
  const platRua = new THREE.Mesh(new THREE.CylinderGeometry(1.8, 1.8, 0.12, 24), new THREE.MeshStandardMaterial({{ color: 0x22c55e, emissive: 0x15803d, emissiveIntensity: 0.45 }}));
  platRua.position.set(27.0, 0.12, 3.2); scene.add(platRua);

  // 5 Zonas Fotográficas (Anéis Azuis no chão)
  const zonasFoto = [
    {{ id: 'foto_esc', x: 11.5, z: 18.0, label: 'Escultura Cinética' }},
    {{ id: 'foto_fonte', x: 27.0, z: 15.2, label: 'Fonte / Espelho d’Água' }},
    {{ id: 'foto_carro', x: 33.0, z: 15.0, label: 'Carro Passando na Via' }},
    {{ id: 'foto_bloco', x: 20.0, z: 24.5, label: 'Bloco 734' }},
    {{ id: 'foto_arvore', x: 13.0, z: 28.5, label: 'Árvore do Bosque' }}
  ];
  zonasFoto.forEach((zf) => {{
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.95, 1.3, 24), new THREE.MeshBasicMaterial({{ color: 0x38bdf8, side: THREE.DoubleSide }}));
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(zf.x, 0.22, zf.z);
    scene.add(ring);
  }});

  // Bloco 734
  const shapeBloco = new THREE.Shape();
  const cx = 37, cz = 42, R = 22, H = 6.6;
  shapeBloco.moveTo(15, -53); shapeBloco.lineTo(15, -42);
  shapeBloco.absarc(cx, -cz, R, Math.PI, 0.5 * Math.PI, true);
  shapeBloco.lineTo(59, -20); shapeBloco.lineTo(59, -53); shapeBloco.closePath();
  const extrudeGeo = new THREE.ExtrudeGeometry(shapeBloco, {{ depth: H, bevelEnabled: false, curveSegments: 32 }});
  extrudeGeo.rotateX(-Math.PI / 2);
  const meshBloco = new THREE.Mesh(extrudeGeo, new THREE.MeshStandardMaterial({{ color: MODO_NOITE ? 0x475569 : 0xf8fafc }}));
  meshBloco.position.y = 0.2; scene.add(meshBloco);

  // Escada Lateral
  const escadaGroup = new THREE.Group();
  escadaGroup.position.set(59.35, 0.2, 24.0);
  const t1 = new THREE.Mesh(new THREE.BoxGeometry(0.12, 6.8, 0.12), new THREE.MeshStandardMaterial({{ color: 0xf59e0b }}));
  t1.position.set(0, 3.4, -0.45); escadaGroup.add(t1);
  const t2 = t1.clone(); t2.position.z = 0.45; escadaGroup.add(t2);
  for (let deg = 0.4; deg <= 6.6; deg += 0.45) {{
    const d = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.08, 0.9), new THREE.MeshStandardMaterial({{ color: 0xfbbf24 }}));
    d.position.set(0, deg, 0); escadaGroup.add(d);
  }}
  scene.add(escadaGroup);

  // Janelas + Janela Assombrada do Final do Bloco (X=59.05, Z=49.0)
  for (let i = 0; i < 9; i++) {{
    const ang = Math.PI + ((i + 0.5) / 9) * (Math.PI * 0.44) + 0.08;
    const vx = cx + (R + 0.05) * Math.cos(ang);
    const vz = cz + (R + 0.05) * Math.sin(ang);
    const janela = new THREE.Mesh(new THREE.BoxGeometry(1.7, 1.9, 0.15), new THREE.MeshStandardMaterial({{ color: 0x0f172a, metalness: 0.85 }}));
    janela.position.set(vx, 2.3, vz); janela.lookAt(cx, 2.3, cz); scene.add(janela);
  }}
  const janelaPanico = new THREE.Mesh(new THREE.BoxGeometry(0.2, 2.0, 1.8), new THREE.MeshStandardMaterial({{ color: 0x090d16 }}));
  janelaPanico.position.set(59.05, 2.2, 49.0); scene.add(janelaPanico);
  const figuraJanela = new THREE.Mesh(new THREE.SphereGeometry(0.42, 12, 12), new THREE.MeshBasicMaterial({{ color: 0xf8fafc }}));
  figuraJanela.position.set(58.6, 2.2, 49.0); scene.add(figuraJanela);

  // Fonte Contaminada
  const bordaFonte = new THREE.Mesh(new THREE.BoxGeometry(15.4, 0.34, 6.4), new THREE.MeshStandardMaterial({{ color: 0x1e293b }}));
  bordaFonte.position.set(27, 0.37, 19.8); scene.add(bordaFonte);
  const laminaAgua = new THREE.Mesh(new THREE.BoxGeometry(14.8, 0.1, 5.8), new THREE.MeshStandardMaterial({{ color: 0x14b8a6, emissive: 0x0d9488, emissiveIntensity: 0.35 }}));
  laminaAgua.position.set(27, 0.51, 19.8); scene.add(laminaAgua);

  // Escultura Cinética
  const grupoEsc = new THREE.Group();
  grupoEsc.position.set(15.0, 0.2, 22.0); grupoEsc.rotation.y = THREE.MathUtils.degToRad(-24);
  const baseEsc = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.28, 3.4), new THREE.MeshStandardMaterial({{ color: 0x64748b }}));
  baseEsc.position.y = 0.14; grupoEsc.add(baseEsc);
  for (let k = 0; k < 5; k++) {{
    const fc = k - 2; const L = 2.35 * (1 - 0.055 * fc);
    const fg = new THREE.Group();
    [[0, L/2, L, 0.14], [0, -L/2, L, 0.14], [L/2, 0, 0.14, L], [-L/2, 0, 0.14, L]].forEach(([bx, by, bw, bh]) => {{
      const bar = new THREE.Mesh(new THREE.BoxGeometry(bw, bh, 0.16), new THREE.MeshStandardMaterial({{ color: 0xf8fafc }}));
      bar.position.set(bx, by, 0); fg.add(bar);
    }});
    fg.rotation.z = Math.PI / 4;
    fg.position.set(fc * 0.22, 0.28 + (L / Math.SQRT2) + 0.05, (k - 2) * 0.58);
    grupoEsc.add(fg);
  }}
  scene.add(grupoEsc);

  // Árvores + Ave Predadora na Árvore 2 (X=11.5, Z=38.0)
  const arvores = [[10.5, 28.5, 1.15], [11.5, 38.0, 1.25], [37.5, 17.5, 0.9]];
  arvores.forEach(([ax, az, esc]) => {{
    const tronco = new THREE.Mesh(new THREE.CylinderGeometry(0.22 * esc, 0.3 * esc, 3.6 * esc, 10), new THREE.MeshStandardMaterial({{ color: 0x5c3a21 }}));
    tronco.position.set(ax, 0.2 + 1.8 * esc, az); scene.add(tronco);
    const copa = new THREE.Mesh(new THREE.DodecahedronGeometry(1.95 * esc, 1), new THREE.MeshStandardMaterial({{ color: MODO_NOITE ? 0x064e3b : 0x16a34a }}));
    copa.position.set(ax, 4.1 * esc, az); scene.add(copa);
  }});

  // Ave Predadora
  const aveGroup = new THREE.Group();
  const aveCorpo = new THREE.Mesh(new THREE.ConeGeometry(0.32, 1.1, 8), new THREE.MeshStandardMaterial({{ color: 0x1e1b4b }}));
  aveCorpo.rotation.x = Math.PI / 2; aveGroup.add(aveCorpo);
  const asaEsq = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.06, 0.45), new THREE.MeshStandardMaterial({{ color: 0x312e81 }}));
  asaEsq.position.x = -0.65; aveGroup.add(asaEsq);
  const asaDir = asaEsq.clone(); asaDir.position.x = 0.65; aveGroup.add(asaDir);
  aveGroup.position.set(11.5, 5.2, 38.0);
  scene.add(aveGroup);
  let aveAtacando = false;

  // NPC Parado Assustador (X=43.0, Z=19.5)
  const npcGroup = new THREE.Group();
  const npcCorpo = new THREE.Mesh(new THREE.BoxGeometry(0.75, 1.2, 0.45), new THREE.MeshStandardMaterial({{ color: 0x111827 }}));
  npcCorpo.position.y = 0.8; npcGroup.add(npcCorpo);
  const npcCabeca = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.52, 0.46), new THREE.MeshStandardMaterial({{ color: 0xf8fafc }}));
  npcCabeca.position.y = 1.68; npcGroup.add(npcCabeca);
  npcGroup.position.set(43.0, 0.2, 19.5);
  scene.add(npcGroup);

  // Postes com fio descascado
  const postes = [[9.5, 21.0, 0.6], [18.5, 16.2, 0.4], [41.0, 16.5, 0.0], [54.5, 17.5, -0.5]];
  const faiscas = [];
  postes.forEach(([px, pz, ox]) => {{
    const mastro = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.1, 6.2, 8), new THREE.MeshStandardMaterial({{ color: 0x334155 }}));
    mastro.position.set(px, 3.3, pz); scene.add(mastro);
    const lum = new THREE.Mesh(new THREE.SphereGeometry(0.32, 12, 12), new THREE.MeshBasicMaterial({{ color: 0xfef08a }}));
    lum.position.set(px + ox, 6.3, pz + 0.4); scene.add(lum);
    const fio = new THREE.Mesh(new THREE.SphereGeometry(0.18, 8, 8), new THREE.MeshBasicMaterial({{ color: 0x38bdf8 }}));
    fio.position.set(px, 5.9, pz); scene.add(fio);
    faiscas.push(fio);
  }});

  // Carros em movimento
  function criarCarro(x, z, rotY, cor, velX) {{
    const g = new THREE.Group();
    const chassi = new THREE.Mesh(new THREE.BoxGeometry(2.0, 0.65, 4.3), new THREE.MeshStandardMaterial({{ color: cor }}));
    chassi.position.y = 0.55; g.add(chassi);
    const cabine = new THREE.Mesh(new THREE.BoxGeometry(1.75, 0.55, 2.2), new THREE.MeshStandardMaterial({{ color: 0x0f172a }}));
    cabine.position.set(0, 1.1, -0.15); g.add(cabine);
    g.position.set(x, 0.1, z); g.rotation.y = rotY; scene.add(g);
    return {{ mesh: g, velX: velX }};
  }}
  const carros = [
    criarCarro(12.0, 7.2, Math.PI / 2, 0xcbd5e1, 13.0),
    criarCarro(56.0, 10.6, -Math.PI / 2, 0xdc2626, -15.0)
  ];

  // Boneco Articulado
  const player = new THREE.Group();
  const matPele = new THREE.MeshStandardMaterial({{ color: 0xfcd34d }});
  const torso = new THREE.Mesh(new THREE.BoxGeometry(0.75, 0.85, 0.4), new THREE.MeshStandardMaterial({{ color: 0x2563eb }}));
  torso.position.set(0, 1.15, 0); player.add(torso);
  const cabeca = new THREE.Mesh(new THREE.BoxGeometry(0.48, 0.48, 0.48), matPele);
  cabeca.position.set(0, 1.85, 0); player.add(cabeca);
  const pernaEsq = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.75, 0.34), new THREE.MeshStandardMaterial({{ color: 0x1e293b }}));
  pernaEsq.position.set(-0.2, 0.38, 0); player.add(pernaEsq);
  const pernaDir = pernaEsq.clone(); pernaDir.position.x = 0.2; player.add(pernaDir);
  const bracoEsq = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.78, 0.28), matPele);
  bracoEsq.position.set(-0.54, 1.15, 0); player.add(bracoEsq);
  const bracoDir = bracoEsq.clone(); bracoDir.position.x = 0.54; player.add(bracoDir);

  const partesBoneco = [
    {{ mesh: torso, posOrig: [0, 1.15, 0], vel: new THREE.Vector3(), rotVel: new THREE.Vector3() }},
    {{ mesh: cabeca, posOrig: [0, 1.85, 0], vel: new THREE.Vector3(), rotVel: new THREE.Vector3() }},
    {{ mesh: pernaEsq, posOrig: [-0.2, 0.38, 0], vel: new THREE.Vector3(), rotVel: new THREE.Vector3() }},
    {{ mesh: pernaDir, posOrig: [0.2, 0.38, 0], vel: new THREE.Vector3(), rotVel: new THREE.Vector3() }},
    {{ mesh: bracoEsq, posOrig: [-0.54, 1.15, 0], vel: new THREE.Vector3(), rotVel: new THREE.Vector3() }},
    {{ mesh: bracoDir, posOrig: [0.54, 1.15, 0], vel: new THREE.Vector3(), rotVel: new THREE.Vector3() }},
  ];
  player.position.set(14.0, 0.2, 14.0);
  scene.add(player);

  let isDead = false, deathCount = 0, velY = 0, walkCycle = 0;
  let tempoNaFonte = 0, subiuArvore = false, subiuPoste = false, atravessouRua = false, estavaNoTelhado = false;

  const chavesMissoes = ['foto_esc', 'foto_fonte', 'foto_carro', 'foto_bloco', 'foto_arvore', 'fonte', 'poste', 'arvore', 'rua', 'telhado'];
  const checks = {{}};
  chavesMissoes.forEach(k => checks[k] = false);

  function atualizarPainelMissoes() {{
    const total = Object.values(checks).filter(Boolean).length;
    document.querySelector('.st-title').textContent = `ESTAÇÕES PARKOUR (${{total}}/10)`;
    chavesMissoes.forEach((k) => {{
      const el = document.getElementById('st-' + k);
      if (checks[k]) {{
        el.textContent = "CHECK ✓";
        el.className = "st-done";
      }} else {{
        el.textContent = "PENDENTE";
        el.className = "st-todo";
      }}
    }});
    document.getElementById('victory-banner').style.display = (total === 10 && !isDead) ? 'block' : 'none';
  }}

  function zerarTodasMissoes() {{
    chavesMissoes.forEach(k => checks[k] = false);
    atravessouRua = false;
    atualizarPainelMissoes();
  }}

  function marcarCheck(chave) {{
    if (checks[chave] || isDead) return;
    checks[chave] = true;
    atualizarPainelMissoes();
  }}

  function triggerDeath(motivo, impactoVel, comJumpscare = false) {{
    if (isDead) return;
    isDead = true;
    deathCount++;
    // Se morrer, zera todas as missões concluídas!
    zerarTodasMissoes();

    document.getElementById('wasted-sub').textContent = motivo;
    document.getElementById('toxic-bar-wrap').style.display = 'none';

    if (comJumpscare) {{
      const jsEl = document.getElementById('jumpscare-overlay');
      jsEl.style.display = 'flex';
      setTimeout(() => {{
        jsEl.style.display = 'none';
        document.getElementById('wasted-overlay').style.display = 'flex';
      }}, 950);
    }} else {{
      document.getElementById('wasted-overlay').style.display = 'flex';
    }}

    partesBoneco.forEach((p) => {{
      p.vel.set((Math.random() - 0.5) * 9 + impactoVel.x, 4.5 + Math.random() * 6, (Math.random() - 0.5) * 9 + impactoVel.z);
      p.rotVel.set((Math.random() - 0.5) * 12, (Math.random() - 0.5) * 12, (Math.random() - 0.5) * 12);
    }});
  }}

  window.resetPlayer = () => {{
    isDead = false;
    tempoNaFonte = 0; subiuArvore = false; subiuPoste = false; estavaNoTelhado = false;
    aveAtacando = false; aveGroup.position.set(11.5, 5.2, 38.0);
    figuraJanela.position.set(58.6, 2.2, 49.0);
    document.getElementById('wasted-overlay').style.display = 'none';
    document.getElementById('jumpscare-overlay').style.display = 'none';
    document.getElementById('toxic-bar-wrap').style.display = 'none';
    player.position.set(14.0, 0.2, 14.0);
    velY = 0;
    partesBoneco.forEach((p) => {{
      p.mesh.position.set(p.posOrig[0], p.posOrig[1], p.posOrig[2]);
      p.mesh.rotation.set(0, 0, 0);
    }});
  }};

  window.resetTotalMissoes = () => {{
    zerarTodasMissoes();
    window.resetPlayer();
  }};

  const keys = {{}};
  window.addEventListener('keydown', (e) => {{
    keys[e.code] = true;
    if (e.code === 'KeyR') window.resetPlayer();
    if (e.code === 'KeyF' && !isDead) {{
      zonasFoto.forEach((zf) => {{
        if (Math.hypot(player.position.x - zf.x, player.position.z - zf.z) < 2.2) {{
          if (zf.id === 'foto_carro') {{
            // Exige um carro passando próximo no momento do clique!
            const temCarroPerto = carros.some(c => Math.abs(c.mesh.position.x - 33.0) < 11.0);
            if (!temCarroPerto) return;
          }}
          const fl = document.getElementById('flash-foto');
          fl.style.display = 'block';
          setTimeout(() => {{ fl.style.display = 'none'; }}, 140);
          marcarCheck(zf.id);
        }}
      }});
    }}
  }});
  window.addEventListener('keyup', (e) => {{ keys[e.code] = false; }});

  let camYaw = 0.65, camPitch = 0.32, camDist = 7.5, camMode = 'third', isDragging = false, prevMouse = {{ x: 0, y: 0 }};
  renderer.domElement.addEventListener('mousedown', (e) => {{ isDragging = true; prevMouse = {{ x: e.clientX, y: e.clientY }}; }});
  window.addEventListener('mouseup', () => {{ isDragging = false; }});
  window.addEventListener('mousemove', (e) => {{
    if (!isDragging || camMode === 'iso') return;
    camYaw -= (e.clientX - prevMouse.x) * 0.007;
    camPitch = Math.max(0.08, Math.min(1.35, camPitch + (e.clientY - prevMouse.y) * 0.006));
    prevMouse = {{ x: e.clientX, y: e.clientY }};
  }});
  renderer.domElement.addEventListener('wheel', (e) => {{ camDist = Math.max(3.0, Math.min(28.0, camDist + e.deltaY * 0.008)); }});
  window.setCameraMode = (mode) => {{ camMode = mode; }};

  function dentroPlantaBloco(nx, nz) {{
    const inBox = (nx > 14.6 && nx < 59.4 && nz > 19.6 && nz < 53.4);
    const distArc = Math.hypot(nx - 37.0, nz - 42.0);
    return inBox && (!(nx < 37.0 && nz < 42.0) || distArc < 22.3);
  }}

  const clock = new THREE.Clock();
  const promptBox = document.getElementById('prompt-box');
  const toxicWrap = document.getElementById('toxic-bar-wrap');
  const toxicBar = document.getElementById('toxic-bar');
  const coordsEl = document.getElementById('coords');

  function animate() {{
    requestAnimationFrame(animate);
    const dt = Math.min(clock.getDelta(), 0.05);

    faiscas.forEach((f, idx) => {{ f.visible = Math.sin(clock.elapsedTime * 22 + idx) > 0.2; }});

    carros.forEach((c) => {{
      c.mesh.position.x += c.velX * dt;
      if (c.velX > 0 && c.mesh.position.x > 66.0) c.mesh.position.x = 4.0;
      if (c.velX < 0 && c.mesh.position.x < 4.0) c.mesh.position.x = 66.0;
      if (!isDead && Math.abs(player.position.x - c.mesh.position.x) < 2.2 && Math.abs(player.position.z - c.mesh.position.z) < 1.15 && player.position.y < 1.5) {{
        triggerDeath("Atropelado na Via do CT! Olhe antes de atravessar.", new THREE.Vector3(c.velX * 0.7, 3.0, 0));
      }}
    }});

    if (isDead) {{
      partesBoneco.forEach((p) => {{
        p.vel.y -= 16.0 * dt;
        p.mesh.position.addScaledVector(p.vel, dt);
        p.mesh.rotation.x += p.rotVel.x * dt;
        p.mesh.rotation.y += p.rotVel.y * dt;
        p.mesh.rotation.z += p.rotVel.z * dt;
        if (p.mesh.position.y < 0.2) {{
          p.mesh.position.y = 0.2; p.vel.y *= -0.45; p.vel.x *= 0.85; p.vel.z *= 0.85;
        }}
      }});
    }} else {{
      let promptTexto = "";

      // Zonas Fotográficas
      zonasFoto.forEach((zf) => {{
        if (Math.hypot(player.position.x - zf.x, player.position.z - zf.z) < 2.2) {{
          if (zf.id === 'foto_carro') {{
            const temCarro = carros.some(c => Math.abs(c.mesh.position.x - 33.0) < 11.0);
            promptTexto = temCarro ? "Carro passando! Aperte [F] agora para fotografar!" : "Aguarde um carro passar e aperte [F]!";
          }} else {{
            promptTexto = `Aperte [F] para tirar foto: ${{zf.label}}`;
          }}
        }}
      }});

      // Perigo 1: Escultura Cinética
      if (Math.hypot(player.position.x - 15.0, player.position.z - 22.0) < 2.1 && player.position.y < 3.8) {{
        triggerDeath("Esmagado / Eletrocutado ao tocar a Escultura Cinética!", new THREE.Vector3(0, 4, -4));
      }}

      // Perigo 2: Ave Predadora na Árvore 2 (X=11.5, Z=38.0)
      const distArvoreAve = Math.hypot(player.position.x - 11.5, player.position.z - 38.0);
      if (distArvoreAve < 4.8) aveAtacando = true;
      if (aveAtacando) {{
        asaEsq.rotation.z = Math.sin(clock.elapsedTime * 28) * 0.6;
        asaDir.rotation.z = -asaEsq.rotation.z;
        const alvoAve = new THREE.Vector3(player.position.x, player.position.y + 1.4, player.position.z);
        aveGroup.lookAt(alvoAve);
        aveGroup.position.lerp(alvoAve, 6.5 * dt);
        if (aveGroup.position.distanceTo(alvoAve) < 0.9) {{
          triggerDeath("Atacado pela ave gigante da Árvore do Fundo!", new THREE.Vector3(3, 3, -3));
        }}
      }}

      // Perigo 3: Jumpscare na Janela Final do Bloco (X=59.5, Z=49.0)
      if (Math.hypot(player.position.x - 60.5, player.position.z - 49.0) < 2.6) {{
        figuraJanela.position.x = 60.2;
        triggerDeath("SUSTO FATAL! Uma figura mascarada saltou da janela final do Bloco 734!", new THREE.Vector3(6, 4, 0), true);
      }}

      // Perigo 4: NPC Parado Assustador (X=43.0, Z=19.5)
      if (Math.hypot(player.position.x - 43.0, player.position.z - 19.5) < 1.85) {{
        npcGroup.lookAt(player.position.x, 0.2, player.position.z);
        triggerDeath("SUSTO FATAL! Você abordou a figura misteriosa parada na praça!", new THREE.Vector3(-4, 4, -4), true);
      }}

      // Fonte Contaminada
      const naAgua = (player.position.x > 19.6 && player.position.x < 34.4 && player.position.z > 16.9 && player.position.z < 22.7 && player.position.y < 0.9);
      if (naAgua) {{
        tempoNaFonte += dt;
        toxicWrap.style.display = 'block';
        toxicBar.style.width = Math.min(100, (tempoNaFonte / 3.5) * 100) + '%';
        if (tempoNaFonte >= 3.5) {{
          triggerDeath("Intoxicado pela água parada contaminada da fonte!", new THREE.Vector3(0, 3, 0));
        }}
      }} else {{
        if (tempoNaFonte > 0.25 && tempoNaFonte < 3.5) marcarCheck('fonte');
        tempoNaFonte = 0;
        toxicWrap.style.display = 'none';
      }}

      // Travessia da Rua
      if (Math.hypot(player.position.x - 27.0, player.position.z - 3.2) < 2.0) {{
        atravessouRua = true;
        promptTexto = "Objetivo alcançado! Volte em segurança para a calçada!";
      }}
      if (atravessouRua && player.position.z >= 16.5) marcarCheck('rua');

      // Escaladas (Escada, Postes, Árvore 1)
      let escalando = false;
      if (Math.hypot(player.position.x - 59.35, player.position.z - 24.0) < 1.35) {{
        promptTexto = "Segure [E] para subir a escada até o telhado!";
        if (keys['KeyE']) {{
          escalando = true; velY = 0; player.position.y += 4.2 * dt;
          if (player.position.y >= 6.85) {{ player.position.y = 6.85; player.position.x = 58.2; }}
        }}
      }}

      postes.forEach(([px, pz]) => {{
        if (Math.hypot(player.position.x - px, player.position.z - pz) < 1.1) {{
          promptTexto = "Segure [E] para escalar o poste (Cuidado com o fio descascado!)";
          if (keys['KeyE']) {{
            escalando = true; velY = 0; player.position.y += 3.6 * dt;
            if (player.position.y > 2.8) subiuPoste = true;
            if (player.position.y >= 5.5) {{
              triggerDeath("Eletrocutado no fio descascado do poste!", new THREE.Vector3(0, 2, -4));
            }}
          }}
        }}
      }});
      if (subiuPoste && !escalando && player.position.y <= 0.35) {{ marcarCheck('poste'); subiuPoste = false; }}

      // Árvore 1 (Bosque Esq X=10.5, Z=28.5)
      if (Math.hypot(player.position.x - 10.5, player.position.z - 28.5) < 1.3) {{
        promptTexto = "Segure [E] para escalar a árvore (Não suba sem parar!)";
        if (keys['KeyE']) {{
          escalando = true; velY = 0; player.position.y += 3.4 * dt;
          if (player.position.y > 2.6) subiuArvore = true;
          if (player.position.y >= 5.4) {{
            triggerDeath("Subiu sem parar! O galho desconectou e você despencou lá de cima!", new THREE.Vector3(2, -6, -2));
          }}
        }}
      }}
      if (subiuArvore && !escalando && player.position.y <= 0.35) {{ marcarCheck('arvore'); subiuArvore = false; }}

      promptBox.textContent = promptTexto;
      promptBox.style.display = promptTexto ? 'block' : 'none';

      // Movimento Horizontal
      let moveForward = 0, moveRight = 0;
      if (keys['KeyW'] || keys['ArrowUp']) moveForward += 1;
      if (keys['KeyS'] || keys['ArrowDown']) moveForward -= 1;
      if (keys['KeyD'] || keys['ArrowRight']) moveRight += 1;
      if (keys['KeyA'] || keys['ArrowLeft']) moveRight -= 1;

      const speed = (keys['ShiftLeft'] || keys['ShiftRight']) ? 10.5 : 5.5;
      if (moveForward !== 0 || moveRight !== 0) {{
        const forwardVec = new THREE.Vector3(-Math.sin(camYaw), 0, -Math.cos(camYaw));
        const rightVec = new THREE.Vector3(Math.cos(camYaw), 0, -Math.sin(camYaw));
        const dir = new THREE.Vector3().addScaledVector(forwardVec, moveForward).addScaledVector(rightVec, moveRight).normalize();

        const nextX = THREE.MathUtils.clamp(player.position.x + dir.x * speed * dt, 3.0, 67.0);
        const nextZ = THREE.MathUtils.clamp(player.position.z + dir.z * speed * dt, 2.5, 58.0);

        if (!(player.position.y < 6.4 && dentroPlantaBloco(nextX, player.position.z))) player.position.x = nextX;
        if (!(player.position.y < 6.4 && dentroPlantaBloco(player.position.x, nextZ))) player.position.z = nextZ;

        player.rotation.y = Math.atan2(dir.x, dir.z);
        walkCycle += dt * speed * 2.2;
        const swing = Math.sin(walkCycle) * 0.55;
        pernaEsq.rotation.x = swing; pernaDir.rotation.x = -swing;
        bracoEsq.rotation.x = -swing * 0.8; bracoDir.rotation.x = swing * 0.8;
      }} else if (escalando) {{
        walkCycle += dt * 10.0;
        const cl = Math.sin(walkCycle) * 0.7;
        bracoEsq.rotation.x = Math.PI + cl; bracoDir.rotation.x = Math.PI - cl;
      }} else {{
        pernaEsq.rotation.x *= 0.8; pernaDir.rotation.x *= 0.8;
        bracoEsq.rotation.x *= 0.8; bracoDir.rotation.x *= 0.8;
      }}

      const noTelhado = dentroPlantaBloco(player.position.x, player.position.z) && (player.position.y >= 6.4);
      if (noTelhado) {{ estavaNoTelhado = true; marcarCheck('telhado'); }}

      const groundY = noTelhado ? 6.8 : (player.position.z >= 16.0 ? 0.2 : 0.1);
      if (!escalando) {{
        if (keys['Space'] && Math.abs(player.position.y - groundY) < 0.08) velY = 6.8;
        velY -= 18.0 * dt;
        player.position.y += velY * dt;
        if (player.position.y <= groundY) {{
          if (estavaNoTelhado && groundY < 1.0 && velY < -8.5) {{
            triggerDeath("Caiu lá de cima do telhado do Bloco 734!", new THREE.Vector3(0, 1, 0));
          }}
          player.position.y = groundY; velY = 0;
          if (groundY < 1.0) estavaNoTelhado = false;
        }}
      }}
    }}

    const targetLook = new THREE.Vector3(player.position.x, player.position.y + 1.4, player.position.z);
    if (camMode === 'iso') {{
      camera.position.set(player.position.x - 14, player.position.y + 14, player.position.z - 14);
      camera.lookAt(targetLook);
    }} else {{
      camera.position.set(
        targetLook.x + camDist * Math.sin(camYaw) * Math.cos(camPitch),
        targetLook.y + camDist * Math.sin(camPitch),
        targetLook.z + camDist * Math.cos(camYaw) * Math.cos(camPitch)
      );
      camera.lookAt(targetLook);
    }}

    coordsEl.textContent = `Mortes: ${{deathCount}} | X=${{player.position.x.toFixed(1)}} | Y=${{player.position.y.toFixed(1)}} | Z=${{player.position.z.toFixed(1)}}`;
    renderer.render(scene, camera);
  }}
  animate();
</script>
</body>
</html>
"""
  components.html(html_code, height=630)


# ==============================================================================
# BARRA LATERAL E CONTROLES
# ==============================================================================
st.sidebar.header("Controles Gerais")
modo_noite = st.sidebar.toggle(
    "Modo Noturno (Postes + Fonte Iluminada)", value=False
)

st.sidebar.divider()
st.sidebar.subheader("Câmera e Projeções (Inspeção 3D e Render)")
estilo_preview = st.sidebar.radio(
    "Estilo da Inspeção 3D (Plotly)",
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
    "Template de Câmera",
    options=list(preset_nomes.keys()),
    format_func=lambda k: preset_nomes[k],
)
renderer_temp = carregar_motor(420)
cfg_atual = renderer_temp.presets[preset_escolhido].copy()

with st.sidebar.expander("Órbita Travada (Azimute & Elevação 0°–90°)"):
  usar_orbita = st.checkbox(
      "Controlar Inspeção 3D por Azimute / Elevação", value=False
  )
  az_deg = st.slider("Azimute (0° a 360°)", 0, 360, 225)
  el_deg = st.slider("Elevação Vertical (2° a 88°)", 2, 88, 28)

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

st.sidebar.divider()
st.sidebar.subheader("Qualidade do Ray Casting")
resolucao = st.sidebar.select_slider(
    "Resolução Base", options=[240, 360, 480, 640], value=480
)
ssaa_rc = st.sidebar.toggle(
    "Supersampling 4x (Nitidez Máxima / Sem Pixelado)", value=True
)
incluir_objs_rc = st.sidebar.toggle(
    "Incluir Malhas .OBJ no Render (Mais pesado)", value=False
)
ativar_sombras_rc = st.sidebar.toggle("Sombras Projetadas", value=True)
ativar_reflexos_rc = st.sidebar.toggle("Reflexos (Vidro / Água / Aço)", value=True)
ativar_caustics_rc = st.sidebar.toggle("Cáusticas da Fonte", value=True)

renderer = carregar_motor(resolucao)

aba_walk, aba_3d, aba_rc = st.tabs([
    "1. Walkthrough & Parkour (Dumb Ways to Die)",
    "2. Inspeção 3D & Picking",
    "3. Render Final (Ray Casting + Picking)",
])

with aba_walk:
  st.caption(
      "Complete as 10 estações sem morrer! Se morrer, o progresso das missões zera. Cuidado com carros, água parada, fios descascados, ave na árvore, NPC parado e a janela final do Bloco 734."
  )
  renderizar_walkthrough_webgl(modo_noite=modo_noite)

with aba_3d:
  modo_insp = st.radio(
      "Motor da Inspeção 3D:",
      options=["webgl", "plotly"],
      format_func=lambda x: {
          "webgl": "Modo WebGL / Estilo Roblox (Sem Raycast + Popup Picking)",
          "plotly": "Modo Plotly (Malhas .OBJ + Templates de Câmera)",
      }[x],
      horizontal=True,
  )

  if modo_insp == "webgl":
    renderizar_inspecao_webgl_roblox(modo_noite=modo_noite)
  else:
    st.caption(
        "Clique sobre qualquer objeto ou vértice na cena 3D abaixo para abrir o card completo de Picking."
    )
    fig_3d = construir_figura_3d_interativa(
        renderer,
        estilo=estilo_preview,
        modo_noite=modo_noite,
        cfg_cam=cfg_atual,
        azimute_deg=az_deg if usar_orbita else None,
        elevacao_deg=el_deg if usar_orbita else None,
    )
    evento_pick = st.plotly_chart(
        fig_3d,
        width="stretch",
        on_select="rerun",
        selection_mode="points",
    )

    if evento_pick and evento_pick.get("selection", {}).get("points"):
      pt = evento_pick["selection"]["points"][0]
      curve_idx = pt.get("curve_number", 0)
      nome_obj = (
          fig_3d.data[curve_idx].name
          if curve_idx < len(fig_3d.data)
          else "Elemento da Cena"
      )
      px_c, pz_c, py_c = pt.get("x", 0.0), pt.get("y", 0.0), pt.get("z", 0.0)
      desc_obj = CATALOGO_OBJETOS_PICKING.get(
          nome_obj,
          "Primitiva geométrica posicionada via matriz homogênea 4x4.",
      )
      st.info(
          f"### Picking 3D: {nome_obj}\n"
          f"- **Coordenadas Mundo:** `X = {px_c:.2f}m` | `Altura Y = {py_c:.2f}m` | `Z = {pz_c:.2f}m`\n"
          f"- **Especificação Técnica:** {desc_obj}"
      )

with aba_rc:
  if st.button(
      "RENDERIZAR RAY CASTING AGORA", type="primary", width="stretch"
  ):
    with st.spinner("Renderizando imagem com Supersampling 4x..."):
      img, titulo_render = renderer.renderizar_config(
          cfg_atual,
          modo_noite=modo_noite,
          ativar_caustics=ativar_caustics_rc,
          ativar_reflexos=ativar_reflexos_rc,
          ativar_sombras=ativar_sombras_rc,
          incluir_objs=incluir_objs_rc,
          ssaa=ssaa_rc,
      )
      st.session_state["img_render"] = img
      st.session_state["tit_render"] = titulo_render
      st.session_state["cfg_render"] = cfg_atual.copy()
      st.session_state["objs_render"] = incluir_objs_rc

  if "img_render" in st.session_state:
    st.subheader(st.session_state["tit_render"])
    img_atual = st.session_state["img_render"]
    h_img, w_img, _ = img_atual.shape

    col_img, col_pick_rc = st.columns([2.2, 1.0])
    with col_pick_rc:
      st.markdown("#### Picking no Render (Raio por Pixel)")
      st.caption(
          "Escolha a coordenada `(Pixel X, Pixel Y)` da imagem renderizada para lançar um raio de inspeção e identificar o objeto atingido:"
      )
      px_sel = st.slider("Pixel X (Horizontal)", 0, w_img - 1, w_img // 2)
      py_sel = st.slider("Pixel Y (Vertical)", 0, h_img - 1, h_img // 2)

      info_rc = renderer.inspecionar_pixel_renderizado(
          st.session_state.get("cfg_render", cfg_atual),
          px_sel,
          py_sel,
          incluir_objs=st.session_state.get("objs_render", incluir_objs_rc),
      )
      if info_rc["hit"]:
        st.success(
            f"**Objeto Atingido:** {info_rc['nome']}\n\n"
            f"- **Primitiva:** `{info_rc['tipo_primitiva']}`\n"
            f"- **Material:** {info_rc['material']}\n"
            f"- **Ponto 3D $(X, Y, Z)$:** `{info_rc['ponto_world']}`\n"
            f"- **Normal $\\vec{{N}}$:** `{info_rc['normal_world']}`\n"
            f"- **Distância $t$:** `{info_rc['distancia']:.2f} m`\n"
            f"- **Difusa $K_d$ / Refl:** `{info_rc['kd']}` / `{info_rc['refletividade']}`"
        )
      else:
        st.warning(f"**Resultado do Raio:** {info_rc['nome']}")

    with col_img:
      # Desenha mira vermelha discreta no pixel inspecionado
      img_marcada = img_atual.copy()
      r_m = max(3, w_img // 75)
      y0, y1 = max(0, py_sel - r_m), min(h_img, py_sel + r_m + 1)
      x0, x1 = max(0, px_sel - r_m), min(w_img, px_sel + r_m + 1)
      img_marcada[y0:y1, max(0, px_sel - 1) : min(w_img, px_sel + 2)] = [
          1.0,
          0.1,
          0.1,
      ]
      img_marcada[max(0, py_sel - 1) : min(h_img, py_sel + 2), x0:x1] = [
          1.0,
          0.1,
          0.1,
      ]
      st.image(img_marcada, width="stretch")
  else:
    st.info(
        "Selecione o template de câmera e clique no botão acima para gerar a imagem final e habilitar o Picking por pixel."
    )
