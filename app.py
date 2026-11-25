from motor_ct import RenderizadorCT
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Bloco 734 CT — Trabalho Final CG I (Prof. Creto Vidal)",
    layout="wide",
)

# ==============================================================================
# CABEÇALHO ACADÊMICO
# ==============================================================================
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #0F172A, #1E293B); padding: 14px 20px; border-radius: 10px; border-left: 5px solid #38BDF8; margin-bottom: 14px; color: #F8FAFC;">
      <div style="font-size: 20px; font-weight: 800; color: #38BDF8;">Trabalho Final — Computação Gráfica I (Bloco 734 CT)</div>
      <div style="font-size: 14px; margin-top: 4px; color: #E2E8F0;">
        <b>Professor:</b> Creto Vidal &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Estudante:</b> João Bosco Pinheiro Dantas Filho
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def carregar_motor(resolucao=420):
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


def exibir_popup_picking(
    renderer, nome_obj, px_c, py_c, pz_c, extra_info=""
):
  """Exibe card detalhado com propriedades ópticas, materiais e coordenadas do objeto selecionado."""
  prim_encontrada = None
  for p in renderer.cena:
    if (
        nome_obj.lower() in p.nome.lower()
        or p.nome.lower() in nome_obj.lower()
    ):
      prim_encontrada = p
      break

  with st.container(border=True):
    st.markdown(f"### 🔍 Inspeção de Objeto (Picking): `{nome_obj}`")
    c1, c2, c3 = st.columns(3)
    c1.metric("Coordenada X (Mundo)", f"{px_c:.2f} m")
    c2.metric("Altura Y (Mundo)", f"{py_c:.2f} m")
    c3.metric("Coordenada Z (Mundo)", f"{pz_c:.2f} m")

    if prim_encontrada is not None:
      m = prim_encontrada.mat
      st.markdown(
          f"""
          - **Primitiva / Modelo:** `{prim_encontrada.tipo}` | **Material:** `{m.nome}`
          - **Coeficiente Difuso ($K_d$):** `({m.kd[0]:.2f}, {m.kd[1]:.2f}, {m.kd[2]:.2f})`
          - **Coeficiente Especular ($K_e$):** `({m.ke[0]:.2f}, {m.ke[1]:.2f}, {m.ke[2]:.2f})` | **Expoente Phong ($m$):** `{m.brilho:.1f}`
          - **Refletividade Espelhada ($k_r$):** `{m.refletividade*100:.0f}%` | **Textura Procedural:** `{m.textura or 'Nenhuma (Sólido)'}`
          """
      )
    if extra_info:
      st.info(extra_info)


def renderizar_inspecao_webgl(modo_noite=False):
  """Modo Roblox/WebGL exclusivo para Inspeção 3D com Raycaster Picking por clique."""
  noite_js = "true" if modo_noite else "false"
  html_code = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8" />
<style>
  body {{ margin: 0; overflow: hidden; background: #0B1120; font-family: system-ui, sans-serif; user-select: none; }}
  #hud-insp {{
    position: absolute; top: 12px; left: 12px; z-index: 10;
    background: rgba(15, 23, 42, 0.88); color: #F8FAFC;
    padding: 10px 14px; border-radius: 10px; border: 1px solid #38BDF8; font-size: 12.5px;
  }}
  #pick-modal {{
    display: none; position: absolute; top: 12px; right: 12px; z-index: 20;
    background: rgba(15, 23, 42, 0.94); color: #F8FAFC; width: 310px;
    padding: 14px 16px; border-radius: 12px; border: 2px solid #38BDF8;
    box-shadow: 0 10px 30px rgba(0,0,0,0.65); font-size: 13px; line-height: 1.5;
  }}
  #pick-modal h4 {{ margin: 0 0 6px 0; color: #38BDF8; font-size: 15px; }}
  #close-pick {{
    float: right; background: #EF4444; color: white; border: none; border-radius: 4px;
    padding: 2px 7px; cursor: pointer; font-weight: bold;
  }}
</style>
</head>
<body>
<div id="hud-insp">
  <b>Inspeção 3D Estilo Roblox (WebGL)</b><br/>
  • <b>Clique em qualquer objeto</b> para abrir a ficha completa (Picking)<br/>
  • <b>Arraste</b> para girar (travado acima do solo) &nbsp;|&nbsp; <b>Scroll</b>: Zoom
</div>

<div id="pick-modal">
  <button id="close-pick" onclick="document.getElementById('pick-modal').style.display='none'">X</button>
  <h4 id="pk-nome">Objeto</h4>
  <div id="pk-detalhes"></div>
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

  scene.add(new THREE.HemisphereLight(0xffffff, 0x475569, MODO_NOITE ? 0.5 : 0.9));
  const dirLight = new THREE.DirectionalLight(0xfffbeb, MODO_NOITE ? 0.3 : 1.1);
  dirLight.position.set(15, 35, 12); dirLight.castShadow = true; scene.add(dirLight);

  const objetosPick = [];
  function registrarPick(mesh, nome, tipo, material, extra) {{
    mesh.userData = {{ nome, tipo, material, extra }};
    objetosPick.push(mesh);
  }}

  const via = new THREE.Mesh(new THREE.BoxGeometry(66, 0.1, 14), new THREE.MeshStandardMaterial({{ color: 0x1e293b }}));
  via.position.set(35, 0.05, 9); scene.add(via);
  registrarPick(via, "Via Asfáltica CT", "Cubo (66m x 14m)", "Asfalto CBUQ (Kd=0.22)", "Faixa dupla amarela e tráfego de veículos");

  const calcada = new THREE.Mesh(new THREE.BoxGeometry(66, 0.2, 44), new THREE.MeshStandardMaterial({{ color: MODO_NOITE ? 0x334155 : 0xcbd5e1 }}));
  calcada.position.set(35, 0.1, 38); scene.add(calcada);
  registrarPick(calcada, "Calçada Bloco 734", "Cubo (66m x 44m)", "Concreto Desempenado (Kd=0.68)", "Piso da praça frontal do Centro de Tecnologia");

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
  registrarPick(meshBloco, "Bloco 734 (CT / Engenharia)", "Edificação Extrudada (Arco R=22m)", "Alvenaria Branca (Kd=0.92, Brilho=16)", "Altura 6.6m, 9 janelas em arco e platibanda nº 734");

  // Janelas
  for (let i = 0; i < 9; i++) {{
    const ang = Math.PI + ((i + 0.5) / 9) * (Math.PI * 0.44) + 0.08;
    const vx = cx + (R + 0.05) * Math.cos(ang);
    const vz = cz + (R + 0.05) * Math.sin(ang);
    const janela = new THREE.Mesh(new THREE.BoxGeometry(1.7, 1.9, 0.15), new THREE.MeshStandardMaterial({{ color: 0x0f172a, metalness: 0.85, roughness: 0.15 }}));
    janela.position.set(vx, 2.3, vz); janela.lookAt(cx, 2.3, cz); scene.add(janela);
    registrarPick(janela, `Janela Vidro Fumê #${{i+1}}`, "Esquadria Curva", "Vidro Fumê Refletivo (kr=55%, Phong=128)", i === 8 ? "ATENÇÃO: Janela final assombrada no modo jogo!" : "Vidro refletivo da fachada do Bloco 734");
  }}

  // Fonte
  const fonte = new THREE.Mesh(new THREE.BoxGeometry(15.0, 0.34, 6.0), new THREE.MeshStandardMaterial({{ color: 0x06b6d4, roughness: 0.1 }}));
  fonte.position.set(27, 0.37, 19.8); scene.add(fonte);
  registrarPick(fonte, "Espelho d'Água (Fonte Frontal)", "Prisma Retangular (15m x 6m)", "Água com Refração n=1.333 e Cáusticas", "Possui 3 spots submersos LED que projetam cáusticas na fachada");

  // Escultura Cinética
  const escMesh = new THREE.Mesh(new THREE.BoxGeometry(3.2, 3.8, 3.4), new THREE.MeshStandardMaterial({{ color: 0xf8fafc, wireframe: false }}));
  escMesh.position.set(15.0, 2.1, 22.0); escMesh.rotation.y = THREE.MathUtils.degToRad(-24); scene.add(escMesh);
  registrarPick(escMesh, "Escultura Cinética (5 Losangos)", "Composição Paramétrica em Aço", "Aço Pintura Branca (kr=14%, Brilho=64)", "5 quadros rotacionados a 45° sobre pedestal de concreto");

  // Árvores e Postes
  [[10.5, 28.5, 1.15], [11.5, 38.0, 1.25], [37.5, 17.5, 0.9]].forEach(([ax, az, esc], idx) => {{
    const copa = new THREE.Mesh(new THREE.DodecahedronGeometry(1.95 * esc, 1), new THREE.MeshStandardMaterial({{ color: 0x16a34a }}));
    copa.position.set(ax, 4.1 * esc, az); scene.add(copa);
    registrarPick(copa, `Árvore do Campus #${{idx+1}}`, "Cilindro + Copa Poliédrica", "Folhagem Verde (Kd=0.14, 0.48, 0.18)", idx === 1 ? "Contém ninho de ave agressiva no modo Walkthrough" : "Vegetação do entorno do Bloco 734");
  }});

  [[9.5, 21.0], [18.5, 16.2], [41.0, 16.5], [54.5, 17.5]].forEach(([px, pz], idx) => {{
    const mastro = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.14, 6.2, 8), new THREE.MeshStandardMaterial({{ color: 0x334155 }}));
    mastro.position.set(px, 3.3, pz); scene.add(mastro);
    registrarPick(mastro, `Poste de Iluminação #${{idx+1}}`, "Cilindro + Luminária Esférica", "Aço Galvanizado Escuro (Brilho=48)", "Fonte pontual de luz noturna com atenuação quadrática");
  }});

  // Carros
  const suv = new THREE.Mesh(new THREE.BoxGeometry(4.3, 1.5, 2.0), new THREE.MeshStandardMaterial({{ color: 0xcbd5e1 }}));
  suv.position.set(23.0, 0.85, 7.2); scene.add(suv);
  registrarPick(suv, "SUV Urbano (Malha .OBJ)", "Malha Triangular Suavizada", "Pintura Automotiva Prata (kr=25%)", "Veículo trafegando na via frontal do CT");

  const sport = new THREE.Mesh(new THREE.BoxGeometry(4.2, 1.3, 1.9), new THREE.MeshStandardMaterial({{ color: 0xdc2626 }}));
  sport.position.set(39.0, 0.75, 10.6); scene.add(sport);
  registrarPick(sport, "Carro Esportivo (Malha .OBJ)", "Malha Triangular Suavizada", "Pintura Vermelha Metálica (kr=28%)", "Veículo trafegando em sentido oposto");

  // Órbita travada acima do solo (0° a 88°)
  let yaw = 2.35, pitch = 0.45, dist = 36.0;
  const target = new THREE.Vector3(32.0, 2.5, 28.0);
  let drag = false, prev = {{ x: 0, y: 0 }};

  renderer.domElement.addEventListener('mousedown', (e) => {{ drag = true; prev = {{ x: e.clientX, y: e.clientY }}; }});
  window.addEventListener('mouseup', () => {{ drag = false; }});
  window.addEventListener('mousemove', (e) => {{
    if (!drag) return;
    yaw -= (e.clientX - prev.x) * 0.007;
    pitch = Math.max(0.05, Math.min(1.45, pitch + (e.clientY - prev.y) * 0.006));
    prev = {{ x: e.clientX, y: e.clientY }};
  }});
  renderer.domElement.addEventListener('wheel', (e) => {{
    dist = Math.max(8.0, Math.min(75.0, dist + e.deltaY * 0.02));
  }});

  // Raycaster Picking ao clicar
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  renderer.domElement.addEventListener('click', (e) => {{
    mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(objetosPick, false);
    if (hits.length > 0) {{
      const h = hits[0];
      const d = h.object.userData;
      document.getElementById('pk-nome').textContent = d.nome;
      document.getElementById('pk-detalhes').innerHTML = `
        <b>Ponto Atingido (Mundo):</b><br/>
        X = <code>${{h.point.x.toFixed(2)}}m</code> | Altura Y = <code>${{h.point.y.toFixed(2)}}m</code> | Z = <code>${{h.point.z.toFixed(2)}}m</code><br/><br/>
        <b>Geometria:</b> ${{d.tipo}}<br/>
        <b>Material:</b> ${{d.material}}<br/>
        <b>Observação:</b> ${{d.extra}}
      `;
      document.getElementById('pick-modal').style.display = 'block';
    }}
  }});

  function loop() {{
    requestAnimationFrame(loop);
    camera.position.set(
      target.x + dist * Math.cos(pitch) * Math.sin(yaw),
      target.y + dist * Math.sin(pitch),
      target.z + dist * Math.cos(pitch) * Math.cos(yaw)
    );
    camera.lookAt(target);
    renderer.render(scene, camera);
  }}
  loop();
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
    backdrop-filter: blur(6px); max-width: 375px; font-size: 12px; line-height: 1.45;
  }}
  #hud b {{ color: #38BDF8; }}
  .btn-cam {{
    margin-top: 6px; margin-right: 4px; padding: 5px 8px;
    background: #1E293B; color: #F8FAFC; border: 1px solid #475569;
    border-radius: 6px; cursor: pointer; font-size: 11px; font-weight: 600;
  }}
  .btn-cam:hover {{ background: #0284C7; border-color: #38BDF8; }}

  #stations-panel {{
    position: absolute; top: 10px; right: 10px; z-index: 10;
    background: rgba(15, 23, 42, 0.90); color: #F8FAFC;
    padding: 10px 13px; border-radius: 10px; border: 1px solid rgba(148, 163, 184, 0.35);
    backdrop-filter: blur(6px); width: 280px; font-size: 11.2px; line-height: 1.45;
  }}
  .st-title {{ font-weight: 800; color: #FACC15; font-size: 12.5px; margin-bottom: 4px; display: flex; justify-content: space-between; align-items: center; }}
  .st-item {{ display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.08); padding: 2px 0; }}
  .st-done {{ color: #4ADE80; font-weight: 700; }}
  .st-todo {{ color: #94A3B8; }}

  #prompt-box {{
    display: none; position: absolute; bottom: 50px; left: 50%; transform: translateX(-50%); z-index: 15;
    background: rgba(2, 132, 199, 0.92); color: #FFFFFF; padding: 8px 16px;
    border-radius: 999px; font-weight: 700; font-size: 13px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.5); border: 1px solid #7DD3FC;
  }}

  #toxic-bar-wrap {{
    display: none; position: absolute; bottom: 92px; left: 50%; transform: translateX(-50%); z-index: 15;
    background: rgba(15,23,42,0.9); padding: 8px 14px; border-radius: 8px; border: 1px solid #22D3EE; width: 240px; text-align: center;
  }}
  #toxic-bar {{ height: 10px; background: #06B6D4; width: 0%; border-radius: 4px; }}

  #coords {{
    position: absolute; bottom: 10px; right: 10px; z-index: 10;
    background: rgba(15, 23, 42, 0.85); color: #E2E8F0;
    padding: 6px 11px; border-radius: 8px; font-family: monospace; font-size: 11.5px;
  }}

  #wasted-overlay {{
    display: none; position: absolute; inset: 0; z-index: 35;
    background: radial-gradient(circle, rgba(127,29,29,0.65) 0%, rgba(15,23,42,0.93) 100%);
    flex-direction: column; align-items: center; justify-content: center; text-align: center;
  }}
  #wasted-title {{ font-size: 50px; font-weight: 900; color: #EF4444; text-shadow: 0 4px 20px rgba(0,0,0,0.9); }}
  #wasted-sub {{ font-size: 18px; color: #FCA5A5; margin-top: 8px; font-weight: 600; max-width: 540px; }}
  #wasted-warn {{ font-size: 14px; color: #FDE047; margin-top: 6px; font-weight: 700; }}
  #btn-respawn {{
    margin-top: 20px; padding: 11px 24px; font-size: 15px; font-weight: 700;
    background: #EF4444; color: white; border: none; border-radius: 8px; cursor: pointer;
  }}

  /* JUMPSCARE MÁSCARA PÂNICO */
  #jumpscare-overlay {{
    display: none; position: absolute; inset: 0; z-index: 40;
    background: #000000; align-items: center; justify-content: center; flex-direction: column;
  }}
  @keyframes shakeScreen {{
    0% {{ transform: translate(6px, 6px) scale(1.05); }}
    25% {{ transform: translate(-8px, -6px) scale(1.12); }}
    50% {{ transform: translate(8px, -4px) scale(1.08); }}
    75% {{ transform: translate(-6px, 8px) scale(1.15); }}
    100% {{ transform: translate(0, 0) scale(1.0); }}
  }}
  .scream-anim {{ animation: shakeScreen 0.14s infinite; }}

  #victory-banner {{
    display: none; position: absolute; top: 80px; left: 50%; transform: translateX(-50%); z-index: 25;
    background: linear-gradient(135deg, #16A34A, #0284C7); color: white;
    padding: 14px 24px; border-radius: 12px; font-weight: 800; font-size: 16px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.6); text-align: center;
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
  <div style="color:#FCA5A5; font-size:11px;"><b>Regra:</b> Morrer zera TODAS as 10 missões! Cuidado com a Janela Final, o NPC suspeito e a Árvore do Fundo!</div>
  <div>
    <button class="btn-cam" onclick="setCameraMode('third')">3ª Pessoa</button>
    <button class="btn-cam" onclick="setCameraMode('iso')">Isométrica</button>
    <button class="btn-cam" onclick="resetTotalMissoes()">Resetar Missões</button>
  </div>
</div>

<div id="stations-panel">
  <div class="st-title">
    <span id="st-counter">MISSÕES (0/10)</span>
    <button onclick="resetTotalMissoes()" style="background:#334155; color:#fff; border:none; border-radius:4px; padding:2px 6px; font-size:10px; cursor:pointer;">Zerar</button>
  </div>
  <div class="st-item"><span>1. Foto da Escultura (F)</span><span id="st-foto-esc" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>2. Foto da Fonte (F)</span><span id="st-foto-fonte" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>3. Foto do Carro na Via (F)</span><span id="st-foto-carro" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>4. Foto do Bloco 734 (F)</span><span id="st-foto-bloco" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>5. Foto da Árvore (F)</span><span id="st-foto-arvore" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>6. Fonte Rápida (&lt;3.5s)</span><span id="st-fonte" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>7. Escalar Poste (E)</span><span id="st-poste" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>8. Escalar Árvore Segura (E)</span><span id="st-arvore" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>9. Atravessar Rua & Voltar</span><span id="st-rua" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>10. Telhado pela Escada (E)</span><span id="st-telhado" class="st-todo">PENDENTE</span></div>
</div>

<div id="prompt-box">Ação disponível</div>
<div id="toxic-bar-wrap">
  <div style="color:#67E8F9; font-size:11px; font-weight:700; margin-bottom:4px;">ÁGUA CONTAMINADA! SAIA JÁ!</div>
  <div id="toxic-bar"></div>
</div>

<div id="victory-banner">
  PARABÉNS! VOCÊ COMPLETOU AS 10 MISSÕES SEM MORRER!<br/>
  <button onclick="resetTotalMissoes()" style="margin-top:8px; padding:6px 14px; border:none; border-radius:6px; background:#0F172A; color:#fff; font-weight:bold; cursor:pointer;">JOGAR NOVAMENTE (RESETAR MISSÕES)</button>
</div>
<div id="flash-foto"></div>

<!-- JUMPSCARE MÁSCARA PÂNICO SVG -->
<div id="jumpscare-overlay">
  <svg class="scream-anim" width="320" height="420" viewBox="0 0 200 260">
    <path d="M35,85 C30,25 170,25 165,85 C162,145 145,215 100,248 C55,215 38,145 35,85 Z" fill="#F8FAFC" stroke="#1E293B" stroke-width="4"/>
    <ellipse cx="72" cy="95" rx="18" ry="28" transform="rotate(-12 72 95)" fill="#090D16"/>
    <ellipse cx="128" cy="95" rx="18" ry="28" transform="rotate(12 128 95)" fill="#090D16"/>
    <path d="M95,122 L88,142 L97,138 Z" fill="#090D16"/>
    <path d="M105,122 L112,142 L103,138 Z" fill="#090D16"/>
    <ellipse cx="100" cy="192" rx="22" ry="44" fill="#090D16"/>
  </svg>
  <div style="color:#EF4444; font-size:34px; font-weight:900; margin-top:10px; letter-spacing:2px;">BUUU!! VOCÊ FOI PEGO!</div>
</div>

<div id="wasted-overlay">
  <div id="wasted-title">DUMB WAYS TO DIE</div>
  <div id="wasted-sub">Você encontrou um fim trágico!</div>
  <div id="wasted-warn">TODAS AS MISSÕES FORAM ZERADAS! TENTE NOVAMENTE SEM MORRER.</div>
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

  scene.add(new THREE.HemisphereLight(MODO_NOITE ? 0x1e293b : 0xffffff, MODO_NOITE ? 0x0f172a : 0x64748b, MODO_NOITE ? 0.5 : 0.85));
  const dirLight = new THREE.DirectionalLight(MODO_NOITE ? 0x3b82f6 : 0xfffbeb, MODO_NOITE ? 0.25 : 1.15);
  dirLight.position.set(12, 32, 10); dirLight.castShadow = true;
  dirLight.shadow.mapSize.width = 2048; dirLight.shadow.mapSize.height = 2048;
  dirLight.shadow.camera.left = -45; dirLight.shadow.camera.right = 45;
  dirLight.shadow.camera.top = 45; dirLight.shadow.camera.bottom = -45;
  scene.add(dirLight);

  // Via e Calçada
  const via = new THREE.Mesh(new THREE.BoxGeometry(66, 0.1, 14), new THREE.MeshStandardMaterial({{ color: 0x1e293b, roughness: 0.85 }}));
  via.position.set(35, 0.05, 9); via.receiveShadow = true; scene.add(via);
  for (let fx = 4; fx < 66; fx += 4.5) {{
    const faixa = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.02, 0.18), new THREE.MeshBasicMaterial({{ color: 0xfacc15 }}));
    faixa.position.set(fx, 0.11, 11.7); scene.add(faixa);
  }}
  const calcada = new THREE.Mesh(new THREE.BoxGeometry(66, 0.2, 44), new THREE.MeshStandardMaterial({{ color: MODO_NOITE ? 0x334155 : 0xcbd5e1, roughness: 0.9 }}));
  calcada.position.set(35, 0.1, 38); calcada.receiveShadow = true; scene.add(calcada);

  // Plataforma Objetivo do Outro Lado da Rua
  const platRua = new THREE.Mesh(new THREE.CylinderGeometry(1.8, 1.8, 0.12, 24), new THREE.MeshStandardMaterial({{ color: 0x22c55e, emissive: 0x15803d, emissiveIntensity: 0.4 }}));
  platRua.position.set(27.0, 0.12, 3.2); scene.add(platRua);

  // Bloco 734
  const shapeBloco = new THREE.Shape();
  const cx = 37, cz = 42, R = 22, H = 6.6;
  shapeBloco.moveTo(15, -53); shapeBloco.lineTo(15, -42);
  shapeBloco.absarc(cx, -cz, R, Math.PI, 0.5 * Math.PI, true);
  shapeBloco.lineTo(59, -20); shapeBloco.lineTo(59, -53); shapeBloco.closePath();
  const extrudeGeo = new THREE.ExtrudeGeometry(shapeBloco, {{ depth: H, bevelEnabled: false, curveSegments: 32 }});
  extrudeGeo.rotateX(-Math.PI / 2);
  const meshBloco = new THREE.Mesh(extrudeGeo, new THREE.MeshStandardMaterial({{ color: MODO_NOITE ? 0x475569 : 0xf8fafc, roughness: 0.45 }}));
  meshBloco.position.y = 0.2; meshBloco.castShadow = true; meshBloco.receiveShadow = true; scene.add(meshBloco);

  // Escada Lateral para o Telhado
  const escadaGroup = new THREE.Group();
  escadaGroup.position.set(59.35, 0.2, 24.0);
  const trilho1 = new THREE.Mesh(new THREE.BoxGeometry(0.12, 6.8, 0.12), new THREE.MeshStandardMaterial({{ color: 0xf59e0b }}));
  trilho1.position.set(0, 3.4, -0.45); escadaGroup.add(trilho1);
  const trilho2 = trilho1.clone(); trilho2.position.z = 0.45; escadaGroup.add(trilho2);
  for (let deg = 0.4; deg <= 6.6; deg += 0.45) {{
    const degrau = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.08, 0.9), new THREE.MeshStandardMaterial({{ color: 0xfbbf24 }}));
    degrau.position.set(0, deg, 0); escadaGroup.add(degrau);
  }}
  scene.add(escadaGroup);

  // 9 Janelas (A 9ª janela no final do arco é a janela do Jumpscare Pânico!)
  let janelaFinalPos = new THREE.Vector3(36.0, 2.3, 20.0);
  for (let i = 0; i < 9; i++) {{
    const ang = Math.PI + ((i + 0.5) / 9) * (Math.PI * 0.44) + 0.08;
    const vx = cx + (R + 0.05) * Math.cos(ang);
    const vz = cz + (R + 0.05) * Math.sin(ang);
    const janela = new THREE.Mesh(
      new THREE.BoxGeometry(1.7, 1.9, 0.15),
      new THREE.MeshStandardMaterial({{ color: i === 8 ? 0x3b0764 : 0x0f172a, metalness: 0.85, roughness: 0.15 }})
    );
    janela.position.set(vx, 2.3, vz); janela.lookAt(cx, 2.3, cz); scene.add(janela);
    if (i === 8) janelaFinalPos.set(vx, 2.3, vz);
  }}

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

  // Árvores (Árvore index 1 em X=11.5, Z=38.0 tem AVE AGRESSIVA!)
  const arvores = [[10.5, 28.5, 1.15], [11.5, 38.0, 1.25], [37.5, 17.5, 0.9]];
  arvores.forEach(([ax, az, esc]) => {{
    const tronco = new THREE.Mesh(new THREE.CylinderGeometry(0.22 * esc, 0.3 * esc, 3.6 * esc, 10), new THREE.MeshStandardMaterial({{ color: 0x5c3a21 }}));
    tronco.position.set(ax, 0.2 + 1.8 * esc, az); scene.add(tronco);
    const copa = new THREE.Mesh(new THREE.DodecahedronGeometry(1.95 * esc, 1), new THREE.MeshStandardMaterial({{ color: MODO_NOITE ? 0x064e3b : 0x16a34a }}));
    copa.position.set(ax, 4.1 * esc, az); scene.add(copa);
  }});

  // AVE AGRESSIVA NA ÁRVORE (X=11.5, Z=38.0)
  const aveGroup = new THREE.Group();
  const corpoAve = new THREE.Mesh(new THREE.ConeGeometry(0.32, 0.9, 8), new THREE.MeshStandardMaterial({{ color: 0x1e293b }}));
  corpoAve.rotation.x = Math.PI / 2; aveGroup.add(corpoAve);
  const asaE = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.06, 0.35), new THREE.MeshStandardMaterial({{ color: 0x991b1b }}));
  asaE.position.x = -0.45; aveGroup.add(asaE);
  const asaD = asaE.clone(); asaD.position.x = 0.45; aveGroup.add(asaD);
  aveGroup.position.set(11.5, 4.6, 38.0);
  scene.add(aveGroup);
  let aveAtacando = false;

  // NPC PARADO MISTERIOSO (X=43.0, Z=19.0) — Se abordar/aproximar, dá susto e mata!
  const npcMisterioso = new THREE.Group();
  const npcCorpo = new THREE.Mesh(new THREE.BoxGeometry(0.75, 1.3, 0.42), new THREE.MeshStandardMaterial({{ color: 0x111827 }}));
  npcCorpo.position.y = 0.85; npcMisterioso.add(npcCorpo);
  const npcCabeca = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.58, 0.52), new THREE.MeshStandardMaterial({{ color: 0xf8fafc }}));
  npcCabeca.position.y = 1.82; npcMisterioso.add(npcCabeca);
  npcMisterioso.position.set(43.0, 0.2, 19.0);
  npcMisterioso.rotation.y = Math.PI; // De costas inicialmente
  scene.add(npcMisterioso);

  // Postes
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
    const chassi = new THREE.Mesh(new THREE.BoxGeometry(2.0, 0.65, 4.3), new THREE.MeshStandardMaterial({{ color: cor, metalness: 0.5 }}));
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

  // Personagem Roblox
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

  // Som de susto sintetizado (Web Audio API)
  function tocarSomSusto() {{
    try {{
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(620, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(110, ctx.currentTime + 0.55);
      gain.gain.setValueAtTime(0.35, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.55);
      osc.connect(gain); gain.connect(ctx.destination);
      osc.start(); osc.stop(ctx.currentTime + 0.56);
    }} catch (e) {{}}
  }}

  let isDead = false, deathCount = 0, velY = 0, walkCycle = 0;
  let tempoNaFonte = 0, subiuArvore = false, subiuPoste = false, atravessouRua = false, estavaNoTelhado = false;

  const missoesMap = {{
    'foto-esc': 'st-foto-esc',
    'foto-fonte': 'st-foto-fonte',
    'foto-carro': 'st-foto-carro',
    'foto-bloco': 'st-foto-bloco',
    'foto-arvore': 'st-foto-arvore',
    'fonte': 'st-fonte',
    'poste': 'st-poste',
    'arvore': 'st-arvore',
    'rua': 'st-rua',
    'telhado': 'st-telhado'
  }};
  const checks = {{}};
  Object.keys(missoesMap).forEach(k => checks[k] = false);

  function atualizarPainelMissoes() {{
    let feitos = 0;
    Object.keys(missoesMap).forEach(k => {{
      const el = document.getElementById(missoesMap[k]);
      if (checks[k]) {{
        feitos++;
        el.textContent = "CHECK ✓";
        el.className = "st-done";
      }} else {{
        el.textContent = "PENDENTE";
        el.className = "st-todo";
      }}
    }});
    document.getElementById('st-counter').textContent = `MISSÕES (${{feitos}}/10)`;
    document.getElementById('victory-banner').style.display = (feitos === 10) ? 'block' : 'none';
  }}

  function marcarCheck(chave) {{
    if (checks[chave] || isDead) return;
    checks[chave] = true;
    atualizarPainelMissoes();
  }}

  window.resetTotalMissoes = () => {{
    Object.keys(checks).forEach(k => checks[k] = false);
    atualizarPainelMissoes();
    window.resetPlayer();
  }};

  function triggerDeath(motivo, impactoVel, comJumpscare = false) {{
    if (isDead) return;
    isDead = true;
    deathCount++;

    // ZERA TODAS AS MISSÕES AO MORRER!
    Object.keys(checks).forEach(k => checks[k] = false);
    atualizarPainelMissoes();

    document.getElementById('toxic-bar-wrap').style.display = 'none';
    partesBoneco.forEach((p) => {{
      p.vel.set((Math.random() - 0.5) * 9 + impactoVel.x, 4.5 + Math.random() * 6, (Math.random() - 0.5) * 9 + impactoVel.z);
      p.rotVel.set((Math.random() - 0.5) * 12, (Math.random() - 0.5) * 12, (Math.random() - 0.5) * 12);
    }});

    if (comJumpscare) {{
      tocarSomSusto();
      const jsEl = document.getElementById('jumpscare-overlay');
      jsEl.style.display = 'flex';
      setTimeout(() => {{
        jsEl.style.display = 'none';
        document.getElementById('wasted-sub').textContent = motivo;
        document.getElementById('wasted-overlay').style.display = 'flex';
      }}, 1100);
    }} else {{
      document.getElementById('wasted-sub').textContent = motivo;
      document.getElementById('wasted-overlay').style.display = 'flex';
    }}
  }}
