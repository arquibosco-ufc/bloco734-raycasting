from motor_ct import RenderizadorCT
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Bloco 734 CT + Walkthrough + Inspeção 3D", layout="wide"
)
st.title("Visualizador 3D, Walkthrough & Ray Casting — Bloco 734 (CT)")


@st.cache_resource
def carregar_motor(resolucao=280):
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


def converter_camera_para_plotly(cfg_cam):
  """Converte eye e at do sistema do motor (X, Y_altura, Z_prof) para Plotly (X, Z, Y)."""
  eye = np.array(cfg_cam["eye"], dtype=float)
  at = np.array(cfg_cam["at"], dtype=float)

  # Centro aproximado do bounding box da cena no Plotly
  centro_cena = np.array([35.0, 31.0, 11.0])
  escala_eixos = np.array([33.0, 29.0, 11.0])

  eye_pl = np.array([eye[0], eye[2], eye[1]])
  at_pl = np.array([at[0], at[2], at[1]])

  cam_eye = (eye_pl - centro_cena) / escala_eixos
  cam_center = (at_pl - centro_cena) / escala_eixos

  # Ajusta fator de zoom conforme 'd' e janela
  fator_zoom = 1.5 / max(float(cfg_cam.get("d", 1.5)), 0.5)
  cam_eye = cam_center + (cam_eye - cam_center) * fator_zoom

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
    renderer, estilo="materiais", modo_noite=False, cfg_cam=None
):
  fig = go.Figure()

  if estilo == "wireframe":
    c_bloco = "#38BDF8" if modo_noite else "#0F172A"
    c_teto = "#0284C7"
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
    c_teto = "#1E293B" if modo_noite else "#94A3B8"
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

  # 1. Piso e Faixas Viárias
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

  # 2. Bloco 734 (Fachada Curva, Teto, Esquadrias, Vidros e Letreiro 734)
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

  # Janelas e Porta de Vidro Fumê no Arco
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
    jy = [1.35, 1.35, 3.25, 3.25]
    fig.add_trace(
        go.Mesh3d(
            x=jx,
            y=jz,
            z=jy,
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color=c_vidro,
            name=f"Janela Vidro Fumê {idx_j+1}",
        )
    )

  # Porta Principal
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

  # Letreiro "734" na Platibanda
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

  # 3. Fonte / Espelho d'Água + Bordas
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

  # 5. Árvores, Postes e Cones
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
                  size=20, color="#064E3B" if modo_noite else cor_e, opacity=0.92
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

  # 6. Malhas (.OBJ contínuas e Abrigo Metálico)
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
  camera_plotly = (
      converter_camera_para_plotly(cfg_cam)
      if cfg_cam
      else dict(eye=dict(x=-1.4, y=-1.4, z=0.65))
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
          zaxis=dict(title="Y (m)", range=[0, 22]),
      ),
  )
  return fig


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
    position: absolute; top: 14px; left: 14px; z-index: 10;
    background: rgba(15, 23, 42, 0.85); color: #F8FAFC;
    padding: 12px 16px; border-radius: 10px; border: 1px solid rgba(148, 163, 184, 0.35);
    backdrop-filter: blur(6px); max-width: 380px; font-size: 13px; line-height: 1.5;
  }}
  #hud b {{ color: #38BDF8; }}
  .danger-txt {{ color: #F87171; font-weight: 700; }}
  .btn-cam {{
    margin-top: 8px; margin-right: 6px; padding: 6px 10px;
    background: #1E293B; color: #F8FAFC; border: 1px solid #475569;
    border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 600;
  }}
  .btn-cam:hover {{ background: #0284C7; border-color: #38BDF8; }}
  #coords {{
    position: absolute; bottom: 12px; right: 14px; z-index: 10;
    background: rgba(15, 23, 42, 0.82); color: #E2E8F0;
    padding: 6px 12px; border-radius: 8px; font-family: monospace; font-size: 12px;
  }}
  #wasted-overlay {{
    display: none; position: absolute; inset: 0; z-index: 30;
    background: radial-gradient(circle, rgba(127,29,29,0.45) 0%, rgba(15,23,42,0.88) 100%);
    flex-direction: column; align-items: center; justify-content: center; text-align: center;
  }}
  #wasted-title {{
    font-size: 56px; font-weight: 900; color: #EF4444;
    text-shadow: 0 4px 20px rgba(0,0,0,0.9); letter-spacing: 3px;
  }}
  #wasted-sub {{
    font-size: 20px; color: #FCA5A5; margin-top: 6px; font-weight: 600;
  }}
  #btn-respawn {{
    margin-top: 22px; padding: 12px 26px; font-size: 16px; font-weight: 700;
    background: #EF4444; color: white; border: none; border-radius: 8px; cursor: pointer;
    box-shadow: 0 4px 14px rgba(239,68,68,0.5);
  }}
  #btn-respawn:hover {{ background: #DC2626; }}
</style>
</head>
<body>
<div id="hud">
  <div style="font-size:15px; font-weight:700; margin-bottom:4px;">Walkthrough 3D — Bloco 734 (60 FPS)</div>
  <div><b>W A S D</b> / <b>Setas</b>: Andar &nbsp;|&nbsp; <b>Shift</b>: Correr &nbsp;|&nbsp; <b>Espaço</b>: Pular</div>
  <div><b>Mouse</b>: Arraste para girar &nbsp;|&nbsp; <span class="danger-txt">Cuidado com Carros e Escultura!</span></div>
  <div>
    <button class="btn-cam" onclick="setCameraMode('third')">3ª Pessoa (Roblox)</button>
    <button class="btn-cam" onclick="setCameraMode('iso')">Isométrica</button>
    <button class="btn-cam" onclick="resetPlayer()">Renascer (R)</button>
  </div>
</div>

<div id="wasted-overlay">
  <div id="wasted-title">DUMB WAYS TO DIE</div>
  <div id="wasted-sub" id="death-reason">Você encostou em um objeto perigoso!</div>
  <button id="btn-respawn" onclick="resetPlayer()">RENASCER AGORA (Tecla R)</button>
</div>

<div id="coords">Mortes: 0 | X=14.0 | Z=14.0</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
  const MODO_NOITE = {noite_js};
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(MODO_NOITE ? 0x070d19 : 0x93c5fd);
  scene.fog = new THREE.FogExp2(MODO_NOITE ? 0x070d19 : 0x93c5fd, 0.0085);

  const camera = new THREE.PerspectiveCamera(58, window.innerWidth / window.innerHeight, 0.1, 300);
  const renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  document.body.appendChild(renderer.domElement);

  const hemiLight = new THREE.HemisphereLight(
    MODO_NOITE ? 0x1e293b : 0xffffff,
    MODO_NOITE ? 0x0f172a : 0x64748b,
    MODO_NOITE ? 0.45 : 0.85
  );
  scene.add(hemiLight);

  const dirLight = new THREE.DirectionalLight(MODO_NOITE ? 0x3b82f6 : 0xfffbeb, MODO_NOITE ? 0.25 : 1.15);
  dirLight.position.set(12, 32, 10);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.width = 2048;
  dirLight.shadow.mapSize.height = 2048;
  const dCam = 45;
  dirLight.shadow.camera.left = -dCam;
  dirLight.shadow.camera.right = dCam;
  dirLight.shadow.camera.top = dCam;
  dirLight.shadow.camera.bottom = -dCam;
  scene.add(dirLight);

  // Via Asfáltica e Calçada
  const via = new THREE.Mesh(
    new THREE.BoxGeometry(66, 0.1, 14),
    new THREE.MeshStandardMaterial({{ color: 0x1e293b, roughness: 0.85 }})
  );
  via.position.set(35, 0.05, 9);
  via.receiveShadow = true;
  scene.add(via);

  for (let fx = 4; fx < 66; fx += 4.5) {{
    const faixa = new THREE.Mesh(
      new THREE.BoxGeometry(2.2, 0.02, 0.18),
      new THREE.MeshBasicMaterial({{ color: 0xfacc15 }})
    );
    faixa.position.set(fx, 0.11, 11.7);
    scene.add(faixa);
  }}

  const calcada = new THREE.Mesh(
    new THREE.BoxGeometry(66, 0.2, 44),
    new THREE.MeshStandardMaterial({{ color: MODO_NOITE ? 0x334155 : 0xcbd5e1, roughness: 0.9 }})
  );
  calcada.position.set(35, 0.1, 38);
  calcada.receiveShadow = true;
  scene.add(calcada);

  // Bloco 734
  const shapeBloco = new THREE.Shape();
  const cx = 37, cz = 42, R = 22, H = 6.6;
  shapeBloco.moveTo(15, -53);
  shapeBloco.lineTo(15, -42);
  shapeBloco.absarc(cx, -cz, R, Math.PI, 0.5 * Math.PI, true);
  shapeBloco.lineTo(59, -20);
  shapeBloco.lineTo(59, -53);
  shapeBloco.closePath();

  const extrudeGeo = new THREE.ExtrudeGeometry(shapeBloco, {{ depth: H, bevelEnabled: false, curveSegments: 32 }});
  extrudeGeo.rotateX(-Math.PI / 2);
  const meshBloco = new THREE.Mesh(
    extrudeGeo,
    new THREE.MeshStandardMaterial({{ color: MODO_NOITE ? 0x475569 : 0xf8fafc, roughness: 0.45 }})
  );
  meshBloco.position.y = 0.2;
  meshBloco.castShadow = true;
  meshBloco.receiveShadow = true;
  scene.add(meshBloco);

  for (let i = 0; i < 9; i++) {{
    const ang = Math.PI + ((i + 0.5) / 9) * (Math.PI * 0.44) + 0.08;
    const vx = cx + (R + 0.05) * Math.cos(ang);
    const vz = cz + (R + 0.05) * Math.sin(ang);
    const janela = new THREE.Mesh(
      new THREE.BoxGeometry(1.7, 1.9, 0.15),
      new THREE.MeshStandardMaterial({{ color: 0x0f172a, metalness: 0.85, roughness: 0.15 }})
    );
    janela.position.set(vx, 2.3, vz);
    janela.lookAt(cx, 2.3, cz);
    scene.add(janela);
  }}

  // Espelho d'Água / Fonte
  const bordaFonte = new THREE.Mesh(
    new THREE.BoxGeometry(15.4, 0.34, 6.4),
    new THREE.MeshStandardMaterial({{ color: 0x1e293b, roughness: 0.4 }})
  );
  bordaFonte.position.set(27, 0.37, 19.8);
  bordaFonte.castShadow = true;
  scene.add(bordaFonte);

  const aguaMat = new THREE.MeshStandardMaterial({{
    color: 0x06b6d4, emissive: MODO_NOITE ? 0x0891b2 : 0x000000,
    emissiveIntensity: MODO_NOITE ? 0.65 : 0.0, roughness: 0.1, metalness: 0.6
  }});
  const laminaAgua = new THREE.Mesh(new THREE.BoxGeometry(14.8, 0.1, 5.8), aguaMat);
  laminaAgua.position.set(27, 0.51, 19.8);
  scene.add(laminaAgua);

  if (MODO_NOITE) {{
    const luzFonte = new THREE.PointLight(0x22d3ee, 2.4, 18);
    luzFonte.position.set(27, 1.5, 19.8);
    scene.add(luzFonte);
  }}

  // Escultura Cinética (OBRA DE ARTE LETAL!)
  const grupoEsc = new THREE.Group();
  grupoEsc.position.set(15.0, 0.2, 22.0);
  grupoEsc.rotation.y = THREE.MathUtils.degToRad(-24);
  const baseEsc = new THREE.Mesh(
    new THREE.BoxGeometry(3.2, 0.28, 3.4),
    new THREE.MeshStandardMaterial({{ color: 0x64748b, roughness: 0.8 }})
  );
  baseEsc.position.y = 0.14;
  baseEsc.castShadow = true;
  grupoEsc.add(baseEsc);

  for (let k = 0; k < 5; k++) {{
    const fc = k - 2;
    const L = 2.35 * (1 - 0.055 * fc);
    const frameGroup = new THREE.Group();
    const bars = [
      [0, L/2, L, 0.14], [0, -L/2, L, 0.14],
      [L/2, 0, 0.14, L], [-L/2, 0, 0.14, L]
    ];
    bars.forEach(([bx, by, bw, bh]) => {{
      const bar = new THREE.Mesh(
        new THREE.BoxGeometry(bw, bh, 0.16),
        new THREE.MeshStandardMaterial({{ color: 0xf8fafc, metalness: 0.3, roughness: 0.25 }})
      );
      bar.position.set(bx, by, 0);
      bar.castShadow = true;
      frameGroup.add(bar);
    }});
    frameGroup.rotation.z = Math.PI / 4;
    frameGroup.position.set(fc * 0.22, 0.28 + (L / Math.SQRT2) + 0.05, (k - 2) * 0.58);
    grupoEsc.add(frameGroup);
  }}
  scene.add(grupoEsc);

  // Lista de Obstáculos Sólidos (Colisão)
  const obstaculosCilindricos = [];
  const obstaculosCaixas = [
    // Fonte
    {{ minX: 19.0, maxX: 35.0, minZ: 16.3, maxZ: 23.3 }},
    // Abrigo Metálico
    {{ minX: 44.3, maxX: 49.2, minZ: 17.0, maxZ: 20.6 }}
  ];

  // Árvores e Postes (com colisão registrada)
  const arvores = [[10.5, 28.5, 1.15], [11.5, 38.0, 1.25], [37.5, 17.5, 0.9]];
  arvores.forEach(([ax, az, esc]) => {{
    const tronco = new THREE.Mesh(
      new THREE.CylinderGeometry(0.2 * esc, 0.28 * esc, 3.2 * esc, 10),
      new THREE.MeshStandardMaterial({{ color: 0x5c3a21 }})
    );
    tronco.position.set(ax, 0.2 + 1.6 * esc, az);
    tronco.castShadow = true;
    scene.add(tronco);

    const copa = new THREE.Mesh(
      new THREE.DodecahedronGeometry(1.9 * esc, 1),
      new THREE.MeshStandardMaterial({{ color: MODO_NOITE ? 0x064e3b : 0x16a34a, roughness: 0.8 }})
    );
    copa.position.set(ax, 3.8 * esc, az);
    copa.castShadow = true;
    scene.add(copa);
    obstaculosCilindricos.push({{ x: ax, z: az, r: 0.65 * esc }});
  }});

  const postes = [[9.5, 21.0, 0.6], [18.5, 16.2, 0.4], [41.0, 16.5, 0.0], [54.5, 17.5, -0.5]];
  postes.forEach(([px, pz, ox]) => {{
    const mastro = new THREE.Mesh(
      new THREE.CylinderGeometry(0.08, 0.1, 6.2, 8),
      new THREE.MeshStandardMaterial({{ color: 0x334155 }})
    );
    mastro.position.set(px, 3.3, pz);
    mastro.castShadow = true;
    scene.add(mastro);

    const lum = new THREE.Mesh(
      new THREE.SphereGeometry(0.28, 12, 12),
      new THREE.MeshBasicMaterial({{ color: MODO_NOITE ? 0xfef08a : 0xe2e8f0 }})
    );
    lum.position.set(px + ox, 6.3, pz + 0.4);
    scene.add(lum);

    if (MODO_NOITE) {{
      const pLight = new THREE.PointLight(0xfde047, 1.8, 22);
      pLight.position.set(px + ox, 6.0, pz + 0.4);
      scene.add(pLight);
    }}
    obstaculosCilindricos.push({{ x: px, z: pz, r: 0.45 }});
  }});

  // Abrigo metálico visual
  const abrigoTeto = new THREE.Mesh(
    new THREE.BoxGeometry(4.6, 0.2, 3.3),
    new THREE.MeshStandardMaterial({{ color: 0x94a3b8, metalness: 0.6, roughness: 0.3 }})
  );
  abrigoTeto.position.set(46.7, 2.6, 18.8);
  abrigoTeto.rotation.z = -0.12;
  abrigoTeto.castShadow = true;
  scene.add(abrigoTeto);

  // Carros em movimento na Via (LETAL AO TOQUE!)
  function criarCarro(x, z, rotY, cor, velX) {{
    const g = new THREE.Group();
    const chassi = new THREE.Mesh(new THREE.BoxGeometry(2.0, 0.65, 4.3), new THREE.MeshStandardMaterial({{ color: cor, metalness: 0.5, roughness: 0.25 }}));
    chassi.position.y = 0.55; chassi.castShadow = true; g.add(chassi);
    const cabine = new THREE.Mesh(new THREE.BoxGeometry(1.75, 0.55, 2.2), new THREE.MeshStandardMaterial({{ color: 0x0f172a, roughness: 0.15 }}));
    cabine.position.set(0, 1.1, -0.15); cabine.castShadow = true; g.add(cabine);
    g.position.set(x, 0.1, z);
    g.rotation.y = rotY;
    scene.add(g);
    return {{ mesh: g, velX: velX }};
  }}
  const carros = [
    criarCarro(12.0, 7.2, Math.PI / 2, 0xcbd5e1, 13.5),
    criarCarro(56.0, 10.6, -Math.PI / 2, 0xdc2626, -15.5)
  ];

  // ==========================================================================
  // PERSONAGEM ARTICULADO ESTILO ROBLOX + RAGDOLL DE MORTE
  // ==========================================================================
  const player = new THREE.Group();
  const matPele = new THREE.MeshStandardMaterial({{ color: 0xfcd34d }});
  const matTroncoP = new THREE.MeshStandardMaterial({{ color: 0x2563eb }});
  const matCalca = new THREE.MeshStandardMaterial({{ color: 0x1e293b }});

  const torso = new THREE.Mesh(new THREE.BoxGeometry(0.75, 0.85, 0.4), matTroncoP);
  torso.position.set(0, 1.15, 0); torso.castShadow = true; player.add(torso);

  const cabeca = new THREE.Mesh(new THREE.BoxGeometry(0.48, 0.48, 0.48), matPele);
  cabeca.position.set(0, 1.85, 0); cabeca.castShadow = true; player.add(cabeca);

  const pernaEsq = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.75, 0.34), matCalca);
  pernaEsq.position.set(-0.2, 0.38, 0); pernaEsq.castShadow = true; player.add(pernaEsq);

  const pernaDir = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.75, 0.34), matCalca);
  pernaDir.position.set(0.2, 0.38, 0); pernaDir.castShadow = true; player.add(pernaDir);

  const bracoEsq = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.78, 0.28), matPele);
  bracoEsq.position.set(-0.54, 1.15, 0); bracoEsq.castShadow = true; player.add(bracoEsq);

  const bracoDir = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.78, 0.28), matPele);
  bracoDir.position.set(0.54, 1.15, 0); bracoDir.castShadow = true; player.add(bracoDir);

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

  let isDead = false;
  let deathCount = 0;
  const wastedOverlay = document.getElementById('wasted-overlay');
  const deathReasonEl = document.getElementById('wasted-sub');

  function triggerDeath(motivo, impactoVel) {{
    if (isDead) return;
    isDead = true;
    deathCount++;
    deathReasonEl.textContent = motivo;
    wastedOverlay.style.display = 'flex';

    // Explode o boneco estilo Roblox OOF / Dumb Ways to Die
    partesBoneco.forEach((p) => {{
      p.vel.set(
        (Math.random() - 0.5) * 9.0 + impactoVel.x,
        4.5 + Math.random() * 6.5,
        (Math.random() - 0.5) * 9.0 + impactoVel.z
      );
      p.rotVel.set(
        (Math.random() - 0.5) * 12,
        (Math.random() - 0.5) * 12,
        (Math.random() - 0.5) * 12
      );
    }});
  }}

  window.resetPlayer = () => {{
    isDead = false;
    wastedOverlay.style.display = 'none';
    player.position.set(14.0, 0.2, 14.0);
    velY = 0;
    partesBoneco.forEach((p) => {{
      p.mesh.position.set(p.posOrig[0], p.posOrig[1], p.posOrig[2]);
      p.mesh.rotation.set(0, 0, 0);
    }});
  }};

  const keys = {{}};
  window.addEventListener('keydown', (e) => {{
    keys[e.code] = true;
    if (e.code === 'KeyR') window.resetPlayer();
  }});
  window.addEventListener('keyup', (e) => {{ keys[e.code] = false; }});

  let camYaw = 0.65;
  let camPitch = 0.32;
  let camDist = 7.5;
  let camMode = 'third';
  let isDragging = false;
  let prevMouse = {{ x: 0, y: 0 }};

  renderer.domElement.addEventListener('mousedown', (e) => {{
    isDragging = true;
    prevMouse = {{ x: e.clientX, y: e.clientY }};
  }});
  window.addEventListener('mouseup', () => {{ isDragging = false; }});
  window.addEventListener('mousemove', (e) => {{
    if (!isDragging || camMode === 'iso') return;
    const dx = e.clientX - prevMouse.x;
    const dy = e.clientY - prevMouse.y;
    camYaw -= dx * 0.007;
    camPitch = Math.max(0.08, Math.min(1.35, camPitch + dy * 0.006));
    prevMouse = {{ x: e.clientX, y: e.clientY }};
  }});
  renderer.domElement.addEventListener('wheel', (e) => {{
    camDist = Math.max(3.0, Math.min(28.0, camDist + e.deltaY * 0.008));
  }});

  window.setCameraMode = (mode) => {{ camMode = mode; }};

  function verificarColisaoSolida(nx, nz) {{
    // 1. Bloco 734
    const inBox = (nx > 14.5 && nx < 59.5 && nz > 19.5 && nz < 53.5);
    const distArc = Math.hypot(nx - 37.0, nz - 42.0);
    if (inBox && (!(nx < 37.0 && nz < 42.0) || distArc < 22.4)) return true;

    // 2. Caixas (Fonte e Abrigo)
    for (const c of obstaculosCaixas) {{
      if (nx > c.minX && nx < c.maxX && nz > c.minZ && nz < c.maxZ) return true;
    }}

    // 3. Cilindros (Árvores e Postes)
    for (const cil of obstaculosCilindricos) {{
      if (Math.hypot(nx - cil.x, nz - cil.z) < (cil.r + 0.35)) return true;
    }}
    return false;
  }}

  let velY = 0;
  let walkCycle = 0;
  const clock = new THREE.Clock();
  const coordsEl = document.getElementById('coords');

  function animate() {{
    requestAnimationFrame(animate);
    const dt = Math.min(clock.getDelta(), 0.05);

    laminaAgua.position.y = 0.50 + Math.sin(clock.elapsedTime * 3.0) * 0.015;

    // Atualiza carros na pista
    carros.forEach((c) => {{
      c.mesh.position.x += c.velX * dt;
      if (c.velX > 0 && c.mesh.position.x > 66.0) c.mesh.position.x = 4.0;
      if (c.velX < 0 && c.mesh.position.x < 4.0) c.mesh.position.x = 66.0;

      // Checa atropelamento
      if (!isDead) {{
        const dxC = Math.abs(player.position.x - c.mesh.position.x);
        const dzC = Math.abs(player.position.z - c.mesh.position.z);
        if (dxC < 2.25 && dzC < 1.15 && player.position.y < 1.5) {{
          triggerDeath("Atropelado na Via do CT! Olhe para os dois lados.", new THREE.Vector3(c.velX * 0.7, 3.0, (Math.random()-0.5)*5));
        }}
      }}
    }});

    if (isDead) {{
      // Física das peças desmontadas
      partesBoneco.forEach((p) => {{
        p.vel.y -= 16.0 * dt;
        p.mesh.position.addScaledVector(p.vel, dt);
        p.mesh.rotation.x += p.rotVel.x * dt;
        p.mesh.rotation.y += p.rotVel.y * dt;
        p.mesh.rotation.z += p.rotVel.z * dt;
        if (p.mesh.position.y < 0.2) {{
          p.mesh.position.y = 0.2;
          p.vel.y *= -0.45;
          p.vel.x *= 0.85;
          p.vel.z *= 0.85;
        }}
      }});
    }} else {{
      // Checa toque na Escultura Cinética (Obra de Arte)
      const distEsc = Math.hypot(player.position.x - 15.0, player.position.z - 22.0);
      if (distEsc < 2.15 && player.position.y < 3.8) {{
        triggerDeath("Eletrocutado / Esmagado ao tocar a Escultura Cinética!", new THREE.Vector3((player.position.x - 15.0)*4, 3.5, (player.position.z - 22.0)*4));
      }}

      let moveForward = 0;
      let moveRight = 0;
      if (keys['KeyW'] || keys['ArrowUp']) moveForward += 1;
      if (keys['KeyS'] || keys['ArrowDown']) moveForward -= 1;
      if (keys['KeyD'] || keys['ArrowRight']) moveRight += 1;
      if (keys['KeyA'] || keys['ArrowLeft']) moveRight -= 1;

      const speed = (keys['ShiftLeft'] || keys['ShiftRight']) ? 10.5 : 5.5;
      const isMoving = (moveForward !== 0 || moveRight !== 0);

      if (isMoving) {{
        const forwardVec = new THREE.Vector3(-Math.sin(camYaw), 0, -Math.cos(camYaw));
        const rightVec = new THREE.Vector3(Math.cos(camYaw), 0, -Math.sin(camYaw));
        const dir = new THREE.Vector3()
          .addScaledVector(forwardVec, moveForward)
          .addScaledVector(rightVec, moveRight)
          .normalize();

        const nextX = THREE.MathUtils.clamp(player.position.x + dir.x * speed * dt, 3.0, 67.0);
        const nextZ = THREE.MathUtils.clamp(player.position.z + dir.z * speed * dt, 3.0, 58.0);

        if (!verificarColisaoSolida(nextX, player.position.z)) player.position.x = nextX;
        if (!verificarColisaoSolida(player.position.x, nextZ)) player.position.z = nextZ;

        player.rotation.y = Math.atan2(dir.x, dir.z);

        walkCycle += dt * speed * 2.2;
        const swing = Math.sin(walkCycle) * 0.55;
        pernaEsq.rotation.x = swing;
        pernaDir.rotation.x = -swing;
        bracoEsq.rotation.x = -swing * 0.8;
        bracoDir.rotation.x = swing * 0.8;
      }} else {{
        pernaEsq.rotation.x *= 0.8;
        pernaDir.rotation.x *= 0.8;
        bracoEsq.rotation.x *= 0.8;
        bracoDir.rotation.x *= 0.8;
      }}

      const groundY = player.position.z >= 16.0 ? 0.2 : 0.1;
      if (keys['Space'] && Math.abs(player.position.y - groundY) < 0.05) {{
        velY = 6.8;
      }}
      velY -= 18.0 * dt;
      player.position.y += velY * dt;
      if (player.position.y <= groundY) {{
        player.position.y = groundY;
        velY = 0;
      }}
    }}

    const targetLook = new THREE.Vector3(player.position.x, player.position.y + 1.4, player.position.z);
    if (camMode === 'iso') {{
      camera.position.set(player.position.x - 14, player.position.y + 14, player.position.z - 14);
      camera.lookAt(targetLook);
    }} else {{
      const cxCam = targetLook.x + camDist * Math.sin(camYaw) * Math.cos(camPitch);
      const cyCam = targetLook.y + camDist * Math.sin(camPitch);
      const czCam = targetLook.z + camDist * Math.cos(camYaw) * Math.cos(camPitch);
      camera.position.set(cxCam, cyCam, czCam);
      camera.lookAt(targetLook);
    }}

    coordsEl.textContent = `Mortes: ${{deathCount}} | X=${{player.position.x.toFixed(1)}} | Z=${{player.position.z.toFixed(1)}}`;
    renderer.render(scene, camera);
  }}

  window.addEventListener('resize', () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }});

  animate();
</script>
</body>
</html>
"""
  components.html(html_code, height=620)


# ==============================================================================
# BARRA LATERAL E CONTROLES
# ==============================================================================
st.sidebar.header("Controles Gerais")
modo_noite = st.sidebar.toggle(
    "Modo Noturno (Postes + Fonte Iluminada)", value=False
)

st.sidebar.divider()
st.sidebar.subheader("Câmera e Projeções (Sincroniza 3D e Render)")
estilo_preview = st.sidebar.radio(
    "Estilo da Inspeção 3D",
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
renderer_temp = carregar_motor(280)
cfg_atual = renderer_temp.presets[preset_escolhido].copy()

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
st.sidebar.subheader("Qualidade e Otimização do Ray Casting")
resolucao = st.sidebar.select_slider(
    "Resolução do Ray Casting", options=[200, 280, 360, 480], value=360
)
suavizar_aa_rc = st.sidebar.toggle("Suavização Anti-Aliasing (Remove Pixelado)", value=True)
incluir_objs_rc = st.sidebar.toggle(
    "Incluir Malhas .OBJ no Render (Mais pesado)", value=False
)
ativar_sombras_rc = st.sidebar.toggle("Sombras Projetadas", value=True)
ativar_reflexos_rc = st.sidebar.toggle("Reflexos (Vidro / Água / Aço)", value=False)
ativar_caustics_rc = st.sidebar.toggle("Cáusticas da Fonte", value=True)

renderer = carregar_motor(resolucao)

aba_walk, aba_3d, aba_rc = st.tabs([
    "1. Walkthrough 3D (Jogo / WASD)",
    "2. Inspeção 3D (Picking & Câmeras)",
    "3. Render Final (Ray Casting)",
])

with aba_walk:
  st.caption(
      "Clique na cena para ativar o teclado (W, A, S, D, Shift, Espaço). Encostar nos carros em movimento ou na Escultura Cinética ativa o modo Dumb Ways to Die!"
  )
  renderizar_walkthrough_webgl(modo_noite=modo_noite)

with aba_3d:
  st.caption(
      "Clique sobre qualquer objeto na cena 3D abaixo para inspecionar seu nome e coordenadas (Picking). Os templates de câmera da barra lateral atualizam esta visão."
  )
  fig_3d = construir_figura_3d_interativa(
      renderer,
      estilo=estilo_preview,
      modo_noite=modo_noite,
      cfg_cam=cfg_atual,
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
    st.success(
        f"**Objeto Selecionado (Picking):** {nome_obj}  |  **Coordenadas Mundo:** X = `{px_c:.2f}m`, Y (Altura) = `{py_c:.2f}m`, Z = `{pz_c:.2f}m`"
    )

with aba_rc:
  if st.button(
      "RENDERIZAR RAY CASTING AGORA", type="primary", width="stretch"
  ):
    with st.spinner("Renderizando cena..."):
      img, titulo_render = renderer.renderizar_config(
          cfg_atual,
          modo_noite=modo_noite,
          ativar_caustics=ativar_caustics_rc,
          ativar_reflexos=ativar_reflexos_rc,
          ativar_sombras=ativar_sombras_rc,
          incluir_objs=incluir_objs_rc,
          suavizar_aa=suavizar_aa_rc,
      )
      st.session_state["img_render"] = img
      st.session_state["tit_render"] = titulo_render

  if "img_render" in st.session_state:
    st.subheader(st.session_state["tit_render"])
    st.image(st.session_state["img_render"], width="stretch")
  else:
    st.info(
        "Selecione o template de câmera e os efeitos na barra lateral e clique no botão acima para renderizar."
    )
