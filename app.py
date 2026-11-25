from motor_ct import RenderizadorCT
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Bloco 734 CT + Dumb Ways Parkour + Inspeção 3D", layout="wide"
)
st.title("Visualizador 3D, Dumb Ways Parkour & Ray Casting — Bloco 734 (CT)")


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
  """Câmera com aspecto isométrico real (sem achatamento) e elevação travada >= 0°."""
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
  # Escala uniforme pelo maior eixo (66m) para preservar proporção real sem achatar
  centro_cena = np.array([35.0, 31.0, 3.5])
  escala_unif = 36.0

  eye_pl = np.array([eye[0], eye[2], max(eye[1], 1.2)])
  at_pl = np.array([at[0], at[2], max(at[1], 1.0)])

  cam_eye = (eye_pl - centro_cena) / escala_unif
  cam_center = (at_pl - centro_cena) / escala_unif
  # Garante que o olho da câmera nunca entre no quadrante inferior (z < 0.05)
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

  # 2. Bloco 734
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

  # Janelas, Porta, Escada e Letreiro 734
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
  camera_plotly = converter_camera_para_plotly(
      cfg_cam, azimute_deg=azimute_deg, elevacao_deg=elevacao_deg
  )

  # Proporção real 1:1:1 (aspectmode="data" sem faixa gigante abaixo de Y=0)
  fig.update_layout(
      height=620,
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
    position: absolute; top: 12px; left: 12px; z-index: 10;
    background: rgba(15, 23, 42, 0.88); color: #F8FAFC;
    padding: 12px 15px; border-radius: 10px; border: 1px solid rgba(148, 163, 184, 0.35);
    backdrop-filter: blur(6px); max-width: 390px; font-size: 12.5px; line-height: 1.45;
  }}
  #hud b {{ color: #38BDF8; }}
  .btn-cam {{
    margin-top: 7px; margin-right: 5px; padding: 5px 9px;
    background: #1E293B; color: #F8FAFC; border: 1px solid #475569;
    border-radius: 6px; cursor: pointer; font-size: 11.5px; font-weight: 600;
  }}
  .btn-cam:hover {{ background: #0284C7; border-color: #38BDF8; }}

  #stations-panel {{
    position: absolute; top: 12px; right: 12px; z-index: 10;
    background: rgba(15, 23, 42, 0.88); color: #F8FAFC;
    padding: 12px 15px; border-radius: 10px; border: 1px solid rgba(148, 163, 184, 0.35);
    backdrop-filter: blur(6px); width: 270px; font-size: 12px; line-height: 1.55;
  }}
  .st-title {{ font-weight: 800; color: #FACC15; font-size: 13px; margin-bottom: 6px; }}
  .st-item {{ display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.08); padding: 2px 0; }}
  .st-done {{ color: #4ADE80; font-weight: 700; }}
  .st-todo {{ color: #94A3B8; }}

  #prompt-box {{
    display: none; position: absolute; bottom: 52px; left: 50%; transform: translateX(-50%); z-index: 15;
    background: rgba(2, 132, 199, 0.92); color: #FFFFFF; padding: 9px 18px;
    border-radius: 999px; font-weight: 700; font-size: 13.5px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.5); border: 1px solid #7DD3FC;
  }}

  #toxic-bar-wrap {{
    display: none; position: absolute; bottom: 95px; left: 50%; transform: translateX(-50%); z-index: 15;
    background: rgba(15,23,42,0.9); padding: 8px 14px; border-radius: 8px; border: 1px solid #22D3EE; width: 240px; text-align: center;
  }}
  #toxic-bar {{ height: 10px; background: #06B6D4; width: 0%; border-radius: 4px; transition: width 0.05s; }}

  #coords {{
    position: absolute; bottom: 12px; right: 12px; z-index: 10;
    background: rgba(15, 23, 42, 0.85); color: #E2E8F0;
    padding: 6px 12px; border-radius: 8px; font-family: monospace; font-size: 12px;
  }}

  #wasted-overlay {{
    display: none; position: absolute; inset: 0; z-index: 30;
    background: radial-gradient(circle, rgba(127,29,29,0.52) 0%, rgba(15,23,42,0.90) 100%);
    flex-direction: column; align-items: center; justify-content: center; text-align: center;
  }}
  #wasted-title {{ font-size: 52px; font-weight: 900; color: #EF4444; text-shadow: 0 4px 20px rgba(0,0,0,0.9); letter-spacing: 2px; }}
  #wasted-sub {{ font-size: 19px; color: #FCA5A5; margin-top: 8px; font-weight: 600; max-width: 520px; }}
  #btn-respawn {{
    margin-top: 22px; padding: 12px 26px; font-size: 15px; font-weight: 700;
    background: #EF4444; color: white; border: none; border-radius: 8px; cursor: pointer;
  }}

  #victory-banner {{
    display: none; position: absolute; top: 80px; left: 50%; transform: translateX(-50%); z-index: 25;
    background: linear-gradient(135deg, #16A34A, #0284C7); color: white;
    padding: 14px 26px; border-radius: 12px; font-weight: 800; font-size: 17px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.6); text-align: center;
  }}
  #flash-foto {{
    display: none; position: absolute; inset: 0; z-index: 28; background: white; opacity: 0.85; pointer-events: none;
  }}
</style>
</head>
<body>
<div id="hud">
  <div style="font-size:14.5px; font-weight:800; margin-bottom:4px;">Walkthrough & Parkour — Bloco 734</div>
  <div><b>WASD / Setas</b>: Andar &nbsp;|&nbsp; <b>Shift</b>: Correr &nbsp;|&nbsp; <b>Espaço</b>: Pular</div>
  <div><b>Segurar E</b>: Escalar Árvore, Poste ou Escada &nbsp;|&nbsp; <b>F</b>: Fotografar</div>
  <div>
    <button class="btn-cam" onclick="setCameraMode('third')">3ª Pessoa</button>
    <button class="btn-cam" onclick="setCameraMode('iso')">Isométrica</button>
    <button class="btn-cam" onclick="resetPlayer()">Renascer (R)</button>
  </div>
</div>

<div id="stations-panel">
  <div class="st-title">ESTAÇÕES PARKOUR (0/6)</div>
  <div class="st-item"><span>1. Foto da Escultura (F)</span><span id="st-foto" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>2. Fonte Rápida (&lt;3.5s)</span><span id="st-fonte" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>3. Escalar Poste (E)</span><span id="st-poste" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>4. Escalar Árvore (E)</span><span id="st-arvore" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>5. Atravessar Rua & Voltar</span><span id="st-rua" class="st-todo">PENDENTE</span></div>
  <div class="st-item"><span>6. Telhado pela Escada (E)</span><span id="st-telhado" class="st-todo">PENDENTE</span></div>
</div>

<div id="prompt-box">Ação disponível</div>
<div id="toxic-bar-wrap">
  <div style="color:#67E8F9; font-size:11px; font-weight:700; margin-bottom:4px;">ÁGUA CONTAMINADA! SAIA JÁ!</div>
  <div id="toxic-bar"></div>
</div>
<div id="victory-banner">PARABÉNS! VOCÊ COMPLETOU TODAS AS 6 ESTAÇÕES DO BLOCO 734!</div>
<div id="flash-foto"></div>

<div id="wasted-overlay">
  <div id="wasted-title">DUMB WAYS TO DIE</div>
  <div id="wasted-sub">Você encontrou um fim trágico!</div>
  <button id="btn-respawn" onclick="resetPlayer()">RENASCER AGORA (Tecla R)</button>
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
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  document.body.appendChild(renderer.domElement);

  scene.add(new THREE.HemisphereLight(MODO_NOITE ? 0x1e293b : 0xffffff, MODO_NOITE ? 0x0f172a : 0x64748b, MODO_NOITE ? 0.5 : 0.85));
  const dirLight = new THREE.DirectionalLight(MODO_NOITE ? 0x3b82f6 : 0xfffbeb, MODO_NOITE ? 0.25 : 1.15);
  dirLight.position.set(12, 32, 10);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.width = 2048;
  dirLight.shadow.mapSize.height = 2048;
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

  // Plataforma Objetivo do Outro Lado da Rua (Estação 5)
  const platRua = new THREE.Mesh(new THREE.CylinderGeometry(1.8, 1.8, 0.12, 24), new THREE.MeshStandardMaterial({{ color: 0x22c55e, emissive: 0x15803d, emissiveIntensity: 0.4 }}));
  platRua.position.set(27.0, 0.12, 3.2); scene.add(platRua);

  // Zona Fotográfica da Escultura (Estação 1)
  const platFoto = new THREE.Mesh(new THREE.RingGeometry(1.0, 1.35, 24), new THREE.MeshBasicMaterial({{ color: 0x38bdf8, side: THREE.DoubleSide }}));
  platFoto.rotation.x = -Math.PI / 2;
  platFoto.position.set(11.5, 0.22, 18.0); scene.add(platFoto);

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

  // Escada Lateral para o Telhado (X=59.35, Z=24.0)
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

  // Janelas
  for (let i = 0; i < 9; i++) {{
    const ang = Math.PI + ((i + 0.5) / 9) * (Math.PI * 0.44) + 0.08;
    const vx = cx + (R + 0.05) * Math.cos(ang);
    const vz = cz + (R + 0.05) * Math.sin(ang);
    const janela = new THREE.Mesh(new THREE.BoxGeometry(1.7, 1.9, 0.15), new THREE.MeshStandardMaterial({{ color: 0x0f172a, metalness: 0.85, roughness: 0.15 }}));
    janela.position.set(vx, 2.3, vz); janela.lookAt(cx, 2.3, cz); scene.add(janela);
  }}

  // Fonte Contaminada (Parada)
  const bordaFonte = new THREE.Mesh(new THREE.BoxGeometry(15.4, 0.34, 6.4), new THREE.MeshStandardMaterial({{ color: 0x1e293b, roughness: 0.4 }}));
  bordaFonte.position.set(27, 0.37, 19.8); bordaFonte.castShadow = true; scene.add(bordaFonte);
  const laminaAgua = new THREE.Mesh(
    new THREE.BoxGeometry(14.8, 0.1, 5.8),
    new THREE.MeshStandardMaterial({{ color: 0x14b8a6, emissive: 0x0d9488, emissiveIntensity: 0.35, roughness: 0.15 }})
  );
  laminaAgua.position.set(27, 0.51, 19.8); scene.add(laminaAgua);

  // Escultura Cinética (Letal ao toque)
  const grupoEsc = new THREE.Group();
  grupoEsc.position.set(15.0, 0.2, 22.0);
  grupoEsc.rotation.y = THREE.MathUtils.degToRad(-24);
  const baseEsc = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.28, 3.4), new THREE.MeshStandardMaterial({{ color: 0x64748b }}));
  baseEsc.position.y = 0.14; grupoEsc.add(baseEsc);
  for (let k = 0; k < 5; k++) {{
    const fc = k - 2; const L = 2.35 * (1 - 0.055 * fc);
    const fg = new THREE.Group();
    [[0, L/2, L, 0.14], [0, -L/2, L, 0.14], [L/2, 0, 0.14, L], [-L/2, 0, 0.14, L]].forEach(([bx, by, bw, bh]) => {{
      const bar = new THREE.Mesh(new THREE.BoxGeometry(bw, bh, 0.16), new THREE.MeshStandardMaterial({{ color: 0xf8fafc, metalness: 0.3 }}));
      bar.position.set(bx, by, 0); fg.add(bar);
    }});
    fg.rotation.z = Math.PI / 4;
    fg.position.set(fc * 0.22, 0.28 + (L / Math.SQRT2) + 0.05, (k - 2) * 0.58);
    grupoEsc.add(fg);
  }}
  scene.add(grupoEsc);

  // Árvores e Postes com Fio Descascado
  const arvores = [[10.5, 28.5, 1.15], [11.5, 38.0, 1.25], [37.5, 17.5, 0.9]];
  arvores.forEach(([ax, az, esc]) => {{
    const tronco = new THREE.Mesh(new THREE.CylinderGeometry(0.22 * esc, 0.3 * esc, 3.6 * esc, 10), new THREE.MeshStandardMaterial({{ color: 0x5c3a21 }}));
    tronco.position.set(ax, 0.2 + 1.8 * esc, az); tronco.castShadow = true; scene.add(tronco);
    const copa = new THREE.Mesh(new THREE.DodecahedronGeometry(1.95 * esc, 1), new THREE.MeshStandardMaterial({{ color: MODO_NOITE ? 0x064e3b : 0x16a34a }}));
    copa.position.set(ax, 4.1 * esc, az); copa.castShadow = true; scene.add(copa);
  }});

  const postes = [[9.5, 21.0, 0.6], [18.5, 16.2, 0.4], [41.0, 16.5, 0.0], [54.5, 17.5, -0.5]];
  const faiscas = [];
  postes.forEach(([px, pz, ox]) => {{
    const mastro = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.1, 6.2, 8), new THREE.MeshStandardMaterial({{ color: 0x334155 }}));
    mastro.position.set(px, 3.3, pz); scene.add(mastro);
    const lum = new THREE.Mesh(new THREE.SphereGeometry(0.32, 12, 12), new THREE.MeshBasicMaterial({{ color: 0xfef08a }}));
    lum.position.set(px + ox, 6.3, pz + 0.4); scene.add(lum);
    // Fio desencapado piscando no topo do poste
    const fio = new THREE.Mesh(new THREE.SphereGeometry(0.18, 8, 8), new THREE.MeshBasicMaterial({{ color: 0x38bdf8 }}));
    fio.position.set(px, 5.9, pz); scene.add(fio);
    faiscas.push(fio);
  }});

  // Carros em movimento na Via
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

  // ==========================================================================
  // BONECO ARTICULADO + RAGDOLL DE MORTE
  // ==========================================================================
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

  // Estado do Jogo e Estações
  let isDead = false;
  let deathCount = 0;
  let velY = 0;
  let walkCycle = 0;
  let tempoNaFonte = 0;
  let subiuArvore = false;
  let subiuPoste = false;
  let atravessouRua = false;
  let estavaNoTelhado = false;

  const checks = {{ foto: false, fonte: false, poste: false, arvore: false, rua: false, telhado: false }};
  function marcarCheck(chave, elId) {{
    if (checks[chave]) return;
    checks[chave] = true;
    const el = document.getElementById(elId);
    el.textContent = "CHECK ✓";
    el.className = "st-done";
    const total = Object.values(checks).filter(Boolean).length;
    document.querySelector('.st-title').textContent = `ESTAÇÕES PARKOUR (${{total}}/6)`;
    if (total === 6) {{
      document.getElementById('victory-banner').style.display = 'block';
    }}
  }}

  function triggerDeath(motivo, impactoVel) {{
    if (isDead) return;
    isDead = true;
    deathCount++;
    document.getElementById('wasted-sub').textContent = motivo;
    document.getElementById('wasted-overlay').style.display = 'flex';
    document.getElementById('toxic-bar-wrap').style.display = 'none';

    partesBoneco.forEach((p) => {{
      p.vel.set((Math.random() - 0.5) * 9 + impactoVel.x, 4.5 + Math.random() * 6, (Math.random() - 0.5) * 9 + impactoVel.z);
      p.rotVel.set((Math.random() - 0.5) * 12, (Math.random() - 0.5) * 12, (Math.random() - 0.5) * 12);
    }});
  }}

  window.resetPlayer = () => {{
    isDead = false;
    tempoNaFonte = 0;
    subiuArvore = false;
    subiuPoste = false;
    estavaNoTelhado = false;
    document.getElementById('wasted-overlay').style.display = 'none';
    document.getElementById('toxic-bar-wrap').style.display = 'none';
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
    if (e.code === 'KeyF' && !isDead) {{
      const dFoto = Math.hypot(player.position.x - 11.5, player.position.z - 18.0);
      if (dFoto < 2.2) {{
        const fl = document.getElementById('flash-foto');
        fl.style.display = 'block';
        setTimeout(() => {{ fl.style.display = 'none'; }}, 140);
        marcarCheck('foto', 'st-foto');
      }}
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

    // Anima faíscas dos fios descascados nos postes
    faiscas.forEach((f, idx) => {{
      f.visible = Math.sin(clock.elapsedTime * 22 + idx) > 0.2;
    }});

    // Carros em movimento
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

      // 1. Checa Escultura Cinética (foto ou morte)
      if (Math.hypot(player.position.x - 11.5, player.position.z - 18.0) < 2.2) {{
        promptTexto = "Aperte [F] para tirar foto da Escultura Cinética!";
      }}
      if (Math.hypot(player.position.x - 15.0, player.position.z - 22.0) < 2.1 && player.position.y < 3.8) {{
        triggerDeath("Eletrocutado / Esmagado ao encostar na Escultura Cinética!", new THREE.Vector3(0, 4, -4));
      }}

      // 2. Checa Fonte Contaminada (entra -> sai rápido = check; demora = morre)
      const naAgua = (player.position.x > 19.6 && player.position.x < 34.4 && player.position.z > 16.9 && player.position.z < 22.7 && player.position.y < 0.9);
      if (naAgua) {{
        tempoNaFonte += dt;
        toxicWrap.style.display = 'block';
        toxicBar.style.width = Math.min(100, (tempoNaFonte / 3.5) * 100) + '%';
        if (tempoNaFonte >= 3.5) {{
          triggerDeath("Intoxicado pela água parada contaminada da fonte!", new THREE.Vector3(0, 3, 0));
        }}
      }} else {{
        if (tempoNaFonte > 0.25 && tempoNaFonte < 3.5) {{
          marcarCheck('fonte', 'st-fonte');
        }}
        tempoNaFonte = 0;
        toxicWrap.style.display = 'none';
      }}

      // 3. Checa Travessia da Rua (Estação 5)
      if (Math.hypot(player.position.x - 27.0, player.position.z - 3.2) < 2.0) {{
        atravessouRua = true;
        promptTexto = "Objetivo alcançado! Agora volte em segurança para a calçada!";
      }}
      if (atravessouRua && player.position.z >= 16.5) {{
        marcarCheck('rua', 'st-rua');
      }}

      // 4. Escalada em Árvore, Poste e Escada do Telhado (Tecla E)
      let escalando = false;

      // Escada do Telhado (X=59.35, Z=24.0)
      if (Math.hypot(player.position.x - 59.35, player.position.z - 24.0) < 1.35) {{
        promptTexto = "Segure [E] para subir a escada até o telhado!";
        if (keys['KeyE']) {{
          escalando = true;
          velY = 0;
          player.position.y += 4.2 * dt;
          if (player.position.y >= 6.85) {{
            player.position.y = 6.85;
            player.position.x = 58.2; // Coloca em cima do telhado
          }}
        }}
      }}

      // Postes (subir = check ao descer; tocar na luz/fio no topo = eletrocutado!)
      postes.forEach(([px, pz]) => {{
        if (Math.hypot(player.position.x - px, player.position.z - pz) < 1.1) {{
          promptTexto = "Segure [E] para escalar o poste (Cuidado com o fio descascado no topo!)";
          if (keys['KeyE']) {{
            escalando = true;
            velY = 0;
            player.position.y += 3.6 * dt;
            if (player.position.y > 2.8) subiuPoste = true;
            if (player.position.y >= 5.5) {{
              triggerDeath("Eletrocutado no fio descascado do poste de iluminação!", new THREE.Vector3((Math.random()-0.5)*6, 2, -4));
            }}
          }}
        }}
      }});
      if (subiuPoste && !escalando && player.position.y <= 0.35) {{
        marcarCheck('poste', 'st-poste');
        subiuPoste = false;
      }}

      // Árvores (subir até a copa e descer = check; subir sem parar = galho quebra e despenca!)
      arvores.forEach(([ax, az, esc]) => {{
        if (Math.hypot(player.position.x - ax, player.position.z - az) < 1.3) {{
          promptTexto = "Segure [E] para escalar a árvore (Não suba além do topo!)";
          if (keys['KeyE']) {{
            escalando = true;
            velY = 0;
            player.position.y += 3.4 * dt;
            if (player.position.y > 2.6) subiuArvore = true;
            if (player.position.y >= 4.7 * esc) {{
              triggerDeath("Subiu demais! O galho quebrou e você despencou lá de cima!", new THREE.Vector3(2, -6, -2));
            }}
          }}
        }}
      }});
      if (subiuArvore && !escalando && player.position.y <= 0.35) {{
        marcarCheck('arvore', 'st-arvore');
        subiuArvore = false;
      }}

      if (promptTexto) {{
        promptBox.textContent = promptTexto;
        promptBox.style.display = 'block';
      }} else {{
        promptBox.style.display = 'none';
      }}

      // Movimento Horizontal WASD
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

        // Só bloqueia parede do prédio se o boneco estiver abaixo da laje do telhado (y < 6.5)
        const bateuParedeX = (player.position.y < 6.4) && dentroPlantaBloco(nextX, player.position.z);
        const bateuParedeZ = (player.position.y < 6.4) && dentroPlantaBloco(player.position.x, nextZ);

        if (!bateuParedeX) player.position.x = nextX;
        if (!bateuParedeZ) player.position.z = nextZ;

        player.rotation.y = Math.atan2(dir.x, dir.z);
        walkCycle += dt * speed * 2.2;
        const swing = Math.sin(walkCycle) * 0.55;
        pernaEsq.rotation.x = swing; pernaDir.rotation.x = -swing;
        bracoEsq.rotation.x = -swing * 0.8; bracoDir.rotation.x = swing * 0.8;
      }} else if (escalando) {{
        walkCycle += dt * 10.0;
        const cl = Math.sin(walkCycle) * 0.7;
        bracoEsq.rotation.x = Math.PI + cl; bracoDir.rotation.x = Math.PI - cl;
        pernaEsq.rotation.x = -cl * 0.5; pernaDir.rotation.x = cl * 0.5;
      }} else {{
        pernaEsq.rotation.x *= 0.8; pernaDir.rotation.x *= 0.8;
        bracoEsq.rotation.x *= 0.8; bracoDir.rotation.x *= 0.8;
      }}

      // Altura do piso (Telhado vs Calçada vs Via)
      const noTelhado = dentroPlantaBloco(player.position.x, player.position.z) && (player.position.y >= 6.4);
      if (noTelhado) {{
        estavaNoTelhado = true;
        marcarCheck('telhado', 'st-telhado');
      }}

      const groundY = noTelhado ? 6.8 : (player.position.z >= 16.0 ? 0.2 : 0.1);

      if (!escalando) {{
        if (keys['Space'] && Math.abs(player.position.y - groundY) < 0.08) {{
          velY = 6.8;
        }}
        velY -= 18.0 * dt;
        player.position.y += velY * dt;
        if (player.position.y <= groundY) {{
          // Se estava no telhado (6.8m) e caiu direto no chão (0.2m), morre de queda!
          if (estavaNoTelhado && groundY < 1.0 && velY < -8.5) {{
            triggerDeath("Caiu do telhado do Bloco 734 sem usar a escada!", new THREE.Vector3(0, 1, 0));
          }}
          player.position.y = groundY;
          velY = 0;
          if (groundY < 1.0) estavaNoTelhado = false;
        }}
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

    coordsEl.textContent = `Mortes: ${{deathCount}} | X=${{player.position.x.toFixed(1)}} | Y=${{player.position.y.toFixed(1)}} | Z=${{player.position.z.toFixed(1)}}`;
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
    "2. Inspeção 3D (Picking & Câmera)",
    "3. Render Final (Ray Casting)",
])

with aba_walk:
  st.caption(
      "Complete as 6 estações de Parkour sem morrer! Segure [E] perto de árvores, postes ou da escada lateral para subir; aperte [F] na marca azul para fotografar a escultura."
  )
  renderizar_walkthrough_webgl(modo_noite=modo_noite)

with aba_3d:
  st.caption(
      "Proporção real 1:1:1 com câmera travada acima do solo (sem capotar para ver embaixo). Clique em qualquer objeto para Picking."
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
    st.success(
        f"**Objeto Selecionado (Picking):** {nome_obj}  |  **Coordenadas:** X = `{px_c:.2f}m`, Altura Y = `{py_c:.2f}m`, Z = `{pz_c:.2f}m`"
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

  if "img_render" in st.session_state:
    st.subheader(st.session_state["tit_render"])
    st.image(st.session_state["img_render"], width="stretch")
  else:
    st.info(
        "Selecione o template de câmera e clique no botão acima para gerar a imagem final."
    )
