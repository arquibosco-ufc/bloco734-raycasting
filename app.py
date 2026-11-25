import io
import time
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

from motor_ct import RenderizadorCT, construir_Mwc

try:
    from streamlit_image_coordinates import streamlit_image_coordinates

    TEM_IMG_COORDS = True
except ImportError:
    TEM_IMG_COORDS = False


# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA & ESTILO GLOBAL
# ==============================================================================
st.set_page_config(
    page_title="Bloco 734 (CT/UFC) — Ray Casting & Simulador Interativo 3D",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container {
        padding-top: 1.0rem;
        padding-bottom: 1.4rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #111827;
        color: #e5e7eb;
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
    }
    .pick-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #38bdf8;
        border-left: 5px solid #38bdf8;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0 12px 0;
        color: #f8fafc;
        box-shadow: 0 8px 24px rgba(2, 6, 23, 0.35);
    }
    .pick-banner-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 4px;
    }
    .pick-banner-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(195px, 1fr));
        gap: 6px 14px;
        font-size: 0.85rem;
        color: #cbd5e1;
        margin-top: 6px;
    }
    .pick-banner-desc {
        font-size: 0.86rem;
        color: #e2e8f0;
        margin-top: 6px;
        padding-top: 6px;
        border-top: 1px solid rgba(148, 163, 184, 0.25);
    }
</style>
""",
    unsafe_allow_html=True,
)


# ==============================================================================
# GERENCIAMENTO DE ESTADO DA SESSÃO
# ==============================================================================
if "resolucao" not in st.session_state:
    st.session_state.resolucao = 500
if "renderer" not in st.session_state or (
    st.session_state.renderer.W != st.session_state.resolucao
):
    st.session_state.renderer = RenderizadorCT(
        largura=st.session_state.resolucao, altura=st.session_state.resolucao
    )
if "ultima_img_arr" not in st.session_state:
    st.session_state.ultima_img_arr = None
if "ultima_legenda" not in st.session_state:
    st.session_state.ultima_legenda = ""
if "ultimo_pick_info" not in st.session_state:
    st.session_state.ultimo_pick_info = None
if "ultima_coord_clicada" not in st.session_state:
    st.session_state.ultima_coord_clicada = None


def renderizar_banner_pick(info, px=None, py=None):
    if not info:
        return
    if not info.get("hit", False):
        st.markdown(
            f"""
            <div class="pick-banner" style="border-left-color:#94a3b8; border-color:#475569;">
                <div class="pick-banner-title" style="color:#cbd5e1;">
                    🎯 Pixel ({px}, {py}) — {info.get('nome', 'Céu / Horizonte')}
                </div>
                <div class="pick-banner-desc">{info.get('descricao', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    coord_txt = (
        f"Pixel ({px}, {py}) • " if (px is not None and py is not None) else ""
    )
    st.markdown(
        f"""
        <div class="pick-banner">
            <div class="pick-banner-title">
                🎯 {coord_txt}{info['nome']} <span style="font-size:0.8rem; color:#94a3b8;">[{info['tipo_primitiva']}]</span>
            </div>
            <div class="pick-banner-grid">
                <div><b>Grupo:</b> {info['grupo']}</div>
                <div><b>Material:</b> {info['material']}</div>
                <div><b>Distância (t):</b> {info['distancia']:.2f} m</div>
                <div><b>Ponto 3D (Mundo):</b> {info['ponto_world']}</div>
                <div><b>Normal N:</b> {info['normal_world']}</div>
                <div><b>Kd / Refl:</b> {info['kd']} | R={info['refletividade']:.2f}</div>
            </div>
            <div class="pick-banner-desc">
                <b>Transformação:</b> {info['transformacao']}<br/>
                <b>Descrição:</b> {info['descricao']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# HTML / THREE.JS / WEB AUDIO — MODO WEBGL COMPLETO & UNIFICADO
# ==============================================================================
def gerar_html_jogo_webgl(
    modo_noite=False, ativar_sol=True, ativar_spots=True
):
    template = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Bloco 734 — Campus Interativo 3D</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; }
  html, body {
    width: 100%; height: 100%; overflow: hidden;
    background: #0b1120; font-family: 'Segoe UI', system-ui, sans-serif;
  }
  #game-wrapper {
    position: relative; width: 100%; height: 730px;
    background: #0b1120; border-radius: 12px; overflow: hidden;
    border: 1px solid #1e293b;
    outline: none;
  }
  #game-wrapper.pseudo-fullscreen {
    position: fixed !important; inset: 0 !important;
    width: 100vw !important; height: 100vh !important;
    z-index: 999999 !important; border-radius: 0 !important;
  }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; outline: none; }
  canvas:active { cursor: grabbing; }

  /* HUD SUPERIOR ESQUERDO */
  #hud-top-left {
    position: absolute; top: 12px; left: 12px;
    display: flex; flex-direction: column; gap: 6px;
    z-index: 12; pointer-events: none;
  }
  .hud-pill {
    background: rgba(15, 23, 42, 0.84);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(148, 163, 184, 0.30);
    color: #f8fafc; padding: 6px 12px; border-radius: 999px;
    font-size: 12px; font-weight: 600;
    display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
    width: fit-content; pointer-events: auto;
    box-shadow: 0 4px 16px rgba(0,0,0,0.35);
  }
  .bar-bg {
    width: 104px; height: 9px; background: rgba(30, 41, 59, 0.95);
    border-radius: 999px; overflow: hidden; border: 1px solid rgba(255,255,255,0.14);
  }
  #energy-fill {
    width: 100%; height: 100%;
    background: linear-gradient(90deg, #22c55e, #38bdf8);
    transition: width 0.18s, background 0.2s;
  }

  /* CHECKLIST COMPACTO DE MINI-QUESTS CT */
  #mini-quest-box {
    background: rgba(15, 23, 42, 0.82);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(56, 189, 248, 0.32);
    border-radius: 10px; padding: 7px 11px;
    font-size: 11.5px; line-height: 1.48;
    color: #e2e8f0; width: fit-content; max-width: 310px;
    pointer-events: auto;
  }
  .mq-done { color: #4ade80; text-decoration: line-through; opacity: 0.85; }
  .mq-todo { color: #e2e8f0; }

  /* HUD SUPERIOR DIREITO */
  #hud-top-right {
    position: absolute; top: 12px; right: 12px;
    display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 7px; z-index: 15;
  }
  .hud-btn {
    background: rgba(15, 23, 42, 0.86);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(56, 189, 248, 0.45);
    color: #f8fafc; padding: 7px 11px; border-radius: 8px;
    font-size: 12px; font-weight: 600; cursor: pointer;
    transition: all 0.15s ease;
  }
  .hud-btn:hover { background: rgba(30, 41, 59, 0.98); border-color: #38bdf8; transform: translateY(-1px); }
  .hud-btn.surv-on {
    background: rgba(127, 29, 29, 0.90);
    border-color: #ef4444; color: #fecaca;
    box-shadow: 0 0 14px rgba(239, 68, 68, 0.4);
  }

  /* PILHA DE ALERTAS DINÂMICOS (IoT, VIGILÂNCIA, CACHORRO, EVENTOS) */
  #alert-stack {
    position: absolute; top: 62px; left: 50%; transform: translateX(-50%);
    display: flex; flex-direction: column; gap: 6px; align-items: center;
    z-index: 24; pointer-events: none; width: min(600px, 88vw);
  }
  .alert-banner {
    padding: 8px 16px; border-radius: 9px; font-weight: 700; font-size: 12.5px;
    text-align: center; box-shadow: 0 8px 24px rgba(0,0,0,0.5);
  }
  .alert-iot { background: rgba(234, 88, 12, 0.94); border: 1.5px solid #fed7aa; color: #fff; }
  .alert-danger { background: rgba(220, 38, 38, 0.95); border: 1.5px solid #fecaca; color: #fff; }
  .alert-info { background: rgba(37, 99, 235, 0.92); border: 1.5px solid #bfdbfe; color: #fff; }
  .alert-success { background: rgba(22, 163, 74, 0.94); border: 1.5px solid #bbf7d0; color: #fff; }

  /* BARRA DE EQUILÍBRIO NA ESCALADA */
  #climb-hud {
    position: absolute; bottom: 118px; left: 50%; transform: translateX(-50%);
    background: rgba(15, 23, 42, 0.92); border: 1px solid #38bdf8;
    border-radius: 12px; padding: 9px 16px;
    display: none; flex-direction: column; align-items: center; gap: 6px; z-index: 18;
  }
  .balance-track {
    position: relative; width: 240px; height: 13px;
    background: linear-gradient(90deg, #ef4444 0%, #22c55e 35%, #22c55e 65%, #ef4444 100%);
    border-radius: 999px; border: 1px solid #fff;
  }
  #balance-needle {
    position: absolute; top: -3px; left: 50%; width: 8px; height: 19px;
    background: #fff; border-radius: 3px; transform: translateX(-50%);
    box-shadow: 0 0 8px #000;
  }

  /* PROMPT DE PROXIMIDADE */
  #interact-prompt {
    position: absolute; bottom: 68px; left: 50%; transform: translateX(-50%);
    background: rgba(15, 23, 42, 0.90);
    border: 1px solid #38bdf8; color: #f8fafc;
    padding: 7px 16px; border-radius: 999px;
    font-size: 12.5px; font-weight: 600;
    display: none; z-index: 14; pointer-events: none;
    box-shadow: 0 4px 16px rgba(0,0,0,0.45);
  }

  /* BOTÕES INFERIORES */
  #hud-bottom-left {
    position: absolute; bottom: 12px; left: 12px;
    display: flex; flex-wrap: wrap; gap: 6px; z-index: 14;
  }
  #hud-bottom-right {
    position: absolute; bottom: 12px; right: 12px;
    display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 6px; z-index: 14;
  }
  .action-btn {
    background: rgba(15, 23, 42, 0.88);
    border: 1px solid rgba(148, 163, 184, 0.42);
    color: #f8fafc; padding: 8px 10px; border-radius: 9px;
    font-size: 11.5px; font-weight: 700; cursor: pointer;
    transition: 0.15s;
  }
  .action-btn:hover { background: #1e293b; border-color: #38bdf8; }
  .action-btn.ar { border-color: #38bdf8; color: #7dd3fc; }
  .action-btn.atk { border-color: #f97316; color: #fdba74; }
  .action-btn.dance { border-color: #ec4899; color: #f9a8d4; }
  .action-btn.photo { border-color: #34d399; color: #6ee7b7; }
  .action-btn.creto { border-color: #a78bfa; color: #ddd6fe; }

  /* CARTÃO DE REALIDADE AUMENTADA */
  #ar-card {
    position: absolute; top: 64px; right: 14px; width: 315px;
    background: rgba(9, 14, 28, 0.92);
    backdrop-filter: blur(10px);
    border: 1px solid #22d3ee; border-left: 4px solid #22d3ee;
    border-radius: 10px; padding: 12px 14px; color: #e2e8f0;
    display: none; z-index: 18;
    box-shadow: 0 10px 30px rgba(34, 211, 238, 0.25);
  }
  #ar-card h4 { color: #22d3ee; font-size: 14px; margin-bottom: 4px; }
  #ar-card .ar-tag {
    display: inline-block; font-size: 10.5px; padding: 2px 7px;
    background: rgba(34, 211, 238, 0.16); color: #67e8f9;
    border-radius: 4px; margin-bottom: 6px; font-weight: 600;
  }
  #ar-card p { font-size: 12px; line-height: 1.45; color: #cbd5e1; margin-bottom: 6px; }
  #ar-card .ar-meta {
    font-size: 11px; color: #94a3b8;
    border-top: 1px solid rgba(148,163,184,0.2); padding-top: 6px;
  }

  /* VISOR FOTOGRÁFICO & FLASH */
  #photo-overlay {
    position: absolute; inset: 0; pointer-events: none; display: none; z-index: 25;
  }
  .viewfinder-box {
    position: absolute; inset: 46px; border: 2px dashed rgba(255,255,255,0.78);
    border-radius: 14px; display: flex; flex-direction: column;
    justify-content: space-between; padding: 14px;
  }
  #flash-screen {
    position: absolute; inset: 0; background: #fff; opacity: 0;
    pointer-events: none; transition: opacity 0.3s ease-out; z-index: 28;
  }

  /* FLASH DE DANO & JUMPSCARE */
  #damage-flash {
    position: absolute; inset: 0; pointer-events: none; z-index: 22;
    background: radial-gradient(circle, transparent 45%, rgba(220, 38, 38, 0.65) 100%);
    opacity: 0; transition: opacity 0.22s ease;
  }
  #jumpscare-overlay {
    position: absolute; inset: 0;
    background: radial-gradient(circle, #450a0a 0%, #000000 85%);
    display: none; flex-direction: column; justify-content: center;
    align-items: center; z-index: 50;
    animation: shakeScreen 0.1s infinite;
  }
  @keyframes shakeScreen {
    0% { transform: translate(0, 0) rotate(0deg); }
    25% { transform: translate(-8px, 6px) rotate(-1.2deg); }
    50% { transform: translate(8px, -6px) rotate(1.2deg); }
    75% { transform: translate(-6px, -5px) rotate(-0.8deg); }
    100% { transform: translate(6px, 5px) rotate(0.8deg); }
  }

  /* LETTERBOX CUTSCENE */
  .letterbox {
    position: absolute; left: 0; width: 100%; height: 0;
    background: #020617; z-index: 30; transition: height 0.4s ease;
  }
  #lb-top { top: 0; }
  #lb-bottom { bottom: 0; }
  #cutscene-caption {
    position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%);
    color: #fef3c7; font-size: 14.5px; font-weight: 700;
    text-shadow: 0 2px 10px #000; z-index: 31; display: none;
    text-align: center; width: 88%;
  }

  /* MODAL QUIZ PROF. CRETO */
  #creto-modal {
    position: absolute; inset: 0; background: rgba(2, 6, 23, 0.80);
    backdrop-filter: blur(6px); display: none; justify-content: center;
    align-items: center; z-index: 42; padding: 16px;
  }
  .creto-card {
    background: #0f172a; border: 2px solid #3b82f6; border-radius: 16px;
    padding: 20px 24px; width: min(560px, 94vw); color: #f8fafc;
    box-shadow: 0 20px 50px rgba(0,0,0,0.75);
  }
  .quiz-opt {
    display: block; width: 100%; text-align: left;
    background: #1e293b; color: #f8fafc; border: 1px solid #475569;
    border-radius: 10px; padding: 10px 14px; margin-top: 8px;
    font-size: 13px; font-weight: 600; cursor: pointer;
  }
  .quiz-opt:hover { background: #2563eb; border-color: #93c5fd; }

  /* TELA GAME OVER (MODO SOBREVIVÊNCIA) */
  #gameover-modal {
    position: absolute; inset: 0; background: rgba(69, 10, 10, 0.90);
    backdrop-filter: blur(8px); display: none; flex-direction: column;
    align-items: center; justify-content: center; z-index: 45;
    color: #fef2f2; text-align: center; padding: 24px;
  }
  #gameover-modal h2 { font-size: 30px; color: #fca5a5; margin-bottom: 8px; }
  #gameover-reason { font-size: 15.5px; color: #fecaca; margin-bottom: 20px; max-width: 520px; }

  /* PAINEL RETRÁTIL DE QUESTS, INVENTÁRIO & INFORMAÇÕES (PAUSA O JOGO) */
  #pause-modal {
    position: absolute; inset: 0; background: rgba(2, 6, 23, 0.86);
    backdrop-filter: blur(8px); display: none; align-items: center;
    justify-content: center; z-index: 40; padding: 18px;
  }
  .modal-box {
    width: 100%; max-width: 880px; max-height: 92%;
    background: #0f172a; border: 1px solid #38bdf8;
    border-radius: 14px; display: flex; flex-direction: column;
    overflow: hidden; box-shadow: 0 20px 50px rgba(0,0,0,0.65);
  }
  .modal-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 18px; background: #1e293b; border-bottom: 1px solid #334155;
  }
  .modal-header h3 { color: #f8fafc; font-size: 15.5px; }
  .modal-tabs {
    display: flex; gap: 6px; padding: 10px 18px 0 18px;
    background: #0f172a; border-bottom: 1px solid #1e293b;
  }
  .tab-btn {
    background: #1e293b; color: #94a3b8; border: 1px solid #334155;
    border-bottom: none; padding: 7px 14px; border-radius: 8px 8px 0 0;
    font-size: 12.5px; font-weight: 600; cursor: pointer;
  }
  .tab-btn.active { background: #0f172a; color: #38bdf8; border-color: #38bdf8; }
  .modal-body {
    padding: 16px 18px; overflow-y: auto; color: #e2e8f0;
    font-size: 13px; line-height: 1.5; user-select: text;
  }
  .act-block {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 8px; padding: 10px 14px; margin-bottom: 10px;
  }
  .act-block h4 { color: #38bdf8; font-size: 13.5px; margin-bottom: 6px; }
  .q-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 3px 0; border-bottom: 1px dashed rgba(148,163,184,0.15);
    font-size: 12.2px;
  }
  .q-badge {
    font-size: 11px; padding: 1px 7px; border-radius: 999px; font-weight: 700;
  }
  .q-done { background: rgba(34, 197, 94, 0.2); color: #4ade80; }
  .q-bonus { background: rgba(250, 204, 21, 0.2); color: #facc15; }
  .inv-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 10px; margin-top: 8px;
  }
  .inv-slot {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 8px; padding: 10px; text-align: center;
  }
  .inv-slot.collected { border-color: #22c55e; background: rgba(20, 83, 45, 0.28); }
</style>
</head>
<body>
<div id="game-wrapper" tabindex="0">
  <canvas id="c" tabindex="0"></canvas>
  <div id="damage-flash"></div>
  <div id="flash-screen"></div>

  <!-- HUD SUPERIOR ESQUERDO -->
  <div id="hud-top-left">
    <div class="hud-pill">
      <span>❤️ Life</span>
      <div class="bar-bg"><div id="energy-fill"></div></div>
      <span id="hp-text" style="color:#4ade80;">100%</span>
      <span id="buff-tag" style="color:#facc15; display:none;">☕ Turbo!</span>
      <span>•</span>
      <span id="vigilante-badge" style="color:#93c5fd;">👮 Vigilância: Calma</span>
    </div>
    <div class="hud-pill">
      <span>🎒 Inv: <b id="inv-count" style="color:#38bdf8;">0/8</b></span>
      <span>•</span>
      <span>🪟 Vidros: <b id="win-broken" style="color:#fb923c;">0/10</b></span>
      <span>•</span>
      <span>📸 Fotos: <b id="foto-count" style="color:#34d399;">0/5</b></span>
      <span>•</span>
      <span id="proj-badge" style="color:#a5f3fc;">📷 3ª Pessoa</span>
    </div>
    <div id="mini-quest-box">
      <div style="font-weight:700; color:#38bdf8; margin-bottom:2px;">📋 Mini-Quests CT (Clique em Quests [Q] para detalhes)</div>
      <div id="mq-foto" class="mq-todo">⬜ 📸 Álbum Fotográfico & Solar (<span id="mq-foto-n">0</span>/5)</div>
      <div id="mq-arvore" class="mq-todo">⬜ 🌳 Subir na Árvore sem Cair [E]</div>
      <div id="mq-poste" class="mq-todo">⬜ 💡 Subir no Poste sem Cair [E]</div>
      <div id="mq-predio" class="mq-todo">⬜ ☀️ Telhado & Foto Painéis Solares</div>
      <div id="mq-rua" class="mq-todo">⬜ 🚸 Atravessar na Faixa e Voltar</div>
      <div id="mq-creto" class="mq-todo">⬜ 🎓 Quiz CG Rep. Prof. Creto (<span id="mq-creto-n">0</span>/4)</div>
    </div>
  </div>

  <!-- HUD SUPERIOR DIREITO -->
  <div id="hud-top-right">
    <button class="hud-btn" id="btn-survival">💀 Sobrevivência: OFF</button>
    <button class="hud-btn" id="btn-audio">🔊 Som: ON</button>
    <button class="hud-btn" id="btn-quests">📜 Inventário & Quests (Q)</button>
    <button class="hud-btn" id="btn-fullscreen">⛶ Tela Cheia</button>
  </div>

  <!-- PILHA DE ALERTAS DINÂMICOS -->
  <div id="alert-stack"></div>

  <!-- MEDIDOR DE EQUILÍBRIO NA ESCALADA -->
  <div id="climb-hud">
    <div style="font-size:12px; font-weight:700; color:#7dd3fc;" id="climb-label">🧗 Escalando! Use [W/S] subir/descer e [A/D] equilibrar</div>
    <div class="balance-track"><div id="balance-needle"></div></div>
  </div>

  <!-- VISOR FOTOGRÁFICO -->
  <div id="photo-overlay">
    <div class="viewfinder-box">
      <div style="display:flex; justify-content:space-between; font-weight:700; font-size:13.5px; text-shadow:0 2px 6px #000; color:#fff;">
        <span>[REC ●] MODO FOTO TÉCNICA CT</span>
        <span id="photo-target-hint">Alvo: Aponte para um ponto de interesse e clique [P]</span>
      </div>
      <div style="text-align:center; font-size:28px; opacity:0.75; color:#fff;">✛</div>
      <div style="text-align:center; font-size:12px; color:#e2e8f0; text-shadow:0 1px 4px #000;">
        Pontos: Painéis Fotovoltaicos (Telhado) • Escultura • Fonte • Cachorro Guardião • Rep. Prof. Creto
      </div>
    </div>
  </div>

  <!-- JUMPSCARE CORUJA -->
  <div id="jumpscare-overlay">
    <div style="font-size:108px; line-height:1;">🦉👁️👁️</div>
    <div style="font-size:30px; font-weight:900; color:#ef4444; margin-top:10px; text-shadow:0 0 20px #000;">
      SUSTO NO CAMPUS CT!
    </div>
    <div style="font-size:14.5px; color:#fecaca; margin-top:6px;">
      A Coruja Guardiã das Copas te pegou desprevenido!
    </div>
  </div>

  <!-- PROMPT DE PROXIMIDADE -->
  <div id="interact-prompt"></div>

  <!-- CARTÃO DE REALIDADE AUMENTADA -->
  <div id="ar-card">
    <span class="ar-tag">📡 REALIDADE AUMENTADA • BOUNDING BOX</span>
    <h4 id="ar-title">Objeto</h4>
    <p id="ar-desc">Descrição</p>
    <div class="ar-meta" id="ar-meta">Primitiva • Transformação</div>
  </div>

  <!-- ATALHOS RÁPIDOS DE CÂMERA / PROJEÇÃO -->
  <div id="hud-bottom-left">
    <button class="action-btn" id="btn-cam-cycle" title="Alternar 3ª Pessoa, 1ª Pessoa, Oblíqua Desobstruída e Aérea">🎥 Câm. [V]</button>
    <button class="action-btn" id="btn-vp" title="Alternar 1, 2 e 3 Pontos de Fuga">📐 Pontos Fuga</button>
    <button class="action-btn" id="btn-iso" title="Alternar Projeção Isométrica / Ortográfica">🧊 Isométrica</button>
    <button class="action-btn" id="btn-obl" title="Alternar Projeção Oblíqua Gabinete">📏 Oblíqua</button>
  </div>

  <!-- BOTÕES DE AÇÃO DO PERSONAGEM -->
  <div id="hud-bottom-right">
    <button class="action-btn photo" id="btn-photo">📸 Foto [P]</button>
    <button class="action-btn creto" id="btn-creto">🎓 Rep. Creto</button>
    <button class="action-btn dance" id="btn-dance">💃 Dançar [C]</button>
    <button class="action-btn ar" id="btn-interact">🔍 Interagir/Escalar [E]</button>
    <button class="action-btn atk" id="btn-attack">⚔️ Atacar [F]</button>
    <button class="action-btn" id="btn-jump">⬆️ Pular [Espaço]</button>
  </div>

  <!-- LETTERBOX CUTSCENE -->
  <div class="letterbox" id="lb-top"></div>
  <div class="letterbox" id="lb-bottom"></div>
  <div id="cutscene-caption">☕ Preparando um cafezinho expresso no Centro de Tecnologia...</div>

  <!-- MODAL QUIZ PROF. CRETO -->
  <div id="creto-modal">
    <div class="creto-card">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:10px;">
        <span style="font-weight:800; font-size:15.5px; color:#60a5fa;">🎓 Representante do Prof. Creto — Computação Gráfica</span>
        <span id="creto-progress" style="background:#1d4ed8; padding:3px 9px; border-radius:6px; font-size:12px; font-weight:700;">Pergunta 1 / 4</span>
      </div>
      <p id="creto-question" style="margin-top:12px; font-size:14px; line-height:1.5; color:#f1f5f9;"></p>
      <div id="creto-options"></div>
      <div id="creto-feedback" style="margin-top:11px; font-size:12.5px; font-weight:700; min-height:20px;"></div>
      <div style="margin-top:12px; display:flex; justify-content:flex-end;">
        <button class="hud-btn" id="btn-close-creto">Fechar [ESC]</button>
      </div>
    </div>
  </div>

  <!-- MODAL GAME OVER (MODO SOBREVIVÊNCIA) -->
  <div id="gameover-modal">
    <h2>💀 DUMB WAYS TO DIE — CT EDITION</h2>
    <p id="gameover-reason">Você sofreu um acidente no campus do Bloco 734!</p>
    <button class="hud-btn" id="btn-respawn" style="font-size:15px; padding:11px 24px; background:#dc2626; border-color:#fca5a5;">
      🔄 Renascer na Praça do Bloco 734
    </button>
  </div>

  <!-- PAINEL RETRÁTIL DE QUESTS, INVENTÁRIO & CONTROLES (PAUSA O JOGO) -->
  <div id="pause-modal">
    <div class="modal-box">
      <div class="modal-header">
        <h3>⏸️ JOGO PAUSADO — Inventário, Quests & Controles (Bloco 734)</h3>
        <button class="hud-btn" id="btn-resume" style="background:#0284c7; border-color:#38bdf8;">▶ Voltar ao Jogo (Q)</button>
      </div>
      <div class="modal-tabs">
        <button class="tab-btn active" data-tab="tab-inv">🎒 Inventário & Mesas (8 Itens)</button>
        <button class="tab-btn" data-tab="tab-quests">📜 Quests CG I (Atos I–VIII) & Camadas CT</button>
        <button class="tab-btn" data-tab="tab-help">🧭 Controles & Mecânicas</button>
      </div>

      <div class="modal-body" id="tab-inv">
        <p>Aproxime-se das <b>Mesas de Trabalho</b> (sob o Abrigo Metálico e na Praça) e pressione <b>[E]</b> para coletar suas entregas. Coletar todas ativa a <b>Cutscene de Comemoração</b>!</p>
        <div class="inv-grid" id="inv-grid-container"></div>
      </div>

      <div class="modal-body" id="tab-quests" style="display:none;">
        <div class="act-block">
          <h4>🏰 ATO I — A Construção do Mundo</h4>
          <div class="q-row"><span>🎯 Sub-quest 1.1 — "Tudo Tem Um Propósito" (Bloco 734 CT Coerente)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🎯 Sub-quest 1.2 — "O Primeiro Octante" (Todos vértices X, Y, Z &ge; 0)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🎯 Quest 2 — "O Zoo dos Objetos" (Esfera, Cilindro, Cone, Caixa, Arco e Malha)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🎯 Sub-quest 3.1 — "Diversidade" (&ge; 4 materiais distintos aplicados)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🎯 Sub-quest 3.2 — "A Pele Pintada" (Asfalto/Faixa, Calçada, Água, Solar, Placa 734)</span><span class="q-badge q-done">✔ Concluída</span></div>
        </div>
        <div class="act-block">
          <h4>⚔️ ATO II — As Transformações Mágicas</h4>
          <div class="q-row"><span>🎯 Quest 4 — "O Ritual da Translação" (Matrizes 4×4)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🎯 Sub-quest 5.1 — "O Eixo Sagrado" (Rotação X, Y e Z)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🎯 Sub-quest 5.2 — "O Eixo Livre" (Quatérnios nas copas/palmeira)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🎯 Quest 6 — "O Encantamento da Escala"</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>✂️ Quest Bônus 6.1 — "A Distorção" (Cisalhamento XY no Abrigo Metálico)</span><span class="q-badge q-bonus">+0.5 pts</span></div>
          <div class="q-row"><span>🪞 Quest Bônus 6.2 — "O Reflexo Perfeito" (Espelho em plano arbitrário nos Cones)</span><span class="q-badge q-bonus">+0.5 pts</span></div>
        </div>
        <div class="act-block">
          <h4>🔦 ATO III — A Luz e a Escuridão (Totens Interativos na Praça)</h4>
          <div class="q-row"><span>💡 Sub-quest 7.1 — "A Chama Pontual" (4 Postes com atenuação quadrática)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🌫️ Sub-quest 7.2 — "O Brilho Ambiente" (Componente Ia)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🔦 Sub-quest Bônus 7.3 — "O Holofote" (Totem Liga/Desliga Spots)</span><span class="q-badge q-bonus" id="q-spot-badge">+1.0 pts</span></div>
          <div class="q-row"><span>☀️ Sub-quest Bônus 7.4 — "O Sol Direcional" (Totem Liga/Desliga Sol)</span><span class="q-badge q-bonus" id="q-sun-badge">+0.5 pts</span></div>
        </div>
        <div class="act-block">
          <h4>📷 ATOS IV & VIII — Câmera, IoT, Vigilância, Fotovoltaica & Quiz Creto</h4>
          <div class="q-row"><span>👁️ Quests 8, 9 & 10 — Eye, At, Up, Distância Focal, Zoom In/Out & Pontos de Fuga</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>☀️ Usina Fotovoltaica & Sensor IoT Perimetral no Telhado + Faixa de Pedestres</span><span class="q-badge q-done">✔ Ativo</span></div>
          <div class="q-row"><span>👮 Vigilância Patrimonial, Cachorro Guardião & Quiz CG Rep. Prof. Creto</span><span class="q-badge q-bonus">+1.0 pts</span></div>
        </div>
      </div>

      <div class="modal-body" id="tab-help" style="display:none;">
        <div class="act-block">
          <h4>🎮 Controles Unificados do Simulador 3D</h4>
          <p>• <b>Clique na tela 3D + W A S D / Setas:</b> Caminhar (<b>Shift</b> corre; arraste o <b>Mouse</b> para girar a câmera e use a <b>Roda</b> para zoom).<br/>
             • <b>[E] Interagir / Escalar / AR:</b> Ação contextual inteligente — coleta itens das mesas, toma café na cafeteira (<code>X=35.8, Z=18.8</code>), fala com o Rep. Prof. Creto (<code>X=28.5, Z=24.5</code>), escala Árvores/Postes/Escada (equilibre com <b>A/D</b>!) ou abre o Raio-X (AR) de qualquer monumento.<br/>
             • <b>[F] Atacar:</b> Quebra as 10 janelas/porta curvas do Bloco 734, derrama asfalto CBUQ e aciona os Vigilantes Patrimoniais (suba numa árvore, poste ou escada para despistá-los!).<br/>
             • <b>[P] Modo Foto Técnica:</b> Enquadra os Painéis Fotovoltaicos no telhado, Escultura, Fonte, Cachorro Guardião ou Rep. Prof. Creto.<br/>
             • <b>[V] Trocar Câmera:</b> Alterna 3ª Pessoa Orbital, 1ª Pessoa, Oblíqua Desobstruída e Aérea Plongée.<br/>
             • <b>[C] Comemorar:</b> Executa a dança comemorativa com música.<br/>
             • <b>[Q] ou [Tab]:</b> Pausa o jogo e abre Inventário, Quests e Ajuda.</p>
        </div>
      </div>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function() {
  const MODO_NOITE = __MODO_NOITE__;
  let sunEnabled = __ATIVAR_SOL__;
  let spotsEnabled = __ATIVAR_SPOTS__;

  const wrapper = document.getElementById('game-wrapper');
  const canvas = document.getElementById('c');

  // Garante foco imediato no teclado ao carregar ou clicar
  setTimeout(() => { canvas.focus(); }, 150);
  wrapper.addEventListener('pointerdown', () => { canvas.focus(); });

  // ============================================================================
  // 1. MOTOR DE ÁUDIO PROCEDURAL (WEB AUDIO API)
  // ============================================================================
  const SoundEngine = {
    ctx: null,
    enabled: true,
    ambientTimer: null,
    init() {
      if (!this.ctx) {
        const AC = window.AudioContext || window.webkitAudioContext;
        if (AC) this.ctx = new AC();
      }
      if (this.ctx && this.ctx.state === 'suspended') {
        this.ctx.resume().catch(() => {});
      }
      if (!this.ambientTimer) {
        this.ambientTimer = setInterval(() => this.playAmbientTick(), 3600);
      }
    },
    tone(freq, dur, type = 'sine', vol = 0.08, delay = 0, slideTo = null) {
      if (!this.enabled) return;
      this.init();
      if (!this.ctx) return;
      const t0 = this.ctx.currentTime + delay;
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, t0);
      if (slideTo) osc.frequency.exponentialRampToValueAtTime(Math.max(20, slideTo), t0 + dur);
      gain.gain.setValueAtTime(0.0001, t0);
      gain.gain.linearRampToValueAtTime(vol, t0 + 0.012);
      gain.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start(t0);
      osc.stop(t0 + dur + 0.02);
    },
    playStep() {
      const pitch = 155 + Math.random() * 35;
      this.tone(pitch, 0.055, 'triangle', 0.075, 0, 75);
    },
    playJump() { this.tone(220, 0.18, 'sine', 0.09, 0, 520); },
    playSwitch() {
      this.tone(580, 0.07, 'square', 0.06, 0);
      this.tone(860, 0.09, 'square', 0.06, 0.07);
    },
    playCollect() {
      [523.25, 659.25, 783.99, 1046.5].forEach((f, i) => {
        this.tone(f, 0.14, 'triangle', 0.09, i * 0.065);
      });
    },
    playGlass() {
      [1400, 1950, 2400, 1100, 1750].forEach((f, i) => {
        this.tone(f + Math.random() * 300, 0.12, 'sawtooth', 0.06, i * 0.025, 320);
      });
      this.tone(110, 0.65, 'sine', 0.08, 0.1, 55);
    },
    playHonk() {
      this.tone(415, 0.22, 'sawtooth', 0.07, 0);
      this.tone(523, 0.22, 'sawtooth', 0.06, 0);
    },
    playScare() {
      this.tone(920, 0.38, 'sawtooth', 0.14, 0, 240);
      this.tone(95, 0.48, 'square', 0.14, 0, 38);
    },
    playIoT() { this.tone(880, 0.14, 'sine', 0.07, 0, 1180); },
    playWhistle() {
      this.tone(2600, 0.14, 'triangle', 0.10, 0, 2900);
      this.tone(2750, 0.22, 'triangle', 0.10, 0.16, 2400);
    },
    playBark() {
      this.tone(190, 0.13, 'sawtooth', 0.11, 0, 95);
      this.tone(220, 0.15, 'sawtooth', 0.11, 0.16, 85);
    },
    playPhoto() {
      this.tone(1200, 0.05, 'square', 0.09, 0, 400);
      this.tone(1600, 0.07, 'sine', 0.08, 0.055, 900);
    },
    playErro() { this.tone(185, 0.32, 'sawtooth', 0.11, 0, 110); },
    playCoffee() {
      [330, 392, 494, 587, 659].forEach((f, i) => {
        this.tone(f, 0.25, 'sine', 0.07, i * 0.22);
      });
    },
    playDanceBeat() {
      const notes = [
        [329.6, 0.0], [329.6, 0.22], [392.0, 0.44], [440.0, 0.66],
        [392.0, 0.96], [329.6, 1.22], [293.7, 1.48], [329.6, 1.75],
        [440.0, 2.15], [392.0, 2.40], [329.6, 2.65], [392.0, 2.95], [523.3, 3.25]
      ];
      notes.forEach(([f, d]) => this.tone(f, 0.18, 'square', 0.065, d));
    },
    playAmbientTick() {
      if (!this.enabled || !this.ctx || this.ctx.state !== 'running') return;
      if (MODO_NOITE || !sunEnabled) {
        this.tone(3600, 0.04, 'sine', 0.012, 0);
        this.tone(3650, 0.04, 'sine', 0.012, 0.08);
      } else {
        const base = 1800 + Math.random() * 500;
        this.tone(base, 0.08, 'sine', 0.015, 0, base * 1.25);
        this.tone(base * 1.15, 0.09, 'sine', 0.015, 0.14, base * 0.9);
      }
    }
  };

  document.getElementById('btn-audio').addEventListener('click', (e) => {
    SoundEngine.enabled = !SoundEngine.enabled;
    e.target.textContent = SoundEngine.enabled ? '🔊 Som: ON' : '🔇 Som: OFF';
    if (SoundEngine.enabled) SoundEngine.playSwitch();
  });

  // ============================================================================
  // 2. RENDERER WEBGL & CÂMERAS
  // ============================================================================
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  const perspCamera = new THREE.PerspectiveCamera(58, wrapper.clientWidth / wrapper.clientHeight, 0.1, 260);
  const frustumSize = 26;
  const aspect0 = wrapper.clientWidth / wrapper.clientHeight;
  const orthoCamera = new THREE.OrthographicCamera(
    -frustumSize * aspect0 * 0.5, frustumSize * aspect0 * 0.5,
    frustumSize * 0.5, -frustumSize * 0.5, -60, 220
  );

  let projMode = 'perspectiva'; // 'perspectiva' | 'isometrica' | 'obliqua'
  let vanishingMode = 2; // 1, 2, 3
  const camViewModes = ['3ª Pessoa', '1ª Pessoa', 'Oblíqua Desobstruída', 'Aérea Plongée'];
  let camViewIdx = 0;

  // ============================================================================
  // 3. ILUMINAÇÃO DINÂMICA (SOL DIRECIONAL & HOLOFOTES / SPOTS)
  // ============================================================================
  const hemiLight = new THREE.HemisphereLight(0xe0f2fe, 0x475569, 0.85);
  scene.add(hemiLight);

  const dirLight = new THREE.DirectionalLight(0xfffbeb, 1.15);
  dirLight.position.set(12, 34, 8);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.set(2048, 2048);
  dirLight.shadow.camera.near = 2;
  dirLight.shadow.camera.far = 110;
  const dCam = 44;
  dirLight.shadow.camera.left = -dCam;
  dirLight.shadow.camera.right = dCam;
  dirLight.shadow.camera.top = dCam;
  dirLight.shadow.camera.bottom = -dCam;
  scene.add(dirLight);

  const spotLightsList = [];
  const spotEsc = new THREE.SpotLight(0xbae6fd, 2.6, 32, Math.PI / 5, 0.35, 1);
  spotEsc.position.set(12.5, 5.2, 18.0);
  spotEsc.target.position.set(15.0, 1.8, 22.0);
  scene.add(spotEsc, spotEsc.target);
  spotLightsList.push(spotEsc);

  const spotBulbs = [];
  [22.0, 27.0, 32.0].forEach((sx) => {
    const sp = new THREE.Mesh(
      new THREE.SphereGeometry(0.17, 12, 12),
      new THREE.MeshBasicMaterial({ color: 0x67e8f9 })
    );
    sp.position.set(sx, 0.34, 17.12);
    scene.add(sp);
    spotBulbs.push(sp);

    const pSub = new THREE.PointLight(0x38bdf8, 1.3, 12);
    pSub.position.set(sx, 0.85, 17.4);
    scene.add(pSub);
    spotLightsList.push(pSub);
  });

  const posteLights = [];
  [[9.5, 21.0, 0.6], [18.5, 16.2, 0.4], [41.0, 16.5, 0.0], [54.5, 17.5, -0.5]].forEach(([px, pz, ox]) => {
    const pt = new THREE.PointLight(0xfde68a, 1.3, 24);
    pt.position.set(px + ox, 6.0, pz + 0.4);
    scene.add(pt);
    posteLights.push(pt);
  });

  function atualizarIluminacaoCena() {
    const isDark = MODO_NOITE || !sunEnabled;
    scene.background = new THREE.Color(isDark ? 0x060b19 : 0x8ec5fc);
    scene.fog = new THREE.FogExp2(isDark ? 0x060b19 : 0x93c5fd, isDark ? 0.013 : 0.006);
    hemiLight.color.setHex(isDark ? 0x1e293b : 0xe0f2fe);
    hemiLight.intensity = isDark ? 0.48 : 0.88;
    dirLight.visible = sunEnabled;
    dirLight.intensity = MODO_NOITE ? 0.25 : 1.15;

    spotLightsList.forEach(l => { l.visible = spotsEnabled; });
    spotBulbs.forEach(b => { b.material.color.setHex(spotsEnabled ? 0x67e8f9 : 0x334155); });
    posteLights.forEach(p => { p.intensity = isDark ? 1.45 : 0.25; });

    const bSun = document.getElementById('q-sun-badge');
    const bSpot = document.getElementById('q-spot-badge');
    if (bSun) bSun.textContent = sunEnabled ? '✔ Sol ON (+0.5 pts)' : '⚠️ Sol OFF';
    if (bSpot) bSpot.textContent = spotsEnabled ? '✔ Spots ON (+1.0 pts)' : '⚠️ Spots OFF';
  }

  // ============================================================================
  // 4. MATERIAIS & CONSTRUÇÃO DO CENÁRIO BLOCO 734
  // ============================================================================
  const matAsfalto = new THREE.MeshStandardMaterial({ color: 0x262930, roughness: 0.9 });
  const matFaixa = new THREE.MeshBasicMaterial({ color: 0xeab308 });
  const matZebra = new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.5 });
  const matCalcada = new THREE.MeshStandardMaterial({ color: 0xb0aea8, roughness: 0.85 });
  const matAlvenaria = new THREE.MeshStandardMaterial({ color: 0xf1f5f9, roughness: 0.65 });
  const matVidro = new THREE.MeshStandardMaterial({
    color: 0x0f172a, roughness: 0.12, metalness: 0.8, transparent: true, opacity: 0.86
  });
  const matEsquadria = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.6, roughness: 0.4 });
  const matAcoBranco = new THREE.MeshStandardMaterial({ color: 0xf8fafc, metalness: 0.35, roughness: 0.25 });
  const matConcreto = new THREE.MeshStandardMaterial({ color: 0x64748b, roughness: 0.8 });
  const matTronco = new THREE.MeshStandardMaterial({ color: 0x5c3a21, roughness: 0.9 });
  const matFolha1 = new THREE.MeshStandardMaterial({ color: 0x15803d, roughness: 0.75 });
  const matFolha2 = new THREE.MeshStandardMaterial({ color: 0x22c55e, roughness: 0.7 });
  const matAgua = new THREE.MeshStandardMaterial({
    color: 0x0ea5e9, roughness: 0.15, metalness: 0.4, transparent: true, opacity: 0.82
  });
  const matGranito = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.6 });
  const matCone = new THREE.MeshStandardMaterial({ color: 0xf97316, roughness: 0.45 });
  const matMetalGalv = new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.7, roughness: 0.3, side: THREE.DoubleSide });
  const matMadeira = new THREE.MeshStandardMaterial({ color: 0x854d27, roughness: 0.65 });
  const matPapel = new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.5 });
  const matTelaPC = new THREE.MeshStandardMaterial({ color: 0x0284c7, emissive: 0x0369a1, emissiveIntensity: 0.6 });
  const matSolar = new THREE.MeshStandardMaterial({ color: 0x1e3a8a, roughness: 0.22, metalness: 0.65 });

  function addBox(w, h, d, mat, x, y, z, ry = 0, cast = true) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
    m.position.set(x, y, z);
    if (ry) m.rotation.y = ry;
    m.castShadow = cast;
    m.receiveShadow = true;
    scene.add(m);
    return m;
  }

  const arObjects = [];
  const clickableTotems = [];
  function registrarAR(nome, tag, desc, meta, center, size, extra = {}) {
    arObjects.push({
      nome, tag, desc, meta,
      center: new THREE.Vector3(...center),
      size: new THREE.Vector3(...size),
      ...extra
    });
  }

  // Piso Urbano + Faixas + Faixa de Pedestres Zebrada (X = 25.5..28.5)
  addBox(66, 0.1, 14, matAsfalto, 35, 0.05, 9, 0, false);
  addBox(64, 0.02, 0.14, matFaixa, 35, 0.11, 11.5, 0, false);
  addBox(64, 0.02, 0.14, matFaixa, 35, 0.11, 11.9, 0, false);
  for (let zf = 3.2; zf <= 14.8; zf += 1.2) {
    addBox(3.0, 0.02, 0.58, matZebra, 27.0, 0.11, zf, 0, false);
  }
  addBox(66, 0.2, 44, matCalcada, 35, 0.10, 38, 0, false);

  // Bloco 734
  addBox(22, 6.6, 33, matAlvenaria, 48, 3.3, 36.5);
  addBox(22, 6.6, 11, matAlvenaria, 26, 3.3, 47.5);
  const arcMesh = new THREE.Mesh(
    new THREE.CylinderGeometry(22, 22, 6.6, 48, 1, false, Math.PI, Math.PI * 0.5),
    matAlvenaria
  );
  arcMesh.position.set(37, 3.3, 42);
  arcMesh.castShadow = true;
  arcMesh.receiveShadow = true;
  scene.add(arcMesh);

  // Painéis Fotovoltaicos na Cobertura (Y = 6.60m) + Balizador IoT de Borda
  [29.0, 34.0].forEach((zf) => {
    const painel = new THREE.Mesh(new THREE.BoxGeometry(11.0, 0.12, 2.2), matSolar);
    painel.position.set(48.0, 7.05, zf);
    painel.rotation.x = THREE.MathUtils.degToRad(22);
    painel.castShadow = true;
    scene.add(painel);

    const moldura = new THREE.Mesh(new THREE.BoxGeometry(11.15, 0.08, 2.32), matAcoBranco);
    moldura.position.set(48.0, 6.98, zf);
    moldura.rotation.x = THREE.MathUtils.degToRad(22);
    scene.add(moldura);
  });
  registrarAR(
    'Usina Fotovoltaica (Cobertura Bloco 734)',
    'ENERGIA SOLAR • PONTO DE FOTO TÉCNICA [P]',
    'Arranjo de módulos fotovoltaicos monocristalinos inclinados a 22° na cobertura. Enquadre e aperte [P]!',
    'Malha Poligonal · T(48, 7.02, 31.5) · Rx(22°)',
    [48.0, 7.05, 31.5], [11.2, 1.2, 7.4]
  );

  const iotBeacon = new THREE.Mesh(
    new THREE.SphereGeometry(0.28, 16, 16),
    new THREE.MeshStandardMaterial({ color: 0xf97316, emissive: 0xea580c, emissiveIntensity: 0.9 })
  );
  iotBeacon.position.set(38.5, 6.88, 21.2);
  scene.add(iotBeacon);
  registrarAR(
    'Balizador IoT de Borda (Telhado)',
    'SENSOR IoT • MONITORAMENTO DE QUEDA',
    'Sensor perimetral que emite alerta sonoro e visual quando o explorador se aproxima da beirada da laje.',
    'Esfera IoT · T(38.5, 6.88, 21.2)',
    [38.5, 6.88, 21.2], [1.2, 1.2, 1.2]
  );

  // Placa 734
  const pCanvas = document.createElement('canvas');
  pCanvas.width = 256; pCanvas.height = 96;
  const pCtx = pCanvas.getContext('2d');
  pCtx.fillStyle = '#f8fafc'; pCtx.fillRect(0, 0, 256, 96);
  pCtx.strokeStyle = '#1e293b'; pCtx.lineWidth = 6; pCtx.strokeRect(3, 3, 250, 90);
  pCtx.fillStyle = '#0f172a'; pCtx.font = 'bold 62px sans-serif';
  pCtx.textAlign = 'center'; pCtx.textBaseline = 'middle';
  pCtx.fillText('734', 128, 52);
  const angPlaca = 0.14 * (Math.PI / 2);
  const placaMesh = new THREE.Mesh(
    new THREE.BoxGeometry(3.4, 1.15, 0.18),
    new THREE.MeshStandardMaterial({ map: new THREE.CanvasTexture(pCanvas) })
  );
  placaMesh.position.set(37 - 22.08 * Math.cos(angPlaca), 5.72, 42 - 22.08 * Math.sin(angPlaca));
  placaMesh.rotation.y = Math.PI / 2 - angPlaca;
  scene.add(placaMesh);

  registrarAR(
    'Bloco 734 (CT / UFC)',
    'EDIFICAÇÃO PRINCIPAL • CSG ANALÍTICO',
    'Edifício sede com fachada arqueada tangencial (R=22m, H=6.6m), 9 janelas quebráveis [F] e platibanda nº 734.',
    'Primitivas: Prisma + Cilindro R=22m • Primeiro Octante',
    [37, 3.3, 36.5], [44, 6.8, 33]
  );

  // Janelas Quebráveis + Derramamento de Asfalto + Poças Ativas (acionam Vigilantes!)
  const breakableWindows = [];
  const asphaltSpills = [];
  const activePuddles = [];
  const particles = [];
  const matAsfaltoLiquido = new THREE.MeshStandardMaterial({ color: 0x090b10, roughness: 0.18, metalness: 0.35 });

  function criarJanelaCurva(nome, sArco, yCentro, larg, alt) {
    const ang = sArco * (Math.PI / 2);
    const r = 22.06;
    const wx = 37 - r * Math.cos(ang);
    const wz = 42 - r * Math.sin(ang);
    const rotY = Math.PI / 2 - ang;

    const grp = new THREE.Group();
    grp.position.set(wx, yCentro, wz);
    grp.rotation.y = rotY;

    const frame = new THREE.Mesh(new THREE.BoxGeometry(larg + 0.18, alt + 0.18, 0.12), matEsquadria);
    const glassMat = matVidro.clone();
    const glass = new THREE.Mesh(new THREE.BoxGeometry(larg, alt, 0.16), glassMat);
    const hpBarBg = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.14, 0.05), new THREE.MeshBasicMaterial({ color: 0x0f172a }));
    hpBarBg.position.set(0, alt * 0.5 + 0.32, 0.12);
    hpBarBg.visible = false;
    const hpBarFill = new THREE.Mesh(new THREE.BoxGeometry(1.34, 0.10, 0.06), new THREE.MeshBasicMaterial({ color: 0x22c55e }));
    hpBarFill.position.set(0, alt * 0.5 + 0.32, 0.13);
    hpBarFill.visible = false;

    grp.add(frame, glass, hpBarBg, hpBarFill);
    scene.add(grp);

    breakableWindows.push({
      nome, grp, glass, glassMat, hpBarBg, hpBarFill,
      pos: new THREE.Vector3(wx, yCentro, wz),
      normal: new THREE.Vector3(-Math.cos(ang), 0, -Math.sin(ang)),
      rotY, larg, alt, hp: 3, maxHp: 3, broken: false, hideBarTimer: 0
    });
  }

  criarJanelaCurva('Porta Principal de Vidro (Bloco 734)', 0.13, 1.48, 2.5, 2.55);
  for (let i = 0; i < 9; i++) {
    const sMid = 0.24 + ((i + 0.5) / 9.0) * 0.72;
    criarJanelaCurva(`Janela Fachada Curva #${i + 1}`, sMid, 2.30, 1.75, 1.88);
  }

  function dispararDerramamentoAsfalto(winObj) {
    const basePos = winObj.pos.clone().addScaledVector(winObj.normal, 0.16);
    const streamMesh = new THREE.Mesh(new THREE.BoxGeometry(winObj.larg * 0.82, 0.1, 0.18), matAsfaltoLiquido);
    streamMesh.position.set(basePos.x, winObj.pos.y - winObj.alt * 0.35, basePos.z);
    streamMesh.rotation.y = winObj.rotY;
    scene.add(streamMesh);

    const puddlePos = winObj.pos.clone().addScaledVector(winObj.normal, 1.15);
    const puddleMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.04, 24), matAsfaltoLiquido);
    puddleMesh.position.set(puddlePos.x, 0.22, puddlePos.z);
    scene.add(puddleMesh);

    asphaltSpills.push({ streamMesh, puddleMesh, topY: winObj.pos.y - winObj.alt * 0.35, progress: 0.0 });
    activePuddles.push({ x: puddlePos.x, z: puddlePos.z, r: 1.35 });
  }

  function spawnParticulas(pos, coresHex, qtd = 20, spread = 5.0) {
    const geo = new THREE.BoxGeometry(0.14, 0.14, 0.06);
    for (let i = 0; i < qtd; i++) {
      const cor = Array.isArray(coresHex) ? coresHex[i % coresHex.length] : coresHex;
      const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: cor }));
      m.position.copy(pos).add(new THREE.Vector3((Math.random()-0.5)*1.2, (Math.random()-0.5)*1.0, (Math.random()-0.5)*0.8));
      scene.add(m);
      particles.push({
        mesh: m,
        vel: new THREE.Vector3((Math.random()-0.5)*spread, Math.random()*4.5 + 1.8, (Math.random()-0.5)*spread),
        life: 1.6
      });
    }
  }

  // Fonte Frontal
  addBox(15.3, 0.34, 0.28, matGranito, 27.0, 0.37, 16.8);
  addBox(15.3, 0.34, 0.28, matGranito, 27.0, 0.37, 22.8);
  addBox(0.28, 0.34, 6.0, matGranito, 19.5, 0.37, 19.8);
  addBox(0.28, 0.34, 6.0, matGranito, 34.5, 0.37, 19.8);
  const aguaMesh = addBox(14.85, 0.08, 5.85, matAgua, 27.0, 0.38, 19.8, 0, false);

  registrarAR(
    "Espelho d'Água (Fonte Cáustica)",
    "REFRAÇÃO n=1.333 • PATRIMÔNIO PROTEGIDO",
    "Lâmina d'água com 3 spots submersos. Saltar do telhado na fonte amortece a queda, mas demorar dentro dela aciona os Vigilantes!",
    "Cubos + Esferas • Textura Ondulatória",
    [27.0, 0.42, 19.8], [15.4, 0.6, 6.2]
  );

  // Escultura Cinética (5 Losangos)
  const escGrp = new THREE.Group();
  escGrp.position.set(15.0, 0.2, 22.0);
  escGrp.rotation.y = THREE.MathUtils.degToRad(-24);
  const baseEsc = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.28, 3.4), matConcreto);
  baseEsc.position.y = 0.14;
  escGrp.add(baseEsc);
  for (let k = 0; k < 5; k++) {
    const fc = k - 2;
    const Lk = 2.35 * (1.0 - 0.055 * fc);
    const rk = Lk / Math.sqrt(2);
    const qGrp = new THREE.Group();
    qGrp.position.set(fc * 0.22, 0.28 + rk + 0.05, -1.16 + k * 0.58);
    qGrp.rotation.z = Math.PI / 4;
    const esp = 0.14, prof = 0.16;
    [
      [Lk + esp, esp, 0, Lk/2], [Lk + esp, esp, 0, -Lk/2],
      [esp, Lk + esp, Lk/2, 0], [esp, Lk + esp, -Lk/2, 0]
    ].forEach(([w, h, bx, by]) => {
      const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, prof), matAcoBranco);
      b.position.set(bx, by, 0);
      b.castShadow = true;
      qGrp.add(b);
    });
    escGrp.add(qGrp);
  }
  scene.add(escGrp);
  registrarAR(
    "Escultura Cinética em Aço",
    "COMPOSIÇÃO PARAMÉTRICA • OBRA PROTEGIDA",
    "Cinco quadros metálicos rotacionados a 45° e -24° em Y. Aproximar-se demais da obra aciona a Vigilância Patrimonial!",
    "Translação · Rotação Z(45°) & Y(-24°) · Escala",
    [15.0, 2.0, 22.0], [3.6, 3.8, 3.8]
  );

  // Escada Telhado
  addBox(0.55, 6.6, 1.1, matEsquadria, 59.35, 3.4, 24.0);
  for (let deg = 0.6; deg <= 6.4; deg += 0.45) {
    addBox(0.65, 0.06, 0.95, matMetalGalv, 59.35, deg, 24.0);
  }
  registrarAR(
    "Escada de Acesso ao Telhado",
    "TRANSPORTE VERTICAL • Y=0.2m → Y=6.8m",
    "Escada lateral (X=59.35, Z=24.0). Pressione [E] para escalar com [W/S] ou suba diretamente!",
    "Cubos Escalonados",
    [59.35, 3.4, 24.0], [1.2, 6.8, 1.5]
  );

  // Árvores Escaláveis + Ave Predadora
  const arvoresCoords = [
    { nome: "Árvore Bosque Esq.", x: 10.5, z: 28.5, h: 4.5, esc: 1.15, temCoruja: false },
    { nome: "Árvore Fundo Esq.", x: 11.5, z: 38.0, h: 4.9, esc: 1.25, temCoruja: true },
    { nome: "Árvore Praça Dir.", x: 37.5, z: 17.5, h: 3.8, esc: 0.90, temCoruja: false }
  ];
  arvoresCoords.forEach((arv) => {
    const tMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.28 * arv.esc, 0.42 * arv.esc, 3.2 * arv.esc, 12), matTronco);
    tMesh.position.set(arv.x, 0.2 + 1.6 * arv.esc, arv.z);
    tMesh.castShadow = true;
    scene.add(tMesh);
    [[0, 3.6, 0, 1.65, matFolha1], [-1.1, 3.2, 0.6, 1.25, matFolha2], [1.1, 3.3, -0.5, 1.30, matFolha2], [0.3, 4.5, 0.4, 1.20, matFolha1]].forEach(([ox, oy, oz, r, m]) => {
      const cMesh = new THREE.Mesh(new THREE.SphereGeometry(r * arv.esc, 14, 12), m);
      cMesh.position.set(arv.x + ox * arv.esc, oy * arv.esc, arv.z + oz * arv.esc);
      cMesh.castShadow = true;
      scene.add(cMesh);
    });
    registrarAR(arv.nome, "VEGETAÇÃO ESCALÁVEL [E] • FUGA VERTICAL", "Pressione [E] para escalar (equilibre com A/D!) e escapar dos Vigilantes ou do Cachorro!", "Cilindro + Esferas (Quatérnios)", [arv.x, 2.8 * arv.esc, arv.z], [3.4 * arv.esc, 5.4 * arv.esc, 3.4 * arv.esc], { climbable: { tipo: 'arvore', nome: arv.nome, x: arv.x, z: arv.z, maxH: arv.h, coruja: arv.temCoruja } });
  });

  // Ave Predadora / Coruja (X=11.5, Z=38.0)
  const aveGrp = new THREE.Group();
  const aveHome = new THREE.Vector3(11.5, 5.3, 38.0);
  aveGrp.position.copy(aveHome);
  const matAve = new THREE.MeshStandardMaterial({ color: 0x312e81, roughness: 0.5 });
  const aveCorpo = new THREE.Mesh(new THREE.ConeGeometry(0.32, 1.1, 10), matAve);
  aveCorpo.rotation.x = Math.PI / 2;
  const asaE = new THREE.Mesh(new THREE.BoxGeometry(1.3, 0.08, 0.45), matAve); asaE.position.x = -0.65;
  const asaD = new THREE.Mesh(new THREE.BoxGeometry(1.3, 0.08, 0.45), matAve); asaD.position.x = 0.65;
  aveGrp.add(aveCorpo, asaE, asaD);
  scene.add(aveGrp);
  let birdScareCooldown = 0;

  // Cones + Abrigo Cisalhado + Postes Escaláveis
  [[35.3, 16.2], [35.9, 15.6], [36.5, 16.9], [37.1, 16.3]].forEach(([cx, cz]) => {
    const c = new THREE.Mesh(new THREE.ConeGeometry(0.32, 0.95, 14), matCone);
    c.position.set(cx, 0.2 + 0.475, cz);
    c.castShadow = true;
    scene.add(c);
  });

  const abrigoMesh = new THREE.Mesh(new THREE.BoxGeometry(4.5, 0.16, 3.2), matMetalGalv);
  abrigoMesh.position.set(46.8, 2.6, 18.8);
  abrigoMesh.rotation.z = -0.09;
  abrigoMesh.castShadow = true;
  scene.add(abrigoMesh);
  [[44.7, 17.4], [48.8, 17.4], [44.7, 20.2], [48.8, 20.2]].forEach(([px, pz]) => {
    const pil = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 2.4, 10), matEsquadria);
    pil.position.set(px, 1.4, pz);
    scene.add(pil);
  });

  const postesData = [
    { nome: 'Poste #1', x: 9.5, z: 21.0, ox: 0.6, h: 5.8 },
    { nome: 'Poste #2', x: 18.5, z: 16.2, ox: 0.4, h: 5.8 },
    { nome: 'Poste #3', x: 41.0, z: 16.5, ox: 0.0, h: 5.8 },
    { nome: 'Poste #4', x: 54.5, z: 17.5, ox: -0.5, h: 5.8 }
  ];
  postesData.forEach((pt) => {
    const p = new THREE.Mesh(new THREE.CylinderGeometry(0.10, 0.14, 6.2, 10), matEsquadria);
    p.position.set(pt.x, 3.3, pt.z);
    scene.add(p);
    const lum = new THREE.Mesh(new THREE.SphereGeometry(0.28, 12, 12), new THREE.MeshBasicMaterial({ color: 0xfef08a }));
    lum.position.set(pt.x + pt.ox, 6.3, pt.z + 0.4);
    scene.add(lum);
    registrarAR(pt.nome, "POSTE ESCALÁVEL [E] • FUGA VERTICAL", "Pressione [E] para escalar o poste (equilibre com A/D!) e concluir a mini-quest.", "Cilindro + Esfera Emissiva", [pt.x, 3.2, pt.z], [1.4, 6.2, 1.4], { climbable: { tipo: 'poste', nome: pt.nome, x: pt.x, z: pt.z, maxH: pt.h } });
  });

  // ============================================================================
  // 5. CARROS ANIMADOS NA VIA CT + CACHORRO GUARDIÃO + VIGILANTES + REP. CRETO
  // ============================================================================
  const animatedCars = [];
  function criarVeiculoAnimado(nome, xInit, zLane, dirX, speed, corHex) {
    const grp = new THREE.Group();
    grp.position.set(xInit, 0.1, zLane);
    grp.rotation.y = dirX > 0 ? Math.PI / 2 : -Math.PI / 2;

    const matCarro = new THREE.MeshStandardMaterial({ color: corHex, metalness: 0.4, roughness: 0.25 });
    const matPneu = new THREE.MeshStandardMaterial({ color: 0x111827, roughness: 0.9 });
    const chassi = new THREE.Mesh(new THREE.BoxGeometry(2.0, 0.65, 4.3), matCarro);
    chassi.position.y = 0.55; chassi.castShadow = true;
    const cabine = new THREE.Mesh(new THREE.BoxGeometry(1.75, 0.54, 2.2), matVidro);
    cabine.position.set(0, 1.12, -0.15);
    grp.add(chassi, cabine);

    const rodas = [];
    [[-1.02, 0.36, 1.35], [1.02, 0.36, 1.35], [-1.02, 0.36, -1.35], [1.02, 0.36, -1.35]].forEach(([rx, ry, rz]) => {
      const roda = new THREE.Mesh(new THREE.CylinderGeometry(0.36, 0.36, 0.22, 12), matPneu);
      roda.rotation.z = Math.PI / 2;
      roda.position.
