import io
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
    page_title="Bloco 734 (CT/UFC) — Ray Casting & Campus 3D Interativo",
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
    @media (pointer: coarse), (max-width: 900px) {
      #xbox-sticks { display: block; }
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

    coord_txt = f"Pixel ({px}, {py}) • " if (px is not None and py is not None) else ""
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
# HTML / THREE.JS / WEB AUDIO — MODO WEBGL COMPLETO & REFINADO
# ==============================================================================
def gerar_html_jogo_webgl(modo_noite=False, ativar_sol=True, ativar_spots=True):
    template = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
<title>Bloco 734 — Campus CT 3D Interativo</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; }
  html, body {
    width: 100%; height: 100%; overflow: hidden;
    background: #0b1120; font-family: 'Segoe UI', system-ui, sans-serif;
  }
  #game-wrapper {
    position: relative; width: 100%; height: clamp(460px, 82dvh, 720px);
    background: #0b1120; border-radius: 12px; overflow: hidden;
    border: 1px solid #1e293b;
  }
  @media (max-width: 900px), (pointer: coarse), (any-pointer: coarse) {
    #game-wrapper { height: clamp(430px, 76dvh, 560px); }
    #xbox-sticks { display: block !important; }
  }
  .char-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 8px; max-height: 250px; overflow-y: auto; margin: 10px 0 14px 0; text-align: left;
  }
  .char-btn {
    background: #020617; border: 1px solid #334155; color: #e2e8f0;
    border-radius: 10px; padding: 8px 10px; font-size: 11.5px; font-weight: 700;
    cursor: pointer; transition: 0.15s;
  }
  .char-btn:hover { border-color: #38bdf8; }
  .char-btn.active { border-color: #38bdf8; background: rgba(37, 99, 235, 0.28); color: #fff; }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; outline: none; }
  canvas:active { cursor: grabbing; }

  /* TELA INICIAL ARCADE START */
  #arcade-start-overlay {
    position: absolute; inset: 0;
    background: radial-gradient(circle at top, rgba(37, 99, 235, 0.28), rgba(2, 6, 23, 0.94) 72%);
    backdrop-filter: blur(8px);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    z-index: 48; padding: 20px; text-align: center; color: #f8fafc;
  }
  .arcade-card {
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid rgba(56, 189, 248, 0.55);
    border-radius: 18px; padding: 24px 28px;
    width: min(460px, 92vw);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.75);
  }
  .arcade-badge {
    display: inline-block; font-size: 11px; font-weight: 800; letter-spacing: 1px;
    color: #38bdf8; background: rgba(56, 189, 248, 0.14);
    border: 1px solid rgba(56, 189, 248, 0.35);
    padding: 3px 10px; border-radius: 999px; margin-bottom: 10px;
  }
  .arcade-title { font-size: 24px; font-weight: 900; margin-bottom: 6px; }
  .arcade-sub { font-size: 12.5px; color: #cbd5e1; margin-bottom: 16px; line-height: 1.45; }
  .arcade-input {
    width: 100%; padding: 10px 12px; border-radius: 10px;
    border: 1px solid #334155; background: #020617; color: #f8fafc;
    font-size: 13px; margin-bottom: 14px; outline: none; user-select: text;
  }
  .arcade-input:focus { border-color: #38bdf8; }
  #btn-arcade-start {
    width: 100%; padding: 12px 18px; border-radius: 12px;
    border: 1px solid #93c5fd;
    background: linear-gradient(135deg, #2563eb, #0284c7);
    color: #ffffff; font-size: 15px; font-weight: 900; letter-spacing: 0.8px;
    cursor: pointer; transition: transform 0.15s, filter 0.15s;
    box-shadow: 0 8px 24px rgba(37, 99, 235, 0.45);
  }
  #btn-arcade-start:hover { transform: translateY(-1px); filter: brightness(1.08); }

  /* TELA RESUMO AO ENCERRAR */
  #summary-modal {
    position: absolute; inset: 0;
    background: rgba(2, 6, 23, 0.92);
    backdrop-filter: blur(8px);
    display: none; flex-direction: column; align-items: center; justify-content: center;
    z-index: 49; padding: 18px; color: #f8fafc;
  }
  .summary-card {
    background: #0f172a; border: 1px solid #38bdf8; border-radius: 16px;
    padding: 20px 24px; width: min(560px, 94vw); max-height: 92%; overflow-y: auto;
    box-shadow: 0 20px 55px rgba(0, 0, 0, 0.8);
  }
  .summary-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 8px; margin: 12px 0;
  }
  .summary-box {
    background: #1e293b; border: 1px solid #334155; border-radius: 10px;
    padding: 10px; text-align: center;
  }
  .summary-box b { display: block; font-size: 16px; color: #38bdf8; margin-top: 2px; }
  .summary-box span { font-size: 11px; color: #cbd5e1; }

  /* HUD SUPERIOR ESQUERDO */
  #hud-top-left {
    position: absolute; top: 12px; left: 12px;
    display: flex; flex-direction: column; gap: 6px;
    pointer-events: none; z-index: 10;
  }
  .hud-pill {
    background: rgba(15, 23, 42, 0.84);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(148, 163, 184, 0.32);
    color: #f8fafc; padding: 6px 12px; border-radius: 999px;
    font-size: 12px; font-weight: 600;
    display: flex; align-items: center; gap: 8px;
    width: fit-content;
    box-shadow: 0 4px 14px rgba(0,0,0,0.35);
  }
  .bar-bg {
    width: 100px; height: 9px; background: rgba(30, 41, 59, 0.95);
    border-radius: 999px; overflow: hidden; border: 1px solid rgba(255,255,255,0.14);
  }
  #energy-fill {
    width: 100%; height: 100%;
    background: linear-gradient(90deg, #22c55e, #38bdf8);
    transition: width 0.18s, background 0.2s;
  }

  /* HUD SUPERIOR DIREITO */
  #hud-top-right {
    position: absolute; top: 12px; right: 12px;
    display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 6px; z-index: 15;
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
    background: rgba(127, 29, 29, 0.9);
    border-color: #ef4444; color: #fecaca;
    box-shadow: 0 0 14px rgba(239, 68, 68, 0.38);
  }

  /* NOTIFICAÇÕES COMPACTAS (SEM SPAM — MÁXIMO 2) */
  #alert-stack {
    position: absolute; top: 82px; left: 50%; transform: translateX(-50%);
    display: flex; flex-direction: column; gap: 5px; align-items: center;
    z-index: 16; pointer-events: none; width: min(560px, 88vw);
  }
  .alert-banner {
    padding: 6px 14px; border-radius: 999px; font-weight: 700; font-size: 12px;
    text-align: center; box-shadow: 0 6px 18px rgba(0,0,0,0.42);
    backdrop-filter: blur(6px);
  }
  .alert-iot { background: rgba(194, 65, 12, 0.88); border: 1px solid #fdba74; color: #fff; }
  .alert-danger { background: rgba(185, 28, 28, 0.90); border: 1px solid #fca5a5; color: #fff; }
  .alert-info { background: rgba(29, 78, 216, 0.88); border: 1px solid #93c5fd; color: #fff; }
  .alert-success { background: rgba(21, 128, 61, 0.90); border: 1px solid #86efac; color: #fff; }

  /* BARRA DE EQUILÍBRIO NA ESCALADA (ÁRVORE / POSTE) */
  #climb-hud {
    position: absolute; bottom: 118px; left: 50%; transform: translateX(-50%);
    background: rgba(15, 23, 42, 0.92); border: 1px solid #38bdf8;
    border-radius: 12px; padding: 9px 16px;
    display: none; flex-direction: column; align-items: center; gap: 6px; z-index: 17;
  }
  .balance-track {
    position: relative; width: 230px; height: 13px;
    background: linear-gradient(90deg, #ef4444 0%, #22c55e 35%, #22c55e 65%, #ef4444 100%);
    border-radius: 999px; border: 1px solid #fff;
  }
  #balance-needle {
    position: absolute; top: -3px; left: 50%; width: 8px; height: 19px;
    background: #ffffff; border-radius: 3px; transform: translateX(-50%);
    box-shadow: 0 0 8px #000;
  }

  /* PROMPT DE PROXIMIDADE */
  #interact-prompt {
    position: absolute; bottom: 68px; left: 50%; transform: translateX(-50%);
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid #38bdf8; color: #f8fafc;
    padding: 7px 16px; border-radius: 999px;
    font-size: 12.5px; font-weight: 600;
    display: none; z-index: 14; pointer-events: none;
    box-shadow: 0 4px 16px rgba(0,0,0,0.45);
  }

  /* BOTÕES INFERIORES */
  #hud-bottom-left {
    position: absolute; bottom: 12px; left: 12px;
    display: flex; flex-wrap: wrap; gap: 5px; z-index: 14;
  }
  #hud-bottom-right {
    position: absolute; bottom: 12px; right: 12px;
    display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 5px; z-index: 14;
  }
  .action-btn {
    background: rgba(15, 23, 42, 0.88);
    border: 1px solid rgba(148, 163, 184, 0.42);
    color: #f8fafc; padding: 7px 9px; border-radius: 9px;
    font-size: 11.5px; font-weight: 700; cursor: pointer;
    transition: 0.15s;
  }
  .action-btn:hover { background: #1e293b; border-color: #38bdf8; }
  .action-btn.ar { border-color: #38bdf8; color: #7dd3fc; }
  .action-btn.climb { border-color: #22c55e; color: #86efac; }
  .action-btn.atk { border-color: #f97316; color: #fdba74; }
  .action-btn.dance { border-color: #ec4899; color: #f9a8d4; }
  .action-btn.photo { border-color: #34d399; color: #6ee7b7; }
  .action-btn.creto { border-color: #a78bfa; color: #ddd6fe; }
  .action-btn.lantern { border-color: #fde047; color: #fef08a; }

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
    position: absolute; inset: 0; pointer-events: none; display: none; z-index: 24;
  }
  .viewfinder-box {
    position: absolute; inset: 46px;
    border: 2px dashed rgba(52, 211, 153, 0.85); border-radius: 14px;
    display: flex; flex-direction: column; justify-content: space-between; padding: 14px;
  }
  #flash-screen {
    position: absolute; inset: 0; background: #ffffff; opacity: 0;
    pointer-events: none; transition: opacity 0.3s ease-out; z-index: 29;
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
    display: none; flex-direction: column; justify-content: center; align-items: center;
    z-index: 50; animation: shakeScreen 0.1s infinite;
  }
  @keyframes shakeScreen {
    0% { transform: translate(0, 0) rotate(0deg); }
    25% { transform: translate(-9px, 7px) rotate(-1.2deg); }
    50% { transform: translate(9px, -7px) rotate(1.2deg); }
    75% { transform: translate(-7px, -5px) rotate(-0.8deg); }
    100% { transform: translate(7px, 5px) rotate(0.8deg); }
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
    position: absolute; inset: 0; background: rgba(2, 6, 23, 0.82);
    backdrop-filter: blur(6px); display: none; justify-content: center; align-items: center; z-index: 42;
  }
  .creto-card {
    background: #0f172a; border: 2px solid #3b82f6; border-radius: 16px;
    padding: 20px 24px; width: min(560px, 92vw); box-shadow: 0 20px 50px rgba(0,0,0,0.75);
  }
  .quiz-opt {
    display: block; width: 100%; text-align: left; background: #1e293b; color: #f8fafc;
    border: 1px solid #475569; border-radius: 10px; padding: 10px 14px; margin-top: 8px;
    font-size: 13px; font-weight: 600; cursor: pointer; transition: background 0.15s;
  }
  .quiz-opt:hover { background: #2563eb; border-color: #93c5fd; }

  /* MODAL GAME OVER */
  #gameover-modal {
    position: absolute; inset: 0; background: rgba(69, 10, 10, 0.90);
    backdrop-filter: blur(8px); display: none; flex-direction: column;
    align-items: center; justify-content: center; z-index: 46;
    color: #fef2f2; text-align: center; padding: 24px;
  }
  #gameover-modal h2 { font-size: 30px; color: #fca5a5; margin-bottom: 8px; }
  #gameover-reason { font-size: 15px; color: #fecaca; margin-bottom: 20px; max-width: 520px; }

  /* MODAL RETRÁTIL DE INVENTÁRIO & QUESTS (PAUSA O JOGO) */
  #pause-modal {
    position: absolute; inset: 0; background: rgba(2, 6, 23, 0.86);
    backdrop-filter: blur(8px); display: none; align-items: center; justify-content: center;
    z-index: 40; padding: 18px;
  }
  .modal-box {
    width: 100%; max-width: 900px; max-height: 92%;
    background: #0f172a; border: 1px solid #38bdf8; border-radius: 14px;
    display: flex; flex-direction: column; overflow: hidden;
    box-shadow: 0 20px 50px rgba(0,0,0,0.65);
  }
  .modal-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 18px; background: #1e293b; border-bottom: 1px solid #334155;
  }
  .modal-header h3 { color: #f8fafc; font-size: 15.5px; }
  .modal-tabs {
    display: flex; flex-wrap: wrap; gap: 6px; padding: 10px 18px 0 18px;
    background: #0f172a; border-bottom: 1px solid #1e293b;
  }
  .tab-btn {
    background: #1e293b; color: #94a3b8; border: 1px solid #334155;
    border-bottom: none; padding: 7px 13px; border-radius: 8px 8px 0 0;
    font-size: 12px; font-weight: 600; cursor: pointer;
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
    padding: 4px 0; border-bottom: 1px dashed rgba(148,163,184,0.15); font-size: 12.2px;
  }
  .q-badge { font-size: 11px; padding: 1px 7px; border-radius: 999px; font-weight: 700; }
  .q-done { background: rgba(34, 197, 94, 0.2); color: #4ade80; }
  .q-todo { background: rgba(148, 163, 184, 0.18); color: #cbd5e1; }
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
<div id="game-wrapper">
  <canvas id="c" tabindex="0"></canvas>
  <div id="damage-flash"></div>
  <div id="flash-screen"></div>

  <!-- TELA INICIAL ARCADE START -->
  <div id="arcade-start-overlay">
    <div class="arcade-card">
      <div class="arcade-badge">🕹️ BLOCO 734 • CT / UFC • GRATUITO</div>
      <div class="arcade-title">CAMPUS 3D INTERATIVO</div>
      <div class="arcade-sub">
        Feito com código aberto/educacional (sem app na loja — roda direto no navegador!).
        Escolha seu avatar, explore o Bloco 734 e dispute o placar 1v1.
      </div>
      <input
        id="player-email-input"
        class="arcade-input"
        type="email"
        placeholder="Seu e-mail (opcional — salvo no resumo da sessão)"
      />
      <button id="btn-arcade-start">▶ START (TELA CHEIA)</button>
    </div>
  </div>

  <!-- TELA DE SELEÇÃO DE PERSONAGEM -->
  <div id="character-select-overlay" style="display:none; position:absolute; inset:0; background:rgba(2,6,23,0.95); backdrop-filter:blur(8px); z-index:48; padding:18px; align-items:center; justify-content:center; color:#f8fafc;">
    <div class="arcade-card" style="width:min(640px, 95vw);">
      <div class="arcade-badge">🎭 ESCOLHA SEU AVATAR • UFC</div>
      <div class="arcade-title" style="font-size:20px;">SELECIONE SEU PERSONAGEM</div>
      <div class="arcade-sub" style="margin-bottom:8px;">
        Escolha quem vai explorar o campus. Pressione <b>ESC</b> a qualquer momento para sair da tela cheia.
      </div>
      <div class="char-grid" id="char-grid-list"></div>
      <button id="btn-confirm-char" style="width:100%; padding:11px 16px; border-radius:12px; border:1px solid #86efac; background:linear-gradient(135deg,#16a34a,#0284c7); color:#fff; font-size:14px; font-weight:900; cursor:pointer;">
        🚀 ENTRAR NO CAMPUS
      </button>
    </div>
  </div>

  <!-- TELA RESUMO AO ENCERRAR -->
  <div id="summary-modal">
    <div class="summary-card">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <h3 style="color:#38bdf8; font-size:18px;">🏁 Resumo da Experiência — Bloco 734</h3>
        <button class="hud-btn" id="btn-close-summary">▶ Continuar</button>
      </div>
      <p style="font-size:12.5px; color:#cbd5e1; margin-top:6px;">
        Obrigado por explorar o <b>Bloco 734 (CT/UFC)</b>! Aqui está o registro consolidado da sua sessão:
      </p>
      <div class="summary-grid">
        <div class="summary-box"><span>⏱️ Tempo Cronometrado</span><b id="sum-time">00:00</b></div>
        <div class="summary-box"><span>🎒 Inventário</span><b id="sum-inv">0/8</b></div>
        <div class="summary-box"><span>📸 Fotos Registradas</span><b id="sum-photos">0/5</b></div>
        <div class="summary-box"><span>🎓 Acertos (Creto)</span><b id="sum-creto">0/4</b></div>
        <div class="summary-box"><span>🪟 Vidros Quebrados</span><b id="sum-windows">0/10</b></div>
        <div class="summary-box"><span>🧭 Mini-Quests</span><b id="sum-mini">0/5</b></div>
      </div>
      <div id="sum-email-box" style="font-size:12px; color:#93c5fd; background:#1e293b; border:1px solid #334155; border-radius:8px; padding:10px; margin-bottom:12px;">
        📧 E-mail: <b id="sum-email-val">Não informado</b>
      </div>
      <div style="display:flex; flex-wrap:wrap; gap:8px; justify-content:flex-end;">
        <a id="btn-mailto-summary" class="hud-btn" style="text-decoration:none; display:none;" target="_blank">✉️ Preparar E-mail</a>
        <button class="hud-btn" id="btn-restart-session" style="background:#16a34a; border-color:#86efac;">🔄 Nova Sessão (START)</button>
      </div>
    </div>
  </div>

  <!-- HUD SUPERIOR ESQUERDO -->
  <div id="hud-top-left">
    <div class="hud-pill">
      <span>⏱️ <b id="timer-badge" style="color:#38bdf8;">00:00</b></span>
      <span>•</span>
      <span id="pvp-badge" style="color:#facc15;">⚔️ 1v1: Você 0 pts vs Rival 0 pts</span>
      <span>❤️ Life</span>
      <div class="bar-bg"><div id="energy-fill"></div></div>
      <span id="hp-text" style="font-size:11.5px; color:#cbd5e1;">100%</span>
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
      <span>🎓 Creto: <b id="creto-count" style="color:#c084fc;">0/4</b></span>
      <span>•</span>
      <span id="proj-badge" style="color:#a5f3fc;">📷 Perspectiva (2 PF)</span>
    </div>
  </div>

  <!-- HUD SUPERIOR DIREITO -->
  <div id="hud-top-right">
    <button class="hud-btn" id="btn-survival">💀 Sobrevivência: OFF</button>
    <button class="hud-btn" id="btn-audio">🔊 Som: ON</button>
    <button class="hud-btn" id="btn-quests">📜 Inventário & Quests (Q)</button>
    <button class="hud-btn" id="btn-summary" style="border-color:#f59e0b; color:#fde68a;">🏁 Resumo / Sair</button>
    <button class="hud-btn" id="btn-fullscreen">⛶ Tela Cheia</button>
  </div>

  <!-- NOTIFICAÇÕES COMPACTAS (MÁX 2, SEM SPAM) -->
  <div id="alert-stack"></div>

  <!-- BARRA DE EQUILÍBRIO NA ESCALADA (ÁRVORE & POSTE) -->
  <div id="climb-hud">
    <div style="font-size:12px; font-weight:700; color:#7dd3fc;" id="climb-label">🧗 Escalando! [W/S] ou Analógico Esq. subir/descer • [A/D] equilibrar • [R] soltar</div>
    <div class="balance-track"><div id="balance-needle"></div></div>
  </div>

  <!-- VISOR FOTOGRÁFICO -->
  <div id="photo-overlay">
    <div class="viewfinder-box">
      <div style="display:flex; justify-content:space-between; font-weight:700; font-size:13.5px; color:#6ee7b7; text-shadow:0 2px 6px #000;">
        <span>[REC ●] MODO FOTO TÉCNICA CT</span>
        <span id="photo-target-hint">Aponte para um alvo e pressione [P] para registrar e sair</span>
      </div>
      <div style="text-align:center; font-size:28px; color:#6ee7b7; opacity:0.8;">✛</div>
      <div style="text-align:center; font-size:12px; color:#e2e8f0; text-shadow:0 1px 4px #000;">
        Alvos: Painéis Fotovoltaicos (Telhado) • Escultura • Fonte • Pássaro CT • Prof. Creto • Cachorro (Fundos)
      </div>
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

  <!-- ATALHOS INFERIORES ESQUERDOS (PROJEÇÃO WEBGL) -->
  <div id="hud-bottom-left">
    <button class="action-btn" id="btn-vp" title="Alternar 1, 2 e 3 Pontos de Fuga">📐 Pontos Fuga</button>
    <button class="action-btn" id="btn-iso" title="Projeção Isométrica">🧊 Isométrica</button>
  </div>

  <div id="xbox-sticks" style="position:absolute;inset:0;pointer-events:none;z-index:25;display:none;touch-action:none;">
    <div id="stick-walk" style="position:absolute;left:20px;bottom:68px;width:116px;height:116px;border-radius:50%;background:rgba(15,23,42,0.55);border:2px solid rgba(56,189,248,0.65);pointer-events:auto;touch-action:none;">
      <div class="knob" style="position:relative;left:35px;top:35px;width:42px;height:42px;border-radius:50%;background:rgba(56,189,248,0.9);box-shadow:0 4px 12px rgba(0,0,0,0.45);"></div>
    </div>
    <div id="stick-look" style="position:absolute;right:20px;bottom:68px;width:116px;height:116px;border-radius:50%;background:rgba(15,23,42,0.55);border:2px solid rgba(167,139,250,0.65);pointer-events:auto;touch-action:none;">
      <div class="knob" style="position:relative;left:35px;top:35px;width:42px;height:42px;border-radius:50%;background:rgba(167,139,250,0.9);box-shadow:0 4px 12px rgba(0,0,0,0.45);"></div>
    </div>
  </div>

  <div id="hud-bottom-right">
    <button class="action-btn lantern" id="btn-lantern">🔦 Lanterna [L]</button>
    <button class="action-btn photo" id="btn-photo">📸 Foto [P]</button>
    <button class="action-btn creto" id="btn-creto">🎓 Falar com Creto [E]</button>
    <button class="action-btn dance" id="btn-dance">💃 Dançar [C]</button>
    <button class="action-btn climb" id="btn-climb">🧗 Escalar [R]</button>
    <button class="action-btn ar" id="btn-interact">🔍 Interagir [E]</button>
    <button class="action-btn atk" id="btn-attack">⚔️ Atacar [F]</button>
    <button class="action-btn" id="btn-jump">⬆️ Pular [Espaço]</button>
  </div>

  <!-- LETTERBOX CUTSCENE -->
  <div class="letterbox" id="lb-top"></div>
  <div class="letterbox" id="lb-bottom"></div>
  <div id="cutscene-caption">☕ Preparando um cafezinho expresso no Centro de Tecnologia...</div>

  <!-- JUMPSCARE -->
  <div id="jumpscare-overlay">
    <div id="js-emoji" style="font-size:104px; line-height:1;">🦉👁️👁️</div>
    <div id="js-title" style="font-size:30px; font-weight:900; color:#ef4444; margin-top:10px; text-shadow:0 0 20px #000;">
      SUSTO NO CAMPUS CT!
    </div>
    <div id="js-sub" style="font-size:14.5px; color:#fecaca; margin-top:6px;">
      Você entrou em uma área oculta vigiada!
    </div>
  </div>

  <!-- MODAL QUIZ PROF. CRETO -->
  <div id="creto-modal">
    <div class="creto-card">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="font-weight:800; font-size:15.5px; color:#60a5fa;">🎓 Prof. Creto — Quiz de Computação Gráfica</span>
        <span id="creto-progress" style="background:#1d4ed8; padding:3px 9px; border-radius:6px; font-size:12px; font-weight:700; color:#fff;">Pergunta 1 / 4</span>
      </div>
      <p id="creto-question" style="margin-top:14px; font-size:14px; line-height:1.5; color:#f1f5f9;"></p>
      <div id="creto-options"></div>
      <div id="creto-feedback" style="margin-top:12px; font-size:13px; font-weight:700; min-height:20px;"></div>
      <div style="margin-top:14px; display:flex; justify-content:flex-end;">
        <button class="hud-btn" id="btn-close-creto">Fechar [ESC]</button>
      </div>
    </div>
  </div>

  <!-- MODAL GAME OVER -->
  <div id="gameover-modal">
    <h2>💀 DUMB WAYS TO DIE — CT EDITION</h2>
    <p id="gameover-reason">Você sofreu um acidente no campus do Bloco 734!</p>
    <button class="hud-btn" id="btn-respawn" style="font-size:15px; padding:11px 24px; background:#dc2626; border-color:#fca5a5;">
      🔄 Renascer na Praça do Bloco 734
    </button>
  </div>

  <!-- MODAL RETRÁTIL DE INVENTÁRIO & QUESTS (PAUSA O JOGO) -->
  <div id="pause-modal">
    <div class="modal-box">
      <div class="modal-header">
        <h3>⏸️ JOGO PAUSADO — Inventário, Mini-Quests CT & Quests CG I</h3>
        <button class="hud-btn" id="btn-resume" style="background:#0284c7; border-color:#38bdf8;">▶ Voltar ao Jogo (Q)</button>
      </div>
      <div class="modal-tabs">
        <button class="tab-btn active" data-tab="tab-inv">🎒 Inventário (8 Itens)</button>
        <button class="tab-btn" data-tab="tab-mini">🧭 Mini-Quests Interativas CT</button>
        <button class="tab-btn" data-tab="tab-quests">🎓 Quests CG I (Atos I–VIII)</button>
        <button class="tab-btn" data-tab="tab-help">🎮 Controles & Mecânicas</button>
      </div>

      <div class="modal-body" id="tab-inv">
        <p>Aproxime-se das <b>Mesas de Trabalho</b> (sob o Abrigo Metálico e na Praça) e pressione <b>[E]</b> para coletar suas entregas. Coletar todos os 8 itens ativa a <b>Dança da Vitória</b>!</p>
        <div class="inv-grid" id="inv-grid-container"></div>
      </div>

      <div class="modal-body" id="tab-mini" style="display:none;">
        <div class="act-block">
          <h4>🏆 Progresso das Camadas Interativas do Campus CT</h4>
          <div class="q-row"><span>📸 Álbum Fotográfico & Solar (<b id="mq-foto-n">0</b>/5 fotos — inclui Painéis Fotovoltaicos)</span><span class="q-badge q-todo" id="mq-foto">Em andamento</span></div>
          <div class="q-row"><span>🌳 Subir na Árvore sem perder o equilíbrio ([R] + equilibrar A/D)</span><span class="q-badge q-todo" id="mq-arvore">Pendente</span></div>
          <div class="q-row"><span>💡 Subir no Poste de Iluminação sem cair ([R] + equilibrar A/D)</span><span class="q-badge q-todo" id="mq-poste">Pendente</span></div>
          <div class="q-row"><span>☀️ Subir ao Telhado pela Escada Lateral e Fotografar Painéis Solares</span><span class="q-badge q-todo" id="mq-solar">Pendente</span></div>
          <div class="q-row"><span>🚸 Atravessar a Rua na Faixa de Pedestres (X=27) e Retornar</span><span class="q-badge q-todo" id="mq-rua">Pendente</span></div>
          <div class="q-row"><span>🎓 Quiz de Computação Gráfica com Prof. Creto (<b id="mq-creto-n">0</b>/4 acertos)</span><span class="q-badge q-todo" id="mq-creto">Pendente</span></div>
        </div>
      </div>

      <div class="modal-body" id="tab-quests" style="display:none;">
        <div class="act-block">
          <h4>🏰 ATO I — A Construção do Mundo</h4>
          <div class="q-row"><span>🎯 Sub-quest 1.1 & 1.2 — Bloco 734 CT Coerente no Primeiro Octante (X, Y, Z &ge; 0)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🎯 Quest 2 & 3 — Zoo de Primitivas, Painéis Solares, Materiais & Texturas Procedurais</span><span class="q-badge q-done">✔ Concluída</span></div>
        </div>
        <div class="act-block">
          <h4>⚔️ ATO II — As Transformações Mágicas</h4>
          <div class="q-row"><span>🎯 Quests 4, 5 & 6 — Translação, Rotação X/Y/Z, Quatérnios, Escala, Cisalhamento & Espelho</span><span class="q-badge q-done">✔ Concluída</span></div>
        </div>
        <div class="act-block">
          <h4>🔦 ATO III — A Luz e a Escuridão (Totens Interativos)</h4>
          <div class="q-row"><span>🔦 Sub-quest Bônus 7.3 — "O Holofote" (Totem Liga/Desliga Spots)</span><span class="q-badge q-bonus" id="q-spot-badge">+1.0 pts</span></div>
          <div class="q-row"><span>☀️ Sub-quest Bônus 7.4 — "O Sol Direcional" (Totem Liga/Desliga Sol)</span><span class="q-badge q-bonus" id="q-sun-badge">+0.5 pts</span></div>
        </div>
        <div class="act-block">
          <h4>📷 ATOS IV–VIII — Projeções, Sombras, Picking & 500×500</h4>
          <div class="q-row"><span>📷 Pontos de Fuga (1/2/3 PF), Projeção Isométrica & Oblíqua (na aba Ray Casting)</span><span class="q-badge q-bonus">+2.5 pts</span></div>
          <div class="q-row"><span>🌑🎯🖼️ Sombras, Picking O(1), Mosaico 500×500 & Sensores IoT</span><span class="q-badge q-done">✔ Concluída</span></div>
        </div>
      </div>

      <div class="modal-body" id="tab-help" style="display:none;">
        <div class="act-block">
          <h4>🎮 Controles Unificados</h4>
          <p>• <b>W A S D / Setas / Analógico Esq.:</b> Caminhar (encoste na <b>Escada Lateral X=59.35, Z=24</b> e suba direto ao telhado!).<br/>
             • <b>Arrastar Mouse / Roda / Analógico Dir.:</b> Girar câmera 360° e aplicar Zoom.<br/>
             • <b>[E] Interagir:</b> Coleta itens das mesas, fala com <b>Prof. Creto</b> (X=9.5, Z=22.2), toma café na ilha (X=35.8, Z=18.8), aciona Totens 3D ou inspeciona monumentos.<br/>
             • <b>[R] Escalar Árvore/Poste:</b> Sobe em árvores ou postes (no celular, use o <b>Analógico Esquerdo</b> para subir/descer e equilibrar; no PC use <b>W/S</b> e <b>A/D</b>; solte com <b>R</b> ou <b>Espaço</b>).<br/>
             • <b>[P] Modo Foto:</b> Primeiro <b>[P]</b> abre o visor; segundo <b>[P]</b> registra a foto e encerra o modo foto.<br/>
             • <b>[F] Atacar:</b> Quebra as janelas curvas do Bloco 734.<br/>
             • <b>[C] Comemorar:</b> Executa dança comemorativa.<br/>
             • <b>[Q] Inventário & Quests:</b> Pausa o jogo e exibe progresso.</p>
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
  let lanternEnabled = MODO_NOITE;
  let isDragging = false, lastMX = 0, lastMY = 0;

  const wrapper = document.getElementById('game-wrapper');
  const canvas = document.getElementById('c');
  wrapper.addEventListener('mouseenter', () => {
    if (gameStarted && !isPaused && !inQuizModal && !inSummaryModal) canvas.focus();
  });
  const AVATAR_PROFILES = [
    { id: "menino", nome: "Menino", tag: "Infantil", torso: 0x38bdf8, cab: 0xfed7aa, calca: 0x1d4ed8, esc: 0.88, prop: "none" },
    { id: "menina", nome: "Menina", tag: "Infantil", torso: 0xf472b6, cab: 0xfed7aa, calca: 0x7c3aed, esc: 0.88, prop: "none" },
    { id: "ado_m", nome: "Adolescente Menino", tag: "Juvenil", torso: 0x0ea5e9, cab: 0xfacc15, calca: 0x1e293b, esc: 0.96, prop: "none" },
    { id: "ado_f", nome: "Adolescente Menina", tag: "Juvenil", torso: 0xec4899, cab: 0xfacc15, calca: 0x334155, esc: 0.96, prop: "none" },
    { id: "cc_m", nome: "Graduação C. Ciências (Menino)", tag: "Centro de Ciências", torso: 0x14b8a6, cab: 0xfed7aa, calca: 0x0f172a, esc: 1.0, prop: "livro" },
    { id: "cc_f", nome: "Graduação C. Ciências (Menina)", tag: "Centro de Ciências", torso: 0x2dd4bf, cab: 0xfed7aa, calca: 0x1e293b, esc: 1.0, prop: "livro" },
    { id: "ic_m", nome: "Bolsista IC (Menino)", tag: "Iniciação Científica", torso: 0x6366f1, cab: 0xfdba74, calca: 0x1e293b, esc: 1.0, prop: "cracha" },
    { id: "ic_f", nome: "Bolsista IC (Menina)", tag: "Iniciação Científica", torso: 0x8b5cf6, cab: 0xfdba74, calca: 0x0f172a, esc: 1.0, prop: "cracha" },
    { id: "ica_m", nome: "Estudante ICA (Menino)", tag: "Cultura e Arte", torso: 0xf97316, cab: 0xfed7aa, calca: 0x3f2e21, esc: 1.0, prop: "arte" },
    { id: "ica_f", nome: "Estudante ICA (Menina)", tag: "Cultura e Arte", torso: 0xf43f5e, cab: 0xfed7aa, calca: 0x27272a, esc: 1.0, prop: "arte" },
    { id: "saude", nome: "Área da Saúde (Porangabuçu)", tag: "Med/Enf/Fisio", torso: 0xf8fafc, cab: 0xfed7aa, calca: 0x0284c7, esc: 1.0, prop: "saude" },
    { id: "ct_eng", nome: "Centro de Tecnologia (Engenharia)", tag: "CT / UFC", torso: 0xeab308, cab: 0xfdba74, calca: 0x1e293b, esc: 1.02, prop: "capacete" },
    { id: "comp", nome: "Estudante de Computação", tag: "CG / Sistemas", torso: 0x2563eb, cab: 0xfed7aa, calca: 0x0f172a, esc: 1.0, prop: "oculos" },
    { id: "mestrado", nome: "Pós-Graduação: Mestrado", tag: "Mestre / UFC", torso: 0x059669, cab: 0xfed7aa, calca: 0x111827, esc: 1.02, prop: "livro" },
    { id: "doutorado", nome: "Pós-Graduação: Doutorado", tag: "Doutor / UFC", torso: 0x4338ca, cab: 0xfed7aa, calca: 0x090d16, esc: 1.03, prop: "livro" }
  ];

  let selectedAvatar = AVATAR_PROFILES[12]; // Computação por padrão
  let avatarPropMesh = null;

  function aplicarAvatarSelecionado(prof) {
    selectedAvatar = prof;
    player.grp.scale.setScalar(prof.esc);
    player.torso.material.color.setHex(prof.torso);
    player.cab.material.color.setHex(prof.cab);
    player.bE.material.color.setHex(prof.cab);
    player.bD.material.color.setHex(prof.cab);
    player.pE.material.color.setHex(prof.calca);
    player.pD.material.color.setHex(prof.calca);

    if (avatarPropMesh) {
      avatarPropMesh.parent && avatarPropMesh.parent.remove(avatarPropMesh);
      avatarPropMesh = null;
    }
    if (prof.prop === "capacete") {
      avatarPropMesh = new THREE.Mesh(new THREE.BoxGeometry(0.54, 0.20, 0.54), new THREE.MeshStandardMaterial({ color: 0xfacc15 }));
      avatarPropMesh.position.set(0, 2.12, 0);
      player.grp.add(avatarPropMesh);
    } else if (prof.prop === "saude") {
      avatarPropMesh = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.10, 0.24), new THREE.MeshStandardMaterial({ color: 0xef4444 }));
      avatarPropMesh.position.set(0, 1.38, 0.22);
      player.grp.add(avatarPropMesh);
    } else if (prof.prop === "cracha" || prof.prop === "livro" || prof.prop === "arte") {
      const corP = prof.prop === "cracha" ? 0x38bdf8 : (prof.prop === "arte" ? 0xf472b6 : 0x4ade80);
      avatarPropMesh = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.06, 0.22), new THREE.MeshStandardMaterial({ color: corP }));
      avatarPropMesh.position.set(0, -0.35, 0.16);
      player.bD.add(avatarPropMesh);
    }
  }

  const charGridList = document.getElementById('char-grid-list');
  AVATAR_PROFILES.forEach((prof, i) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'char-btn' + (i === 12 ? ' active' : '');
    btn.innerHTML = `<div>${prof.nome}</div><div style="font-size:10px; color:#94a3b8; font-weight:600;">${prof.tag}</div>`;
    btn.onclick = () => {
      document.querySelectorAll('.char-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      aplicarAvatarSelecionado(prof);
      SoundEngine.playSwitch();
    };
    charGridList.appendChild(btn);
  });

  function entrarTelaCheiaJogo() {
    if (!document.fullscreenElement && !wrapper.classList.contains('pseudo-fullscreen')) {
      if (wrapper.requestFullscreen) {
        wrapper.requestFullscreen().catch(() => wrapper.classList.add('pseudo-fullscreen'));
      } else {
        wrapper.classList.add('pseudo-fullscreen');
      }
    }
  }

  document.getElementById('btn-arcade-start').addEventListener('click', () => {
    SoundEngine.init();
    playerEmail = (document.getElementById('player-email-input').value || '').trim();
    document.getElementById('arcade-start-overlay').style.display = 'none';
    document.getElementById('character-select-overlay').style.display = 'flex';
    entrarTelaCheiaJogo();
    SoundEngine.playSwitch();
  });

  document.getElementById('btn-confirm-char').addEventListener('click', () => {
    aplicarAvatarSelecionado(selectedAvatar);
    document.getElementById('character-select-overlay').style.display = 'none';
    gameStarted = true;
    atualizarHUD();
    SoundEngine.playCollect();
    mostrarAlerta(`🎭 Jogando como ${selectedAvatar.nome}!`, 'alert-info', 2200, 'info');
    canvas.focus();
  });
  // ============================================================================
  // 1. MOTOR DE ÁUDIO PROCEDURAL (WEB AUDIO API)
  // ============================================================================
  const SoundEngine = {
    ctx: null,
    enabled: true,
    init() {
      if (!this.ctx) {
        const AC = window.AudioContext || window.webkitAudioContext;
        if (AC) this.ctx = new AC();
      }
      if (this.ctx && this.ctx.state === 'suspended') {
        this.ctx.resume().catch(() => {});
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
    playStep() { const p = 155 + Math.random() * 35; this.tone(p, 0.055, 'triangle', 0.07, 0, 75); },
    playJump() { this.tone(220, 0.18, 'sine', 0.09, 0, 520); },
    playSwitch() { this.tone(580, 0.07, 'square', 0.06, 0); this.tone(860, 0.09, 'square', 0.06, 0.07); },
    playCollect() { [523.25, 659.25, 783.99, 1046.5].forEach((f, i) => this.tone(f, 0.14, 'triangle', 0.09, i * 0.065)); },
    playError() { this.tone(185, 0.32, 'sawtooth', 0.14, 0, 110); },
    playIoT() { this.tone(880, 0.12, 'sine', 0.06, 0, 1180); },
    playWhistle() {
      this.tone(2600, 0.14, 'triangle', 0.11, 0, 2900);
      this.tone(2750, 0.22, 'triangle', 0.11, 0.16, 2400);
    },
    playBark() {
      this.tone(190, 0.13, 'sawtooth', 0.14, 0, 95);
      this.tone(220, 0.15, 'sawtooth', 0.14, 0.17, 85);
    },
    playChirp() {
      this.tone(1950, 0.08, 'sine', 0.04, 0, 2650);
      this.tone(2400, 0.09, 'sine', 0.04, 0.10, 1850);
    },
    playPhoto() {
      this.tone(1200, 0.05, 'square', 0.10, 0, 400);
      this.tone(1600, 0.07, 'sine', 0.09, 0.055, 900);
    },
    playGlass() {
      [1400, 1950, 2400, 1100, 1750].forEach((f, i) => this.tone(f + Math.random()*300, 0.12, 'sawtooth', 0.06, i * 0.025, 320));
      this.tone(110, 0.65, 'sine', 0.08, 0.1, 55);
    },
    playHonk() { this.tone(415, 0.22, 'sawtooth', 0.07, 0); this.tone(523, 0.22, 'sawtooth', 0.06, 0); },
    playScare() {
      this.tone(110, 0.85, 'sawtooth', 0.22, 0, 520);
      this.tone(920, 0.38, 'sawtooth', 0.14, 0, 220);
    },
    playCoffee() { [330, 392, 494, 587, 659].forEach((f, i) => this.tone(f, 0.25, 'sine', 0.07, i * 0.22)); },
    playDanceBeat() {
      const notes = [
        [329.6, 0.0], [329.6, 0.22], [392.0, 0.44], [440.0, 0.66],
        [392.0, 0.96], [329.6, 1.22], [293.7, 1.48], [329.6, 1.75],
        [440.0, 2.15], [392.0, 2.40], [329.6, 2.65], [392.0, 2.95], [523.3, 3.25]
      ];
      notes.forEach(([f, d]) => this.tone(f, 0.18, 'square', 0.065, d));
    }
  };

  document.getElementById('btn-audio').addEventListener('click', (e) => {
    SoundEngine.enabled = !SoundEngine.enabled;
    e.target.textContent = SoundEngine.enabled ? '🔊 Som: ON' : '🔇 Som: OFF';
    if (SoundEngine.enabled) SoundEngine.playSwitch();
    canvas.focus();
  });

  // ============================================================================
  // 2. RENDERER WEBGL & CÂMERAS (PERSPECTIVA & ISOMÉTRICA)
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

  let projMode = 'perspectiva'; // 'perspectiva' | 'isometrica'
  let vanishingMode = 2;

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

    document.getElementById('q-sun-badge').textContent = sunEnabled ? '✔ Sol ON (+0.5 pts)' : '⚠️ Sol OFF';
    document.getElementById('q-spot-badge').textContent = spotsEnabled ? '✔ Spots ON (+1.0 pts)' : '⚠️ Spots OFF';
  }

  // ============================================================================
  // 4. MATERIAIS & CONSTRUÇÃO DO CENÁRIO BLOCO 734 (1º OCTANTE)
  // ============================================================================
  const matAsfalto = new THREE.MeshStandardMaterial({ color: 0x262930, roughness: 0.9 });
  const matFaixa = new THREE.MeshBasicMaterial({ color: 0xeab308 });
  const matFaixaBranca = new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.5 });
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

  function criarPlacaFlutuante(texto, corFundo = '#0f172a', corBorda = '#38bdf8', largura = 3.6, altura = 0.72) {
    const cv = document.createElement('canvas');
    cv.width = 512; cv.height = 104;
    const ctx = cv.getContext('2d');
    ctx.fillStyle = corFundo;
    ctx.fillRect(4, 4, 504, 96);
    ctx.strokeStyle = corBorda;
    ctx.lineWidth = 8;
    ctx.strokeRect(4, 4, 504, 96);
    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 31px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(texto, 256, 54);
    const tex = new THREE.CanvasTexture(cv);
    const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true }));
    spr.scale.set(largura, altura, 1);
    return spr;
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

  // Piso Urbano + Faixa de Pedestres em X = 25.5..28.5 + Calçada Ampliada na Frente do Bloco
  addBox(68, 0.1, 11.5, matAsfalto, 35, 0.05, 7.25, 0, false);
  addBox(66, 0.02, 0.14, matFaixa, 35, 0.11, 9.8, 0, false);
  addBox(66, 0.02, 0.14, matFaixa, 35, 0.11, 10.2, 0, false);
  for (let zf = 2.2; zf <= 12.2; zf += 1.15) {
    addBox(3.2, 0.02, 0.56, matFaixaBranca, 27.0, 0.11, zf, 0, false);
  }
  addBox(68, 0.2, 48, matCalcada, 35, 0.10, 37.0, 0, false);
  addBox(68, 0.2, 4.0, matCalcada, 35, 0.10, 14.5, 0, false);

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

  // Painéis Fotovoltaicos no Telhado (Y = 6.60m) + Balizador IoT de Borda
  [29.0, 34.0].forEach((zf) => {
    const painel = new THREE.Mesh(new THREE.BoxGeometry(11.0, 0.12, 2.2), matSolar);
    painel.position.set(48.0, 7.05, zf);
    painel.rotation.x = THREE.MathUtils.degToRad(22);
    painel.castShadow = true;
    scene.add(painel);

    const moldura = new THREE.Mesh(new THREE.BoxGeometry(11.15, 0.08, 2.32), matMetalGalv);
    moldura.position.set(48.0, 6.98, zf);
    moldura.rotation.x = THREE.MathUtils.degToRad(22);
    scene.add(moldura);
  });

  registrarAR(
    'Usina Solar Fotovoltaica (Cobertura Bloco 734)',
    'ENERGIA RENOVÁVEL • ALVO FOTOGRÁFICO [P]',
    'Arranjo fotovoltaico inclinado a 22° na cobertura do Bloco 734. Suba pela escada lateral e pressione [P] para fotografar!',
    'Primitivas: Prismas Rotacionados (Rx=22°) • Y=7.0m',
    [48.0, 7.0, 31.5], [11.5, 1.2, 7.5]
  );

  const iotBeacon = new THREE.Mesh(
    new THREE.SphereGeometry(0.28, 16, 16),
    new THREE.MeshStandardMaterial({ color: 0xf97316, emissive: 0xea580c, emissiveIntensity: 0.9 })
  );
  iotBeacon.position.set(38.5, 6.88, 21.2);
  scene.add(iotBeacon);

  // Janelas Quebráveis + Derramamento de Asfalto
  const breakableWindows = [];
  const asphaltSpills = [];
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
      rotY, larg, alt, hp: 3, maxHp: 3, broken: false
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

    asphaltSpills.push({
      streamMesh, puddleMesh,
      x: puddlePos.x, z: puddlePos.z,
      topY: winObj.pos.y - winObj.alt * 0.35, progress: 0.0
    });
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
    "REFRAÇÃO n=1.333 • ÁREA VIGIADA",
    "Lâmina d'água com 3 spots submersos. Saltar do telhado na água amortece a queda, mas permanecer dentro da fonte aciona os vigilantes!",
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
    "PATRIMÔNIO PROTEGIDO • 5 LOSANGOS",
    "Cinco quadros metálicos rotacionados a 45° e orientados a -24° em Y. Evite encostar nela ou os vigilantes intervirão!",
    "Translação · Rotação Z(45°) & Y(-24°) · Escala",
    [15.0, 2.0, 22.0], [3.6, 3.8, 3.8]
  );

  // Escada Lateral de Acesso Direto ao Telhado (X=59.35, Z=24.0)
  addBox(0.55, 6.6, 1.1, matEsquadria, 59.35, 3.4, 24.0);
  for (let deg = 0.6; deg <= 6.4; deg += 0.45) {
    addBox(0.65, 0.06, 0.95, matMetalGalv, 59.35, deg, 24.0);
  }
  const placaEscada = criarPlacaFlutuante("🪜 ESCADA TELHADO (Subida Direta)", "#0f172a", "#38bdf8", 3.4, 0.65);
  placaEscada.position.set(59.6, 2.6, 24.0);
  scene.add(placaEscada);

  // Árvores Escaláveis ([R]) + Coruja do Jumpscare + Arbustos Leves
  const arvoresCoords = [
    { nome: "Árvore Bosque Esq.", x: 8.5, z: 25.0, h: 4.5, esc: 1.15, temCoruja: false },
    { nome: "Árvore Sombria (Coruja)", x: 11.5, z: 38.0, h: 4.9, esc: 1.25, temCoruja: true },
    { nome: "Árvore Praça Dir.", x: 37.5, z: 16.2, h: 3.8, esc: 0.90, temCoruja: false },
    { nome: "Árvore Calçada Oeste", x: 6.2, z: 17.5, h: 4.2, esc: 1.05, temCoruja: false },
    { nome: "Árvore Jardim Lateral", x: 7.5, z: 32.5, h: 4.4, esc: 1.10, temCoruja: false }
  ];
  const troncoGeo = new THREE.CylinderGeometry(0.28, 0.42, 3.2, 10);
  const copaGeo = new THREE.SphereGeometry(1.0, 10, 8);
  const arbustoGeo = new THREE.DodecahedronGeometry(0.62, 0);

  arvoresCoords.forEach((arv) => {
    const tMesh = new THREE.Mesh(troncoGeo, matTronco);
    tMesh.scale.set(arv.esc, arv.esc, arv.esc);
    tMesh.position.set(arv.x, 0.2 + 1.6 * arv.esc, arv.z);
    tMesh.castShadow = true;
    scene.add(tMesh);
    [[0, 3.6, 0, 1.65, matFolha1], [-1.1, 3.2, 0.6, 1.25, matFolha2], [1.1, 3.3, -0.5, 1.30, matFolha2], [0.3, 4.5, 0.4, 1.20, matFolha1]].forEach(([ox, oy, oz, r, m]) => {
      const cMesh = new THREE.Mesh(copaGeo, m);
      cMesh.scale.setScalar(r * arv.esc);
      cMesh.position.set(arv.x + ox * arv.esc, oy * arv.esc, arv.z + oz * arv.esc);
      cMesh.castShadow = true;
      scene.add(cMesh);
    });
    if (arv.temCoruja) {
      const coruja = new THREE.Mesh(
        new THREE.ConeGeometry(0.35, 0.95, 10),
        new THREE.MeshStandardMaterial({ color: 0x312e81, emissive: 0x1e1b4b })
      );
      coruja.position.set(arv.x, 0.2 + 5.1 * arv.esc, arv.z);
      scene.add(coruja);
    }
    registrarAR(arv.nome, "ESCALÁVEL [R] • REFÚGIO CONTRA VIGILANTES", "Pressione [R] próximo ao tronco para escalar usando W/S (ou Analógico Esq. no celular) e equilibrando com A/D!", "Cilindro + Esferas (Quatérnios)", [arv.x, 2.8 * arv.esc, arv.z], [3.4 * arv.esc, 5.4 * arv.esc, 3.4 * arv.esc]);
  });

  const arbustosPos = [
    [6.5, 21.5, 0.95], [9.8, 21.2, 0.80], [7.2, 28.8, 1.05], [10.2, 34.5, 0.90],
    [13.5, 15.2, 0.85], [17.8, 15.0, 0.78], [39.5, 15.4, 0.85], [51.5, 15.6, 0.95]
  ];
  arbustosPos.forEach(([ax, az, sc], idx) => {
    const arb = new THREE.Mesh(arbustoGeo, idx % 2 === 0 ? matFolha2 : matFolha1);
    arb.scale.set(sc * 1.2, sc * 0.85, sc);
    arb.position.set(ax, 0.2 + 0.45 * sc, az);
    arb.castShadow = true;
    arb.receiveShadow = true;
    scene.add(arb);
  });

  // Postes de Iluminação Escaláveis ([R])
  const postesData = [
    { nome: "Poste #1", x: 9.5, z: 21.0, ox: 0.6, h: 5.8 },
    { nome: "Poste #2", x: 18.5, z: 16.2, ox: 0.4, h: 5.8 },
    { nome: "Poste #3", x: 41.0, z: 16.5, ox: 0.0, h: 5.8 },
    { nome: "Poste #4", x: 54.5, z: 17.5, ox: -0.5, h: 5.8 }
  ];
  postesData.forEach((pt) => {
    const p = new THREE.Mesh(new THREE.CylinderGeometry(0.10, 0.14, 6.2, 10), matEsquadria);
    p.position.set(pt.x, 3.3, pt.z);
    p.castShadow = true;
    scene.add(p);
    const lum = new THREE.Mesh(new THREE.SphereGeometry(0.28, 12, 12), new THREE.MeshBasicMaterial({ color: 0xfef08a }));
    lum.position.set(pt.x + pt.ox, 6.3, pt.z + 0.4);
    scene.add(lum);
  });

  // Cones Frontais + Cones de Área Restrita nos Fundos do Bloco
  [[35.3, 16.2], [35.9, 15.6], [36.5, 16.9], [37.1, 16.3],
   [59.5, 53.0], [58.0, 53.6], [36.0, 53.6], [14.5, 53.0]].forEach(([cx, cz]) => {
    const c = new THREE.Mesh(new THREE.ConeGeometry(0.32, 0.95, 14), matCone);
    c.position.set(cx, 0.2 + 0.475, cz);
    c.castShadow = true;
    scene.add(c);
  });

  const placaFundos = criarPlacaFlutuante("⛔ ÁREA RESTRITA • CÃO DE GUARDA", "#7f1d1d", "#fca5a5", 3.6, 0.65);
  placaFundos.position.set(58.8, 2.3, 53.4);
  scene.add(placaFundos);

  // Abrigo Cisalhado
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

  // ============================================================================
  // 5. PÁSSARO ANIMADO, CARROS, AVATARES, PROF. CRETO, VIGILANTES & CÃO OCULTO
  // ============================================================================
  const birdGroup = new THREE.Group();
  const matPassaroCorpo = new THREE.MeshStandardMaterial({ color: 0x0284c7, roughness: 0.4 });
  const matPassaroPeito = new THREE.MeshStandardMaterial({ color: 0xfacc15, roughness: 0.5 });
  const matBico = new THREE.MeshStandardMaterial({ color: 0xf97316, roughness: 0.4 });

  const birdBody = new THREE.Mesh(new THREE.SphereGeometry(0.26, 12, 10), matPassaroCorpo);
  birdBody.scale.set(1.0, 0.85, 1.6);
  const birdBelly = new THREE.Mesh(new THREE.SphereGeometry(0.22, 10, 8), matPassaroPeito);
  birdBelly.position.set(0, -0.08, 0.05);
  const birdHead = new THREE.Mesh(new THREE.SphereGeometry(0.17, 10, 10), matPassaroCorpo);
  birdHead.position.set(0, 0.14, 0.34);
  const birdBeak = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.22, 8), matBico);
  birdBeak.rotation.x = Math.PI / 2;
  birdBeak.position.set(0, 0.13, 0.54);
  const birdTail = new THREE.Mesh(new THREE.ConeGeometry(0.11, 0.35, 6), matPassaroCorpo);
  birdTail.rotation.x = -Math.PI / 2.2;
  birdTail.position.set(0, 0.05, -0.46);

  const wingGeo = new THREE.BoxGeometry(0.68, 0.04, 0.30);
  const wingL = new THREE.Mesh(wingGeo, matPassaroCorpo);
  wingL.position.set(-0.36, 0.06, 0.04);
  const wingR = new THREE.Mesh(wingGeo, matPassaroCorpo);
  wingR.position.set(0.36, 0.06, 0.04);

  birdGroup.add(birdBody, birdBelly, birdHead, birdBeak, birdTail, wingL, wingR);
  birdGroup.position.set(30.0, 9.2, 22.0);
  scene.add(birdGroup);

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
      roda.position.set(rx, ry, rz);
      grp.add(roda);
      rodas.push(roda);
    });
    [-0.65, 0.65].forEach((fx) => {
      const f = new THREE.Mesh(new THREE.SphereGeometry(0.14, 8, 8), new THREE.MeshBasicMaterial({ color: 0xfef9c3 }));
      f.position.set(fx, 0.62, 2.15);
      const l = new THREE.Mesh(new THREE.SphereGeometry(0.14, 8, 8), new THREE.MeshBasicMaterial({ color: 0xef4444 }));
      l.position.set(fx, 0.62, -2.15);
      grp.add(f, l);
    });
    scene.add(grp);
    animatedCars.push({ nome, grp, rodas, dirX, speed, honkTimer: 0 });
  }

  criarVeiculoAnimado("SUV Prata na Via CT", 12.0, 5.6, 1, 8.5, 0xcbd5e1);
  criarVeiculoAnimado("Carro Esportivo Vermelho", 58.0, 8.6, -1, 11.5, 0xdc2626);

  function criarAvatarRoblox(x, y, z, rotDeg, corTorso, corCabeca, corCalca, esc = 1.0) {
    const grp = new THREE.Group();
    grp.position.set(x, y, z);
    grp.rotation.y = THREE.MathUtils.degToRad(rotDeg);
    grp.scale.setScalar(esc);

    const mTorso = new THREE.MeshStandardMaterial({ color: corTorso, roughness: 0.6 });
    const mCab = new THREE.MeshStandardMaterial({ color: corCabeca, roughness: 0.5 });
    const mCalca = new THREE.MeshStandardMaterial({ color: corCalca, roughness: 0.8 });

    const pE = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.76, 0.34), mCalca); pE.position.set(-0.20, 0.38, 0);
    const pD = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.76, 0.34), mCalca); pD.position.set(0.20, 0.38, 0);
    const torso = new THREE.Mesh(new THREE.BoxGeometry(0.76, 0.84, 0.40), mTorso); torso.position.set(0, 1.16, 0);
    const cab = new THREE.Mesh(new THREE.BoxGeometry(0.48, 0.48, 0.48), mCab); cab.position.set(0, 1.85, 0);
    const bE = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.78, 0.28), mCab); bE.position.set(-0.54, 1.15, 0);
    const bD = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.78, 0.28), mCab); bD.position.set(0.54, 1.15, 0);
    [pE, pD, torso, cab, bE, bD].forEach(part => { part.castShadow = true; grp.add(part); });
    scene.add(grp);
    return { grp, bE, bD, pE, pD, torso, cab };
  }

  // Sistema de Notificações Limpo (Sem Spam — máximo 2 banners e cooldown por canal)
  const alertStack = document.getElementById('alert-stack');
  const activeChannelBanners = {};
  const channelCooldowns = {};

  function mostrarAlerta(texto, classe = 'alert-info', duracao = 2500, channel = 'general') {
    const now = performance.now();
    if (activeChannelBanners[channel] && activeChannelBanners[channel].el.parentNode) {
      activeChannelBanners[channel].el.textContent = texto;
      activeChannelBanners[channel].el.className = 'alert-banner ' + classe;
      clearTimeout(activeChannelBanners[channel].timer);
      activeChannelBanners[channel].timer = setTimeout(() => {
        if (activeChannelBanners[channel] && activeChannelBanners[channel].el.parentNode) {
          activeChannelBanners[channel].el.remove();
        }
        delete activeChannelBanners[channel];
      }, duracao);
      return;
    }

    if (channelCooldowns[channel] && now < channelCooldowns[channel]) return;
    channelCooldowns[channel] = now + 3400;

    while (alertStack.children.length >= 2) {
      alertStack.firstChild.remove();
    }

    const div = document.createElement('div');
    div.className = 'alert-banner ' + classe;
    div.textContent = texto;
    alertStack.appendChild(div);

    const timer = setTimeout(() => {
      if (div.parentNode) div.remove();
      delete activeChannelBanners[channel];
    }, duracao);
    activeChannelBanners[channel] = { el: div, timer };
  }

  // Quiz Prof. Creto
  const bancoPerguntasCreto = [
    {
      q: "No ciclo trigonométrico usado em rotações 2D/3D, quais são os sinais de sin(θ) e cos(θ) no 2º Quadrante (90° < θ < 180°)?",
      opts: [
        "sin(θ) > 0 (positivo) e cos(θ) < 0 (negativo)",
        "sin(θ) < 0 (negativo) e cos(θ) > 0 (positivo)",
        "sin(θ) > 0 (positivo) e cos(θ) > 0 (positivo)",
        "sin(θ) < 0 (negativo) e cos(θ) < 0 (negativo)"
      ],
      ans: 0,
      exp: "No 2º Quadrante, Y (seno) é positivo e X (cosseno) é negativo."
    },
    {
      q: "No 3º Quadrante (180° < θ < 270°), quais são os sinais de sin(θ) e cos(θ)?",
      opts: [
        "sin(θ) > 0 e cos(θ) < 0",
        "sin(θ) < 0 (negativo) e cos(θ) < 0 (negativo)",
        "sin(θ) < 0 e cos(θ) > 0",
        "sin(θ) > 0 e cos(θ) > 0"
      ],
      ans: 1,
      exp: "No 3º Quadrante, tanto X (cos θ) quanto Y (sin θ) são negativos."
    },
    {
      q: "No 4º Quadrante (270° < θ < 360°), quais são os sinais de sin(θ) e cos(θ)?",
      opts: [
        "sin(θ) > 0 e cos(θ) > 0",
        "sin(θ) < 0 e cos(θ) < 0",
        "sin(θ) < 0 (negativo) e cos(θ) > 0 (positivo)",
        "sin(θ) > 0 e cos(θ) < 0"
      ],
      ans: 2,
      exp: "No 4º Quadrante, X (cosseno) é positivo e Y (seno) é negativo."
    },
    {
      q: "No 1º Quadrante (0° < θ < 90°), quais são os sinais de sin(θ) e cos(θ)?",
      opts: [
        "sin(θ) > 0 (positivo) e cos(θ) > 0 (positivo)",
        "sin(θ) < 0 (negativo) e cos(θ) > 0 (positivo)",
        "sin(θ) > 0 (positivo) e cos(θ) < 0 (negativo)",
        "sin(θ) = 0 e cos(θ) < 0"
      ],
      ans: 0,
      exp: "No 1º Quadrante, ambas as coordenadas são positivas."
    },
    {
      q: "Por que utilizamos Coordenadas Homogêneas 4D (x, y, z, w) nas matrizes de transformação da Computação Gráfica?",
      opts: [
        "Para representar Translação, Rotação, Escala e Projeção como multiplicação de matrizes 4×4",
        "Para eliminar o eixo Z e renderizar apenas em 2D",
        "Para substituir o cálculo de vetores normais na equação de Phong",
        "Para evitar o uso de seno e cosseno em rotações"
      ],
      ans: 0,
      exp: "Com w=1 (pontos) e w=0 (vetores), a translação torna-se uma operação matricial 4×4."
    },
    {
      q: "Na iluminação de Phong, qual componente depende do ângulo entre o vetor refletido R e o vetor do observador V?",
      opts: [
        "Componente Ambiente (Ia · Ka)",
        "Componente Difusa Lambertiana (Id · Kd · (N·L))",
        "Componente Especular (Ie · Ke · (R·V)^n)",
        "Atenuação atmosférica"
      ],
      ans: 2,
      exp: "O brilho especular concentra-se na direção do reflexo espelho R ponderado pelo expoente n."
    },
    {
      q: "Na câmera sintética (M_wc), como são obtidos os eixos ortonormais (xc, yc, zc) a partir de Eye, At e Up?",
      opts: [
        "zc = normalizar(Eye - At), xc = normalizar(Up × zc), yc = zc × xc",
        "zc = Eye + At, xc = Up · At, yc = Eye × Up",
        "xc = Eye - At, yc = Eye + Up, zc = (0,0,0)",
        "zc = Up, xc = At, yc = Eye"
      ],
      ans: 0,
      exp: "O eixo zc aponta da cena para a câmera (Eye - At) no sistema de mão direita."
    }
  ];

  let quizDeck = [];
  let quizIdx = 0;
  let gameStarted = false;
  let sessionSeconds = 0;
  let playerEmail = "";
  let inQuizModal = false;
  let inSummaryModal = false;
  let isPaused = false;
  let survivalMode = false;
  let isDead = false;
  let inCoffeeCutscene = false;
  let coffeeTimer = 0;
  let inDanceCutscene = false;
  let danceTimer = 0;
  let coffeeBuffTimer = 0;
  let energy = 100;
  let windowsBrokenCount = 0;
  let attackTimer = 0;
  let shakeTimer = 0;
  let stepAccumulator = 0;

  let rivalRun = {
  nome: "Rival CT",
  score: 450,
  tempo: 55
};

function calcularScoreJogador() {
  const invCount = collectibles.filter(c => c.coletado).length;
  const pontos =
    invCount * 100 +
    pState.quests.fotos.size * 150 +
    pState.quests.cretoAcertos * 200 +
    windowsBrokenCount * 50;
  const penalidade = Math.max(0, (100 - Math.round(energy)) * 2);
  return Math.max(0, pontos - penalidade);
}

let ultimoScorePvP = 0;
function atualizarPlacar1v1() {
  const meuScore = calcularScoreJogador();
  const badge = document.getElementById('pvp-badge');
  if (badge) {
    badge.textContent = `⚔️ 1v1: Você ${meuScore} pts vs ${rivalRun.nome} ${rivalRun.score} pts`;
    badge.style.color = meuScore >= rivalRun.score ? '#4ade80' : '#facc15';
  }
  if (meuScore > rivalRun.score && ultimoScorePvP <= rivalRun.score) {
    SoundEngine.playCollect();
    mostrarAlerta(`🔥 VOCÊ ASSUMIU A LIDERANÇA DO 1v1 (${meuScore} vs ${rivalRun.score} pts)!`, "alert-success", 2800, "pvp");
  }
  ultimoScorePvP = meuScore;
}

  const pState = {
    pos: new THREE.Vector3(24.0, 0.2, 15.2),
    velY: 0,
    yaw: Math.PI,
    camYaw: Math.PI,
    camPitch: 0.34,
    camDist: 7.4,
    onGround: true,
    wasOnRoof: false,
    onLadderAuto: false,
    climbing: null,
    fountainTimer: 0,
    damageCooldown: 0,
    jumpscareCooldown: 0,
    crossedOut: false,
    quests: {
      fotos: new Set(),
      arvore: false,
      poste: false,
      predioSolar: false,
      travessia: false,
      cretoAcertos: 0
    }
  };

  function formatarTempo(seg) {
    const total = Math.max(0, Math.floor(seg));
    const mm = String(Math.floor(total / 60)).padStart(2, '0');
    const ss = String(total % 60).padStart(2, '0');
    return `${mm}:${ss}`;
  }

  function atualizarHUD() {
    document.getElementById('timer-badge').textContent = formatarTempo(sessionSeconds);
    const pct = Math.max(0, Math.min(100, Math.round(energy)));
    const fill = document.getElementById('energy-fill');
    fill.style.width = `${pct}%`;
    fill.style.background = pct > 50
      ? 'linear-gradient(90deg, #22c55e, #38bdf8)'
      : (pct > 25 ? 'linear-gradient(90deg, #f59e0b, #facc15)' : '#ef4444');
    document.getElementById('hp-text').textContent = `${pct}%`;

    const invCount = collectibles.filter(c => c.coletado).length;
    document.getElementById('inv-count').textContent = `${invCount}/${collectibles.length}`;
    document.getElementById('foto-count').textContent = `${pState.quests.fotos.size}/5`;
    document.getElementById('creto-count').textContent = `${pState.quests.cretoAcertos}/4`;

    const grid = document.getElementById('inv-grid-container');
    grid.innerHTML = collectibles.map(c => `
      <div class="inv-slot ${c.coletado ? 'collected' : ''}">
        <div style="font-size:20px; margin-bottom:4px;">${c.coletado ? '✅' : '📦'}</div>
        <div style="font-weight:700; color:${c.coletado ? '#4ade80' : '#f8fafc'};">${c.nome}</div>
        <div style="font-size:11px; color:#94a3b8; margin-top:2px;">
          ${c.coletado ? 'Coletado!' : `Mesa X=${c.x.toFixed(1)}, Z=${c.z.toFixed(1)}`}
        </div>
      </div>
    `).join('');

    document.getElementById('mq-foto-n').textContent = pState.quests.fotos.size;
    document.getElementById('mq-creto-n').textContent = pState.quests.cretoAcertos;
    const setBadge = (id, ok) => {
      const el = document.getElementById(id);
      if (ok) { el.className = 'q-badge q-done'; el.textContent = '✔ Concluída'; }
    };
    setBadge('mq-foto', pState.quests.fotos.size >= 3);
    setBadge('mq-arvore', pState.quests.arvore);
    setBadge('mq-poste', pState.quests.poste);
    setBadge('mq-solar', pState.quests.predioSolar);
    setBadge('mq-rua', pState.quests.travessia);
    setBadge('mq-creto', pState.quests.cretoAcertos >= 4);
    atualizarPlacar1v1();
  }

  function abrirResumoSessao() {
    const invCount = collectibles.filter(c => c.coletado).length;
    const miniDone = [
      pState.quests.fotos.size >= 3,
      pState.quests.arvore,
      pState.quests.poste,
      pState.quests.predioSolar,
      pState.quests.travessia
    ].filter(Boolean).length;

    const tempoStr = formatarTempo(sessionSeconds);
    document.getElementById('sum-time').textContent = tempoStr;
    document.getElementById('sum-inv').textContent = `${invCount}/${collectibles.length}`;
    document.getElementById('sum-photos').textContent = `${pState.quests.fotos.size}/5`;
    document.getElementById('sum-creto').textContent = `${pState.quests.cretoAcertos}/4`;
    document.getElementById('sum-windows').textContent = `${windowsBrokenCount}/${breakableWindows.length}`;
    document.getElementById('sum-mini').textContent = `${miniDone}/5`;

    const emailLimpo = (document.getElementById('player-email-input').value || playerEmail || '').trim();
    playerEmail = emailLimpo;
    document.getElementById('sum-email-val').textContent = emailLimpo || 'Não informado';

    const meuScoreFinal = calcularScoreJogador();
    const venceuPvP = meuScoreFinal >= rivalRun.score;
    mostrarAlerta(
      venceuPvP
        ? `🏆 Vitória no 1v1! Você (${meuScoreFinal} pts) venceu ${rivalRun.nome} (${rivalRun.score} pts).`
        : `⚔️ ${rivalRun.nome} venceu por ${rivalRun.score} vs ${meuScoreFinal} pts — tente de novo!`,
      venceuPvP ? "alert-success" : "alert-info",
      3200,
      "pvp_resumo"
    );
    if (venceuPvP) {
      rivalRun = {
        nome: "Seu Recorde",
        score: meuScoreFinal,
        tempo: Math.max(1, Math.floor(sessionSeconds))
      };
    }

    const mailBtn = document.getElementById('btn-mailto-summary');
    if (emailLimpo) {
      const corpo = encodeURIComponent(
        `Resumo Bloco 734 (CT/UFC)\nTempo: ${tempoStr}\nInventário: ${invCount}/${collectibles.length}\nFotos: ${pState.quests.fotos.size}/5\nAcertos Creto: ${pState.quests.cretoAcertos}/4\nVidros quebrados: ${windowsBrokenCount}/${breakableWindows.length}\nMini-quests: ${miniDone}/5\n\nObrigado por jogar!`
      );
      mailBtn.href = `mailto:${encodeURIComponent(emailLimpo)}?subject=${encodeURIComponent('Resumo da Sessão — Bloco 734 (CT/UFC)')}&body=${corpo}`;
      mailBtn.style.display = 'inline-block';
    } else {
      mailBtn.style.display = 'none';
    }

    inSummaryModal = true;
    document.getElementById('summary-modal').style.display = 'flex';
  }

  function fecharResumoSessao() {
    inSummaryModal = false;
    document.getElementById('summary-modal').style.display = 'none';
    if (gameStarted) canvas.focus();
  }

  document.getElementById('btn-arcade-start').addEventListener('click', () => {
    SoundEngine.init();
    playerEmail = (document.getElementById('player-email-input').value || '').trim();
    gameStarted = true;
    document.getElementById('arcade-start-overlay').style.display = 'none';
    SoundEngine.playSwitch();
    atualizarHUD();
    canvas.focus();
  });

  document.getElementById('btn-summary').addEventListener('click', () => {
    abrirResumoSessao();
  });
  document.getElementById('btn-close-summary').addEventListener('click', fecharResumoSessao);
  document.getElementById('btn-restart-session').addEventListener('click', () => {
    sessionSeconds = 0;
    energy = 100;
    windowsBrokenCount = 0;
    pState.pos.set(24.0, 0.2, 15.2);
    pState.velY = 0;
    pState.wasOnRoof = false;
    pState.climbing = null;
    pState.quests.fotos.clear();
    pState.quests.arvore = false;
    pState.quests.poste = false;
    pState.quests.predioSolar = false;
    pState.quests.travessia = false;
    pState.quests.cretoAcertos = 0;
    collectibles.forEach(c => { c.coletado = false; c.mesh.visible = true; });
    fecharResumoSessao();
    gameStarted = false;
    document.getElementById('arcade-start-overlay').style.display = 'flex';
    atualizarHUD();
  });

  function aplicarDano(qtd, motivo, isScare = false) {
    if (isDead || inCoffeeCutscene || pState.damageCooldown > 0) return;
    pState.damageCooldown = 1.2;
    if (isScare) SoundEngine.playScare();
    shakeTimer = 0.28;
    const flash = document.getElementById('damage-flash');
    flash.style.opacity = '1';
    setTimeout(() => { flash.style.opacity = '0'; }, 260);

    if (survivalMode) {
      energy = Math.max(0, energy - qtd);
      atualizarHUD();
      if (energy <= 0) {
        isDead = true;
        document.getElementById('gameover-reason').textContent = motivo;
        document.getElementById('gameover-modal').style.display = 'flex';
        return;
      }
      mostrarAlerta(`💔 -${qtd} Life! ${motivo}`, 'alert-danger', 2600, 'damage');
    } else {
      energy = Math.max(10, energy - qtd);
      atualizarHUD();
      mostrarAlerta(`⚠️ -${qtd} Life! ${motivo} (Café em X=35.8, Z=18.8)`, 'alert-danger', 2600, 'damage');
    }
  }

  function abrirQuizCreto() {
    quizDeck = [...bancoPerguntasCreto].sort(() => Math.random() - 0.5).slice(0, 4);
    quizIdx = 0;
    pState.quests.cretoAcertos = 0;
    inQuizModal = true;
    document.getElementById('creto-modal').style.display = 'flex';
    renderizarPerguntaCreto();
  }

  function fecharModalCreto() {
    inQuizModal = false;
    document.getElementById('creto-modal').style.display = 'none';
    canvas.focus();
  }

  function renderizarPerguntaCreto() {
    const item = quizDeck[quizIdx];
    document.getElementById('creto-progress').textContent = `Pergunta ${quizIdx + 1} / 4 • Acertos: ${pState.quests.cretoAcertos}`;
    document.getElementById('creto-question').textContent = item.q;
    document.getElementById('creto-feedback').textContent = '';
    const box = document.getElementById('creto-options');
    box.innerHTML = '';
    const letras = ['A', 'B', 'C', 'D'];
    item.opts.forEach((optText, idx) => {
      const btn = document.createElement('button');
      btn.className = 'quiz-opt';
      btn.textContent = `${letras[idx]}) ${optText}`;
      btn.onclick = () => responderCreto(idx);
      box.appendChild(btn);
    });
  }

  function responderCreto(escolha) {
    const item = quizDeck[quizIdx];
    const fb = document.getElementById('creto-feedback');

    if (escolha === item.ans) {
      SoundEngine.playCollect();
      if (quizTentativaAtual === 0) {
        pState.quests.cretoAcertos++;
        energy = Math.min(100, energy + 10);
        fb.style.color = '#4ade80';
        fb.textContent = `🎉 ACERTOU DE PRIMEIRA! ${item.exp}`;
      } else {
        fb.style.color = '#38bdf8';
        fb.textContent = `✔ Resposta encontrada (sem ponto extra de tentativa): ${item.exp}`;
      }
      atualizarHUD();
      setTimeout(() => {
        quizIdx++;
        if (quizIdx < 4) {
          renderizarPerguntaCreto();
        } else {
          pState.quests.cretoConcluido = true;
          atualizarHUD();
          fecharModalCreto();
          iniciarCutsceneDanca("🎓 MISSÃO DO PROF. CRETO CONCLUÍDA! Comemorando no Bloco 734!");
        }
      }, 1300);
    } else {
      quizTentativaAtual++;
      SoundEngine.playError();
      fb.style.color = '#f87171';
      fb.textContent = `❌ Não foi dessa vez! Tente outra opção (sem ponto extra nesta questão). ${item.exp}`;
      aplicarDano(5, 'Tentativa incorreta no Quiz do Prof. Creto!');
    }
  }

  // Alunos na Praça
  const alunoSegundaVezPos = new THREE.Vector3(17.8, 0.2, 16.5);
  const alunoSegundaVez = criarAvatarRoblox(alunoSegundaVezPos.x, alunoSegundaVezPos.y, alunoSegundaVezPos.z, -25, 0xd97706, 0xfacc15, 0x1e293b, 0.96);
  const cadernoRep = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.05, 0.24), matPapel);
  cadernoRep.position.set(0, -0.35, 0.16);
  alunoSegundaVez.bD.add(cadernoRep);
  const placaAluno1 = criarPlacaFlutuante("😅 ALUNO (2ª VEZ) • DICAS [E]", "#78350f", "#fbbf24", 3.5, 0.65);
  placaAluno1.position.set(alunoSegundaVezPos.x, 2.55, alunoSegundaVezPos.z);
  scene.add(placaAluno1);

  const alunoEstudiosoPos = new THREE.Vector3(19.8, 0.2, 16.2);
  const alunoEstudioso = criarAvatarRoblox(alunoEstudiosoPos.x, alunoEstudiosoPos.y, alunoEstudiosoPos.z, 20, 0x2563eb, 0xfed7aa, 0x0f172a, 0.95);
  const livroEst = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.06, 0.26), new THREE.MeshStandardMaterial({ color: 0x38bdf8 }));
  livroEst.position.set(0, -0.35, 0.16);
  alunoEstudioso.bD.add(livroEst);
  const placaAluno2 = criarPlacaFlutuante("📚 ESTUDANDO MUITO • DICAS [E]", "#1e3a8a", "#38bdf8", 3.6, 0.65);
  placaAluno2.position.set(alunoEstudiosoPos.x, 2.55, alunoEstudiosoPos.z);
  scene.add(placaAluno2);

  const dicaAluno1Action = {
    id: "aluno_2a_vez",
    titulo: "Aluno (Fazendo a Cadeira pela 2ª Vez)",
    x: alunoSegundaVezPos.x,
    z: alunoSegundaVezPos.z,
    acaoFn: () => {
      SoundEngine.playCollect();
      mostrarAlerta(
        "😅 Aluno (2ª vez): 'Na 1ª eu enrolei! Faça o Trabalho Final cedo, entregue todas as atividades e revise pela autocorreção da prova!'",
        "alert-iot", 4600, "dica_aluno"
      );
    }
  };
  const dicaAluno2Action = {
    id: "aluno_estudioso",
    titulo: "Aluno (Estudando Muito para Passar)",
    x: alunoEstudiosoPos.x,
    z: alunoEstudiosoPos.z,
    acaoFn: () => {
      SoundEngine.playCollect();
      mostrarAlerta(
        "📚 Aluno (Focado): 'Dica de ouro: avance o Trabalho Final, não perca o prazo das atividades e use a autocorreção da prova para fixar!'",
        "alert-info", 4600, "dica_aluno"
      );
    }
  };

  [alunoSegundaVez.torso, alunoSegundaVez.cab].forEach(m => { m.userData.totem = dicaAluno1Action; clickableTotems.push(m); });
  [alunoEstudioso.torso, alunoEstudioso.cab].forEach(m => { m.userData.totem = dicaAluno2Action; clickableTotems.push(m); });

  registrarAR(
    "Colega Fazendo a Cadeira pela 2ª Vez",
    "ALUNO • DICAS DE APROVAÇÃO [E]",
    "Faz a cadeira pela 2ª vez e compartilha dicas: foque no Trabalho Final, entregue as atividades e faça a autocorreção da prova.",
    "Avatar Articulado + Caderno • [E] Ouvir Dica",
    [alunoSegundaVezPos.x, 1.1, alunoSegundaVezPos.z], [1.4, 2.3, 1.4],
    { totem: dicaAluno1Action }
  );
  registrarAR(
    "Colega Estudando Muito para Passar",
    "ALUNO • DICAS DE ESTUDO [E]",
    "Estudando firme para passar de primeira: dá dicas sobre Trabalho Final, atividades práticas e autocorreção da prova.",
    "Avatar Articulado + Apostila • [E] Ouvir Dica",
    [alunoEstudiosoPos.x, 1.1, alunoEstudiosoPos.z], [1.4, 2.3, 1.4],
    { totem: dicaAluno2Action }
  );

  // Personagem do Prof. Creto (X=9.5, Z=22.2)
  const cretoPos = new THREE.Vector3(9.5, 0.2, 22.2);
  const profCreto = criarAvatarRoblox(cretoPos.x, cretoPos.y, cretoPos.z, 180, 0xf8fafc, 0xfed7aa, 0x1e3a8a, 1.12);
  const oculos = new THREE.Mesh(new THREE.BoxGeometry(0.44, 0.11, 0.08), matEsquadria);
  oculos.position.set(0, 1.88, 0.25);
  profCreto.grp.add(oculos);

  const cretoBeacon = new THREE.Mesh(
    new THREE.ConeGeometry(0.26, 0.52, 6),
    new THREE.MeshBasicMaterial({ color: 0x38bdf8 })
  );
  cretoBeacon.rotation.x = Math.PI;
  cretoBeacon.position.set(0, 2.52, 0);
  profCreto.grp.add(cretoBeacon);

  const cretoAro = new THREE.Mesh(
    new THREE.TorusGeometry(1.05, 0.08, 8, 32),
    new THREE.MeshBasicMaterial({ color: 0x38bdf8 })
  );
  cretoAro.rotation.x = Math.PI / 2;
  cretoAro.position.set(cretoPos.x, 0.25, cretoPos.z);
  scene.add(cretoAro);

  const placaCreto = criarPlacaFlutuante("🎓 PROF. CRETO • INTERAGIR [E]", "#1e3a8a", "#60a5fa", 4.1, 0.82);
  placaCreto.position.set(cretoPos.x, 3.15, cretoPos.z);
  scene.add(placaCreto);

  const cretoTotemAction = { id: "prof_creto_npc", titulo: "Prof. Creto — Quiz de CG", x: cretoPos.x, z: cretoPos.z, acaoFn: abrirQuizCreto };
  [profCreto.torso, profCreto.cab, cretoBeacon].forEach((m) => {
    m.userData.totem = cretoTotemAction;
    clickableTotems.push(m);
  });
  registrarAR(
    "Prof. Creto (Área Lateral do Bloco 734)",
    "NPC QUALIFICADO • INTERAÇÃO [E] OU CLIQUE",
    "Professor de Computação Gráfica na área lateral (X=9.5, Z=22.2). Aproxime-se e pressione [E] ou clique nele para iniciar o quiz!",
    "Avatar Articulado + Marcador 3D • [E] Falar & Quiz",
    [cretoPos.x, 1.2, cretoPos.z], [1.6, 2.6, 1.6],
    { totem: cretoTotemAction }
  );

  // Vigilantes Patrimoniais
  const vigilantes = [
    { avatar: criarAvatarRoblox(7.8, 0.2, 20.4, 180, 0x1e3a8a, 0xfdba74, 0x0f172a, 1.02), home: new THREE.Vector3(7.8, 0.2, 20.4), alertado: false },
    { avatar: criarAvatarRoblox(32.5, 0.2, 15.2, 180, 0x1e3a8a, 0xfdba74, 0x0f172a, 1.02), home: new THREE.Vector3(32.5, 0.2, 15.2), alertado: false }
  ];

  // Cachorro Guardião (Fundos do Bloco 734: X=46.0, Z=56.2)
  const caoGroup = new THREE.Group();
  const matCao = new THREE.MeshStandardMaterial({ color: 0x7c2d12, roughness: 0.8 });
  const caoCorpo = new THREE.Mesh(new THREE.BoxGeometry(1.05, 0.50, 0.50), matCao);
  caoCorpo.position.y = 0.55; caoCorpo.castShadow = true;
  const caoCabeca = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.40, 0.40), matCao);
  caoCabeca.position.set(0.60, 0.82, 0);
  const olhoCao1 = new THREE.Mesh(new THREE.SphereGeometry(0.05, 6, 6), new THREE.MeshBasicMaterial({ color: 0xef4444 }));
  olhoCao1.position.set(0.82, 0.88, 0.12);
  const olhoCao2 = olhoCao1.clone();
  olhoCao2.position.z = -0.12;
  caoGroup.add(caoCorpo, caoCabeca, olhoCao1, olhoCao2);
  [[-0.34, -0.16], [-0.34, 0.16], [0.34, -0.16], [0.34, 0.16]].forEach(([lx, lz]) => {
    const pata = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.34, 0.15), matCao);
    pata.position.set(lx, 0.17, lz);
    caoGroup.add(pata);
  });
  const caoHome = new THREE.Vector3(46.0, 0.2, 56.2);
  caoGroup.position.copy(caoHome);
  scene.add(caoGroup);

  // Palmeira
  [[0, 0.2, 0, 0.24, 1.6, 0], [0.05, 1.75, 0, 0.20, 1.6, -0.07], [0.15, 3.3, 0, 0.16, 1.5, -0.12]].forEach(([ox, oy, oz, r, h, rz]) => {
    const seg = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.85, r, h, 10), matTronco);
    seg.position.set(42.5 + ox, oy + h * 0.5, 17.5 + oz);
    seg.rotation.z = rz;
    scene.add(seg);
  });
  for (let k = 0; k < 6; k++) {
    const ang = k * (Math.PI * 2 / 6);
    const palma = new THREE.Mesh(new THREE.ConeGeometry(0.42, 2.2, 8), matFolha2);
    palma.position.set(42.74 + Math.cos(ang) * 0.9, 4.65, 17.5 + Math.sin(ang) * 0.9);
    palma.rotation.z = -Math.cos(ang) * 1.15;
    palma.rotation.x = Math.sin(ang) * 1.15;
    scene.add(palma);
  }

  // ============================================================================
  // 6. TOTENS INTERATIVOS 3D NA PRAÇA (CLIQUE DO MOUSE OU [E])
  // ============================================================================
  function criarTotemInterativo(id, titulo, x, z, corHex, acaoFn) {
    const grp = new THREE.Group();
    grp.position.set(x, 0.2, z);

    const pedestal = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.34, 1.05, 14), matEsquadria);
    pedestal.position.y = 0.525;
    pedestal.castShadow = true;

    const btnMat = new THREE.MeshStandardMaterial({
      color: corHex, emissive: corHex, emissiveIntensity: 0.45, roughness: 0.3
    });
    const botaoTopo = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.26, 0.52), btnMat);
    botaoTopo.position.y = 1.18;
    botaoTopo.castShadow = true;

    grp.add(pedestal, botaoTopo);
    scene.add(grp);

    const totemData = { id, titulo, x, z, grp, botaoTopo, acaoFn };
    botaoTopo.userData.totem = totemData;
    pedestal.userData.totem = totemData;
    clickableTotems.push(botaoTopo, pedestal);

    registrarAR(
      titulo,
      "TOTEM INTERATIVO • CLIQUE OU [E]",
      "Painel de controle físico na praça do CT. Clique com o mouse sobre ele ou pressione [E] ao se aproximar!",
      "Cilindro + Cubo Emissivo",
      [x, 0.8, z], [1.0, 1.6, 1.0],
      { totem: totemData }
    );
    return totemData;
  }

  function alternarPontosDeFuga() {
    SoundEngine.playSwitch();
    vanishingMode = (vanishingMode % 3) + 1;
    projMode = 'perspectiva';
    if (vanishingMode === 1) {
      pState.pos.set(27.0, 0.2, 13.5);
      pState.yaw = 0;
      pState.camYaw = Math.PI;
      pState.camPitch = 0.08;
      pState.camDist = 7.5;
      mostrarAlerta("1️⃣ 1 Ponto de Fuga ativado (Eixo Z alinhado com Bloco 734)", "alert-info", 2400, "info");
    } else if (vanishingMode === 2) {
      pState.camYaw = Math.PI * 0.78;
      pState.camPitch = 0.14;
      pState.camDist = 7.8;
      mostrarAlerta("2️⃣ 2 Pontos de Fuga ativados", "alert-info", 2200, "info");
    } else {
      pState.camYaw = Math.PI * 0.76;
      pState.camPitch = 0.72;
      pState.camDist = 11.5;
      mostrarAlerta("3️⃣ 3+ Pontos de Fuga ativados (Visada aérea)", "alert-info", 2200, "info");
    }
    document.getElementById('proj-badge').textContent = `📷 Perspectiva (${vanishingMode} PF)`;
  }

  function alternarIsometrica() {
    SoundEngine.playSwitch();
    if (projMode === 'isometrica') {
      projMode = 'perspectiva';
      document.getElementById('proj-badge').textContent = `📷 Perspectiva (${vanishingMode} PF)`;
      mostrarAlerta("📷 Retornou para Projeção Perspectiva", "alert-info", 2000, "info");
    } else {
      projMode = 'isometrica';
      pState.camYaw = Math.PI * 0.75;
      pState.camPitch = 0.615;
      document.getElementById('proj-badge').textContent = "🧊 Projeção Isométrica";
      mostrarAlerta("🧊 Projeção Isométrica ativada (Oblíqua disponível na aba Ray Casting)", "alert-info", 2600, "info");
    }
  }

  criarTotemInterativo("totem_spots", "Totem: Liga/Desliga Holofotes (Spots)", 21.5, 15.2, 0x38bdf8, () => {
    spotsEnabled = !spotsEnabled;
    SoundEngine.playSwitch();
    atualizarIluminacaoCena();
    mostrarAlerta(`🔦 Holofotes / Spots: ${spotsEnabled ? 'LIGADOS' : 'DESLIGADOS'}`, "alert-info", 2200, "info");
  });

  criarTotemInterativo("totem_sol", "Totem: Liga/Desliga Sol Direcional", 23.5, 15.2, 0xfacc15, () => {
    sunEnabled = !sunEnabled;
    SoundEngine.playSwitch();
    atualizarIluminacaoCena();
    mostrarAlerta(`☀️ Sol Direcional: ${sunEnabled ? 'LIGADO' : 'DESLIGADO'}`, "alert-info", 2200, "info");
  });

  criarTotemInterativo("totem_vp", "Totem: Alternar Pontos de Fuga (1/2/3 PF)", 25.5, 15.2, 0xa855f7, alternarPontosDeFuga);
  criarTotemInterativo("totem_iso", "Totem: Projeção Isométrica", 27.5, 15.2, 0x22c55e, alternarIsometrica);

  document.getElementById('btn-vp').addEventListener('click', () => { alternarPontosDeFuga(); canvas.focus(); });
  document.getElementById('btn-iso').addEventListener('click', () => { alternarIsometrica(); canvas.focus(); });

  // ============================================================================
  // 7. MESAS DE TRABALHO, INVENTÁRIO (8 ITENS) & CAFETEIRA EM ILHA ABERTA
  // ============================================================================
  const mesasConfig = [
    { nome: "Mesa Trabalho 1 (PC • Atividades 1–3)", x: 45.6, z: 18.2, tipo: "pc" },
    { nome: "Mesa Trabalho 2 (Papéis • Atividades 4–5)", x: 47.8, z: 18.2, tipo: "papel" },
    { nome: "Mesa Avaliações (Papéis • AP1 e AP2)", x: 45.6, z: 19.6, tipo: "papel" },
    { nome: "Estação Trabalho Final (PC • Bloco 734)", x: 39.2, z: 18.6, tipo: "pc" }
  ];
  mesasConfig.forEach((m) => {
    addBox(1.45, 0.08, 0.85, matMadeira, m.x, 0.96, m.z);
    [[-0.64, -0.34], [0.64, -0.34], [-0.64, 0.34], [0.64, 0.34]].forEach(([dx, dz]) => {
      const pe = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.74, 8), matEsquadria);
      pe.position.set(m.x + dx, 0.57, m.z + dz);
      scene.add(pe);
    });
    if (m.tipo === "pc") {
      addBox(0.62, 0.40, 0.06, matTelaPC, m.x, 1.25, m.z - 0.12);
      addBox(0.20, 0.36, 0.38, matEsquadria, m.x + 0.46, 1.16, m.z - 0.08);
    } else {
      addBox(0.34, 0.06, 0.44, matPapel, m.x - 0.12, 1.03, m.z + 0.05, 0.24);
    }
  });

  const collectibles = [
    { id: "atv1", nome: "Atividade 1", x: 45.15, y: 1.42, z: 18.2, cor: 0x38bdf8, coletado: false },
    { id: "atv2", nome: "Atividade 2", x: 45.60, y: 1.42, z: 18.2, cor: 0x38bdf8, coletado: false },
    { id: "atv3", nome: "Atividade 3", x: 46.05, y: 1.42, z: 18.2, cor: 0x38bdf8, coletado: false },
    { id: "atv4", nome: "Atividade 4", x: 47.55, y: 1.42, z: 18.2, cor: 0x4ade80, coletado: false },
    { id: "atv5", nome: "Atividade 5", x: 48.05, y: 1.42, z: 18.2, cor: 0x4ade80, coletado: false },
    { id: "ap1",  nome: "AP1",         x: 45.35, y: 1.42, z: 19.6, cor: 0xfacc15, coletado: false },
    { id: "ap2",  nome: "AP2",         x: 45.85, y: 1.42, z: 19.6, cor: 0xfacc15, coletado: false },
    { id: "tf",   nome: "Trabalho Final", x: 39.20, y: 1.50, z: 18.6, cor: 0xf472b6, coletado: false }
  ];
  collectibles.forEach((item) => {
    const geom = item.id === "tf" ? new THREE.OctahedronGeometry(0.22, 0) : new THREE.BoxGeometry(0.22, 0.22, 0.22);
    const matI = new THREE.MeshStandardMaterial({ color: item.cor, emissive: item.cor, emissiveIntensity: 0.45 });
    const mesh = new THREE.Mesh(geom, matI);
    mesh.position.set(item.x, item.y, item.z);
    scene.add(mesh);
    item.mesh = mesh;
  });

  // Cafeteira na Ilha Aberta (X=35.8, Z=18.8)
  const cafePos = new THREE.Vector3(35.8, 0.2, 18.8);
  addBox(0.85, 0.90, 0.75, matMadeira, cafePos.x, 0.65, cafePos.z);
  addBox(0.46, 0.56, 0.44, new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.75, roughness: 0.25 }), cafePos.x, 1.38, cafePos.z);
  const xicaraMesa = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.05, 0.11, 12), matPapel);
  xicaraMesa.position.set(cafePos.x, 1.15, cafePos.z - 0.14);
  scene.add(xicaraMesa);
  const ledCafe = new THREE.Mesh(new THREE.SphereGeometry(0.045, 8, 8), new THREE.MeshBasicMaterial({ color: 0x22c55e }));
  ledCafe.position.set(cafePos.x + 0.14, 1.54, cafePos.z - 0.23);
  scene.add(ledCafe);

  registrarAR(
    "Máquina de Café Expresso CT",
    "ILHA ABERTA • RECUPERA 100% LIFE",
    "Cafeteira automática no centro da praça (X=35.8, Z=18.8) com câmera livre 360°. Recupera toda a vida e ativa Turbo!",
    "Cubos + Cilindro + Esfera LED · [E] Tomar Café",
    [cafePos.x, 1.0, cafePos.z], [1.1, 1.7, 1.0],
    { isCoffeeMachine: true }
  );

  // ============================================================================
  // 8. JOGADOR ARTICULADO, LANTERNA, MINI-QUESTS, VIGILANTES & MODO FOTO
  // ============================================================================
  const player = criarAvatarRoblox(24.0, 0.2, 15.2, 0, 0x0284c7, 0xfacc15, 0x0f172a, 1.0);
  const xicaraMao = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.055, 0.13, 12), matPapel);
  xicaraMao.position.set(0, -0.38, 0.14);
  xicaraMao.visible = false;
  player.bD.add(xicaraMao);

  const lanternaMesh = new THREE.Mesh(
    new THREE.CylinderGeometry(0.045, 0.065, 0.24, 10),
    new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.7, roughness: 0.3 })
  );
  lanternaMesh.rotation.x = Math.PI / 2;
  lanternaMesh.position.set(0, -0.38, 0.18);
  player.bE.add(lanternaMesh);

  const playerLantern = new THREE.SpotLight(0xfffbeb, lanternEnabled ? 3.2 : 0.0, 26, Math.PI / 6.5, 0.45, 1.2);
  playerLantern.position.set(0, 1.35, 0.35);
  const lanternTarget = new THREE.Object3D();
  lanternTarget.position.set(0, 0.65, 8.5);
  player.grp.add(playerLantern);
  player.grp.add(lanternTarget);
  playerLantern.target = lanternTarget;

  function alternarLanterna() {
    lanternEnabled = !lanternEnabled;
    playerLantern.intensity = lanternEnabled ? 3.2 : 0.0;
    SoundEngine.playSwitch();
    mostrarAlerta(
      lanternEnabled ? "🔦 Lanterna LIGADA (cone de luz ativo)" : "🔦 Lanterna DESLIGADA",
      "alert-info", 1800, "info"
    );
  }
  document.getElementById('btn-lantern').addEventListener('click', () => { alternarLanterna(); canvas.focus(); });

  const steamParticles = [];
  for (let i = 0; i < 12; i++) {
    const sp = new THREE.Mesh(
      new THREE.SphereGeometry(0.05, 6, 6),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.65 })
    );
    sp.visible = false;
    scene.add(sp);
    steamParticles.push({ mesh: sp, t: Math.random() });
  }

  const arBoxHelper = new THREE.Box3Helper(new THREE.Box3(), 0x22d3ee);
  arBoxHelper.visible = false;
  scene.add(arBoxHelper);
  let arTimer = 0;

  atualizarHUD();

  function acionarVigilantes(motivo) {
    let novosAlertados = false;
    vigilantes.forEach((v) => {
      if (!v.alertado) novosAlertados = true;
      v.alertado = true;
    });
    if (novosAlertados) SoundEngine.playWhistle();
    document.getElementById('vigilante-badge').textContent = '🚨 Vigilância: PERSEGUINDO!';
    document.getElementById('vigilante-badge').style.color = '#f87171';
    mostrarAlerta(`👮 ALERTA (${motivo})! Suba numa árvore [R], poste [R] ou escada para escapar!`, 'alert-danger', 2800, 'vigilante');
  }

  function dispararJumpscare(emoji = '🦉👁️👁️', titulo = 'SUSTO NO CAMPUS CT!', sub = 'A Coruja Guardiã das Copas te pegou desprevenido!') {
    if (!MODO_NOITE) return;
    if (pState.jumpscareCooldown > 0) return;
    pState.jumpscareCooldown = 8.0;
    SoundEngine.playScare();
    document.getElementById('js-emoji').textContent = emoji;
    document.getElementById('js-title').textContent = titulo;
    document.getElementById('js-sub').textContent = sub;
    const ov = document.getElementById('jumpscare-overlay');
    ov.style.display = 'flex';
    setTimeout(() => {
      ov.style.display = 'none';
      mostrarAlerta(`😱 ${titulo}`, 'alert-danger', 2600, 'scare');
    }, 1450);
  }

  document.getElementById('btn-survival').addEventListener('click', (e) => {
    survivalMode = !survivalMode;
    SoundEngine.playSwitch();
    e.target.textContent = survivalMode ? '💀 Sobrevivência: ON' : '💀 Sobrevivência: OFF';
    e.target.classList.toggle('surv-on', survivalMode);
    mostrarAlerta(
      survivalMode
        ? '💀 Modo Sobrevivência ATIVADO!'
        : '🛡️ Modo Sobrevivência DESATIVADO.',
      'alert-info', 2400, 'info'
    );
    canvas.focus();
  });

  document.getElementById('btn-respawn').addEventListener('click', () => {
    isDead = false;
    energy = 100;
    pState.pos.set(24.0, 0.2, 15.2);
    pState.velY = 0;
    pState.wasOnRoof = false;
    pState.climbing = null;
    document.getElementById('climb-hud').style.display = 'none';
    document.getElementById('gameover-modal').style.display = 'none';
    vigilantes.forEach(v => { v.alertado = false; v.avatar.grp.position.copy(v.home); });
    document.getElementById('vigilante-badge').textContent = '👮 Vigilância: Calma';
    document.getElementById('vigilante-badge').style.color = '#93c5fd';
    atualizarHUD();
    SoundEngine.playCollect();
    mostrarAlerta("🔄 Você renasceu na Praça do Bloco 734!", 'alert-success', 2200, 'info');
    canvas.focus();
  });

  let visorFotoAtivo = false;
  function abrirVisorFoto() {
    visorFotoAtivo = true;
    document.getElementById('photo-overlay').style.display = 'block';
    verificarAlvoFotografico();
    mostrarAlerta("📸 Visor aberto — pressione [P] novamente para tirar a foto e fechar!", "alert-info", 2200, "photo_mode");
  }

  function fecharVisorFoto() {
    visorFotoAtivo = false;
    document.getElementById('photo-overlay').style.display = 'none';
  }

  function verificarAlvoFotografico() {
    const p = pState.pos;
    let alvo = null;
    if (p.y >= 6.2 && Math.hypot(p.x - 48.0, p.z - 31.5) < 9.5) {
      alvo = 'Painéis Fotovoltaicos (Cobertura Bloco 734)';
    } else if (Math.hypot(p.x - 15.0, p.z - 22.0) < 6.5) {
      alvo = 'Escultura Cinética em Aço';
    } else if (Math.hypot(p.x - 27.0, p.z - 19.8) < 8.0) {
      alvo = "Espelho d'Água & Fachada Arqueada 734";
    } else if (Math.hypot(p.x - birdGroup.position.x, p.z - birdGroup.position.z) < 12.0) {
      alvo = 'Pássaro do Campus CT em Voo';
    } else if (Math.hypot(p.x - caoGroup.position.x, p.z - caoGroup.position.z) < 7.5) {
      alvo = 'Cachorro Guardião (Fundos Bloco 734)';
    } else if (Math.hypot(p.x - cretoPos.x, p.z - cretoPos.z) < 6.5) {
      alvo = 'Prof. Creto na Praça do CT';
    }
    document.getElementById('photo-target-hint').textContent = alvo
      ? `🎯 Enquadrado: ${alvo} — Pressione [P] para registrar e sair`
      : 'Aproxime-se de um alvo e pressione [P] para registrar e sair';
    return alvo;
  }

  function tirarFotoEFechar() {
    SoundEngine.playPhoto();
    const flash = document.getElementById('flash-screen');
    flash.style.opacity = '0.92';
    setTimeout(() => { flash.style.opacity = '0'; }, 120);

    const alvo = verificarAlvoFotografico() || 'Panorama Geral Bloco 734';
    pState.quests.fotos.add(alvo);
    if (alvo.includes('Fotovoltaicos')) {
      pState.quests.predioSolar = true;
    }
    fecharVisorFoto();
    mostrarAlerta(`📸 Foto salva & visor encerrado: "${alvo}" (${pState.quests.fotos.size}/5)!`, 'alert-success', 2600, 'success');
    atualizarHUD();
  }

  function acionarTeclaFoto() {
    if (!visorFotoAtivo) abrirVisorFoto();
    else tirarFotoEFechar();
  }

  document.getElementById('btn-photo').addEventListener('click', () => {
    acionarTeclaFoto();
    canvas.focus();
  });

  document.getElementById('btn-creto').addEventListener('click', () => {
    const distCreto = Math.hypot(pState.pos.x - cretoPos.x, pState.pos.z - cretoPos.z);
    if (distCreto <= 4.2 && pState.pos.y < 2.2) {
      abrirQuizCreto();
    } else {
      mostrarAlerta("🎓 Aproxime-se do Prof. Creto na área lateral (X=9.5, Z=22.2) e pressione [E] para falar com ele!", "alert-info", 2500, "info");
    }
    canvas.focus();
  });
  document.getElementById('btn-close-creto').addEventListener('click', fecharModalCreto);

  function iniciarCutsceneDanca(mensagem = "💃 VITÓRIA NO BLOCO 734! Comemorando conquista!") {
    if (inCoffeeCutscene || isDead) return;
    inDanceCutscene = true;
    danceTimer = 3.6;
    SoundEngine.playDanceBeat();
    document.getElementById('lb-top').style.height = '44px';
    document.getElementById('lb-bottom').style.height = '44px';
    const cap = document.getElementById('cutscene-caption');
    cap.textContent = mensagem;
    cap.style.display = 'block';
    spawnParticulas(pState.pos.clone().add(new THREE.Vector3(0, 2.2, 0)), [0x38bdf8, 0xfacc15, 0xf472b6, 0x4ade80], 34, 6.2);
  }

  function encerrarCutsceneDanca() {
    inDanceCutscene = false;
    player.bE.rotation.set(0, 0, 0);
    player.bD.rotation.set(0, 0, 0);
    player.pE.rotation.set(0, 0, 0);
    player.pD.rotation.set(0, 0, 0);
    player.torso.rotation.z = 0;
    document.getElementById('lb-top').style.height = '0';
    document.getElementById('lb-bottom').style.height = '0';
    document.getElementById('cutscene-caption').style.display = 'none';
  }

  document.getElementById('btn-dance').addEventListener('click', () => {
    iniciarCutsceneDanca("💃 MODO COMEMORAÇÃO — Dança da Vitória no Centro de Tecnologia!");
    canvas.focus();
  });

  function iniciarCutsceneCafe() {
    inCoffeeCutscene = true;
    coffeeTimer = 4.0;
    xicaraMao.visible = true;
    SoundEngine.playCoffee();
    document.getElementById('lb-top').style.height = '44px';
    document.getElementById('lb-bottom').style.height = '44px';
    const cap = document.getElementById('cutscene-caption');
    cap.textContent = "☕ Tomando um cafezinho expresso na Praça do CT... Life 100% restaurado!";
    cap.style.display = 'block';
    document.getElementById('ar-card').style.display = 'none';
  }

  function encerrarCutsceneCafe() {
    inCoffeeCutscene = false;
    xicaraMao.visible = false;
    steamParticles.forEach(s => s.mesh.visible = false);
    document.getElementById('lb-top').style.height = '0';
    document.getElementById('lb-bottom').style.height = '0';
    document.getElementById('cutscene-caption').style.display = 'none';
    energy = 100;
    atualizarHUD();
    coffeeBuffTimer = 30.0;
    document.getElementById('buff-tag').style.display = 'inline';
    mostrarAlerta("☕ Cafezinho concluído! Life 100% + Turbo de Velocidade por 30s!", 'alert-success', 2800, 'success');
  }

  document.getElementById('btn-fullscreen').addEventListener('click', () => {
    if (!document.fullscreenElement && !wrapper.classList.contains('pseudo-fullscreen')) {
      if (wrapper.requestFullscreen) {
        wrapper.requestFullscreen().catch(() => wrapper.classList.add('pseudo-fullscreen'));
      } else {
        wrapper.classList.add('pseudo-fullscreen');
      }
    } else {
      if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
      wrapper.classList.remove('pseudo-fullscreen');
    }
    if (gameStarted) canvas.focus();
  });

  const pauseModal = document.getElementById('pause-modal');
  function togglePauseModal(forceState = null) {
    if (!gameStarted || inCoffeeCutscene || inDanceCutscene || isDead || inQuizModal || inSummaryModal) return;
    isPaused = forceState !== null ? forceState : !isPaused;
    pauseModal.style.display = isPaused ? 'flex' : 'none';
    if (isPaused) atualizarHUD();
    else canvas.focus();
  }
  document.getElementById('btn-quests').addEventListener('click', () => togglePauseModal(true));
  document.getElementById('btn-resume').addEventListener('click', () => togglePauseModal(false));

  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const target = btn.getAttribute('data-tab');
      ['tab-inv', 'tab-mini', 'tab-quests', 'tab-help'].forEach(id => {
        document.getElementById(id).style.display = (id === target) ? 'block' : 'none';
      });
    });
  });

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let downX = 0, downY = 0;

  canvas.addEventListener('mousedown', (e) => {
    if (!gameStarted) return;
    SoundEngine.init();
    canvas.focus();
    downX = e.clientX; downY = e.clientY;
    isDragging = true; lastMX = e.clientX; lastMY = e.clientY;
  });

  window.addEventListener('mouseup', (e) => {
    isDragging = false;
    if (gameStarted && Math.hypot(e.clientX - downX, e.clientY - downY) < 5 && !isPaused && !isDead && !inQuizModal && !inSummaryModal) {
      const rect = canvas.getBoundingClientRect();
      pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      const activeCam = (projMode === 'perspectiva') ? perspCamera : orthoCamera;
      raycaster.setFromCamera(pointer, activeCam);
      const hits = raycaster.intersectObjects(clickableTotems, false);
      if (hits.length > 0 && hits[0].object.userData.totem) {
        hits[0].object.userData.totem.acaoFn();
      }
    }
  });

  const keys = {};
  window.addEventListener('keydown', (e) => {
    if (!gameStarted) return;
    SoundEngine.init();
    const k = e.key.toLowerCase();
    if (['arrowup', 'arrowdown', 'arrowleft', 'arrowright', ' '].includes(k)) {
      e.preventDefault();
    }
    if (k === 'q' || k === 'tab') {
      e.preventDefault();
      togglePauseModal();
      return;
    }
    if (k === 'escape') {
      if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
      wrapper.classList.remove('pseudo-fullscreen');
      if (inSummaryModal) fecharResumoSessao();
      else if (inQuizModal) fecharModalCreto();
      else if (visorFotoAtivo) fecharVisorFoto();
      else if (isPaused) togglePauseModal(false);
      return;
    }
    keys[k] = true;
    if (isPaused || inCoffeeCutscene || inDanceCutscene || isDead || inQuizModal || inSummaryModal) return;

    if (e.code === 'Space') {
      if (pState.climbing) {
        pState.climbing = null;
        pState.onGround = false;
        pState.velY = 3.5;
        document.getElementById('climb-hud').style.display = 'none';
      } else {
        tentarPular();
      }
    } else if (k === 'e') {
      executarInteracaoE();
    } else if (k === 'r') {
      executarEscalarArvorePosteR();
    } else if (k === 'f') {
      executarAtaque();
    } else if (k === 'p') {
      acionarTeclaFoto();
    } else if (k === 'l') {
      alternarLanterna();
    } else if (k === 'g') {
      const distCreto = Math.hypot(pState.pos.x - cretoPos.x, pState.pos.z - cretoPos.z);
      if (distCreto <= 4.2 && pState.pos.y < 2.2) {
        abrirQuizCreto();
      } else {
        mostrarAlerta("🎓 Vá até o Prof. Creto na área lateral (X=9.5, Z=22.2) e pressione [E] para interagir!", "alert-info", 2500, "info");
      }
    } else if (k === 'c') {
      iniciarCutsceneDanca("💃 Comemorando no Bloco 734!");
    }
  });

  const analog = { moveX: 0, moveY: 0, lookX: 0, lookY: 0 };

  function bindXboxStick(el, onStick) {
    const knob = el.querySelector('.knob');
    const maxR = 35;
    let pid = null;

    function setFromPoint(cx, cy) {
      const r = el.getBoundingClientRect();
      let dx = cx - (r.left + r.width / 2);
      let dy = cy - (r.top + r.height / 2);
      const dist = Math.hypot(dx, dy);
      if (dist > maxR) { dx = (dx / dist) * maxR; dy = (dy / dist) * maxR; }
      knob.style.transform = `translate(${dx}px, ${dy}px)`;
      onStick(dx / maxR, -dy / maxR);
    }

    function clearStick() {
      pid = null;
      knob.style.transform = 'translate(0px, 0px)';
      onStick(0, 0);
    }

    el.addEventListener('pointerdown', (e) => {
      if (!gameStarted) return;
      e.preventDefault();
      SoundEngine.init();
      pid = e.pointerId;
      el.setPointerCapture(pid);
      setFromPoint(e.clientX, e.clientY);
    });
    el.addEventListener('pointermove', (e) => {
      if (e.pointerId === pid) { e.preventDefault(); setFromPoint(e.clientX, e.clientY); }
    });
    el.addEventListener('pointerup', (e) => { if (e.pointerId === pid) clearStick(); });
    el.addEventListener('pointercancel', (e) => { if (e.pointerId === pid) clearStick(); });
  }

  bindXboxStick(document.getElementById('stick-walk'), (x, y) => { analog.moveX = x; analog.moveY = y; });
  bindXboxStick(document.getElementById('stick-look'), (x, y) => { analog.lookX = x; analog.lookY = y; });
  window.addEventListener('keyup', (e) => { keys[e.key.toLowerCase()] = false; });

  window.addEventListener('mousemove', (e) => {
    if (!gameStarted || !isDragging || isPaused || inCoffeeCutscene || inDanceCutscene || isDead || inQuizModal || inSummaryModal) return;
    const dx = e.clientX - lastMX;
    const dy = e.clientY - lastMY;
    lastMX = e.clientX; lastMY = e.clientY;
    pState.camYaw -= dx * 0.0065;
    pState.camPitch = Math.max(0.05, Math.min(1.25, pState.camPitch + dy * 0.0055));
  });
  canvas.addEventListener('wheel', (e) => {
    if (!gameStarted) return;
    e.preventDefault();
    pState.camDist = Math.max(3.4, Math.min(18.0, pState.camDist + e.deltaY * 0.005));
  }, { passive: false });

  function tentarPular() {
    if (pState.climbing) return;
    if (pState.onGround) {
      pState.velY = coffeeBuffTimer > 0 ? 8.8 : 7.5;
      pState.onGround = false;
      SoundEngine.playJump();
    }
  }

  function executarAtaque() {
    if (attackTimer > 0 || isPaused || inCoffeeCutscene || inDanceCutscene || isDead || pState.climbing) return;
    attackTimer = 0.36;
    const forward = new THREE.Vector3(Math.sin(pState.yaw), 0, Math.cos(pState.yaw));
    const hitPoint = pState.pos.clone().add(new THREE.Vector3(0, 1.3, 0)).addScaledVector(forward, 1.5);

    let acertou = false;
    breakableWindows.forEach((w) => {
      if (w.broken) return;
      const distXZ = Math.hypot(hitPoint.x - w.pos.x, hitPoint.z - w.pos.z);
      if (distXZ < 2.35 && Math.abs(hitPoint.y - w.pos.y) < 2.2) {
        acertou = true;
        w.hp -= 1;
        w.hpBarBg.visible = true;
        w.hpBarFill.visible = true;
        const ratio = Math.max(0, w.hp / w.maxHp);
        w.hpBarFill.scale.x = Math.max(0.01, ratio);
        w.hpBarFill.material.color.setHex(ratio > 0.5 ? 0x22c55e : (ratio > 0.2 ? 0xfacc15 : 0xef4444));
        spawnParticulas(w.pos, 0x93c5fd, 8);
        shakeTimer = 0.14;

        if (w.hp <= 0) {
          w.broken = true;
          w.glass.visible = false;
          w.hpBarBg.visible = false;
          w.hpBarFill.visible = false;
          windowsBrokenCount++;
          document.getElementById('win-broken').textContent = `${windowsBrokenCount}/${breakableWindows.length}`;
          atualizarPlacar1v1();
          SoundEngine.playGlass();
          spawnParticulas(w.pos, 0xe0f2fe, 24);
          dispararDerramamentoAsfalto(w);
          acionarVigilantes(`Quebra de ${w.nome}`);
          mostrarAlerta(`💥 ${w.nome} quebrada! Derramando asfalto CBUQ...`, 'alert-danger', 2600, 'danger');
        } else {
          SoundEngine.tone(420, 0.09, 'triangle', 0.08);
          w.glassMat.color.setHex(w.hp === 2 ? 0x1e3a5f : 0x3b82f6);
          mostrarAlerta(`⚔️ Impacto em ${w.nome}! Durabilidade: ${w.hp}/${w.maxHp}`, 'alert-iot', 1600, 'info');
        }
      }
    });
    if (!acertou) SoundEngine.tone(180, 0.08, 'sine', 0.05, 0, 90);
  }

  function encontrarArvoreOuPosteProximo() {
    for (const arv of arvoresCoords) {
      if (Math.hypot(pState.pos.x - arv.x, pState.pos.z - arv.z) < 3.0) {
        return { tipo: 'arvore', nome: arv.nome, x: arv.x, z: arv.z, maxH: arv.h, balance: 0, coruja: arv.temCoruja };
      }
    }
    for (const pt of postesData) {
      if (Math.hypot(pState.pos.x - pt.x, pState.pos.z - pt.z) < 2.8) {
        return { tipo: 'poste', nome: pt.nome, x: pt.x, z: pt.z, maxH: pt.h, balance: 0 };
      }
    }
    return null;
  }

  function executarEscalarArvorePosteR() {
    if (!gameStarted || isPaused || inCoffeeCutscene || inDanceCutscene || isDead || inQuizModal || inSummaryModal) return;
    if (pState.climbing) {
      pState.climbing = null;
      pState.onGround = false;
      pState.velY = 2.0;
      document.getElementById('climb-hud').style.display = 'none';
      mostrarAlerta('🧗 Você soltou da escalada!', 'alert-info', 1800, 'info');
      return;
    }
    const alvoClimb = encontrarArvoreOuPosteProximo();
    if (!alvoClimb) {
      mostrarAlerta("🌳 Aproxime-se de uma Árvore ou Poste de Iluminação e pressione [R]!", 'alert-info', 2200, 'info');
      return;
    }
    pState.climbing = alvoClimb;
    pState.pos.x = alvoClimb.x;
    pState.pos.z = alvoClimb.z;
    pState.pos.y = Math.max(0.4, pState.pos.y);
    document.getElementById('climb-hud').style.display = 'flex';
    document.getElementById('climb-label').textContent =
      `🧗 Escalando ${alvoClimb.nome}! [W/S] ou Analógico Esq. subir/descer • [A/D] equilibrar • [R] soltar`;
    SoundEngine.playSwitch();
  }

  function encontrarInteracaoE() {
    for (const item of collectibles) {
      if (!item.coletado && pState.pos.distanceTo(new THREE.Vector3(item.x, pState.pos.y, item.z)) < 2.6) {
        return { tipo: 'collectible', item };
      }
    }
    if (Math.hypot(pState.pos.x - cretoPos.x, pState.pos.z - cretoPos.z) < 4.2 && pState.pos.y < 2.2) {
      return { tipo: 'creto' };
    }
    if (Math.hypot(pState.pos.x - cafePos.x, pState.pos.z - cafePos.z) < 2.6 && pState.pos.y < 1.8) {
      return { tipo: 'coffee' };
    }
    for (const t of clickableTotems) {
      const td = t.userData.totem;
      if (td && Math.hypot(pState.pos.x - td.x, pState.pos.z - td.z) < 2.5 && pState.pos.y < 1.8) {
        return { tipo: 'totem', totem: td };
      }
    }
    let melhor = null;
    let menorDist = 5.4;
    for (const obj of arObjects) {
      const d = Math.hypot(pState.pos.x - obj.center.x, pState.pos.z - obj.center.z);
      const raioEfetivo = Math.max(obj.size.x, obj.size.z) * 0.5 + 2.4;
      if (d < raioEfetivo && d < menorDist) {
        menorDist = d;
        melhor = { tipo: 'ar', obj };
      }
    }
    return melhor;
  }

  function executarInteracaoE() {
    if (!gameStarted || isPaused || inCoffeeCutscene || inDanceCutscene || isDead || inQuizModal || inSummaryModal) return;

    const alvo = encontrarInteracaoE();
    if (!alvo) {
      mostrarAlerta("🔍 Aproxime-se de uma mesa, Prof. Creto, totem, cafeteira ou monumento e pressione [E]!", 'alert-info', 2300, 'info');
      return;
    }

    if (alvo.tipo === 'collectible') {
      alvo.item.coletado = true;
      alvo.item.mesh.visible = false;
      atualizarHUD();
      const totalCol = collectibles.filter(c => c.coletado).length;
      SoundEngine.playCollect();
      spawnParticulas(new THREE.Vector3(alvo.item.x, alvo.item.y + 0.3, alvo.item.z), [alvo.item.cor, 0xffffff], 16, 3.6);
      if (totalCol === collectibles.length) {
        iniciarCutsceneDanca("🏆 TODAS AS 8 ATIVIDADES E PROVAS CONCLUÍDAS! Parabéns!");
      } else {
        mostrarAlerta(`🎒 Coletado: ${alvo.item.nome} (${totalCol}/8)!`, 'alert-success', 2400, 'success');
      }
      return;
    }

    if (alvo.tipo === 'creto') {
      abrirQuizCreto();
      return;
    }

    if (alvo.tipo === 'coffee') {
      iniciarCutsceneCafe();
      return;
    }

    if (alvo.tipo === 'totem') {
      alvo.totem.acaoFn();
      return;
    }

    const obj = alvo.obj;
    SoundEngine.playSwitch();
    const minV = obj.center.clone().sub(obj.size.clone().multiplyScalar(0.5));
    const maxV = obj.center.clone().add(obj.size.clone().multiplyScalar(0.5));
    arBoxHelper.box.set(minV, maxV);
    arBoxHelper.visible = true;
    arTimer = 6.0;

    document.getElementById('ar-title').textContent = obj.nome;
    document.getElementById('ar-desc').textContent = obj.desc;
    document.getElementById('ar-meta').textContent = obj.meta;
    document.getElementById('ar-card').style.display = 'block';
  }

  document.getElementById('btn-interact').addEventListener('click', () => { executarInteracaoE(); canvas.focus(); });
  document.getElementById('btn-climb').addEventListener('click', () => { executarEscalarArvorePosteR(); canvas.focus(); });
  document.getElementById('btn-attack').addEventListener('click', () => { executarAtaque(); canvas.focus(); });
  document.getElementById('btn-jump').addEventListener('click', () => { tentarPular(); canvas.focus(); });

  function dentroBloco734(x, z) {
    if (x < 15.0 || x > 59.0 || z < 20.0 || z > 53.0) return false;
    if (x < 37.0 && z < 42.0) return ((x - 37.0) ** 2 + (z - 42.0) ** 2) <= (22.0 ** 2);
    return true;
  }

  function getGroundY(x, z, curY) {
    if (dentroBloco734(x, z) && curY >= 6.1) return 6.60;
    if (x >= 19.5 && x <= 34.5 && z >= 16.8 && z <= 22.8) return 0.42;
    return z >= 16.0 ? 0.20 : 0.10;
  }

  // ============================================================================
  // 9. LOOP PRINCIPAL DE ANIMAÇÃO, CRONÔMETRO E FÍSICA
  // ============================================================================
  const clock = new THREE.Clock();
  let timerAccum = 0;
  atualizarIluminacaoCena();

  function renderActiveCamera() {
    if (projMode === 'perspectiva') {
      renderer.render(scene, perspCamera);
      return;
    }
    const target = new THREE.Vector3(pState.pos.x, pState.pos.y + 1.4, pState.pos.z);
    const dist = 28;
    orthoCamera.position.set(
      target.x + Math.sin(pState.camYaw) * Math.cos(pState.camPitch) * dist,
      target.y + Math.sin(pState.camPitch) * dist,
      target.z + Math.cos(pState.camYaw) * Math.cos(pState.camPitch) * dist
    );
    orthoCamera.lookAt(target);
    orthoCamera.updateProjectionMatrix();
    renderer.render(scene, orthoCamera);
  }

  function animate() {
    requestAnimationFrame(animate);
    const dt = Math.min(clock.getDelta(), 0.05);
    const elapsed = clock.getElapsedTime();

    if (!gameStarted || isPaused || isDead || inQuizModal || inSummaryModal) {
      renderActiveCamera();
      return;
    }

    if (!inCoffeeCutscene && !inDanceCutscene) {
      timerAccum += dt;
      if (timerAccum >= 1.0) {
        const inc = Math.floor(timerAccum);
        timerAccum -= inc;
        sessionSeconds += inc;
        document.getElementById('timer-badge').textContent = formatarTempo(sessionSeconds);
      }
    }

    if (pState.damageCooldown > 0) pState.damageCooldown -= dt;
    if (pState.jumpscareCooldown > 0) pState.jumpscareCooldown -= dt;

    const birdAng = elapsed * 0.42;
    const prevBirdX = birdGroup.position.x;
    const prevBirdZ = birdGroup.position.z;
    birdGroup.position.set(
      34.0 + Math.cos(birdAng) * 18.0,
      9.4 + Math.sin(elapsed * 1.6) * 0.9,
      26.0 + Math.sin(birdAng * 2.0) * 9.5
    );
    birdGroup.rotation.y = Math.atan2(birdGroup.position.x - prevBirdX, birdGroup.position.z - prevBirdZ);
    const flap = Math.sin(elapsed * 15.0) * 0.55;
    wingL.rotation.z = flap;
    wingR.rotation.z = -flap;
    if (Math.sin(elapsed * 0.7) > 0.996) SoundEngine.playChirp();

    iotBeacon.scale.setScalar(1.0 + 0.25 * Math.sin(elapsed * 8.0));
    cretoAro.scale.setScalar(1.0 + 0.12 * Math.sin(elapsed * 4.0));
    cretoBeacon.position.y = 2.52 + Math.sin(elapsed * 3.5) * 0.10;
    cretoBeacon.rotation.y += dt * 1.8;
    placaCreto.position.y = 3.15 + Math.sin(elapsed * 2.5) * 0.08;
    aguaMesh.position.y = 0.38 + Math.sin(elapsed * 2.8) * 0.015;
    collectibles.forEach((c, idx) => {
      if (!c.coletado) {
        c.mesh.rotation.y += dt * 2.2;
        c.mesh.position.y = c.y + Math.sin(elapsed * 3.0 + idx) * 0.07;
      }
    });

    asphaltSpills.forEach((sp) => {
      if (sp.progress < 1.0) {
        sp.progress = Math.min(1.0, sp.progress + dt * 0.45);
        const h = (sp.topY - 0.20) * sp.progress;
        sp.streamMesh.scale.y = Math.max(0.1, h / 0.1);
        sp.streamMesh.position.y = sp.topY - h * 0.5;
        const r = 0.1 + 1.35 * sp.progress;
        sp.puddleMesh.scale.set(r / 0.1, 1, r / 0.1);
      }
      if (Math.hypot(pState.pos.x - sp.x, pState.pos.z - sp.z) < 1.3 && pState.pos.y < 1.0) {
        aplicarDano(12, "Contato com asfalto CBUQ derramado nas janelas!");
        acionarVigilantes("Ocorrência de Asfalto nas Janelas");
      }
    });

    for (let i = particles.length - 1; i >= 0; i--) {
      const s = particles[i];
      s.life -= dt;
      s.vel.y -= 13.5 * dt;
      s.mesh.position.addScaledVector(s.vel, dt);
      s.mesh.rotation.x += dt * 6;
      s.mesh.rotation.y += dt * 5;
      if (s.life <= 0 || s.mesh.position.y < 0.12) {
        scene.remove(s.mesh);
        particles.splice(i, 1);
      }
    }

    const pedestreNaFaixa = (pState.pos.x >= 25.0 && pState.pos.x <= 29.0 && pState.pos.z <= 13.0 && pState.pos.y < 1.6);

    animatedCars.forEach((car) => {
      let bloqueadoFaixa = false;
      if (pedestreNaFaixa) {
        const distAteFaixa = car.dirX > 0 ? (24.4 - car.grp.position.x) : (car.grp.position.x - 29.6);
        if (distAteFaixa > 0 && distAteFaixa < 8.5) {
          bloqueadoFaixa = true;
        }
      }
      const velAtual = bloqueadoFaixa ? 0 : car.speed;
      car.grp.position.x += car.dirX * velAtual * dt;
      if (car.dirX > 0 && car.grp.position.x > 65.0) car.grp.position.x = 5.0;
      if (car.dirX < 0 && car.grp.position.x < 5.0) car.grp.position.x = 65.0;
      car.rodas.forEach(r => { r.rotation.x += car.dirX * velAtual * dt * 2.5; });

      if (car.honkTimer > 0) car.honkTimer -= dt;
      const dx = Math.abs(pState.pos.x - car.grp.position.x);
      const dz = Math.abs(pState.pos.z - car.grp.position.z);
      if (dx < 5.5 && dz < 2.2 && car.honkTimer <= 0 && pState.pos.y < 1.5) {
        SoundEngine.playHonk();
        car.honkTimer = 2.5;
      }
      if (dx < 2.35 && dz < 1.25 && pState.pos.y < 1.5) {
        pState.pos.z = 14.5;
        aplicarDano(45, `Atropelado pelo ${car.nome} na pista do CT!`, true);
      }
    });

    if (inDanceCutscene) {
      danceTimer -= dt;
      const beat = elapsed * 8.5;
      player.bE.rotation.x = -1.2;
      player.bD.rotation.x = -1.2;
      player.bE.rotation.z = -0.35 + Math.abs(Math.sin(beat)) * 0.45;
      player.bD.rotation.z = 0.35 - Math.abs(Math.sin(beat)) * 0.45;
      player.torso.rotation.z = Math.sin(beat * 0.5) * 0.16;
      player.pE.rotation.z = Math.max(0, Math.sin(beat * 0.5)) * 0.25;
      player.pD.rotation.z = Math.min(0, Math.sin(beat * 0.5)) * 0.25;
      player.grp.position.y = pState.pos.y + Math.abs(Math.sin(beat)) * 0.14;

      const angD = elapsed * 1.1;
      perspCamera.position.set(
        pState.pos.x + Math.cos(angD) * 3.1,
        pState.pos.y + 1.9,
        pState.pos.z + Math.sin(angD) * 3.1
      );
      perspCamera.lookAt(pState.pos.x, pState.pos.y + 1.3, pState.pos.z);
      if (danceTimer <= 0) encerrarCutsceneDanca();
      renderActiveCamera();
      return;
    }

    if (inCoffeeCutscene) {
      coffeeTimer -= dt;
      player.bD.rotation.x = -1.35 + Math.sin(elapsed * 6.0) * 0.22;
      steamParticles.forEach((sp, i) => {
        sp.mesh.visible = true;
        sp.t = (sp.t + dt * 0.7) % 1.0;
        sp.mesh.position.set(
          pState.pos.x + Math.sin(elapsed * 4 + i) * 0.12,
          pState.pos.y + 1.55 + sp.t * 0.75,
          pState.pos.z + Math.cos(elapsed * 4 + i) * 0.12
        );
        sp.mesh.material.opacity = (1.0 - sp.t) * 0.7;
      });
      const angCam = elapsed * 0.95;
      perspCamera.position.set(
        pState.pos.x + Math.cos(angCam) * 2.6,
        pState.pos.y + 1.85,
        pState.pos.z + Math.sin(angCam) * 2.6
      );
      perspCamera.lookAt(pState.pos.x, pState.pos.y + 1.35, pState.pos.z);
      if (coffeeTimer <= 0) {
        player.bD.rotation.x = 0;
        encerrarCutsceneCafe();
      }
      renderActiveCamera();
      return;
    }

    if (coffeeBuffTimer > 0) {
      coffeeBuffTimer -= dt;
      if (coffeeBuffTimer <= 0) document.getElementById('buff-tag').style.display = 'none';
    }

    if (arTimer > 0) {
      arTimer -= dt;
      if (arTimer <= 0) {
        arBoxHelper.visible = false;
        document.getElementById('ar-card').style.display = 'none';
      }
    }

    let isMoving = false;
    pState.onLadderAuto = false;

    if (pState.climbing) {
      const cl = pState.climbing;
      pState.pos.x = cl.x;
      pState.pos.z = cl.z;

      let climbVert = 0;
      if (keys['w'] || keys['arrowup']) climbVert += 1;
      if (keys['s'] || keys['arrowdown']) climbVert -= 1;
      if (Math.abs(analog.moveY) > 0.20) climbVert += analog.moveY;
      climbVert = Math.max(-1, Math.min(1, climbVert));

      if (Math.abs(climbVert) > 0.12) {
        pState.pos.y += climbVert * 3.1 * dt;
        isMoving = true;
      }

      cl.balance += (Math.sin(elapsed * 3.4) * 0.48) * dt;
      if (keys['a'] || keys['arrowleft']) cl.balance -= 1.45 * dt;
      if (keys['d'] || keys['arrowright']) cl.balance += 1.45 * dt;
      if (Math.abs(analog.moveX) > 0.15) cl.balance += analog.moveX * 1.45 * dt;
      const pct = 50 + cl.balance * 50;
      document.getElementById('balance-needle').style.left = `${Math.max(4, Math.min(96, pct))}%`;

      if (Math.abs(cl.balance) > 0.96) {
        pState.climbing = null;
        pState.onGround = false;
        pState.velY = -2.0;
        document.getElementById('climb-hud').style.display = 'none';
        aplicarDano(15, `Perdeu o equilíbrio e caiu de ${cl.nome}!`, true);
      }

      if (pState.climbing && pState.pos.y >= cl.maxH) {
        pState.pos.y = cl.maxH;
        if (cl.tipo === 'arvore') {
          pState.quests.arvore = true;
          mostrarAlerta('🌳 Mini-Quest Concluída: Subiu na Árvore sem cair!', 'alert-success', 2600, 'success');
          if (cl.coruja) dispararJumpscare('🦉👁️👁️', 'SUSTO DA CORUJA!', 'A Coruja Guardiã estava escondida no topo da árvore!');
        } else if (cl.tipo === 'poste') {
          pState.quests.poste = true;
          mostrarAlerta('💡 Mini-Quest Concluída: Subiu no Poste sem cair!', 'alert-success', 2600, 'success');
        }
        atualizarHUD();
      }
      if (pState.climbing && pState.pos.y <= 0.22) {
        pState.pos.y = 0.20;
        pState.climbing = null;
        document.getElementById('climb-hud').style.display = 'none';
      }
    } else {
      const speed = coffeeBuffTimer > 0 ? 9.8 : 7.0;
      let moveF = 0, moveR = 0;
      if (keys['w'] || keys['arrowup']) moveF += 1;
      if (keys['s'] || keys['arrowdown']) moveF -= 1;
      if (keys['d'] || keys['arrowright']) moveR += 1;
      if (keys['a'] || keys['arrowleft']) moveR -= 1;

      if (Math.abs(analog.lookX) > 0.04 || Math.abs(analog.lookY) > 0.04) {
        pState.camYaw -= analog.lookX * 2.5 * dt;
        pState.camPitch = Math.max(0.05, Math.min(1.25, pState.camPitch - analog.lookY * 1.8 * dt));
      }
      if (Math.abs(analog.moveY) > 0.06) moveF += analog.moveY;
      if (Math.abs(analog.moveX) > 0.06) moveR += analog.moveX;

      const fwd = new THREE.Vector3(-Math.sin(pState.camYaw), 0, -Math.cos(pState.camYaw));
      const right = new THREE.Vector3(Math.cos(pState.camYaw), 0, -Math.sin(pState.camYaw));
      const moveVec = new THREE.Vector3().addScaledVector(fwd, moveF).addScaledVector(right, moveR);

      const distEscada = Math.hypot(pState.pos.x - 59.35, pState.pos.z - 24.0);
      if (distEscada < 1.35 && (moveF > 0 || keys[' '])) {
        pState.onLadderAuto = true;
        isMoving = true;
        pState.pos.x = 59.35;
        pState.pos.z = 24.0;
        pState.pos.y += 5.2 * dt;
        pState.velY = 0;
        pState.onGround = false;
        if (pState.pos.y >= 6.65) {
          pState.pos.set(57.4, 6.60, 24.0);
          pState.wasOnRoof = true;
          pState.onGround = true;
          mostrarAlerta('☀️ Chegou ao telhado! Visite os Painéis Fotovoltaicos e pressione [P]!', 'alert-info', 2800, 'info');
        }
      } else {
        isMoving = moveVec.lengthSq() > 0.001;
        if (isMoving) {
          moveVec.normalize();
          pState.yaw = Math.atan2(moveVec.x, moveVec.z);
          const nextX = Math.max(3.0, Math.min(66.0, pState.pos.x + moveVec.x * speed * dt));
          const nextZ = Math.max(3.0, Math.min(58.5, pState.pos.z + moveVec.z * speed * dt));

          if (!(dentroBloco734(nextX, nextZ) && pState.pos.y < 6.1)) {
            pState.pos.x = nextX;
            pState.pos.z = nextZ;
          }
          if (pState.onGround) {
            stepAccumulator += speed * dt;
            if (stepAccumulator >= 2.2) {
              stepAccumulator = 0;
              SoundEngine.playStep();
            }
          }
        }

        const groundY = getGroundY(pState.pos.x, pState.pos.z, pState.pos.y);
        if (pState.pos.y > groundY || pState.velY > 0) {
          pState.velY -= 19.5 * dt;
          pState.pos.y += pState.velY * dt;
          pState.onGround = false;
        }

        if (pState.pos.y <= groundY) {
          if (pState.wasOnRoof && groundY < 1.5) {
            const naFonte = (pState.pos.x >= 19.5 && pState.pos.x <= 34.5 && pState.pos.z >= 16.8 && pState.pos.z <= 22.8);
            const naArvore = arvoresCoords.some(a => Math.hypot(pState.pos.x - a.x, pState.pos.z - a.z) < 3.4) ||
                             (Math.hypot(pState.pos.x - 42.5, pState.pos.z - 17.5) < 3.0);
            const naEscada = Math.hypot(pState.pos.x - 59.35, pState.pos.z - 24.0) < 2.2;

            if (naFonte) {
              SoundEngine.playCollect();
              spawnParticulas(pState.pos.clone().add(new THREE.Vector3(0, 0.4, 0)), 0x38bdf8, 28);
              mostrarAlerta("💦 Pouso perfeito na Fonte do CT! Queda 100% amortecida na água.", 'alert-success', 2800, 'success');
            } else if (naArvore) {
              SoundEngine.playCollect();
              spawnParticulas(pState.pos.clone().add(new THREE.Vector3(0, 1.2, 0)), 0x22c55e, 24);
              mostrarAlerta("🌿 Pouso amortecido pela copa da Árvore!", 'alert-success', 2800, 'success');
            } else if (!naEscada) {
              aplicarDano(55, "Queda livre do telhado do Bloco 734! Use a Escada, a Árvore ou a Fonte.", true);
            }
          }
          pState.pos.y = groundY;
          pState.velY = 0;
          pState.onGround = true;
          pState.wasOnRoof = (groundY >= 6.4);
        }
      }
    }

    player.grp.position.copy(pState.pos);
    player.grp.rotation.y = pState.yaw;

    if (pState.climbing || pState.onLadderAuto) {
      const clSwing = Math.sin(elapsed * 10.0) * 0.65;
      player.bE.rotation.x = -2.2 + clSwing;
      player.bD.rotation.x = -2.2 - clSwing;
      player.pE.rotation.x = -clSwing * 0.5;
      player.pD.rotation.x = clSwing * 0.5;
    } else if (isMoving && pState.onGround) {
      const swing = Math.sin(elapsed * 11.0) * 0.55;
      player.pE.rotation.x = swing;
      player.pD.rotation.x = -swing;
      player.bE.rotation.x = -swing * 0.7;
      if (attackTimer <= 0) player.bD.rotation.x = swing * 0.7;
    } else {
      player.pE.rotation.x *= 0.8;
      player.pD.rotation.x *= 0.8;
      player.bE.rotation.x *= 0.8;
      if (attackTimer <= 0) player.bD.rotation.x *= 0.8;
    }

    if (attackTimer > 0) {
      attackTimer -= dt;
      player.bD.rotation.x = -Math.sin((1.0 - attackTimer / 0.36) * Math.PI) * 2.1;
    }

    if (pState.pos.y >= 6.2) {
      const distArco = Math.abs(Math.hypot(pState.pos.x - 37.0, pState.pos.z - 42.0) - 22.0);
      const pertoBorda =
        (pState.pos.x > 57.5) || (pState.pos.z < 21.4) || (pState.pos.z > 51.5) || (pState.pos.x < 16.5) ||
        (pState.pos.x <= 37.2 && pState.pos.z <= 42.2 && distArco < 1.6);
      if (pertoBorda) {
        SoundEngine.playIoT();
        mostrarAlerta('📡 SENSOR IoT TELHADO: Risco de queda na beirada do prédio!', 'alert-iot', 2200, 'iot');
      }
    }

    if (pState.pos.y < 1.5 && pState.pos.z < 12.8) {
      SoundEngine.playIoT();
      mostrarAlerta('📡 ALERTA IoT VIÁRIO: Use a faixa de pedestres (X=27) para atravessar!', 'alert-iot', 2200, 'iot');
      if (pState.pos.x >= 25.0 && pState.pos.x <= 29.0 && pState.pos.z <= 4.0) {
        pState.crossedOut = true;
        mostrarAlerta('🚸 Metade da travessia concluída — agora retorne pela faixa!', 'alert-info', 2200, 'info');
      }
    } else if (pState.crossedOut && pState.pos.z >= 13.2) {
      pState.crossedOut = false;
      pState.quests.travessia = true;
      mostrarAlerta('✅ Mini-Quest Concluída: Travessia na faixa e retorno em segurança!', 'alert-success', 2800, 'success');
      atualizarHUD();
    }

    const naFonteAtual = (pState.pos.x >= 19.65 && pState.pos.x <= 34.35 && pState.pos.z >= 16.95 && pState.pos.z <= 22.65 && pState.pos.y < 1.2);
    if (naFonteAtual) {
      pState.fountainTimer += dt;
      mostrarAlerta('⚠️ ÁREA RESTRITA: Não permaneça dentro da fonte!', 'alert-iot', 1800, 'iot');
      if (pState.fountainTimer > 1.8) {
        aplicarDano(10, 'Permanência não autorizada dentro da fonte!');
        acionarVigilantes('Invasão da Fonte');
      }
    } else {
      pState.fountainTimer = Math.max(0, pState.fountainTimer - dt);
    }

    if (Math.hypot(pState.pos.x - 15.0, pState.pos.z - 22.0) < 2.25 && pState.pos.y < 2.5) {
      acionarVigilantes('Aproximação na Escultura Cinética');
    }

    if (pState.pos.x < 15.5 && pState.pos.z > 37.5 && pState.pos.y < 2.0) {
      dispararJumpscare('🦉👁️👁️', 'SUSTO NO BECO ESCURO!', 'Você entrou no corredor sombrio da Ala Sul!');
    }

    const distCaoPlayer = Math.hypot(pState.pos.x - caoHome.x, pState.pos.z - caoHome.z);
    if ((pState.pos.z > 53.2 && pState.pos.x > 22.0 && pState.pos.y < 2.0) || distCaoPlayer < 7.5) {
      if (distCaoPlayer < 6.0 && pState.jumpscareCooldown <= 0) {
        dispararJumpscare('🐕🔥', 'EMBOSCADA NOS FUNDOS DO BLOCO!', 'O Cachorro Guardião protege a área restrita atrás do Bloco 734!');
      }
      const dirCao = new THREE.Vector3(pState.pos.x - caoGroup.position.x, 0, pState.pos.z - caoGroup.position.z);
      if (dirCao.length() > 0.1) {
        dirCao.normalize();
        caoGroup.position.addScaledVector(dirCao, 6.4 * dt);
        caoGroup.rotation.y = Math.atan2(-dirCao.z, dirCao.x);
      }
      if (Math.sin(elapsed * 12) > 0.92) SoundEngine.playBark();
      mostrarAlerta('🐕 ÁREA RESTRITA DOS FUNDOS! O Cachorro Guardião está avançando!', 'alert-danger', 2200, 'dog');
      if (Math.hypot(pState.pos.x - caoGroup.position.x, pState.pos.z - caoGroup.position.z) < 1.3 && pState.pos.y < 1.4) {
        aplicarDano(20, 'Mordida do Cachorro Guardião nos fundos do Bloco 734!', true);
      }
    } else {
      const back = new THREE.Vector3().subVectors(caoHome, caoGroup.position);
      if (back.length() > 0.2) {
        back.normalize();
        caoGroup.position.addScaledVector(back, 2.8 * dt);
      }
    }

    const jogadorEscapouNoAlto = (pState.pos.y >= 2.2) || (pState.climbing !== null && pState.pos.y >= 2.0);
    vigilantes.forEach((v) => {
      if (v.alertado) {
        if (jogadorEscapouNoAlto) {
          v.alertado = false;
          document.getElementById('vigilante-badge').textContent = '👮 Vigilância: Despistada (Alto)';
          document.getElementById('vigilante-badge').style.color = '#4ade80';
          mostrarAlerta('🧗 Você escapou do vigilante subindo para o alto!', 'alert-success', 2500, 'vigilante');
        } else {
          const dirV = new THREE.Vector3(pState.pos.x - v.avatar.grp.position.x, 0, pState.pos.z - v.avatar.grp.position.z);
          const distV = dirV.length();
          if (distV > 0.1) {
            dirV.normalize();
            v.avatar.grp.position.addScaledVector(dirV, 4.7 * dt);
            v.avatar.grp.rotation.y = Math.atan2(dirV.x, dirV.z);
            const sw = Math.sin(elapsed * 11) * 0.5;
            v.avatar.pE.rotation.x = sw; v.avatar.pD.rotation.x = -sw;
          }
          if (distV < 1.15) {
            v.alertado = false;
            document.getElementById('vigilante-badge').textContent = '👮 Vigilância: Calma';
            document.getElementById('vigilante-badge').style.color = '#93c5fd';
            aplicarDano(20, 'Detido pelo Vigilante Patrimonial!', true);
            pState.pos.set(24.0, 0.2, 15.2);
          }
        }
      } else {
        const backV = new THREE.Vector3().subVectors(v.home, v.avatar.grp.position);
        if (backV.length() > 0.25) {
          backV.normalize();
          v.avatar.grp.position.addScaledVector(backV, 2.2 * dt);
        }
      }
    });

    const proxE = encontrarInteracaoE();
    const proxR = encontrarArvoreOuPosteProximo();
    const promptEl = document.getElementById('interact-prompt');
    if (proxR && !pState.climbing) {
      promptEl.style.display = 'block';
      promptEl.textContent = `🧗 Pressione [R] para Escalar ${proxR.nome}`;
    } else if (proxE && !pState.climbing) {
      promptEl.style.display = 'block';
      if (proxE.tipo === 'collectible') {
        promptEl.textContent = `🎒 [E] Coletar "${proxE.item.nome}" para o Inventário`;
      } else if (proxE.tipo === 'creto') {
        promptEl.textContent = `🎓 [E] Falar com Prof. Creto e iniciar o Quiz`;
      } else if (proxE.tipo === 'coffee') {
        promptEl.textContent = `☕ [E] Tomar Cafezinho Expresso (Recuperar 100% Life)`;
      } else if (proxE.tipo === 'totem') {
        promptEl.textContent = `🎛️ [E] ou Clique — ${proxE.totem.titulo}`;
      } else {
        promptEl.textContent = `🔍 [E] Inspecionar em RA — ${proxE.obj.nome}`;
      }
    } else {
      promptEl.style.display = 'none';
    }

    if (visorFotoAtivo) verificarAlvoFotografico();

    const targetCam = new THREE.Vector3(
      pState.pos.x + Math.sin(pState.camYaw) * Math.cos(pState.camPitch) * pState.camDist,
      pState.pos.y + 1.5 + Math.sin(pState.camPitch) * pState.camDist,
      pState.pos.z + Math.cos(pState.camYaw) * Math.cos(pState.camPitch) * pState.camDist
    );
    if (shakeTimer > 0) {
      shakeTimer -= dt;
      targetCam.x += (Math.random() - 0.5) * 0.25;
      targetCam.y += (Math.random() - 0.5) * 0.25;
    }
    perspCamera.position.lerp(targetCam, 0.22);
    perspCamera.lookAt(pState.pos.x, pState.pos.y + 1.35, pState.pos.z);

    renderActiveCamera();
  }

  window.addEventListener('resize', () => {
    const asp = wrapper.clientWidth / wrapper.clientHeight;
    perspCamera.aspect = asp;
    perspCamera.updateProjectionMatrix();
    orthoCamera.left = -frustumSize * asp * 0.5;
    orthoCamera.right = frustumSize * asp * 0.5;
    orthoCamera.updateProjectionMatrix();
    renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
  });

  animate();
})();
</script>
</body>
</html>
"""
    return (
        template.replace("__MODO_NOITE__", "true" if modo_noite else "false")
        .replace("__ATIVAR_SOL__", "true" if ativar_sol else "false")
        .replace("__ATIVAR_SPOTS__", "true" if ativar_spots else "false")
    )


# ==============================================================================
# BARRA LATERAL — PARÂMETROS DO MOTOR RAY CASTING & CÂMERA (ATOS I–VIII)
# ==============================================================================
renderer = st.session_state.renderer

with st.sidebar:
    st.header("🎛️ Painel de Controle — Bloco 734")
    preset_key = st.selectbox(
        "Preset de Câmera / Projeção",
        options=list(renderer.presets.keys()),
        format_func=lambda k: renderer.presets[k]["titulo"],
        index=0,
    )
    preset_base = renderer.presets[preset_key]

    col_res, col_turno = st.columns(2)
    with col_res:
        opcoes_res = [250, 350, 500, 600]
        idx_res = opcoes_res.index(st.session_state.resolucao) if st.session_state.resolucao in opcoes_res else 2
        nova_res = st.selectbox("Resolução (px)", options=opcoes_res, index=idx_res)
        if nova_res != st.session_state.resolucao:
            st.session_state.resolucao = nova_res
            st.session_state.renderer = RenderizadorCT(largura=nova_res, altura=nova_res)
            st.session_state.ultima_img_arr = None
            st.session_state.ultimo_pick_info = None
            st.rerun()

    with col_turno:
        turno = st.radio("Turno", ["Dia", "Noite"], index=0)
        modo_noite = turno == "Noite"

    st.subheader("🔦 Luzes & Efeitos Ópticos")
    c1, c2 = st.columns(2)
    with c1:
        ativar_sol = st.checkbox("☀️ Sol Direcional", value=True)
        ativar_sombras = st.checkbox("🌑 Sombras", value=True)
        ativar_reflexos = st.checkbox("🪞 Reflexos", value=True)
    with c2:
        ativar_spots = st.checkbox("🔦 Holofotes/Spots", value=True)
        ativar_caustics = st.checkbox("🌊 Cáusticas", value=True)
        ssaa_2x = st.checkbox("✨ SSAA 2x", value=False)

    with st.expander("📷 Ajuste Fino da Câmera (Eye, At, d, Janela)", expanded=False):
        ex = st.number_input("Eye X", value=float(preset_base["eye"][0]), step=0.5)
        ey = st.number_input("Eye Y", value=float(preset_base["eye"][1]), step=0.5)
        ez = st.number_input("Eye Z", value=float(preset_base["eye"][2]), step=0.5)

        ax = st.number_input("At X", value=float(preset_base["at"][0]), step=0.5)
        ay = st.number_input("At Y", value=float(preset_base["at"][1]), step=0.5)
        az = st.number_input("At Z", value=float(preset_base["at"][2]), step=0.5)

        d_focal = st.slider("Distância Focal (d)", 0.5, 4.0, float(preset_base["d"]), 0.05)
        w_escala = st.slider("Escala da Janela (FOV)", 0.5, 2.0, 1.0, 0.05)

    w0, w1, w2, w3 = preset_base["win"]
    cfg_atual = {
        "titulo": preset_base["titulo"],
        "eye": [ex, ey, ez],
        "at": [ax, ay, az],
        "up": preset_base["up"],
        "d": d_focal,
        "win": (w0 * w_escala, w1 * w_escala, w2 * w_escala, w3 * w_escala),
        "proj": preset_base["proj"],
    }

    btn_renderizar = st.button("🚀 Renderizar Cena (Ray Casting)", type="primary", width="stretch")


# ==============================================================================
# ABAS PRINCIPAIS DA APLICAÇÃO
# ==============================================================================
st.title("🏛️ Bloco 734 (CT/UFC) — Motor Ray Casting & Campus 3D Interativo")

aba_webgl, aba_rc, aba_quests = st.tabs([
    "🎮 Campus CT 3D Interativo (Inventário, Quests, IoT, Vigilância & Quiz Creto)",
    "🖼️ Renderizador Ray Casting (Projeção Oblíqua, Picking & Alta Qualidade)",
    "📜 Painel de Quests (Atos I–VIII) & Matrizes",
])


# ------------------------------------------------------------------------------
# ABA 1: MODO WEBGL / ESTILO ROBLOX COMPLETO & REFINADO
# ------------------------------------------------------------------------------
with aba_webgl:
    components.html(
        gerar_html_jogo_webgl(
            modo_noite=modo_noite,
            ativar_sol=ativar_sol,
            ativar_spots=ativar_spots,
        ),
        height=735,
        scrolling=False,
    )


# ------------------------------------------------------------------------------
# ABA 2: RENDERIZADOR RAY CASTING + PICKING COM BANNER NO TOPO
# ------------------------------------------------------------------------------
with aba_rc:
    st.info(
        "📐 **Dica para Apresentação:** Para demonstrar a **Projeção Oblíqua** ao professor, selecione o preset correspondente em **Preset de Câmera / Projeção** na barra lateral esquerda e clique em **Renderizar Cena (Ray Casting)**."
    )

    if btn_renderizar or st.session_state.ultima_img_arr is None:
        with st.spinner(f"Traçando raios analíticos em {renderer.W}×{renderer.H}px..."):
            img_arr, legenda = renderer.renderizar_config(
                cfg_atual,
                modo_noite=modo_noite,
                ativar_caustics=ativar_caustics,
                ativar_reflexos=ativar_reflexos,
                ativar_sombras=ativar_sombras,
                ativar_sol=ativar_sol,
                ativar_spots=ativar_spots,
                ssaa=ssaa_2x,
            )
            st.session_state.ultima_img_arr = img_arr
            st.session_state.ultima_legenda = legenda
            st.session_state.ultimo_pick_info = None
            st.session_state.ultima_coord_clicada = None

    img_uint8 = (np.clip(st.session_state.ultima_img_arr, 0.0, 1.0) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)

    col_img, col_ctrl = st.columns([2.2, 1.0])

    with col_img:
        st.caption(
            f"📌 **{st.session_state.ultima_legenda}** — Clique sobre qualquer ponto da imagem para inspecionar o objeto."
        )

        if st.session_state.ultimo_pick_info is not None:
            px_c, py_c = st.session_state.ultima_coord_clicada or (None, None)
            renderizar_banner_pick(st.session_state.ultimo_pick_info, px_c, py_c)

        if TEM_IMG_COORDS:
            coords = streamlit_image_coordinates(pil_img, key="rc_pick_canvas")
            if coords is not None:
                px, py = int(coords["x"]), int(coords["y"])
                if st.session_state.ultima_coord_clicada != (px, py):
                    st.session_state.ultima_coord_clicada = (px, py)
                    st.session_state.ultimo_pick_info = renderer.consultar_pick_buffer(px, py)
        else:
            st.image(pil_img, width="stretch")

    with col_ctrl:
        st.subheader("🎯 Inspetor de Pixel (Picking)")
        st.write("Clique na imagem ao lado ou consulte manualmente uma coordenada `(X, Y)`:")
        p_x = st.number_input("Pixel X", 0, renderer.W - 1, renderer.W // 2)
        p_y = st.number_input("Pixel Y", 0, renderer.H - 1, renderer.H // 2)
        if st.button("🔍 Inspecionar Coordenada (X, Y)", width="stretch"):
            st.session_state.ultima_coord_clicada = (int(p_x), int(p_y))
            st.session_state.ultimo_pick_info = renderer.consultar_pick_buffer(int(p_x), int(p_y))

        st.divider()
        buf_png = io.BytesIO()
        pil_img.save(buf_png, format="PNG")
        st.download_button(
            label=f"⬇️ Baixar Imagem PNG ({renderer.W}×{renderer.H})",
            data=buf_png.getvalue(),
            file_name=f"bloco734_raycasting_{renderer.W}x{renderer.H}.png",
            mime="image/png",
            width="stretch",
        )


# ------------------------------------------------------------------------------
# ABA 3: PAINEL DE QUESTS (ATOS I–VIII) & AUDITORIA MATEMÁTICA
# ------------------------------------------------------------------------------
with aba_quests:
    st.subheader("📊 Auditoria Automática — Quests da Apresentação Final (CG I)")
    status_quests = renderer.obter_status_quests(
        cfg_atual=cfg_atual,
        resolucao=renderer.W,
        ativar_sombras=ativar_sombras,
        ativar_sol=ativar_sol,
        ativar_spots=ativar_spots,
    )

    for ato_key, ato_data in status_quests.items():
        with st.expander(ato_data["titulo"], expanded=True):
            for desc_q, ok_q, tag_q in ato_data["items"]:
                c_q1, c_q2 = st.columns([4.5, 1.0])
                with c_q1:
                    st.markdown(f"{'✅' if ok_q else '⚠️'} {desc_q}")
                with c_q2:
                    st.caption(f"**{tag_q}**")

    st.subheader("🧮 Matriz de Transformação Mundo → Câmera ($M_{wc}$)")
    M_wc_atual = construir_Mwc(cfg_atual["eye"], cfg_atual["at"], cfg_atual["up"])
    st.code(np.array2string(M_wc_atual, precision=4, suppress_small=True))
