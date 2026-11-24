import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import ResizeEvent
from matplotlib.widgets import Button, TextBox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
import os

if not hasattr(ResizeEvent, 'inaxes'):
    ResizeEvent.inaxes = None

# ==============================================================================
# 1. ÁLGEBRA LINEAR, QUATÉRNIOS, MATRIZES AFINS (4x4) E BOUNDING BOX (AABB)
# ==============================================================================
def mat_identidade():
    return np.eye(4, dtype=np.float64)

def mat_translacao(tx, ty, tz):
    M = np.eye(4, dtype=np.float64)
    M[0, 3], M[1, 3], M[2, 3] = tx, ty, tz
    return M

def mat_escala(sx, sy, sz):
    M = np.eye(4, dtype=np.float64)
    M[0, 0], M[1, 1], M[2, 2] = sx, sy, sz
    return M

def mat_rotacao_y(ang_rad):
    c, s = np.cos(ang_rad), np.sin(ang_rad)
    return np.array([
        [ c, 0,  s, 0],
        [ 0, 1,  0, 0],
        [-s, 0,  c, 0],
        [ 0, 0,  0, 1]
    ], dtype=np.float64)

def mat_rotacao_z(ang_rad):
    c, s = np.cos(ang_rad), np.sin(ang_rad)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ], dtype=np.float64)

def mat_rotacao_eixo_arbitrario_quaternion(eixo, ang_rad, ponto_pivo=(0, 0, 0)):
    u = np.array(eixo, dtype=np.float64)
    u /= np.linalg.norm(u)
    half = ang_rad / 2.0
    w = np.cos(half)
    x, y, z = u * np.sin(half)

    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y), 0],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x), 0],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y), 0],
        [                0,                 0,                 0, 1]
    ], dtype=np.float64)

    px, py, pz = ponto_pivo
    return mat_translacao(px, py, pz) @ R @ mat_translacao(-px, -py, -pz)

def mat_cisalhamento_xy(sh_xy):
    M = np.eye(4, dtype=np.float64)
    M[0, 1] = sh_xy
    return M

def mat_espelho_plano_arbitrario(ponto_plano, normal_plano):
    p0 = np.array(ponto_plano, dtype=np.float64)
    n = np.array(normal_plano, dtype=np.float64)
    n /= np.linalg.norm(n)

    R_esp = np.eye(4, dtype=np.float64)
    R_esp[:3, :3] = np.eye(3) - 2.0 * np.outer(n, n)
    return mat_translacao(*p0) @ R_esp @ mat_translacao(*(-p0))

def construir_Mwc(eye, at, up):
    eye = np.array(eye, dtype=np.float64)
    at = np.array(at, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    k = eye - at
    k /= np.linalg.norm(k)
    i = np.cross(up, k)
    i /= np.linalg.norm(i)
    j = np.cross(k, i)

    R = np.array([
        [i[0], i[1], i[2], 0.0],
        [j[0], j[1], j[2], 0.0],
        [k[0], k[1], k[2], 0.0],
        [0.0,  0.0,  0.0,  1.0]
    ], dtype=np.float64)
    return R @ mat_translacao(-eye[0], -eye[1], -eye[2])

def testar_aabb(O, D, bmin, bmax, t_limite=np.inf):
    invD = 1.0 / np.where(np.abs(D) > 1e-8, D, 1e-8)
    t1 = (bmin - O) * invD
    t2 = (bmax - O) * invD
    tmin = np.max(np.minimum(t1, t2), axis=-1)
    tmax = np.min(np.maximum(t1, t2), axis=-1)
    return (tmax >= np.maximum(tmin, 1e-4)) & (tmin < t_limite)

# ==============================================================================
# 2. MATERIAIS, TEXTURAS PROCEDURAIS E CÁUSTICA POR REFRAÇÃO (LEI DE SNELL)
# ==============================================================================
class Material:
    def __init__(self, nome, kd, ke, ka, brilho=32.0, refletividade=0.0, textura=None):
        self.nome = nome
        self.kd = np.array(kd, dtype=np.float64)
        self.ke = np.array(ke, dtype=np.float64)
        self.ka = np.array(ka, dtype=np.float64)
        self.brilho = brilho
        self.refletividade = refletividade
        self.textura = textura

def avaliar_textura(tipo_tex, P_world, kd_base):
    if tipo_tex is None:
        return np.broadcast_to(kd_base, P_world.shape).copy()

    x, y, z = P_world[..., 0], P_world[..., 1], P_world[..., 2]

    if tipo_tex == 'asfalto':
        ruido = (np.sin(x * 19.7) * np.cos(z * 23.1) + np.sin((x + z) * 41.3)) * 0.5
        base = 0.14 + 0.04 * ruido
        cor = np.stack([base, base, base * 1.05], axis=-1)
        faixa = (np.abs(z - 11.5) < 0.12) | (np.abs(z - 11.9) < 0.12)
        cor = np.where(faixa[..., None], np.array([0.88, 0.72, 0.12]), cor)
        return np.clip(cor, 0.0, 1.0)

    elif tipo_tex == 'calcada':
        junta_x = np.abs((x % 1.5) - 0.75) > 0.72
        junta_z = np.abs((z % 1.5) - 0.75) > 0.72
        rejunte = junta_x | junta_z
        tom = 0.68 + 0.03 * (((np.floor(x / 1.5) + np.floor(z / 1.5)) % 2))
        cor = np.stack([tom, tom * 0.98, tom * 0.95], axis=-1)
        return np.where(rejunte[..., None], np.array([0.42, 0.42, 0.42]), cor)

    elif tipo_tex == 'placa_734':
        u = np.clip((x - 19.2) / 3.6, 0.0, 0.999)
        v = np.clip((y - 5.15) / 1.15, 0.0, 0.999)
        grid = np.array([
            [1,1,1,1, 0, 1,1,1,1, 0, 1,0,0,1],
            [0,0,0,1, 0, 0,0,0,1, 0, 1,0,0,1],
            [0,0,1,0, 0, 0,1,1,1, 0, 1,1,1,1],
            [0,1,0,0, 0, 0,0,0,1, 0, 0,0,0,1],
            [0,1,0,0, 0, 1,1,1,1, 0, 0,0,0,1],
        ], dtype=bool)
        gh, gw = grid.shape
        iy = np.clip(((1.0 - v) * gh).astype(int), 0, gh - 1)
        ix = np.clip((u * gw).astype(int), 0, gw - 1)
        letra = grid[iy, ix]
        cor_fundo = np.array([0.94, 0.95, 0.96])
        cor_num = np.array([0.08, 0.16, 0.38])
        return np.where(letra[..., None], cor_num, cor_fundo)

    elif tipo_tex == 'agua_fonte':
        # Ondulações suaves na superfície da lâmina d'água do espelho d'água
        w1 = np.sin(x * 3.8 + z * 2.7)
        w2 = np.cos(x * 5.2 - z * 4.4)
        w3 = np.sin((x - 27.0)**2 * 0.45 + (z - 19.8)**2 * 0.45)
        onda = 0.5 + 0.22 * w1 + 0.18 * w2 + 0.10 * w3
        cor_fundo = np.array([0.06, 0.28, 0.36])
        cor_crista = np.array([0.32, 0.72, 0.85])
        t = np.clip(onda, 0.0, 1.0)[..., None]
        return (1.0 - t) * cor_fundo + t * cor_crista

    return np.broadcast_to(kd_base, P_world.shape).copy()


def calcular_caustica_fonte(P_world, N_world, fonte_info, modo_noite):
    """
    Calcula a projeção de cáusticas geradas pelos spots submersos na parede interna
    da fonte atravessando a lâmina d'água ondulada (Lei de Snell + convergência de frente de onda).
    """
    x_min, x_max = fonte_info['x_min'], fonte_info['x_max']
    z_min, z_max = fonte_info['z_min'], fonte_info['z_max']
    y_agua = fonte_info['y_agua']
    spots = fonte_info['spots_submersos']

    intensidade_global = 1.45 if modo_noite else 0.25
    cor_caustica_base = np.array([0.62, 0.92, 1.00]) if modo_noite else np.array([0.75, 0.95, 1.00])

    acum_caustica = np.zeros_like(P_world)
    px, py, pz = P_world[..., 0], P_world[..., 1], P_world[..., 2]

    for spot_pos, spot_dir in spots:
        sx, sy, sz = spot_pos
        # Vetor do spot submerso até o ponto P_world na fachada/cenário
        vx = px - sx
        vy = py - sy
        vz = pz - sz
        dist = np.sqrt(vx**2 + vy**2 + vz**2) + 1e-8
        dx, dy, dz = vx / dist, vy / dist, vz / dist

        # Só projeta para frente/cima em direção ao prédio e escultura
        cos_cone = dx * spot_dir[0] + dy * spot_dir[1] + dz * spot_dir[2]
        dentro_cone = (cos_cone > 0.35) & (dy > 0.02)

        # Interseção do raio refratado com a superfície da água (y = y_agua)
        t_agua = (y_agua - sy) / np.where(np.abs(dy) > 1e-6, dy, 1e-6)
        wx = sx + t_agua * dx
        wz = sz + t_agua * dz

        na_lamina = (
            dentro_cone &
            (t_agua > 0.05) &
            (t_agua < dist) &
            (wx >= x_min) & (wx <= x_max) &
            (wz >= z_min) & (wz <= z_max)
        )

        # Campo de ondas multi-frequência na superfície da água h(wx, wz)
        k1x, k1z = 4.6, 3.9
        k2x, k2z = -6.3, 5.1
        k3x, k3z = 8.7, 7.4

        fase1 = k1x * wx + k1z * wz
        fase2 = k2x * wx + k2z * wz + 1.1
        fase3 = k3x * wx + k3z * wz + 2.4

        # Gradiente da onda (inclinação da normal da água -> refração de Snell)
        dh_dx = 0.045 * k1x * np.cos(fase1) + 0.032 * k2x * np.cos(fase2) + 0.020 * k3x * np.cos(fase3)
        dh_dz = 0.045 * k1z * np.cos(fase1) + 0.032 * k2z * np.cos(fase2) + 0.020 * k3z * np.cos(fase3)

        # Laplaciano da superfície (d²h/dx² + d²h/dz²) mede convergência de raios refratados (cáustica!)
        laplaciano = -(
            0.045 * (k1x**2 + k1z**2) * np.sin(fase1) +
            0.032 * (k2x**2 + k2z**2) * np.sin(fase2) +
            0.020 * (k3x**2 + k3z**2) * np.sin(fase3)
        )

        # Filamentos luminosos nítidos onde os raios refratados convergem (cristas côncavas)
        convergencia = np.maximum(0.0, laplaciano * 0.28 + 0.35 * np.sqrt(dh_dx**2 + dh_dz**2))
        filamento = np.power(np.clip(convergencia, 0.0, 2.2), 2.3)

        # Atenuação angular do spot e pela distância até a fachada
        atenuacao_cone = np.power(np.maximum(0.0, (cos_cone - 0.35) / 0.65), 1.6)
        atenuacao_dist = 1.0 / (1.0 + 0.012 * (dist**2))
        ndl_fachada = np.maximum(0.0, -(N_world[..., 0]*dx + N_world[..., 1]*dy + N_world[..., 2]*dz))

        contrib = na_lamina * filamento * atenuacao_cone * atenuacao_dist * (0.35 + 0.65 * ndl_fachada)
        acum_caustica += contrib[..., None] * cor_caustica_base * intensidade_global

    # Também ilumina o fundo interno da própria fonte com cáustica subaquática
    no_fundo_fonte = (
        (px >= x_min) & (px <= x_max) &
        (pz >= z_min) & (pz <= z_max) &
        (np.abs(py - y_agua) < 0.08)
    )
    if np.any(no_fundo_fonte):
        sub_wave = np.sin(px * 6.5 + pz * 5.2) * np.cos(px * 5.8 - pz * 6.7)
        c_sub = np.power(np.clip(0.55 + 0.65 * sub_wave, 0.0, 1.4), 2.6)
        acum_caustica += no_fundo_fonte[..., None] * c_sub[..., None] * cor_caustica_base * (1.1 if modo_noite else 0.35)

    return acum_caustica

# ==============================================================================
# 3. PRIMITIVAS GEOMÉTRICAS COM BOUNDING BOX (AABB)
# ==============================================================================
class Primitiva:
    def __init__(self, nome, tipo, matriz_M, material, dados_extra=None):
        self.nome = nome
        self.tipo = tipo
        self.M = matriz_M
        self.mat = material
        self.dados = dados_extra or {}
        self.M_total = self.M.copy()
        self.M_inv = None
        self.M_inv_T = None
        self.aabb_cam = None

    def _calcular_aabb_camera(self, M_wc):
        if self.tipo == 'bloco_ct_arqueado':
            d = self.dados
            pts_w = np.array([
                [d['x_min'], 0.0, d['z_min']], [d['x_max'], 0.0, d['z_min']],
                [d['x_max'], 0.0, d['z_max']], [d['x_min'], 0.0, d['z_max']],
                [d['x_min'], d['altura'], d['z_min']], [d['x_max'], d['altura'], d['z_min']],
                [d['x_max'], d['altura'], d['z_max']], [d['x_min'], d['altura'], d['z_max']],
            ])
        elif self.tipo == 'malha':
            pts_w = self.dados['vertices_world']
        else:
            ymin = -0.5 if self.tipo == 'cubo' else (-1.0 if self.tipo == 'esfera' else 0.0)
            ymax =  0.5 if self.tipo == 'cubo' else 1.0
            xmin = -0.5 if self.tipo == 'cubo' else -1.0
            xmax =  0.5 if self.tipo == 'cubo' else 1.0
            pts_loc = np.array([
                [xmin, ymin, xmin], [xmax, ymin, xmin], [xmax, ymin, xmax], [xmin, ymin, xmax],
                [xmin, ymax, xmin], [xmax, ymax, xmin], [xmax, ymax, xmax], [xmin, ymax, xmax],
            ])
            pts_w = (self.M @ np.hstack([pts_loc, np.ones((8, 1))]).T).T[:, :3]

        pts_c = (M_wc @ np.hstack([pts_w, np.ones((len(pts_w), 1))]).T).T[:, :3]
        margem = 0.08
        self.aabb_cam = (pts_c.min(axis=0) - margem, pts_c.max(axis=0) + margem)

    def preparar_camera(self, M_wc):
        self.M_total = M_wc @ self.M
        self.M_inv = np.linalg.inv(self.M_total)
        self.M_inv_T = self.M_inv.T

        if self.tipo == 'malha':
            verts_w = self.dados['vertices_world']
            verts_h = np.hstack([verts_w, np.ones((len(verts_w), 1))])
            self.verts_cam = (M_wc @ verts_h.T).T[:, :3]

        self._calcular_aabb_camera(M_wc)

def criar_malha_cobertura_metalica(M_inst):
    v_loc = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0], [1.0, 0.6, 0.0], [1.0, 0.6, 1.0], [0.0, 1.0, 1.0],
    ], dtype=np.float64)
    v_h = np.hstack([v_loc, np.ones((8, 1))])
    v_world = (M_inst @ v_h.T).T[:, :3]

    faces = np.array([
        [4, 5, 6], [4, 6, 7],
        [0, 2, 1], [0, 3, 2],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5],
        [3, 0, 4], [3, 4, 7],
    ], dtype=int)
    return {'vertices_world': v_world, 'faces': faces}
def carregar_malha_obj(caminho_obj, M_inst):
    vertices = []
    faces = []
    with open(caminho_obj, 'r', encoding='utf-8', errors='ignore') as arq:
        for linha in arq:
            partes = linha.strip().split()
            if not partes:
                continue
            if partes[0] == 'v':
                vertices.append([float(partes[1]), float(partes[2]), float(partes[3])])
            elif partes[0] == 'f':
                # Converte polígonos de 3 ou 4 lados em triângulos (índices base 0)
                idxs = [int(p.split('/')[0]) - 1 for p in partes[1:]]
                for k in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[k], idxs[k + 1]])

    v_loc = np.array(vertices, dtype=np.float64)
    # Normaliza para altura = 1.0m e base apoiada em Y = 0
    v_min, v_max = v_loc.min(axis=0), v_loc.max(axis=0)
    centro_xz = 0.5 * (v_min + v_max)
    escala_norm = 1.0 / max(v_max[1] - v_min[1], 1e-6)
    v_loc[:, 0] = (v_loc[:, 0] - centro_xz[0]) * escala_norm
    v_loc[:, 1] = (v_loc[:, 1] - v_min[1]) * escala_norm
    v_loc[:, 2] = (v_loc[:, 2] - centro_xz[2]) * escala_norm

    v_h = np.hstack([v_loc, np.ones((len(v_loc), 1))])
    v_world = (M_inst @ v_h.T).T[:, :3]
    return {'vertices_world': v_world, 'faces': np.array(faces, dtype=int)}
# ==============================================================================
# 3B. MODELO PARAMÉTRICO: ESCULTURA CINÉTICA EM AÇO (5 LOSANGOS ROTACIONADOS A 45°)
# ==============================================================================
def gerar_parametros_escultura_cinetica(
    origem_base=(15.0, 0.2, 22.0),
    qtd_quadros=5,
    lado_base=2.35,
    espessura_perfil=0.14,
    profundidade_perfil=0.16,
    espacamento_z=0.58,
    deslocamento_x_passo=0.22,
    variacao_escala_passo=0.055,
    rotacao_conjunto_y_deg=-24.0
):
    ox, oy, oz = origem_base
    M_conjunto = mat_translacao(ox, oy, oz) @ mat_rotacao_y(np.deg2rad(rotacao_conjunto_y_deg))

    base_larg = 3.20
    base_alt  = 0.28
    base_prof = 3.40

    quadros = []
    z_inicio = -((qtd_quadros - 1) * espacamento_z) / 2.0

    for k in range(qtd_quadros):
        fator_centro = k - (qtd_quadros - 1) / 2.0
        L_k = lado_base * (1.0 - variacao_escala_passo * fator_centro)
        r_k = L_k / np.sqrt(2.0)

        thetas = np.deg2rad([0.0, 90.0, 180.0, 270.0])
        vx = r_k * np.cos(thetas)
        vy = r_k * np.sin(thetas)

        cx_k = fator_centro * deslocamento_x_passo
        cy_k = base_alt + r_k + espessura_perfil * 0.35
        cz_k = z_inicio + k * espacamento_z

        verts_loc = np.column_stack([cx_k + vx, cy_k + vy, np.full(4, cz_k)])
        verts_h = np.hstack([verts_loc, np.ones((4, 1))])
        verts_world = (M_conjunto @ verts_h.T).T[:, :3]

        barras_M = []
        for i in range(4):
            p1 = verts_loc[i]
            p2 = verts_loc[(i + 1) % 4]
            vetor_aresta = p2 - p1
            comp = np.linalg.norm(vetor_aresta)
            ang_aresta = np.arctan2(vetor_aresta[1], vetor_aresta[0])
            pm = 0.5 * (p1 + p2)

            M_barra_loc = (
                mat_translacao(pm[0], pm[1], pm[2]) @
                mat_rotacao_z(ang_aresta) @
                mat_escala(comp + espessura_perfil * 0.72, espessura_perfil, profundidade_perfil)
            )
            barras_M.append(M_conjunto @ M_barra_loc)

        p_inf = verts_loc[3]
        M_sapata_loc = (
            mat_translacao(p_inf[0], base_alt + 0.04, p_inf[2]) @
            mat_escala(espessura_perfil * 1.4, 0.12, profundidade_perfil * 1.3)
        )

        quadros.append({
            'indice': k + 1,
            'lado': L_k,
            'vertices_world': verts_world,
            'barras_M': barras_M,
            'haste_M': M_conjunto @ M_sapata_loc,
            'base_haste_world': (M_conjunto @ np.array([p_inf[0], base_alt, p_inf[2], 1.0]))[:3],
            'vertice_inf_world': verts_world[3]
        })

    M_base = M_conjunto @ mat_translacao(0.0, base_alt * 0.5, 0.0) @ mat_escala(base_larg, base_alt, base_prof)

    bx, by, bz = base_larg * 0.5, base_alt, base_prof * 0.5
    base_pts_loc = np.array([
        [-bx, 0.0, -bz], [ bx, 0.0, -bz], [ bx, 0.0,  bz], [-bx, 0.0,  bz],
        [-bx,  by, -bz], [ bx,  by, -bz], [ bx,  by,  bz], [-bx,  by,  bz]
    ])
    base_pts_world = (M_conjunto @ np.hstack([base_pts_loc, np.ones((8, 1))]).T).T[:, :3]

    return {
        'M_base': M_base,
        'base_pts_world': base_pts_world,
        'quadros': quadros,
        'espessura_perfil': espessura_perfil
    }

# ==============================================================================
# 4. MODELAGEM DO BLOCO CT ASFALTO (33m x 44m COM ARCO TANGENCIAL)
# ==============================================================================
def construir_geometria_bloco_ct():
    return {
        'cx': 37.0, 'cz': 42.0, 'raio': 22.0,
        'x_min': 15.0, 'x_max': 59.0, 'z_min': 20.0, 'z_max': 53.0,
        'altura': 6.6, 'altura_platibanda': 5.0
    }

def intersect_bloco_ct(O_cam, D_cam, prim):
    M_inv = prim.M_inv
    O_w = np.einsum('ij,...j->...i', M_inv[:3, :3], O_cam) + M_inv[:3, 3]
    D_w = np.einsum('ij,...j->...i', M_inv[:3, :3], D_cam)

    d = prim.dados
    cx, cz, R = d['cx'], d['cz'], d['raio']
    x_min, x_max = d['x_min'], d['x_max']
    z_min, z_max = d['z_min'], d['z_max']
    H = d['altura']

    shape = D_cam.shape[:-1]
    t_best = np.full(shape, np.inf)
    N_best = np.zeros_like(D_w)

    ox, oy, oz = O_w[..., 0], O_w[..., 1], O_w[..., 2]
    dx, dy, dz = D_w[..., 0], D_w[..., 1], D_w[..., 2]

    def dentro_planta(px, pz):
        in_box = (px >= x_min - 1e-4) & (px <= x_max + 1e-4) & (pz >= z_min - 1e-4) & (pz <= z_max + 1e-4)
        in_quad = (px < cx) & (pz < cz)
        in_arc = ((px - cx)**2 + (pz - cz)**2) <= (R**2 + 1e-3)
        return in_box & (~in_quad | in_arc)

    m_top = np.abs(dy) > 1e-7
    t_top = (H - oy) / np.where(m_top, dy, 1.0)
    px, pz = ox + t_top * dx, oz + t_top * dz
    v_top = m_top & (t_top > 1e-4) & dentro_planta(px, pz) & (t_top < t_best)
    t_best = np.where(v_top, t_top, t_best)
    N_top = np.zeros_like(N_best); N_top[..., 1] = 1.0
    N_best = np.where(v_top[..., None], N_top, N_best)

    planos = [
        ('z', z_min, -1.0, lambda x, z: (x >= cx - 1e-4) & (x <= x_max + 1e-4)),
        ('x', x_min, -1.0, lambda x, z: (z >= cz - 1e-4) & (z <= z_max + 1e-4)),
        ('x', x_max,  1.0, lambda x, z: (z >= z_min - 1e-4) & (z <= z_max + 1e-4)),
        ('z', z_max,  1.0, lambda x, z: (x >= x_min - 1e-4) & (x <= x_max + 1e-4)),
    ]
    for eixo_p, val_p, s_norm, cond_fn in planos:
        if eixo_p == 'z':
            mp = np.abs(dz) > 1e-7
            tp = (val_p - oz) / np.where(mp, dz, 1.0)
        else:
            mp = np.abs(dx) > 1e-7
            tp = (val_p - ox) / np.where(mp, dx, 1.0)
        hx, hy, hz = ox + tp * dx, oy + tp * dy, oz + tp * dz
        vp = mp & (tp > 1e-4) & (hy >= 0.0) & (hy <= H) & cond_fn(hx, hz) & (tp < t_best)
        t_best = np.where(vp, tp, t_best)
        Np = np.zeros_like(N_best)
        if eixo_p == 'z': Np[..., 2] = s_norm
        else: Np[..., 0] = s_norm
        N_best = np.where(vp[..., None], Np, N_best)

    rx0, rz0 = ox - cx, oz - cz
    a = dx**2 + dz**2
    b = 2.0 * (rx0 * dx + rz0 * dz)
    c = rx0**2 + rz0**2 - R**2
    delta = b**2 - 4.0 * a * c
    mc = (delta >= 0) & (a > 1e-8)
    sq = np.sqrt(np.maximum(delta, 0.0))

    for tc in [(-b - sq) / (2.0 * np.where(mc, a, 1.0)),
               (-b + sq) / (2.0 * np.where(mc, a, 1.0))]:
        hx, hy, hz = ox + tc * dx, oy + tc * dy, oz + tc * dz
        vc = (mc & (tc > 1e-4) & (hy >= 0.0) & (hy <= H) &
              (hx <= cx + 1e-3) & (hz <= cz + 1e-3) & (tc < t_best))
        if np.any(vc):
            t_best = np.where(vc, tc, t_best)
            Nc = np.stack([(hx - cx) / R, np.zeros_like(tc), (hz - cz) / R], axis=-1)
            N_best = np.where(vc[..., None], Nc, N_best)

    P_hit = O_w + np.where(np.isfinite(t_best)[..., None], t_best[..., None], 0.0) * D_w
    px, py, pz = P_hit[..., 0], P_hit[..., 1], P_hit[..., 2]

    ang = np.arctan2(cz - pz, cx - px)
    s_arco = np.clip(ang / (np.pi / 2.0), 0.0, 1.0)
    na_fachada_arco = (px <= cx + 0.05) & (pz <= cz + 0.05) & (py < H - 0.05)

    na_porta = na_fachada_arco & (s_arco >= 0.06) & (s_arco <= 0.20) & (py <= 2.75)
    borda_porta = na_porta & ((s_arco < 0.07) | (s_arco > 0.19) | (py > 2.63))

    s_rel = (s_arco - 0.24) / (0.96 - 0.24)
    idx_jan = np.floor(s_rel * 9.0)
    frac_jan = (s_rel * 9.0) - idx_jan
    na_janela = (na_fachada_arco & (idx_jan >= 0) & (idx_jan < 9) &
                 (frac_jan >= 0.16) & (frac_jan <= 0.84) &
                 (py >= 1.35) & (py <= 3.25))
    borda_janela = na_janela & ((frac_jan < 0.22) | (frac_jan > 0.78) | (py < 1.45) | (py > 3.15))

    na_placa_734 = na_fachada_arco & (s_arco >= 0.05) & (s_arco <= 0.23) & (py >= 5.15) & (py <= 6.30)

    sub_mat = np.where(na_porta | na_janela, 1, 0)
    sub_mat = np.where(borda_porta | borda_janela, 3, sub_mat)
    sub_mat = np.where(na_placa_734, 2, sub_mat)

    N_cam = np.einsum('ij,...j->...i', prim.M_inv_T[:3, :3], N_best)
    N_cam /= np.where(np.linalg.norm(N_cam, axis=-1, keepdims=True) > 1e-12,
                      np.linalg.norm(N_cam, axis=-1, keepdims=True), 1.0)

    return t_best, N_cam, P_hit, sub_mat, N_best

# ==============================================================================
# 5. INTERSEÇÃO GERAL DAS PRIMITIVAS COM FILTRO BOUNDING BOX
# ==============================================================================
def intersect_primitiva(O_cam, D_cam, prim, t_atual_min=np.inf):
    shape = D_cam.shape[:-1]

    if prim.aabb_cam is not None:
        if not np.any(testar_aabb(O_cam, D_cam, prim.aabb_cam[0], prim.aabb_cam[1], t_atual_min)):
            return (np.full(shape, np.inf), np.zeros_like(D_cam),
                    np.zeros_like(D_cam), np.zeros(shape, dtype=int), np.zeros_like(D_cam))

    if prim.tipo == 'bloco_ct_arqueado':
        return intersect_bloco_ct(O_cam, D_cam, prim)

    if prim.tipo == 'malha':
        verts = prim.verts_cam
        faces = prim.dados['faces']
        t_best = np.full(shape, np.inf)
        N_best = np.zeros_like(D_cam)

        for f in faces:
            v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
            e1 = v1 - v0
            e2 = v2 - v0
            h = np.cross(D_cam, e2)
            a = np.einsum('...i,i->...', h, e1)
            mask_a = np.abs(a) > 1e-7
            inv_a = 1.0 / np.where(mask_a, a, 1.0)
            s = O_cam - v0
            u = inv_a * np.einsum('...i,...i->...', s, h)
            q = np.cross(s, e1)
            v = inv_a * np.einsum('...i,...i->...', D_cam, q)
            t = inv_a * np.einsum('...i,i->...', q, e2)

            hit_f = mask_a & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-4) & (t < t_best)
            if np.any(hit_f):
                t_best = np.where(hit_f, t, t_best)
                nf = np.cross(e1, e2)
                nf /= np.linalg.norm(nf)
                N_best = np.where(hit_f[..., None], nf, N_best)

        M_inv = prim.M_inv
        P_w = np.einsum('ij,...j->...i', M_inv[:3, :3], O_cam + np.where(np.isfinite(t_best)[..., None], t_best[..., None], 0.0) * D_cam) + M_inv[:3, 3]
        return t_best, N_best, P_w, np.zeros(shape, dtype=int), N_best

    M_inv = prim.M_inv
    O_loc = np.einsum('ij,...j->...i', M_inv[:3, :3], O_cam) + M_inv[:3, 3]
    D_loc = np.einsum('ij,...j->...i', M_inv[:3, :3], D_cam)

    t_hit = np.full(shape, np.inf)
    N_loc = np.zeros_like(D_loc)

    if prim.tipo == 'cubo':
        invD = 1.0 / np.where(np.abs(D_loc) > 1e-8, D_loc, 1e-8)
        t1 = (-0.5 - O_loc) * invD
        t2 = ( 0.5 - O_loc) * invD
        tmin = np.max(np.minimum(t1, t2), axis=-1)
        tmax = np.min(np.maximum(t1, t2), axis=-1)
        valid = (tmax >= tmin) & (tmax > 1e-4)
        t_hit = np.where(valid, np.where(tmin > 1e-4, tmin, tmax), np.inf)
        P_loc = O_loc + t_hit[..., None] * D_loc
        max_ax = np.argmax(np.abs(P_loc), axis=-1)
        for ax in range(3):
            N_loc[..., ax] = np.where(max_ax == ax, np.sign(P_loc[..., ax]), 0.0)

    elif prim.tipo == 'esfera':
        a = np.einsum('...i,...i->...', D_loc, D_loc)
        b = 2.0 * np.einsum('...i,...i->...', D_loc, O_loc)
        c = np.einsum('...i,...i->...', O_loc, O_loc) - 1.0
        delta = b**2 - 4.0 * a * c
        mask = delta >= 0
        sq = np.sqrt(np.maximum(delta, 0.0))
        t1 = (-b - sq) / (2.0 * a)
        t2 = (-b + sq) / (2.0 * a)
        t_val = np.where(t1 > 1e-4, t1, np.where(t2 > 1e-4, t2, np.inf))
        t_hit = np.where(mask, t_val, np.inf)
        N_loc = O_loc + t_hit[..., None] * D_loc

    elif prim.tipo == 'cilindro':
        ox, oy, oz = O_loc[..., 0], O_loc[..., 1], O_loc[..., 2]
        dx, dy, dz = D_loc[..., 0], D_loc[..., 1], D_loc[..., 2]
        a = dx**2 + dz**2
        b = 2.0 * (ox * dx + oz * dz)
        c = ox**2 + oz**2 - 1.0
        delta = b**2 - 4.0 * a * c
        mask = (delta >= 0) & (a > 1e-8)
        sq = np.sqrt(np.maximum(delta, 0.0))
        for tc in [(-b - sq) / (2.0 * np.where(mask, a, 1.0)),
                   (-b + sq) / (2.0 * np.where(mask, a, 1.0))]:
            yh = oy + tc * dy
            v = mask & (tc > 1e-4) & (yh >= 0.0) & (yh <= 1.0) & (tc < t_hit)
            t_hit = np.where(v, tc, t_hit)
        P_loc = O_loc + t_hit[..., None] * D_loc
        N_loc = np.stack([P_loc[..., 0], np.zeros_like(t_hit), P_loc[..., 2]], axis=-1)
        for yt, ny in [(0.0, -1.0), (1.0, 1.0)]:
            mt = np.abs(dy) > 1e-6
            tt = (yt - oy) / np.where(mt, dy, 1.0)
            xh, zh = ox + tt * dx, oz + tt * dz
            vt = mt & (tt > 1e-4) & ((xh**2 + zh**2) <= 1.0) & (tt < t_hit)
            t_hit = np.where(vt, tt, t_hit)
            Nt = np.zeros_like(N_loc); Nt[..., 1] = ny
            N_loc = np.where(vt[..., None], Nt, N_loc)

    elif prim.tipo == 'cone':
        ox, oy, oz = O_loc[..., 0], O_loc[..., 1] - 1.0, O_loc[..., 2]
        dx, dy, dz = D_loc[..., 0], D_loc[..., 1], D_loc[..., 2]
        a = dx**2 + dz**2 - dy**2
        b = 2.0 * (ox * dx + oz * dz - oy * dy)
        c = ox**2 + oz**2 - oy**2
        delta = b**2 - 4.0 * a * c
        mask = (delta >= 0) & (np.abs(a) > 1e-8)
        sq = np.sqrt(np.maximum(delta, 0.0))
        for tc in [(-b - sq) / (2.0 * np.where(mask, a, 1.0)),
                   (-b + sq) / (2.0 * np.where(mask, a, 1.0))]:
            yh = O_loc[..., 1] + tc * dy
            v = mask & (tc > 1e-4) & (yh >= 0.0) & (yh <= 1.0) & (tc < t_hit)
            t_hit = np.where(v, tc, t_hit)
        P_loc = O_loc + t_hit[..., None] * D_loc
        N_loc = np.stack([P_loc[..., 0], 1.0 - P_loc[..., 1], P_loc[..., 2]], axis=-1)
        mb = np.abs(dy) > 1e-6
        tb = -O_loc[..., 1] / np.where(mb, dy, 1.0)
        xb, zb = ox + tb * dx, oz + tb * dz
        vb = mb & (tb > 1e-4) & ((xb**2 + zb**2) <= 1.0) & (tb < t_hit)
        t_hit = np.where(vb, tb, t_hit)
        Nb = np.zeros_like(N_loc); Nb[..., 1] = -1.0
        N_loc = np.where(vb[..., None], Nb, N_loc)

    P_loc_hit = O_loc + np.where(np.isfinite(t_hit)[..., None], t_hit[..., None], 0.0) * D_loc
    P_world = np.einsum('ij,...j->...i', prim.M[:3, :3], P_loc_hit) + prim.M[:3, 3]

    M_w_inv_T = np.linalg.inv(prim.M).T
    N_world = np.einsum('ij,...j->...i', M_w_inv_T[:3, :3], N_loc)
    nw_len = np.linalg.norm(N_world, axis=-1, keepdims=True)
    N_world /= np.where(nw_len > 1e-12, nw_len, 1.0)

    N_cam = np.einsum('ij,...j->...i', prim.M_inv_T[:3, :3], N_loc)
    nlen = np.linalg.norm(N_cam, axis=-1, keepdims=True)
    N_cam /= np.where(nlen > 1e-12, nlen, 1.0)

    return t_hit, N_cam, P_world, np.zeros(shape, dtype=int), N_world

# ==============================================================================
# 6. MONTAGEM DO CENÁRIO: BLOCO 734 + ESCULTURA + ESPELHO D'ÁGUA COM SPOTS
# ==============================================================================
def montar_cenario():
    mat_alvenaria_branca = Material("Alvenaria Branca (Bloco 734)", (0.92, 0.93, 0.94), (0.25, 0.25, 0.25), (0.85, 0.85, 0.88), brilho=16.0)
    mat_aco_branco       = Material("Aço Pintura Branca (Escultura Cinética)", (0.96, 0.97, 0.98), (0.65, 0.68, 0.72), (0.90, 0.91, 0.93), brilho=64.0, refletividade=0.14)
    mat_concreto_base    = Material("Concreto Aparente (Base Escultura)", (0.52, 0.53, 0.55), (0.12, 0.12, 0.12), (0.45, 0.46, 0.48), brilho=12.0)
    mat_vidro_fume       = Material("Vidro Fumê Refletivo", (0.04, 0.05, 0.06), (0.95, 0.95, 0.98), (0.03, 0.03, 0.04), brilho=128.0, refletividade=0.55)
    mat_esquadria        = Material("Esquadria Alumínio Escuro", (0.12, 0.13, 0.15), (0.50, 0.50, 0.55), (0.10, 0.10, 0.12), brilho=48.0)
    mat_placa_734        = Material("Platibanda Nº 734", (0.92, 0.93, 0.94), (0.25, 0.25, 0.25), (0.85, 0.85, 0.88), brilho=16.0, textura='placa_734')
    mat_asfalto          = Material("Pavimento Asfáltico (CBUQ)", (0.22, 0.22, 0.24), (0.08, 0.08, 0.08), (0.18, 0.18, 0.20), brilho=8.0, textura='asfalto')
    mat_calcada          = Material("Calçada de Concreto", (0.68, 0.67, 0.65), (0.10, 0.10, 0.10), (0.60, 0.60, 0.58), brilho=10.0, textura='calcada')
    mat_tronco           = Material("Casca de Árvore", (0.36, 0.22, 0.11), (0.05, 0.05, 0.05), (0.30, 0.18, 0.09), brilho=8.0)
    mat_folha_1          = Material("Folhagem Verde Escuro", (0.08, 0.34, 0.12), (0.12, 0.18, 0.12), (0.07, 0.28, 0.10), brilho=14.0)
    mat_folha_2          = Material("Folhagem Verde Médio", (0.14, 0.48, 0.18), (0.15, 0.22, 0.15), (0.12, 0.40, 0.15), brilho=16.0)
    mat_folha_3          = Material("Folhagem Verde Claro", (0.22, 0.60, 0.24), (0.18, 0.25, 0.18), (0.18, 0.50, 0.20), brilho=18.0)
    mat_cone_laranja     = Material("Cone Sinalização Viária", (0.95, 0.35, 0.04), (0.45, 0.40, 0.35), (0.85, 0.30, 0.04), brilho=36.0)
    mat_metal_galv       = Material("Aço Galvanizado (Cobertura)", (0.58, 0.62, 0.66), (0.85, 0.88, 0.90), (0.50, 0.54, 0.58), brilho=80.0, refletividade=0.25)
    mat_agua_fonte       = Material("Água Espelho d'Água (Refração n=1.333)", (0.12, 0.48, 0.62), (0.95, 0.98, 1.00), (0.08, 0.32, 0.42), brilho=140.0, refletividade=0.42, textura='agua_fonte')
    mat_borda_fonte      = Material("Borda Granito Escuro (Fonte)", (0.22, 0.24, 0.27), (0.30, 0.32, 0.35), (0.18, 0.20, 0.22), brilho=36.0)
    mat_spot_submerso    = Material("Spot Submerso LED (Fonte)", (0.75, 0.96, 1.00), (1.00, 1.00, 1.00), (0.70, 0.95, 1.00), brilho=160.0)

    materiais_bloco = {
        0: mat_alvenaria_branca,
        1: mat_vidro_fume,
        2: mat_placa_734,
        3: mat_esquadria
    }

    cena = []

    M_via = mat_translacao(35.0, 0.05, 9.0) @ mat_escala(66.0, 0.10, 14.0)
    cena.append(Primitiva("Via Asfáltica CT", "cubo", M_via, mat_asfalto))

    M_calc = mat_translacao(35.0, 0.10, 38.0) @ mat_escala(66.0, 0.20, 44.0)
    cena.append(Primitiva("Calçada Bloco 734", "cubo", M_calc, mat_calcada))

    dados_bloco = construir_geometria_bloco_ct()
    cena.append(Primitiva("Bloco 734 (CT/Asfalto)", "bloco_ct_arqueado", mat_identidade(),
                          mat_alvenaria_branca, dados_extra=dados_bloco))

    # --- ESPELHO D'ÁGUA / FONTE FRONTAL COM SPOTS SUBMERSOS PARA CÁUSTICA ---
    fx_min, fx_max = 19.5, 34.5
    fz_min, fz_max = 16.8, 22.8
    f_cx, f_cz = 0.5 * (fx_min + fx_max), 0.5 * (fz_min + fz_max)
    f_larg, f_prof = fx_max - fx_min, fz_max - fz_min
    y_agua = 0.42

    # 4 muretas baixas da borda da fonte (altura 0.50m sobre calçada y=0.20m)
    esp_borda = 0.28
    alt_borda = 0.34
    y_centro_borda = 0.20 + alt_borda * 0.5
    bordas_fonte = [
        ("Borda Frontal Fonte",  mat_translacao(f_cx, y_centro_borda, fz_min) @ mat_escala(f_larg + esp_borda, alt_borda, esp_borda)),
        ("Borda Traseira Fonte", mat_translacao(f_cx, y_centro_borda, fz_max) @ mat_escala(f_larg + esp_borda, alt_borda, esp_borda)),
        ("Borda Esq. Fonte",     mat_translacao(fx_min, y_centro_borda, f_cz) @ mat_escala(esp_borda, alt_borda, f_prof)),
        ("Borda Dir. Fonte",     mat_translacao(fx_max, y_centro_borda, f_cz) @ mat_escala(esp_borda, alt_borda, f_prof)),
    ]
    for nome_b, M_b in bordas_fonte:
        cena.append(Primitiva(nome_b, "cubo", M_b, mat_borda_fonte))

    # Lâmina d'água ondulada do espelho d'água
    M_agua = mat_translacao(f_cx, y_agua - 0.04, f_cz) @ mat_escala(f_larg - 0.15, 0.08, f_prof - 0.15)
    cena.append(Primitiva("Espelho d'Água (Fonte Cáustica)", "cubo", M_agua, mat_agua_fonte))

    # 3 spots submersos fixados na parede interna frontal da fonte apontados para o Bloco 734
    spots_submersos = []
    for idx_sp, sx in enumerate([22.0, 27.0, 32.0], start=1):
        sy = 0.32
        sz = fz_min + 0.32
        # Direção inclinada para cima (+Y) e para o fundo/fachada (+Z, +X)
        alvo_sp = np.array([sx + 2.5, 3.6, 28.0])
        dir_sp = alvo_sp - np.array([sx, sy, sz])
        dir_sp /= np.linalg.norm(dir_sp)
        spots_submersos.append((np.array([sx, sy, sz]), dir_sp))

        M_sp = mat_translacao(sx, sy, sz) @ mat_escala(0.18, 0.14, 0.18)
        cena.append(Primitiva(f"Spot Submerso Fonte {idx_sp}", "esfera", M_sp, mat_spot_submerso))

    fonte_info = {
        'x_min': fx_min, 'x_max': fx_max,
        'z_min': fz_min, 'z_max': fz_max,
        'y_agua': y_agua,
        'spots_submersos': spots_submersos
    }

    # --- ESCULTURA CINÉTICA EM AÇO (5 LOSANGOS A 45°) AO LADO DA FONTE ---
    dados_escultura = gerar_parametros_escultura_cinetica(origem_base=(15.0, 0.2, 22.0))
    cena.append(Primitiva("Base Concreto (Escultura Cinética)", "cubo", dados_escultura['M_base'], mat_concreto_base))

    for q in dados_escultura['quadros']:
        idx_q = q['indice']
        for b_idx, M_barra in enumerate(q['barras_M']):
            cena.append(Primitiva(f"Escultura Cinética - Losango {idx_q} (Perfil {b_idx+1})", "cubo", M_barra, mat_aco_branco))
        cena.append(Primitiva(f"Escultura Cinética - Losango {idx_q} (Sapata)", "cubo", q['haste_M'], mat_aco_branco))

    # --- ÁRVORES NO ENTORNO ---
    arvores_cfg = [
        ("Árvore Bosque Esq.",   10.5, 28.5, 1.15),
        ("Árvore Fundo Esq.",    11.5, 38.0, 1.25),
        ("Árvore Praça Dir.",    37.5, 17.5, 0.90),
    ]
    esferas_copa_rel = [
        ((0.0, 3.6, 0.0), 1.65, mat_folha_1),
        ((-1.1, 3.2, 0.6), 1.25, mat_folha_2),
        ((1.1, 3.3, -0.5), 1.30, mat_folha_2),
        ((0.3, 4.5, 0.4), 1.20, mat_folha_3),
        ((-0.7, 4.1, -0.7), 1.15, mat_folha_3),
        ((0.8, 2.9, 0.8), 1.10, mat_folha_1),
    ]
    arvores_3d_info = []
    for nome_arv, tx_arv, tz_arv, esc_arv in arvores_cfg:
        alt_t = 3.2 * esc_arv
        M_tronco = mat_translacao(tx_arv, 0.2, tz_arv) @ mat_escala(0.42 * esc_arv, alt_t, 0.42 * esc_arv)
        cena.append(Primitiva(f"Tronco {nome_arv}", "cilindro", M_tronco, mat_tronco))
        esfs_3d = []
        for idx_e, (off, r_esf, mat_f) in enumerate(esferas_copa_rel):
            M_rot_q = mat_rotacao_eixo_arbitrario_quaternion((1.0, 0.5, 0.3), idx_e * 0.4, (tx_arv, 3.5 * esc_arv, tz_arv))
            px_e = tx_arv + off[0] * esc_arv
            py_e = off[1] * esc_arv
            pz_e = tz_arv + off[2] * esc_arv
            r_final = r_esf * esc_arv
            M_esf = M_rot_q @ mat_translacao(px_e, py_e, pz_e) @ mat_escala(r_final, r_final, r_final)
            cena.append(Primitiva(f"Copa {nome_arv} ({idx_e+1})", "esfera", M_esf, mat_f))
            if idx_e < 4:
                esfs_3d.append((px_e, py_e, pz_e, r_final, '#16A34A' if idx_e % 2 == 0 else '#15803D'))
        arvores_3d_info.append({'x': tx_arv, 'z': tz_arv, 'alt': alt_t, 'esferas': esfs_3d})

    # --- CONES DE SINALIZAÇÃO ---
    M_espelho = mat_espelho_plano_arbitrario((36.2, 0.0, 16.0), (0.8, 0.0, 0.6))
    cones_base = [(35.3, 0.2, 16.2), (35.9, 0.2, 15.6)]
    cones_3d_info = []
    for i_c, (cx_c, cy_c, cz_c) in enumerate(cones_base):
        M_cone = mat_translacao(cx_c, cy_c, cz_c) @ mat_escala(0.32, 0.95, 0.32)
        M_cone_esp = M_espelho @ M_cone
        cena.append(Primitiva(f"Cone Sinalização {i_c+1}", "cone", M_cone, mat_cone_laranja))
        cena.append(Primitiva(f"Cone Espelhado {i_c+1}", "cone", M_cone_esp, mat_cone_laranja))
        cones_3d_info.append((cx_c, cz_c))
        cones_3d_info.append((M_cone_esp[0, 3], M_cone_esp[2, 3]))

    # --- ABRIGO METÁLICO ---
    M_malha = (mat_translacao(44.5, 0.2, 17.2) @
               mat_cisalhamento_xy(0.15) @
               mat_escala(4.5, 2.6, 3.2))
    dados_m = criar_malha_cobertura_metalica(M_malha)
    cena.append(Primitiva("Abrigo Metálico (Malha)", "malha", M_malha, mat_metal_galv, dados_extra=dados_m))

    for px_p, pz_p in [(44.7, 17.4), (48.8, 17.4), (44.7, 20.2), (48.8, 20.2)]:
        M_pil = mat_translacao(px_p, 0.2, pz_p) @ mat_escala(0.12, 2.3, 0.12)
        cena.append(Primitiva("Pilar Abrigo Metálico", "cilindro", M_pil, mat_esquadria))

    # --- POSTES DE ILUMINAÇÃO PÚBLICA ---
    postes_coords = [
        (9.5, 21.0, 0.6),
        (18.5, 16.2, 0.4),
        (41.0, 16.5, 0.0),
        (54.5, 17.5, -0.5),
    ]
    for idx_p, (px_pos, pz_pos, off_lum_x) in enumerate(postes_coords, start=1):
        M_poste = mat_translacao(px_pos, 0.2, pz_pos) @ mat_escala(0.14, 6.2, 0.14)
        cena.append(Primitiva(f"Poste Iluminação {idx_p}", "cilindro", M_poste, mat_esquadria))
        M_lum = mat_translacao(px_pos + off_lum_x, 6.3, pz_pos + 0.4) @ mat_escala(0.35, 0.24, 0.35)
        cena.append(Primitiva(f"Luminária Poste {idx_p}", "esfera", M_lum, mat_alvenaria_branca))
# --- ELEMENTOS 3D EXTERNOS (.OBJ LOW-POLY) ---
    mat_suv       = Material("SUV Urbano (Pintura Prata)", (0.72, 0.75, 0.78), (0.80, 0.82, 0.85), (0.60, 0.62, 0.65), brilho=90.0, refletividade=0.25)
    mat_esportivo = Material("Carro Esportivo (Vermelho)", (0.82, 0.12, 0.14), (0.85, 0.85, 0.88), (0.68, 0.10, 0.12), brilho=100.0, refletividade=0.28)
    mat_estudante1= Material("Estudante 1 (Casual)", (0.18, 0.42, 0.72), (0.20, 0.20, 0.22), (0.15, 0.35, 0.60), brilho=20.0)
    mat_estudante2= Material("Estudante 2 (Vestido)", (0.78, 0.48, 0.22), (0.20, 0.20, 0.22), (0.65, 0.40, 0.18), brilho=20.0)
    mat_palmeira  = Material("Palmeira Tropical (OBJ)", (0.14, 0.46, 0.18), (0.15, 0.20, 0.15), (0.12, 0.38, 0.15), brilho=16.0)

    # 1. Carros na via asfáltica (Y = 0.10m)
    M_suv = mat_translacao(23.0, 0.10, 7.2) @ mat_rotacao_y(np.deg2rad(90.0)) @ mat_escala(1.65, 1.65, 1.65)
    cena.append(Primitiva("SUV na Via CT (OBJ)", "malha", M_suv, mat_suv,
                          dados_extra=carregar_malha_obj("SUV.obj", M_suv)))

    M_sport = mat_translacao(39.0, 0.10, 10.6) @ mat_rotacao_y(np.deg2rad(-90.0)) @ mat_escala(1.32, 1.32, 1.32)
    cena.append(Primitiva("Carro Esportivo Via CT (OBJ)", "malha", M_sport, mat_esportivo,
                          dados_extra=carregar_malha_obj("SportsCar.obj", M_sport)))

    # 2. Grupo de estudantes na calçada (Y = 0.20m) junto à fonte e escultura
    M_p1 = mat_translacao(18.6, 0.20, 18.8) @ mat_rotacao_y(np.deg2rad(-35.0)) @ mat_escala(1.68, 1.68, 1.68)
    cena.append(Primitiva("Estudante Casual (OBJ)", "malha", M_p1, mat_estudante1,
                          dados_extra=carregar_malha_obj("Female_Casual.obj", M_p1)))

    M_p2 = mat_translacao(19.4, 0.20, 18.3) @ mat_rotacao_y(np.deg2rad(-55.0)) @ mat_escala(1.65, 1.65, 1.65)
    cena.append(Primitiva("Estudante Vestido (OBJ)", "malha", M_p2, mat_estudante2,
                          dados_extra=carregar_malha_obj("Female_Dress.obj", M_p2)))

    # 3. Palmeira na praça (Y = 0.20m)
    M_palm = mat_translacao(42.5, 0.20, 17.5) @ mat_escala(4.8, 4.8, 4.8)
    cena.append(Primitiva("Palmeira Praça CT (OBJ)", "malha", M_palm, mat_palmeira,
                          dados_extra=carregar_malha_obj("PalmTree_2.obj", M_palm)))
    extras_3d = {
        'arvores': arvores_3d_info,
        'postes': postes_coords,
        'cones': cones_3d_info,
        'fonte': fonte_info
    }
    return cena, materiais_bloco, dados_escultura, extras_3d

# ==============================================================================
# 7. MOTOR DE RENDERIZAÇÃO (DIA/NOITE + CÁUSTICA + BOUNDING BOX AABB)
# ==============================================================================
class RenderizadorCT:
    def __init__(self, largura=500, altura=500):
        self.W = largura
        self.H = altura
        self.cena, self.materiais_bloco, self.dados_escultura, self.extras_3d = montar_cenario()

        self.centro_domo = np.array([35.0, 2.5, 34.0], dtype=np.float64)
        self.raio_domo = 38.0

        self.presets = {
            '2_pontos_fuga': {
                'titulo': 'Perspectiva - 2 Pontos de Fuga (Bloco 734 + Escultura + Fonte)',
                'eye': [6.5, 3.4, 5.5], 'at': [28.0, 2.8, 28.0], 'up': [0.0, 1.0, 0.0],
                'd': 1.5, 'win': (-1.02, 1.02, -1.02, 1.02), 'proj': 'perspectiva'
            },
            '1_ponto_fuga': {
                'titulo': 'Perspectiva - 1 Ponto de Fuga (Frontal Fonte e Fachada)',
                'eye': [27.0, 3.1, 4.5], 'at': [27.0, 2.8, 26.0], 'up': [0.0, 1.0, 0.0],
                'd': 1.5, 'win': (-1.05, 1.05, -1.05, 1.05), 'proj': 'perspectiva'
            },
            '3_pontos_fuga': {
                'titulo': 'Perspectiva - 3 Pontos de Fuga (Vista Aérea Inclinada)',
                'eye': [7.5, 18.0, 6.0], 'at': [30.0, 2.5, 30.0], 'up': [0.0, 1.0, 0.0],
                'd': 1.5, 'win': (-0.98, 0.98, -0.98, 0.98), 'proj': 'perspectiva'
            },
            'zoom_in': {
                'titulo': 'Zoom In (Escultura 5 Losangos + Espelho d\'Água)',
                'eye': [10.5, 2.5, 13.0], 'at': [18.5, 2.2, 22.0], 'up': [0.0, 1.0, 0.0],
                'd': 1.85, 'win': (-0.88, 0.88, -0.88, 0.88), 'proj': 'perspectiva'
            },
            'zoom_out': {
                'titulo': 'Zoom Out (Campo de Visão Ampliado)',
                'eye': [6.5, 3.4, 5.5], 'at': [28.0, 2.8, 28.0], 'up': [0.0, 1.0, 0.0],
                'd': 1.1, 'win': (-1.28, 1.28, -1.28, 1.28), 'proj': 'perspectiva'
            },
            'ortografica': {
                'titulo': 'Projeção Ortográfica (Raios Paralelos)',
                'eye': [10.0, 8.0, 6.0], 'at': [29.0, 3.0, 30.0], 'up': [0.0, 1.0, 0.0],
                'd': 1.5, 'win': (-23.0, 23.0, -23.0, 23.0), 'proj': 'ortografica'
            },
            'obliqua': {
                'titulo': 'Projeção Oblíqua (Cavaleira / Gabinete)',
                'eye': [25.0, 6.5, 4.0], 'at': [25.0, 6.5, 30.0], 'up': [0.0, 1.0, 0.0],
                'd': 1.5, 'win': (-23.0, 23.0, -23.0, 23.0), 'proj': 'obliqua'
            }
        }
        self.ultimo_buffer_pick = None

    def _aabb_global_cena(self):
        mins = np.min([p.aabb_cam[0] for p in self.cena], axis=0)
        maxs = np.max([p.aabb_cam[1] for p in self.cena], axis=0)
        return mins, maxs

    def tracar_raios(self, O_cam, D_cam):
        shape = D_cam.shape[:-1]
        t_min = np.full(shape, np.inf)
        N_min = np.zeros_like(D_cam)
        N_w_min = np.zeros_like(D_cam)
        P_w_min = np.zeros_like(D_cam)
        obj_idx = np.full(shape, -1, dtype=int)
        sub_mat_min = np.zeros(shape, dtype=int)

        g_min, g_max = self._aabb_global_cena()
        if not np.any(testar_aabb(O_cam, D_cam, g_min, g_max)):
            return t_min, N_min, P_w_min, obj_idx, sub_mat_min, N_w_min

        for idx, prim in enumerate(self.cena):
            t_o, N_o, P_w, sub_m, N_w = intersect_primitiva(O_cam, D_cam, prim, t_min)
            m = t_o < t_min
            if np.any(m):
                t_min = np.where(m, t_o, t_min)
                N_min = np.where(m[..., None], N_o, N_min)
                N_w_min = np.where(m[..., None], N_w, N_w_min)
                P_w_min = np.where(m[..., None], P_w, P_w_min)
                obj_idx = np.where(m, idx, obj_idx)
                sub_mat_min = np.where(m, sub_m, sub_mat_min)

        return t_min, N_min, P_w_min, obj_idx, sub_mat_min, N_w_min

    def sombrear(self, O_cam, D_cam, t_min, N_min, P_w_min, obj_idx, sub_mat_min, N_w_min, luzes, modo_noite=False, permitir_reflexo=True):
        shape = D_cam.shape[:-1]
        hit = np.isfinite(t_min)
        P_cam = np.where(hit[..., None], O_cam + t_min[..., None] * D_cam, 0.0)
        V_cam = -D_cam

        flip = np.einsum('...i,...i->...', N_min, V_cam) < 0
        N_min = np.where(flip[..., None], -N_min, N_min)

        Kd = np.zeros_like(D_cam)
        Ke = np.zeros_like(D_cam)
        Ka = np.zeros_like(D_cam)
        brilho = np.full(shape, 32.0)
        refl = np.zeros(shape)
        emissivo = np.zeros_like(D_cam)

        for idx, prim in enumerate(self.cena):
            m = (obj_idx == idx)
            if not np.any(m):
                continue
            if prim.tipo == 'bloco_ct_arqueado':
                for sm_id, mat_sm in self.materiais_bloco.items():
                    ms = m & (sub_mat_min == sm_id)
                    if np.any(ms):
                        kd_val = avaliar_textura(mat_sm.textura, P_w_min, mat_sm.kd)
                        Kd = np.where(ms[..., None], kd_val, Kd)
                        Ke = np.where(ms[..., None], mat_sm.ke, Ke)
                        Ka = np.where(ms[..., None], mat_sm.ka, Ka)
                        brilho = np.where(ms, mat_sm.brilho, brilho)
                        refl = np.where(ms, mat_sm.refletividade, refl)
            else:
                kd_val = avaliar_textura(prim.mat.textura, P_w_min, prim.mat.kd)
                Kd = np.where(m[..., None], kd_val, Kd)
                Ke = np.where(m[..., None], prim.mat.ke, Ke)
                Ka = np.where(m[..., None], prim.mat.ka, Ka)
                brilho = np.where(m, prim.mat.brilho, brilho)
                refl = np.where(m, prim.mat.refletividade, refl)

                # Luminárias dos postes e spots submersos brilham diretamente à noite
                if "Luminária Poste" in prim.nome:
                    emissivo = np.where(m[..., None], np.array([1.0, 0.90, 0.62]) * (1.25 if modo_noite else 0.15), emissivo)
                elif "Spot Submerso" in prim.nome:
                    emissivo = np.where(m[..., None], np.array([0.65, 0.96, 1.00]) * (1.40 if modo_noite else 0.35), emissivo)

        cor = Ka * luzes['I_amb'] + emissivo
        P_eps = P_cam + N_min * 0.02

        # 1. Luz Solar Direcional (ativa apenas no modo DIA)
        if not modo_noite:
            L_dir = -luzes['dir_cam']
            sombra_dir = np.zeros(shape, dtype=bool)
            D_dir_bcast = np.broadcast_to(L_dir, D_cam.shape)
            for prim in self.cena[2:]:
                ts, _, _, _, _ = intersect_primitiva(P_eps, D_dir_bcast, prim)
                sombra_dir |= (hit & np.isfinite(ts))
            ndl_d = np.maximum(0.0, np.einsum('...i,i->...', N_min, L_dir))
            R_d = 2.0 * ndl_d[..., None] * N_min - L_dir
            rdv_d = np.maximum(0.0, np.einsum('...i,...i->...', R_d, V_cam))
            cor += (~sombra_dir)[..., None] * (Kd * luzes['I_dir'] * ndl_d[..., None] +
                                               Ke * luzes['I_dir'] * (rdv_d[..., None] ** brilho[..., None]))

        # 2. Luzes Artificiais dos Postes (Iluminação Pública da Praça)
        for p_cam, i_poste in luzes['postes_cam']:
            L_p_vec = p_cam - P_cam
            dist_p = np.linalg.norm(L_p_vec, axis=-1)
            L_p = L_p_vec / np.where(dist_p[..., None] > 1e-8, dist_p[..., None], 1.0)
            aten_p = 1.0 / (1.0 + 0.008 * (dist_p**2))
            ndl_p = np.maximum(0.0, np.einsum('...i,...i->...', N_min, L_p))
            R_p = 2.0 * ndl_p[..., None] * N_min - L_p
            rdv_p = np.maximum(0.0, np.einsum('...i,...i->...', R_p, V_cam))
            cor += aten_p[..., None] * (Kd * i_poste * ndl_p[..., None] +
                                        Ke * i_poste * (rdv_p[..., None] ** brilho[..., None]))

        # 3. Spot Direcionado para a Escultura Cinética
        L_s_vec = luzes['spot_pos_cam'] - P_cam
        dist_s = np.linalg.norm(L_s_vec, axis=-1)
        L_s = L_s_vec / np.where(dist_s[..., None] > 1e-8, dist_s[..., None], 1.0)
        cos_ang = np.einsum('...i,i->...', -L_s, luzes['spot_dir_cam'])
        fator_spot = np.where(cos_ang >= luzes['spot_cos_cutoff'],
                              np.maximum(0.0, cos_ang) ** 6.0, 0.0)
        aten_s = 1.0 / (1.0 + 0.006 * (dist_s**2))
        ndl_s = np.maximum(0.0, np.einsum('...i,...i->...', N_min, L_s))
        cor += (fator_spot * aten_s)[..., None] * (Kd * luzes['I_spot'] * ndl_s[..., None])

        # 4. Cáusticas Refratadas pelo Espelho d'Água na Fachada e Escultura
        cor_caustica = calcular_caustica_fonte(P_w_min, N_w_min, self.extras_3d['fonte'], modo_noite)
        cor += cor_caustica

        # 5. Reflexão Especular (Água, Vidro Fumê, Aço)
        if permitir_reflexo and np.any(refl > 0.01):
            D_refl = D_cam - 2.0 * np.einsum('...i,...i->...', D_cam, N_min)[..., None] * N_min
            D_refl /= np.linalg.norm(D_refl, axis=-1, keepdims=True)
            t_r, N_r, P_wr, idx_r, sub_r, Nw_r = self.tracar_raios(P_eps, D_refl)
            cor_r = self.sombrear(P_eps, D_refl, t_r, N_r, P_wr, idx_r, sub_r, Nw_r, luzes, modo_noite=modo_noite, permitir_reflexo=False)
            cor = (1.0 - refl[..., None]) * cor + refl[..., None] * cor_r

        # Fundo Céu (Dia claro vs Noite profunda)
        ceu_t = np.clip(0.5 + 0.5 * D_cam[..., 1], 0.0, 1.0)[..., None]
        if modo_noite:
            ceu = (1.0 - ceu_t) * np.array([0.04, 0.07, 0.14]) + ceu_t * np.array([0.01, 0.02, 0.06])
        else:
            ceu = (1.0 - ceu_t) * np.array([0.82, 0.90, 0.98]) + ceu_t * np.array([0.32, 0.58, 0.88])

        return np.where(hit[..., None], np.clip(cor, 0.0, 1.0), ceu)

    def renderizar_config(self, cfg, modo_noite=False):
        t0 = time.time()
        M_wc = construir_Mwc(cfg['eye'], cfg['at'], cfg['up'])

        for prim in self.cena:
            prim.preparar_camera(M_wc)

        dir_w = np.array([0.45, -0.75, 0.48, 0.0])
        dir_c = (M_wc @ dir_w)[:3]; dir_c /= np.linalg.norm(dir_c)

        # Converte posições dos 4 postes de iluminação pública para coordenadas de câmera
        intensidade_poste = np.array([1.45, 1.25, 0.85]) if modo_noite else np.array([0.18, 0.18, 0.16])
        postes_cam = []
        for px_pos, pz_pos, off_lum_x in self.extras_3d['postes']:
            pw = np.array([px_pos + off_lum_x, 6.1, pz_pos + 0.4, 1.0])
            postes_cam.append(((M_wc @ pw)[:3], intensidade_poste))

        spot_pos_w = np.array([18.5, 6.1, 16.2, 1.0])
        spot_alvo_w = np.array([15.0, 2.2, 22.0, 1.0])
        spot_pos_c = (M_wc @ spot_pos_w)[:3]
        spot_dir_c = (M_wc @ (spot_alvo_w - spot_pos_w))[:3]
        spot_dir_c /= np.linalg.norm(spot_dir_c)

        luzes = {
            'I_amb': np.array([0.06, 0.08, 0.14]) if modo_noite else np.array([0.26, 0.27, 0.30]),
            'dir_cam': dir_c,
            'I_dir': np.array([0.0, 0.0, 0.0]) if modo_noite else np.array([0.75, 0.73, 0.68]),
            'postes_cam': postes_cam,
            'spot_pos_cam': spot_pos_c,
            'spot_dir_cam': spot_dir_c,
            'spot_cos_cutoff': np.cos(np.deg2rad(42.0)),
            'I_spot': np.array([1.85, 1.70, 1.35]) if modo_noite else np.array([0.45, 0.45, 0.42])
        }

        xmin, xmax, ymin, ymax = cfg['win']
        d = cfg['d']
        u = np.linspace(xmin, xmax, self.W)
        v = np.linspace(ymax, ymin, self.H)
        U, V = np.meshgrid(u, v)

        if cfg['proj'] == 'perspectiva':
            O_cam = np.zeros((self.H, self.W, 3), dtype=np.float64)
            D_cam = np.stack([U, V, np.full_like(U, -d)], axis=-1)
            D_cam /= np.linalg.norm(D_cam, axis=-1, keepdims=True)
        elif cfg['proj'] == 'ortografica':
            O_cam = np.stack([U, V, np.zeros_like(U)], axis=-1)
            D_cam = np.broadcast_to(np.array([0.0, 0.0, -1.0]), O_cam.shape).copy()
        elif cfg['proj'] == 'obliqua':
            O_cam = np.stack([U, V, np.zeros_like(U)], axis=-1)
            l_ob = 0.5
            ang_ob = np.deg2rad(45.0)
            d_vec = np.array([-l_ob * np.cos(ang_ob), -l_ob * np.sin(ang_ob), -1.0])
            d_vec /= np.linalg.norm(d_vec)
            D_cam = np.broadcast_to(d_vec, O_cam.shape).copy()

        t_min, N_min, P_w_min, obj_idx, sub_mat_min, N_w_min = self.tracar_raios(O_cam, D_cam)
        img = self.sombrear(O_cam, D_cam, t_min, N_min, P_w_min, obj_idx, sub_mat_min, N_w_min, luzes, modo_noite=modo_noite)

        self.ultimo_buffer_pick = {
            'obj_idx': obj_idx,
            'sub_mat': sub_mat_min,
            'P_world': P_w_min,
            't_min': t_min
        }
        turno_txt = "NOITE (Luz Artificial + Cáusticas)" if modo_noite else "DIA (Sol + Cáusticas Suaves)"
        print(f"[{cfg['titulo']} | {turno_txt}] Renderizado em {time.time() - t0:.2f}s ({self.W}x{self.H} px)")
        return img, f"{cfg['titulo']} [{turno_txt}]"

# ==============================================================================
# 8. VIEWPORT INTERATIVA 3D (SUPORTE A DIA / NOITE + ESPELHO D'ÁGUA)
# ==============================================================================
def desenhar_cena_3d_interativa(ax_3d, renderer, estilo_visual='materiais', cfg_cam=None, modo_noite=False):
    ax_3d.clear()
    cx, cz, R = 37.0, 42.0, 22.0
    H = 6.6

    ax_3d.set_facecolor('#0B1120' if modo_noite else '#FFFFFF')

    if estilo_visual == 'wireframe':
        c_bloco, c_teto, c_vidro, c_734 = 'none', 'none', 'none', 'none'
        c_asfalto, c_calcada, c_tronco, c_poste = 'none', 'none', '#64748B', '#94A3B8'
        c_abrigo, c_base_esc, c_esc_aco, c_agua, edge_c, alpha_f = 'none', 'none', '#38BDF8' if modo_noite else '#0F172A', 'none', '#38BDF8' if modo_noite else '#0F172A', 0.0
    elif estilo_visual == 'shaded':
        c_bloco, c_teto, c_vidro, c_734 = ('#334155', '#1E293B', '#0F172A', '#1E3A8A') if modo_noite else ('#CBD5E1', '#94A3B8', '#64748B', '#475569')
        c_asfalto, c_calcada, c_tronco, c_poste = ('#1E293B', '#475569', '#475569', '#94A3B8') if modo_noite else ('#94A3B8', '#E2E8F0', '#64748B', '#475569')
        c_abrigo, c_base_esc, c_esc_aco, c_agua, edge_c, alpha_f = '#475569', '#64748B', '#E0F2FE', '#0284C7', '#1E293B', 0.95
    else:
        if modo_noite:
            c_bloco, c_teto, c_vidro, c_734 = '#3B4D6B', '#1E293B', '#090D16', '#1D4ED8'
            c_asfalto, c_calcada, c_tronco, c_poste = '#111827', '#334155', '#3B2314', '#64748B'
            c_abrigo, c_base_esc, c_esc_aco, c_agua, edge_c, alpha_f = '#475569', '#334155', '#E0F2FE', '#06B6D4', '#0F172A', 0.95
        else:
            c_bloco, c_teto, c_vidro, c_734 = '#F8FAFC', '#E2E8F0', '#1E293B', '#1E3A8A'
            c_asfalto, c_calcada, c_tronco, c_poste = '#334155', '#CBD5E1', '#5C3A21', '#1E293B'
            c_abrigo, c_base_esc, c_esc_aco, c_agua, edge_c, alpha_f = '#94A3B8', '#64748B', '#FFFFFF', '#38BDF8', '#475569', 0.95

    # 1. Piso
    if estilo_visual != 'wireframe':
        gx_v, gz_v = np.meshgrid([2.0, 68.0], [2.0, 16.0])
        ax_3d.plot_surface(gx_v, gz_v, np.full_like(gx_v, 0.05), color=c_asfalto, alpha=0.9, shade=False)
        gx_c, gz_c = np.meshgrid([2.0, 68.0], [16.0, 60.0])
        ax_3d.plot_surface(gx_c, gz_c, np.full_like(gx_c, 0.15), color=c_calcada, alpha=0.9, shade=False)
    else:
        ax_3d.plot([2, 68, 68, 2, 2], [2, 2, 60, 60, 2], [0, 0, 0, 0, 0], color='#64748B', lw=0.8)

    # 2. Bloco 734
    thetas = np.linspace(np.pi, 1.5 * np.pi, 28)
    arco_x = cx + R * np.cos(thetas)
    arco_z = cz + R * np.sin(thetas)

    pts_x = np.concatenate([[15.0], arco_x, [59.0, 59.0, 15.0]])
    pts_z = np.concatenate([[53.0], arco_z, [20.0, 53.0, 53.0]])

    verts_teto = [list(zip(pts_x, pts_z, np.full_like(pts_x, H)))]
    ax_3d.add_collection3d(Poly3DCollection(verts_teto, facecolors=c_teto, edgecolors=edge_c, alpha=alpha_f, linewidths=1.0))

    paredes = []
    for k in range(len(pts_x) - 1):
        paredes.append([
            (pts_x[k], pts_z[k], 0.2),
            (pts_x[k+1], pts_z[k+1], 0.2),
            (pts_x[k+1], pts_z[k+1], H),
            (pts_x[k], pts_z[k], H)
        ])
    ax_3d.add_collection3d(Poly3DCollection(paredes, facecolors=c_bloco, edgecolors=edge_c, alpha=alpha_f, linewidths=0.6))

    # 3. Porta, Janelas e Placa 734
    esquadrias = []
    ang_p = np.pi + np.linspace(0.06, 0.20, 4) * (np.pi / 2.0)
    for k in range(len(ang_p) - 1):
        esquadrias.append([
            (cx + (R+0.08)*np.cos(ang_p[k]),   cz + (R+0.08)*np.sin(ang_p[k]),   0.2),
            (cx + (R+0.08)*np.cos(ang_p[k+1]), cz + (R+0.08)*np.sin(ang_p[k+1]), 0.2),
            (cx + (R+0.08)*np.cos(ang_p[k+1]), cz + (R+0.08)*np.sin(ang_p[k+1]), 2.75),
            (cx + (R+0.08)*np.cos(ang_p[k]),   cz + (R+0.08)*np.sin(ang_p[k]),   2.75),
        ])
    for j_idx in range(9):
        s0 = 0.24 + (j_idx + 0.18) * (0.72 / 9.0)
        s1 = 0.24 + (j_idx + 0.82) * (0.72 / 9.0)
        a0, a1 = np.pi + s0 * (np.pi / 2.0), np.pi + s1 * (np.pi / 2.0)
        esquadrias.append([
            (cx + (R+0.08)*np.cos(a0), cz + (R+0.08)*np.sin(a0), 1.35),
            (cx + (R+0.08)*np.cos(a1), cz + (R+0.08)*np.sin(a1), 1.35),
            (cx + (R+0.08)*np.cos(a1), cz + (R+0.08)*np.sin(a1), 3.25),
            (cx + (R+0.08)*np.cos(a0), cz + (R+0.08)*np.sin(a0), 3.25),
        ])
    ax_3d.add_collection3d(Poly3DCollection(esquadrias, facecolors=c_vidro, edgecolors='#0F172A', alpha=0.95 if estilo_visual != 'wireframe' else 0.0, linewidths=0.8))

    a_pl0, a_pl1 = np.pi + 0.05 * (np.pi / 2.0), np.pi + 0.23 * (np.pi / 2.0)
    placa_poly = [[
        (cx + (R+0.12)*np.cos(a_pl0), cz + (R+0.12)*np.sin(a_pl0), 5.15),
        (cx + (R+0.12)*np.cos(a_pl1), cz + (R+0.12)*np.sin(a_pl1), 5.15),
        (cx + (R+0.12)*np.cos(a_pl1), cz + (R+0.12)*np.sin(a_pl1), 6.30),
        (cx + (R+0.12)*np.cos(a_pl0), cz + (R+0.12)*np.sin(a_pl0), 6.30),
    ]]
    ax_3d.add_collection3d(Poly3DCollection(placa_poly, facecolors=c_734, edgecolors='#1E3A8A', alpha=0.95 if estilo_visual != 'wireframe' else 0.0, linewidths=1.0))

    # 3A. ESPELHO D'ÁGUA (FONTE) + SPOTS SUBMERSOS + FEIXES DE CÁUSTICA NO PREVIEW
    f_inf = renderer.extras_3d['fonte']
    fx0, fx1, fz0, fz1, ya = f_inf['x_min'], f_inf['x_max'], f_inf['z_min'], f_inf['z_max'], f_inf['y_agua']
    poly_agua = [[(fx0, fz0, ya), (fx1, fz0, ya), (fx1, fz1, ya), (fx0, fz1, ya)]]
    ax_3d.add_collection3d(Poly3DCollection(poly_agua, facecolors=c_agua, edgecolors='#0284C7', alpha=0.85 if estilo_visual != 'wireframe' else 0.0, linewidths=1.2))
    if estilo_visual == 'wireframe':
        ax_3d.plot([fx0, fx1, fx1, fx0, fx0], [fz0, fz0, fz1, fz1, fz0], [ya]*5, color='#0284C7', lw=1.2)

    for sp_pos, sp_dir in f_inf['spots_submersos']:
        ax_3d.scatter([sp_pos[0]], [sp_pos[2]], [sp_pos[1]], color='#67E8F9', s=45)
        if modo_noite:
            # Feixe indicativo dos spots da fonte projetando cáusticas na fachada
            p_fim = sp_pos + sp_dir * 9.5
            ax_3d.plot([sp_pos[0], p_fim[0]], [sp_pos[2], p_fim[2]], [sp_pos[1], p_fim[1]],
                       color='#67E8F9', linestyle='--', linewidth=1.1, alpha=0.65)

    # 3B. ESCULTURA CINÉTICA EM AÇO (5 Losangos a 45° + Base Concreto)
    esc = renderer.dados_escultura
    bp = esc['base_pts_world']
    faces_base_idx = [
        [4, 5, 6, 7], [0, 1, 2, 3],
        [0, 1, 5, 4], [1, 2, 6, 5],
        [2, 3, 7, 6], [3, 0, 4, 7]
    ]
    polys_base = [[(bp[i, 0], bp[i, 2], bp[i, 1]) for i in f] for f in faces_base_idx]
    ax_3d.add_collection3d(Poly3DCollection(polys_base, facecolors=c_base_esc, edgecolors=edge_c, alpha=alpha_f, linewidths=0.9))

    for q in esc['quadros']:
        vw = q['vertices_world']
        xs = np.append(vw[:, 0], vw[0, 0])
        zs = np.append(vw[:, 2], vw[0, 2])
        ys = np.append(vw[:, 1], vw[0, 1])
        ax_3d.plot(xs, zs, ys, color=c_esc_aco, linewidth=3.4 if estilo_visual != 'wireframe' else 1.8)
        if estilo_visual != 'wireframe':
            ax_3d.plot(xs, zs, ys, color='#334155', linewidth=0.7, alpha=0.65)

    # 4. Árvores
    u_s, v_s = np.mgrid[0:2*np.pi:10j, 0:np.pi:6j]
    for arv in renderer.extras_3d['arvores']:
        ax_3d.plot([arv['x'], arv['x']], [arv['z'], arv['z']], [0.2, arv['alt'] + 0.2], color=c_tronco, linewidth=4.5)
        for (px_e, py_e, pz_e, r_e, cor_e) in arv['esferas']:
            xs = px_e + r_e * np.cos(u_s) * np.sin(v_s)
            zs = pz_e + r_e * np.sin(u_s) * np.sin(v_s)
            ys = py_e + r_e * np.cos(v_s)
            if estilo_visual == 'wireframe':
                ax_3d.plot_wireframe(xs, zs, ys, color='#15803D', linewidth=0.5)
            else:
                cor_atual = '#475569' if estilo_visual == 'shaded' else ('#064E3B' if modo_noite else cor_e)
                ax_3d.plot_surface(xs, zs, ys, color=cor_atual, alpha=0.9, linewidth=0)

    # 5. Postes
    for px_pos, pz_pos, off_lx in renderer.extras_3d['postes']:
        ax_3d.plot([px_pos, px_pos], [pz_pos, pz_pos], [0.2, 6.4], color=c_poste, linewidth=2.5)
        ax_3d.plot([px_pos, px_pos + off_lx], [pz_pos, pz_pos + 0.4], [6.4, 6.3], color=c_poste, linewidth=2.0)
        cor_luz = '#FDE047' if (modo_noite or estilo_visual == 'materiais') else '#475569'
        ax_3d.scatter([px_pos + off_lx], [pz_pos + 0.4], [6.3], color=cor_luz, s=75 if modo_noite else 35)

    # 6. Abrigo Metálico
    prim_malha = next(p for p in renderer.cena if p.tipo == 'malha')
    vw = prim_malha.dados['vertices_world']
    fc = prim_malha.dados['faces']
    polys_m = [[(vw[f[0], 0], vw[f[0], 2], vw[f[0], 1]),
                (vw[f[1], 0], vw[f[1], 2], vw[f[1], 1]),
                (vw[f[2], 0], vw[f[2], 2], vw[f[2], 1])] for f in fc]
    ax_3d.add_collection3d(Poly3DCollection(polys_m, facecolors=c_abrigo, edgecolors=edge_c, alpha=alpha_f, linewidths=0.8))

    # 7. Cones
    for cx_c, cz_c in renderer.extras_3d['cones']:
        ax_3d.scatter([cx_c], [cz_c], [0.65], color='#64748B' if estilo_visual == 'shaded' else '#EA580C', s=50, marker='^')

    # 8. Abóbada Semi-Esférica
    c_domo = renderer.centro_domo
    r_domo = renderer.raio_domo
    az_g = np.linspace(0, 2 * np.pi, 28)
    el_g = np.linspace(0.0, np.pi / 2.0, 9)
    AZ, EL = np.meshgrid(az_g, el_g)
    XD = c_domo[0] + r_domo * np.cos(EL) * np.cos(AZ)
    ZD = c_domo[2] + r_domo * np.cos(EL) * np.sin(AZ)
    YD = c_domo[1] + r_domo * np.sin(EL)
    ax_3d.plot_wireframe(XD, ZD, YD, color='#38BDF8' if modo_noite else '#0284C7', alpha=0.14, linewidth=0.6)

    ax_3d.set_xlim(2, 68)
    ax_3d.set_ylim(2, 60)
    ax_3d.set_zlim(0, 26)
    ax_3d.set_box_aspect((66, 58, 26))
    cor_eixo = '#E2E8F0' if modo_noite else '#0F172A'
    ax_3d.set_xlabel('X (m)', color=cor_eixo)
    ax_3d.set_ylabel('Z (m)', color=cor_eixo)
    ax_3d.set_zlabel('Y Altura (m)', color=cor_eixo)
    ax_3d.tick_params(colors=cor_eixo)

    if cfg_cam is not None:
        eye = np.array(cfg_cam['eye'], dtype=np.float64)
        at = np.array(cfg_cam['at'], dtype=np.float64)
        vec = eye - at
        horiz = np.hypot(vec[0], vec[2]) + 1e-6
        elev_ini = float(np.clip(np.rad2deg(np.arctan2(vec[1], horiz)), 0.0, 90.0))
        azim_ini = float(np.rad2deg(np.arctan2(vec[2], vec[0])))
        ax_3d.view_init(elev=elev_ini, azim=azim_ini)
        if cfg_cam.get('proj') == 'ortografica':
            ax_3d.set_proj_type('ortho')
        else:
            ax_3d.set_proj_type('persp')

# ==============================================================================
# 9. INTERFACE GRÁFICA INTERATIVA (BOTÃO GLOBAL DIA / NOITE + CÁUSTICAS + RENDER)
# ==============================================================================
def iniciar_aplicacao():
    renderer = RenderizadorCT(largura=500, altura=500)

    fig = plt.figure(figsize=(9.5, 9.6))
    plt.subplots_adjust(bottom=0.21, top=0.83, left=0.05, right=0.95)

    ax_rc = fig.add_subplot(111)
    im_plot = ax_rc.imshow(np.ones((500, 500, 3)))
    ax_rc.axis('off')
    ax_rc.set_visible(False)

    ax_3d = fig.add_subplot(111, projection='3d')
    ax_3d.set_visible(True)

    estado = {
        'em_render': False,
        'modo_painel': False,
        'modo_noite': False,
        'estilo_revit': 'materiais',
        'cfg_atual': renderer.presets['2_pontos_fuga'].copy(),
        'usou_orbita_livre': False
    }

    desenhar_cena_3d_interativa(ax_3d, renderer, estilo_visual=estado['estilo_revit'], cfg_cam=estado['cfg_atual'], modo_noite=estado['modo_noite'])
    ax_3d.set_title(
        f"PREVIEW INTERATIVO [DIA] ({estado['estilo_revit'].upper()}) - {estado['cfg_atual']['titulo']}\n"
        "Alterne [Dia / Noite] para acender postes + spots da fonte (Cáusticas) e clique em [RENDER FINAL]",
        fontsize=9.0, color='#0F172A'
    )

    ax_painel_bg = fig.add_axes([0.12, 0.24, 0.76, 0.57])
    ax_painel_bg.set_facecolor('#F1F5F9')
    ax_painel_bg.set_xticks([]); ax_painel_bg.set_yticks([])
    ax_painel_bg.text(
        0.5, 0.92,
        "CONFIGURAÇÃO MANUAL DA CÂMERA\nDefina Eye, At, Up, Distância Focal (d) e Janela de Visão",
        ha='center', va='top', fontsize=10.5, fontweight='bold', color='#0F172A'
    )
    ax_painel_bg.set_visible(False)

    campos_cfg = [
        ('Eye (x y z):',                  '6.5 3.4 5.5',           [0.44, 0.65, 0.38, 0.055]),
        ('At  (x y z):',                  '28.0 2.8 28.0',         [0.44, 0.57, 0.38, 0.055]),
        ('Up  (x y z):',                  '0.0 1.0 0.0',           [0.44, 0.49, 0.38, 0.055]),
        ('Distância Focal (d):',          '1.5',                   [0.44, 0.41, 0.38, 0.055]),
        ('Janela (xmin xmax ymin ymax):', '-1.02 1.02 -1.02 1.02', [0.44, 0.33, 0.38, 0.055]),
    ]
    textboxes = []
    painel_axes = [ax_painel_bg]

    for label_txt, val_inicial, rect in campos_cfg:
        ax_tb = fig.add_axes(rect)
        tb = TextBox(ax_tb, label_txt + '  ', initial=val_inicial, color='#FFFFFF', hovercolor='#E0F2FE')
        tb.label.set_fontsize(9.5)
        ax_tb.set_visible(False)
        textboxes.append(tb)
        painel_axes.append(ax_tb)

    ax_btn_aplicar = fig.add_axes([0.28, 0.255, 0.22, 0.055])
    btn_aplicar = Button(ax_btn_aplicar, 'Aplicar Preview', color='#86EFAC', hovercolor='#4ADE80')
    ax_btn_aplicar.set_visible(False)
    painel_axes.append(ax_btn_aplicar)

    ax_btn_cancelar = fig.add_axes([0.52, 0.255, 0.20, 0.055])
    btn_cancelar = Button(ax_btn_cancelar, 'Cancelar', color='#FECACA', hovercolor='#F87171')
    ax_btn_cancelar.set_visible(False)
    painel_axes.append(ax_btn_cancelar)

    def mostrar_painel(visivel):
        estado['modo_painel'] = visivel
        for ax_item in painel_axes:
            ax_item.set_visible(visivel)
        ax_rc.set_visible(not visivel and estado['em_render'])
        ax_3d.set_visible(not visivel and not estado['em_render'])
        fig.canvas.draw_idle()

    def on_aplicar_camera(_event):
        try:
            eye_in = [float(v) for v in textboxes[0].text.replace(',', ' ').split()]
            at_in  = [float(v) for v in textboxes[1].text.replace(',', ' ').split()]
            up_in  = [float(v) for v in textboxes[2].text.replace(',', ' ').split()]
            d_in   = float(textboxes[3].text.replace(',', '.'))
            win_in = tuple(float(v) for v in textboxes[4].text.replace(',', ' ').split())
            if len(eye_in) != 3 or len(at_in) != 3 or len(up_in) != 3 or len(win_in) != 4:
                raise ValueError("Quantidade incorreta de coordenadas.")

            estado['cfg_atual'] = {
                'titulo': f'Câmera Manual: Eye={eye_in}, At={at_in}, d={d_in}',
                'eye': eye_in, 'at': at_in, 'up': up_in,
                'd': d_in, 'win': win_in, 'proj': 'perspectiva'
            }
            estado['usou_orbita_livre'] = False
            estado['em_render'] = False
            mostrar_painel(False)
            desenhar_cena_3d_interativa(ax_3d, renderer, estilo_visual=estado['estilo_revit'], cfg_cam=estado['cfg_atual'], modo_noite=estado['modo_noite'])
            ax_3d.set_title(
                f"PREVIEW ({estado['estilo_revit'].upper()}): {estado['cfg_atual']['titulo']}\n"
                "Clique em [RENDER FINAL] para calcular Ray Casting desta câmera",
                fontsize=9.2, color='#0F172A'
            )
            fig.canvas.draw_idle()
        except Exception as erro:
            ax_painel_bg.set_title(f"Erro: {erro}", fontsize=9, color='red')
            fig.canvas.draw_idle()

    btn_aplicar.on_clicked(on_aplicar_camera)
    btn_cancelar.on_clicked(lambda _e: mostrar_painel(False))

    def travar_elevacao_0_90(event=None):
        if estado['em_render'] or estado['modo_painel']:
            return
        if event is not None and event.inaxes == ax_3d and getattr(event, 'button', None) in (1, 3):
            estado['usou_orbita_livre'] = True
        elev_atual = ax_3d.elev
        azim_atual = ax_3d.azim
        elev_travada = float(np.clip(elev_atual, 0.0, 90.0))
        if elev_atual < 0.0 or elev_atual > 90.0:
            ax_3d.elev = elev_travada
            ax_3d.view_init(elev=elev_travada, azim=azim_atual)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', travar_elevacao_0_90)
    fig.canvas.mpl_connect('button_release_event', travar_elevacao_0_90)

    def on_click(event):
        if not estado['em_render'] or estado['modo_painel'] or event.inaxes != ax_rc or event.xdata is None:
            return
        px = int(np.clip(event.xdata, 0, renderer.W - 1))
        py = int(np.clip(event.ydata, 0, renderer.H - 1))
        buf = renderer.ultimo_buffer_pick
        oid = buf['obj_idx'][py, px]
        if oid < 0:
            msg = f"PICK ({px}, {py}): Céu / Fundo"
        else:
            prim = renderer.cena[oid]
            pw = buf['P_world'][py, px]
            if prim.tipo == 'bloco_ct_arqueado':
                mat_nome = renderer.materiais_bloco[buf['sub_mat'][py, px]].nome
            else:
                mat_nome = prim.mat.nome
            msg = (f"PICK ({px},{py}): {prim.nome} | Material: {mat_nome} | "
                   f"Mundo=({pw[0]:.2f}, {pw[1]:.2f}, {pw[2]:.2f})m")
        print(msg)
        ax_rc.set_title(msg, fontsize=9, color='darkblue')
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)

    botoes_topo = [
        ('Linhas (Wire)',  'wireframe',   [0.03, 0.90, 0.14, 0.055], '#F1F5F9'),
        ('Planos (Shade)', 'shaded',      [0.18, 0.90, 0.14, 0.055], '#E2E8F0'),
        ('Materiais',      'materiais',   [0.33, 0.90, 0.13, 0.055], '#DBEAFE'),
        ('Cam Manual',     'cam_manual',  [0.47, 0.90, 0.13, 0.055], '#BAE6FD'),
        ('Modo: DIA',      'dia_noite',   [0.61, 0.90, 0.15, 0.055], '#FEF08A'),
        ('RENDER FINAL',   'render_now',  [0.77, 0.90, 0.20, 0.055], '#86EFAC'),
    ]

    botoes_base = [
        ('1 PF',           '1_ponto_fuga',  [0.04, 0.10, 0.21, 0.055]),
        ('2 PF',           '2_pontos_fuga', [0.27, 0.10, 0.21, 0.055]),
        ('3 PF',           '3_pontos_fuga', [0.50, 0.10, 0.21, 0.055]),
        ('Zoom In',        'zoom_in',       [0.73, 0.10, 0.23, 0.055]),
        ('Zoom Out',       'zoom_out',      [0.04, 0.03, 0.21, 0.055]),
        ('Ortográf.',      'ortografica',   [0.27, 0.03, 0.21, 0.055]),
        ('Oblíqua',        'obliqua',       [0.50, 0.03, 0.21, 0.055]),
        ('Abóbada 3D',     'abobada_3d',    [0.73, 0.03, 0.23, 0.055]),
    ]
    botoes_refs = {}

    def alternar_estilo_revit(novo_estilo):
        def callback(_event):
            if estado['modo_painel']:
                mostrar_painel(False)
            estado['estilo_revit'] = novo_estilo
            estado['em_render'] = False
            ax_rc.set_visible(False)
            ax_3d.set_visible(True)
            desenhar_cena_3d_interativa(ax_3d, renderer, estilo_visual=novo_estilo, cfg_cam=None, modo_noite=estado['modo_noite'])
            turno = "NOITE" if estado['modo_noite'] else "DIA"
            ax_3d.set_title(
                f"ESTILO: {novo_estilo.upper()} [{turno}] (Giro 0° a 90°)\n"
                "Clique em [RENDER FINAL] para calcular Ray Casting e Cáusticas",
                fontsize=9.2, color='#0F172A'
            )
            fig.canvas.draw_idle()
        return callback

    def executar_render_final(_event=None):
        if estado['modo_painel']:
            mostrar_painel(False)

        if not estado['em_render'] and estado['usou_orbita_livre']:
            elev_deg = float(np.clip(ax_3d.elev, 0.0, 90.0))
            azim_deg = float(ax_3d.azim)
            el, az = np.deg2rad(elev_deg), np.deg2rad(azim_deg)
            c_domo, r_domo = renderer.centro_domo, renderer.raio_domo
            eye_domo = [
                c_domo[0] + r_domo * np.cos(el) * np.cos(az),
                max(0.8, c_domo[1] + r_domo * np.sin(el)),
                c_domo[2] + r_domo * np.cos(el) * np.sin(az)
            ]
            cfg_render = {
                'titulo': f'RayCast Abóbada 3D (Az={azim_deg:.0f}°, El={elev_deg:.0f}°)',
                'eye': eye_domo, 'at': c_domo.tolist(), 'up': [0.0, 1.0, 0.0],
                'd': 1.5, 'win': (-0.95, 0.95, -0.95, 0.95),
                'proj': estado['cfg_atual'].get('proj', 'perspectiva')
            }
        else:
            cfg_render = estado['cfg_atual']

        estado['em_render'] = True
        ax_3d.set_visible(False)
        ax_rc.set_visible(True)
        turno_lbl = "NOITE (Postes + Spots Submersos + Cáusticas)" if estado['modo_noite'] else "DIA (Sol + Cáusticas)"
        ax_rc.set_title(f"Renderizando Ray Casting 500x500 — {turno_lbl}...", fontsize=9.5, color='darkred')
        fig.canvas.draw_idle()
        plt.pause(0.05)

        nova_img, novo_tit = renderer.renderizar_config(cfg_render, modo_noite=estado['modo_noite'])
        im_plot.set_data(nova_img)
        ax_rc.set_title(f"RENDERIZADO: {novo_tit}\n(Clique na imagem para PICK | Alterne [Modo: DIA/NOITE] ou estilos acima)", fontsize=8.8, color='black')
        fig.canvas.draw_idle()

    def alternar_dia_noite(_event):
        estado['modo_noite'] = not estado['modo_noite']
        btn_dn = botoes_refs['dia_noite']
        if estado['modo_noite']:
            btn_dn.label.set_text('Modo: NOITE')
            btn_dn.color = '#1E293B'
            btn_dn.hovercolor = '#334155'
            btn_dn.label.set_color('#38BDF8')
        else:
            btn_dn.label.set_text('Modo: DIA')
            btn_dn.color = '#FEF08A'
            btn_dn.hovercolor = '#FDE047'
            btn_dn.label.set_color('#0F172A')

        # Se já estiver na imagem renderizada, re-renderiza imediatamente no novo ciclo Dia/Noite!
        if estado['em_render']:
            executar_render_final()
        else:
            desenhar_cena_3d_interativa(ax_3d, renderer, estilo_visual=estado['estilo_revit'], cfg_cam=None, modo_noite=estado['modo_noite'])
            turno = "NOITE (Luz Artificial + Cáusticas Ativas)" if estado['modo_noite'] else "DIA (Luz Solar)"
            ax_3d.set_title(
                f"CICLO ALTERADO: {turno}\n"
                "Clique em [RENDER FINAL] para visualizar as cáusticas na fachada do Bloco 734",
                fontsize=9.2, color='#0369A1' if estado['modo_noite'] else '#0F172A'
            )
            fig.canvas.draw_idle()

    def selecionar_preset_leve(chave):
        def callback(_event):
            if estado['modo_painel']:
                mostrar_painel(False)
            if chave == 'abobada_3d':
                estado['usou_orbita_livre'] = True
                estado['em_render'] = False
                ax_rc.set_visible(False)
                ax_3d.set_visible(True)
                desenhar_cena_3d_interativa(ax_3d, renderer, estilo_visual=estado['estilo_revit'], cfg_cam=None, modo_noite=estado['modo_noite'])
                ax_3d.set_title(
                    f"ABÓBADA SEMI-ESFÉRICA 3D ({estado['estilo_revit'].upper()}) - Elevação 0° a 90°\n"
                    "Gire ao redor e clique em [RENDER FINAL] para renderizar",
                    fontsize=9.2, color='#0369A1'
                )
                fig.canvas.draw_idle()
                return

            cfg_p = renderer.presets[chave].copy()
            estado['cfg_atual'] = cfg_p
            estado['usou_orbita_livre'] = False
            estado['em_render'] = False

            textboxes[0].set_val(f"{cfg_p['eye'][0]} {cfg_p['eye'][1]} {cfg_p['eye'][2]}")
            textboxes[1].set_val(f"{cfg_p['at'][0]} {cfg_p['at'][1]} {cfg_p['at'][2]}")
            textboxes[2].set_val(f"{cfg_p['up'][0]} {cfg_p['up'][1]} {cfg_p['up'][2]}")
            textboxes[3].set_val(f"{cfg_p['d']}")
            textboxes[4].set_val(f"{cfg_p['win'][0]} {cfg_p['win'][1]} {cfg_p['win'][2]} {cfg_p['win'][3]}")

            ax_rc.set_visible(False)
            ax_3d.set_visible(True)
            desenhar_cena_3d_interativa(ax_3d, renderer, estilo_visual=estado['estilo_revit'], cfg_cam=cfg_p, modo_noite=estado['modo_noite'])
            ax_3d.set_title(
                f"PREVIEW ({estado['estilo_revit'].upper()}): {cfg_p['titulo']}\n"
                "Clique em [RENDER FINAL] para gerar o Ray Casting desta vista",
                fontsize=9.2, color='#0F172A'
            )
            fig.canvas.draw_idle()
        return callback

    for label, chave, rect, cor_b in botoes_topo:
        ax_b = fig.add_axes(rect)
        btn = Button(ax_b, label, color=cor_b)
        btn.label.set_fontsize(8.8)
        if chave in ('wireframe', 'shaded', 'materiais'):
            btn.on_clicked(alternar_estilo_revit(chave))
        elif chave == 'cam_manual':
            btn.on_clicked(lambda _e: mostrar_painel(not estado['modo_painel']))
        elif chave == 'dia_noite':
            btn.on_clicked(alternar_dia_noite)
        elif chave == 'render_now':
            btn.on_clicked(executar_render_final)
        botoes_refs[chave] = btn

    for label, chave, rect in botoes_base:
        ax_b = fig.add_axes(rect)
        btn = Button(ax_b, label)
        btn.on_clicked(selecionar_preset_leve(chave))
        botoes_refs[chave] = btn

    plt.show()

if __name__ == "__main__":
    iniciar_aplicacao()
