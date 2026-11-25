import math
import numpy as np


# ==============================================================================
# ÁLGEBRA LINEAR, QUATÉRNIOS E MATRIZES DE TRANSFORMAÇÃO 4x4 (ATO II)
# ==============================================================================
def mat_identidade():
    return np.eye(4, dtype=np.float64)


def mat_translacao(tx, ty, tz):
    M = np.eye(4, dtype=np.float64)
    M[0, 3] = tx
    M[1, 3] = ty
    M[2, 3] = tz
    return M


def mat_escala(sx, sy, sz):
    M = np.eye(4, dtype=np.float64)
    M[0, 0] = sx
    M[1, 1] = sy
    M[2, 2] = sz
    return M


def mat_rotacao_x(rad):
    c, s = math.cos(rad), math.sin(rad)
    M = np.eye(4, dtype=np.float64)
    M[1, 1], M[1, 2] = c, -s
    M[2, 1], M[2, 2] = s, c
    return M


def mat_rotacao_y(rad):
    c, s = math.cos(rad), math.sin(rad)
    M = np.eye(4, dtype=np.float64)
    M[0, 0], M[0, 2] = c, s
    M[2, 0], M[2, 2] = -s, c
    return M


def mat_rotacao_z(rad):
    c, s = math.cos(rad), math.sin(rad)
    M = np.eye(4, dtype=np.float64)
    M[0, 0], M[0, 1] = c, -s
    M[1, 0], M[1, 1] = s, c
    return M


def quat_para_matriz(eixo, angulo_rad):
    """Sub-quest 5.2 (Caminho B): Rotação em eixo arbitrário via Quatérnios unitários."""
    ax = np.asarray(eixo, dtype=np.float64)
    norm = np.linalg.norm(ax)
    if norm < 1e-12:
        return mat_identidade()
    ux, uy, uz = ax / norm
    half = 0.5 * angulo_rad
    w = math.cos(half)
    s = math.sin(half)
    x, y, z = ux * s, uy * s, uz * s

    M = np.eye(4, dtype=np.float64)
    M[0, 0] = 1.0 - 2.0 * (y * y + z * z)
    M[0, 1] = 2.0 * (x * y - w * z)
    M[0, 2] = 2.0 * (x * z + w * y)
    M[1, 0] = 2.0 * (x * y + w * z)
    M[1, 1] = 1.0 - 2.0 * (x * x + z * z)
    M[1, 2] = 2.0 * (y * z - w * x)
    M[2, 0] = 2.0 * (x * z - w * y)
    M[2, 1] = 2.0 * (y * z + w * x)
    M[2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return M


def mat_cisalhamento_xy(sh_xy=0.15, sh_zy=0.0):
    """Quest Bônus 6.1: Matriz de Cisalhamento (Shear)."""
    M = np.eye(4, dtype=np.float64)
    M[1, 0] = sh_xy
    M[1, 2] = sh_zy
    return M


def mat_espelho_plano(ponto_plano, normal_plano):
    """Quest Bônus 6.2: Matriz de Reflexão Especular em Plano Arbitrário."""
    p0 = np.asarray(ponto_plano, dtype=np.float64)
    n = np.asarray(normal_plano, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    d = -np.dot(n, p0)

    H = np.eye(4, dtype=np.float64)
    H[:3, :3] -= 2.0 * np.outer(n, n)
    H[:3, 3] = -2.0 * d * n
    return H


def aplicar_matriz_pontos(M, pts):
    """Aplica matriz 4x4 a um array (N, 3) de pontos."""
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
    res = (M @ pts_h.T).T
    return res[:, :3]


def construir_Mwc(eye, at, up):
    """Ato IV (Quests 8 e 9): Matriz Mundo -> Câmera (M_wc)."""
    eye = np.asarray(eye, dtype=np.float64)
    at = np.asarray(at, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    zc = eye - at
    zc /= np.linalg.norm(zc) + 1e-12

    xc = np.cross(up, zc)
    if np.linalg.norm(xc) < 1e-8:
        up_alt = np.array([0.0, 0.0, 1.0])
        xc = np.cross(up_alt, zc)
    xc /= np.linalg.norm(xc) + 1e-12

    yc = np.cross(zc, xc)

    M = np.eye(4, dtype=np.float64)
    M[0, :3] = xc
    M[1, :3] = yc
    M[2, :3] = zc
    M[0, 3] = -np.dot(xc, eye)
    M[1, 3] = -np.dot(yc, eye)
    M[2, 3] = -np.dot(zc, eye)
    return M


def construir_Mcw(eye, at, up):
    """Matriz inversa Câmera -> Mundo (M_cw)."""
    Mwc = construir_Mwc(eye, at, up)
    Mcw = np.eye(4, dtype=np.float64)
    R = Mwc[:3, :3]
    t = Mwc[:3, 3]
    Mcw[:3, :3] = R.T
    Mcw[:3, 3] = -R.T @ t
    return Mcw


# ==============================================================================
# MATERIAIS E TEXTURAS PROCEDURAIS (ATO I - QUEST 3.1 & 3.2)
# ==============================================================================
class Material:
    def __init__(
        self,
        nome,
        kd,
        ka=None,
        ke=(0.25, 0.25, 0.25),
        shininess=24.0,
        refletividade=0.0,
        textura=None,
        emissivo=(0.0, 0.0, 0.0),
    ):
        self.nome = nome
        self.kd = np.asarray(kd, dtype=np.float64)
        self.ka = (
            np.asarray(ka, dtype=np.float64) if ka is not None else self.kd * 0.45
        )
        self.ke = np.asarray(ke, dtype=np.float64)
        self.shininess = float(shininess)
        self.refletividade = float(refletividade)
        self.textura = textura
        self.emissivo = np.asarray(emissivo, dtype=np.float64)


def criar_textura_placa_734():
    """Gera bitmap procedural da placa '734' sem dependências externas."""
    H, W = 64, 160
    tex = np.ones((H, W, 3), dtype=np.float64) * 0.96
    tex[:4, :] = [0.10, 0.14, 0.22]
    tex[-4:, :] = [0.10, 0.14, 0.22]
    tex[:, :4] = [0.10, 0.14, 0.22]
    tex[:, -4:] = [0.10, 0.14, 0.22]

    cor_num = np.array([0.07, 0.10, 0.18])
    # Dígito '7'
    tex[14:21, 22:52] = cor_num
    for r in range(21, 52):
        c = int(50 - (r - 21) * 0.72)
        tex[r, c - 4 : c + 3] = cor_num
    # Dígito '3'
    tex[14:20, 66:96] = cor_num
    tex[30:36, 72:96] = cor_num
    tex[46:52, 66:96] = cor_num
    tex[14:52, 90:96] = cor_num
    # Dígito '4'
    tex[14:36, 110:116] = cor_num
    tex[30:36, 110:142] = cor_num
    tex[14:52, 132:138] = cor_num
    return tex


# ==============================================================================
# PRIMITIVAS ANALÍTICAS PARA RAY CASTING VETORIZADO (ATO I - QUEST 2)
# ==============================================================================
class Primitiva:
    def __init__(
        self,
        pid,
        nome,
        tipo,
        grupo,
        material,
        transformacao_txt,
        descricao,
        recebe_caustica=False,
    ):
        self.pid = pid
        self.nome = nome
        self.tipo = tipo
        self.grupo = grupo
        self.material = material
        self.transformacao_txt = transformacao_txt
        self.descricao = descricao
        self.recebe_caustica = recebe_caustica


class CaixaAABB(Primitiva):
    def __init__(
        self,
        pid,
        nome,
        grupo,
        bmin,
        bmax,
        material,
        transformacao_txt="T(x,y,z) · S(sx,sy,sz)",
        descricao="",
        recebe_caustica=False,
    ):
        super().__init__(
            pid, nome, "Caixa / Prisma", grupo, material,
            transformacao_txt, descricao, recebe_caustica,
        )
        self.bmin = np.asarray(bmin, dtype=np.float64)
        self.bmax = np.asarray(bmax, dtype=np.float64)

    def intersectar(self, O, D):
        invD = np.where(np.abs(D) > 1e-9, 1.0 / D, 1e9)
        t0 = (self.bmin - O) * invD
        t1 = (self.bmax - O) * invD
        tmin = np.minimum(t0, t1)
        tmax = np.maximum(t0, t1)

        t_enter = np.max(tmin, axis=1)
        t_exit = np.min(tmax, axis=1)

        hit = (t_exit >= np.maximum(t_enter, 1e-4)) & (t_enter > 1e-4)
        t_hit = np.where(hit, t_enter, np.inf)

        N = np.zeros_like(D)
        if np.any(hit):
            P = O[hit] + t_hit[hit, None] * D[hit]
            eps = 1e-4
            nh = np.zeros_like(P)
            nh[np.abs(P[:, 0] - self.bmin[0]) < eps, 0] = -1.0
            nh[np.abs(P[:, 0] - self.bmax[0]) < eps, 0] = 1.0
            nh[np.abs(P[:, 1] - self.bmin[1]) < eps, 1] = -1.0
            nh[np.abs(P[:, 1] - self.bmax[1]) < eps, 1] = 1.0
            nh[np.abs(P[:, 2] - self.bmin[2]) < eps, 2] = -1.0
            nh[np.abs(P[:, 2] - self.bmax[2]) < eps, 2] = 1.0
            norm = np.linalg.norm(nh, axis=1, keepdims=True)
            nh = np.where(norm > 1e-8, nh / norm, np.array([0.0, 1.0, 0.0]))
            N[hit] = nh
        return t_hit, N


class Esfera(Primitiva):
    def __init__(
        self,
        pid,
        nome,
        grupo,
        centro,
        raio,
        material,
        transformacao_txt="T(cx,cy,cz) · Q(eixo, θ) · S(r)",
        descricao="",
    ):
        super().__init__(
            pid, nome, "Esfera", grupo, material, transformacao_txt, descricao
        )
        self.c = np.asarray(centro, dtype=np.float64)
        self.r = float(raio)

    def intersectar(self, O, D):
        oc = O - self.c
        b = np.sum(oc * D, axis=1)
        c = np.sum(oc * oc, axis=1) - self.r * self.r
        disc = b * b - c
        valid = disc >= 0.0
        sqrt_d = np.sqrt(np.maximum(disc, 0.0))
        t1 = -b - sqrt_d
        t2 = -b + sqrt_d
        t_hit = np.where(
            valid & (t1 > 1e-4), t1,
            np.where(valid & (t2 > 1e-4), t2, np.inf),
        )
        hit = np.isfinite(t_hit)
        N = np.zeros_like(D)
        if np.any(hit):
            P = O[hit] + t_hit[hit, None] * D[hit]
            N[hit] = (P - self.c) / self.r
        return t_hit, N


class CilindroY(Primitiva):
    def __init__(
        self,
        pid,
        nome,
        grupo,
        cx,
        cz,
        y0,
        y1,
        raio,
        material,
        transformacao_txt="T(cx,y,cz) · S(r,h,r)",
        descricao="",
    ):
        super().__init__(
            pid, nome, "Cilindro", grupo, material, transformacao_txt, descricao
        )
        self.cx = float(cx)
        self.cz = float(cz)
        self.y0 = float(y0)
        self.y1 = float(y1)
        self.r = float(raio)

    def intersectar(self, O, D):
        ox = O[:, 0] - self.cx
        oz = O[:, 2] - self.cz
        dx = D[:, 0]
        dz = D[:, 2]

        a = dx * dx + dz * dz
        b = 2.0 * (ox * dx + oz * dz)
        c = ox * ox + oz * oz - self.r * self.r
        disc = b * b - 4.0 * a * c

        valid = (disc >= 0.0) & (a > 1e-10)
        sqrt_d = np.sqrt(np.maximum(disc, 0.0))
        inv_2a = np.where(a > 1e-10, 0.5 / a, 0.0)

        t_hit = np.full(O.shape[0], np.inf, dtype=np.float64)
        N = np.zeros_like(D)

        for cand in [(-b - sqrt_d) * inv_2a, (-b + sqrt_d) * inv_2a]:
            y_cand = O[:, 1] + cand * D[:, 1]
            ok = (
                valid
                & (cand > 1e-4)
                & (cand < t_hit)
                & (y_cand >= self.y0)
                & (y_cand <= self.y1)
            )
            if np.any(ok):
                t_hit[ok] = cand[ok]
                px = O[ok, 0] + cand[ok] * D[ok, 0] - self.cx
                pz = O[ok, 2] + cand[ok] * D[ok, 2] - self.cz
                N[ok, 0] = px / self.r
                N[ok, 1] = 0.0
                N[ok, 2] = pz / self.r

        dy_ok = np.abs(D[:, 1]) > 1e-8
        for y_cap, ny in [(self.y1, 1.0), (self.y0, -1.0)]:
            tc = np.where(dy_ok, (y_cap - O[:, 1]) / D[:, 1], np.inf)
            px = O[:, 0] + tc * D[:, 0] - self.cx
            pz = O[:, 2] + tc * D[:, 2] - self.cz
            ok_c = (
                dy_ok
                & (tc > 1e-4)
                & (tc < t_hit)
                & (px * px + pz * pz <= self.r * self.r)
            )
            if np.any(ok_c):
                t_hit[ok_c] = tc[ok_c]
                N[ok_c] = [0.0, ny, 0.0]
        return t_hit, N


class ArcoBloco734(Primitiva):
    """Parede arqueada tangencial (R=22m, centro=(37,42), quadrante X<=37, Z<=42)."""

    def __init__(
        self,
        pid,
        nome,
        grupo,
        cx,
        cz,
        y0,
        y1,
        raio,
        material,
        transformacao_txt="T(37,0,42) · Cilindro Parcial R=22m",
        descricao="",
    ):
        super().__init__(
            pid, nome, "Cilindro Parcial (Arco R=22m)",
            grupo, material, transformacao_txt, descricao,
            recebe_caustica=True,
        )
        self.cx = float(cx)
        self.cz = float(cz)
        self.y0 = float(y0)
        self.y1 = float(y1)
        self.r = float(raio)

    def intersectar(self, O, D):
        ox = O[:, 0] - self.cx
        oz = O[:, 2] - self.cz
        dx = D[:, 0]
        dz = D[:, 2]

        a = dx * dx + dz * dz
        b = 2.0 * (ox * dx + oz * dz)
        c = ox * ox + oz * oz - self.r * self.r
        disc = b * b - 4.0 * a * c
        valid = (disc >= 0.0) & (a > 1e-10)
        sqrt_d = np.sqrt(np.maximum(disc, 0.0))
        inv_2a = np.where(a > 1e-10, 0.5 / a, 0.0)

        t_hit = np.full(O.shape[0], np.inf, dtype=np.float64)
        N = np.zeros_like(D)

        for cand in [(-b - sqrt_d) * inv_2a, (-b + sqrt_d) * inv_2a]:
            px = O[:, 0] + cand * D[:, 0]
            py = O[:, 1] + cand * D[:, 1]
            pz = O[:, 2] + cand * D[:, 2]
            ok = (
                valid
                & (cand > 1e-4)
                & (cand < t_hit)
                & (py >= self.y0)
                & (py <= self.y1)
                & (px <= self.cx + 0.05)
                & (pz <= self.cz + 0.05)
            )
            if np.any(ok):
                t_hit[ok] = cand[ok]
                N[ok, 0] = (px[ok] - self.cx) / self.r
                N[ok, 1] = 0.0
                N[ok, 2] = (pz[ok] - self.cz) / self.r

        dy_ok = np.abs(D[:, 1]) > 1e-8
        tc = np.where(dy_ok, (self.y1 - O[:, 1]) / D[:, 1], np.inf)
        px = O[:, 0] + tc * D[:, 0]
        pz = O[:, 2] + tc * D[:, 2]
        r2 = (px - self.cx) ** 2 + (pz - self.cz) ** 2
        ok_teto = (
            dy_ok
            & (tc > 1e-4)
            & (tc < t_hit)
            & (r2 <= self.r * self.r)
            & (px <= self.cx + 0.05)
            & (pz <= self.cz + 0.05)
        )
        if np.any(ok_teto):
            t_hit[ok_teto] = tc[ok_teto]
            N[ok_teto] = [0.0, 1.0, 0.0]
        return t_hit, N


class ConeY(Primitiva):
    def __init__(
        self,
        pid,
        nome,
        grupo,
        cx,
        cz,
        y_base,
        altura,
        raio_base,
        material,
        transformacao_txt="T(cx,y,cz) · S(r,h,r)",
        descricao="",
    ):
        super().__init__(
            pid, nome, "Cone", grupo, material, transformacao_txt, descricao
        )
        self.cx = float(cx)
        self.cz = float(cz)
        self.y0 = float(y_base)
        self.h = float(altura)
        self.y1 = self.y0 + self.h
        self.r = float(raio_base)
        self.k = (self.r / self.h) ** 2

    def intersectar(self, O, D):
        ox = O[:, 0] - self.cx
        oz = O[:, 2] - self.cz
        oy = O[:, 1] - self.y1

        dx, dy, dz = D[:, 0], D[:, 1], D[:, 2]
        a = dx * dx + dz * dz - self.k * dy * dy
        b = 2.0 * (ox * dx + oz * dz - self.k * oy * dy)
        c = ox * ox + oz * oz - self.k * oy * oy

        disc = b * b - 4.0 * a * c
        valid = (disc >= 0.0) & (np.abs(a) > 1e-10)
        sqrt_d = np.sqrt(np.maximum(disc, 0.0))
        inv_2a = np.where(np.abs(a) > 1e-10, 0.5 / a, 0.0)

        t_hit = np.full(O.shape[0], np.inf, dtype=np.float64)
        N = np.zeros_like(D)

        for cand in [(-b - sqrt_d) * inv_2a, (-b + sqrt_d) * inv_2a]:
            py = O[:, 1] + cand * D[:, 1]
            ok = (
                valid
                & (cand > 1e-4)
                & (cand < t_hit)
                & (py >= self.y0)
                & (py <= self.y1)
            )
            if np.any(ok):
                t_hit[ok] = cand[ok]
                px = O[ok, 0] + cand[ok] * D[ok, 0] - self.cx
                pz = O[ok, 2] + cand[ok] * D[ok, 2] - self.cz
                ry = np.sqrt(px * px + pz * pz) + 1e-9
                nx = px / ry
                nz = pz / ry
                ny = self.r / self.h
                nh = np.stack([nx, np.full_like(nx, ny), nz], axis=1)
                nh /= np.linalg.norm(nh, axis=1, keepdims=True) + 1e-9
                N[ok] = nh

        dy_ok = np.abs(D[:, 1]) > 1e-8
        tc = np.where(dy_ok, (self.y0 - O[:, 1]) / D[:, 1], np.inf)
        px = O[:, 0] + tc * D[:, 0] - self.cx
        pz = O[:, 2] + tc * D[:, 2] - self.cz
        ok_b = (
            dy_ok
            & (tc > 1e-4)
            & (tc < t_hit)
            & (px * px + pz * pz <= self.r * self.r)
        )
        if np.any(ok_b):
            t_hit[ok_b] = tc[ok_b]
            N[ok_b] = [0.0, -1.0, 0.0]
        return t_hit, N


class MalhaTriangulos(Primitiva):
    """Malha de triângulos com interseção Möller–Trumbore vetorizada em lote."""

    def __init__(
        self,
        pid,
        nome,
        grupo,
        triangulos,
        material,
        transformacao_txt="Malha Poligonal Transformada (M 4x4)",
        descricao="",
    ):
        super().__init__(
            pid, nome, "Malha Triangular", grupo, material,
            transformacao_txt, descricao,
        )
        self.tris = np.asarray(triangulos, dtype=np.float64)
        e1 = self.tris[:, 1] - self.tris[:, 0]
        e2 = self.tris[:, 2] - self.tris[:, 0]
        fn = np.cross(e1, e2)
        norms = np.linalg.norm(fn, axis=1, keepdims=True)
        self.normals = np.where(norms > 1e-9, fn / norms, np.array([0.0, 1.0, 0.0]))

        all_v = self.tris.reshape(-1, 3)
        self.bmin = np.min(all_v, axis=0) - 0.05
        self.bmax = np.max(all_v, axis=0) + 0.05

    def intersectar(self, O, D):
        N_rays = O.shape[0]
        t_hit = np.full(N_rays, np.inf, dtype=np.float64)
        N = np.zeros_like(D)

        invD = np.where(np.abs(D) > 1e-9, 1.0 / D, 1e9)
        t0 = (self.bmin - O) * invD
        t1 = (self.bmax - O) * invD
        t_enter = np.max(np.minimum(t0, t1), axis=1)
        t_exit = np.min(np.maximum(t0, t1), axis=1)
        mask_box = (t_exit >= np.maximum(t_enter, 1e-4)) & (t_enter > 1e-4)
        if not np.any(mask_box):
            return t_hit, N

        idx_box = np.where(mask_box)[0]
        Ob = O[idx_box]
        Db = D[idx_box]
        t_enter_b = t_enter[idx_box]

        for ti in range(self.tris.shape[0]):
            v0, v1, v2 = self.tris[ti]
            e1 = v1 - v0
            e2 = v2 - v0
            h = np.cross(Db, e2)
            a = np.sum(h * e1, axis=1)
            valid = np.abs(a) > 1e-8
            f = np.where(valid, 1.0 / a, 0.0)
            s = Ob - v0
            u = f * np.sum(s * h, axis=1)
            valid &= (u >= 0.0) & (u <= 1.0)
            q = np.cross(s, e1)
            v = f * np.sum(Db * q, axis=1)
            valid &= (v >= 0.0) & (u + v <= 1.0)
            t = f * np.sum(e2 * q, axis=1)
            valid &= (t > 1e-4) & (t >= t_enter_b) & (t < t_hit[idx_box])

            if np.any(valid):
                sub = idx_box[valid]
                t_hit[sub] = t[valid]
                fn = self.normals[ti]
                flip = np.sum(Db[valid] * fn, axis=1) > 0
                N[sub] = np.where(flip[:, None], -fn, fn)
        return t_hit, N


def gerar_caixa_triangulos(w, h, d, M):
    """Constrói uma caixa centrada na origem transformada pela matriz 4x4 M."""
    hw, hh, hd = w * 0.5, h * 0.5, d * 0.5
    v = np.array(
        [
            [-hw, -hh, -hd],
            [hw, -hh, -hd],
            [hw, hh, -hd],
            [-hw, hh, -hd],
            [-hw, -hh, hd],
            [hw, -hh, hd],
            [hw, hh, hd],
            [-hw, hh, hd],
        ],
        dtype=np.float64,
    )
    vt = aplicar_matriz_pontos(M, v)
    faces = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
        (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5),
    ]
    return [np.array([vt[a], vt[b], vt[c]]) for (a, b, c) in faces]


# ==============================================================================
# RENDERIZADOR PRINCIPAL — BLOCO 734 (CT/UFC)
# ==============================================================================
class RenderizadorCT:
    # Constantes do arco (usadas no mapeamento UV da placa 734)
    ARCO_CX = 37.0
    ARCO_CZ = 42.0
    ARCO_RAIO_PLACA = 22.08
    ARCO_ANG_PLACA = 0.14 * (math.pi / 2.0)
    PLACA_MEIA_LARGURA = 1.55
    PLACA_Y_INI = 5.20
    PLACA_Y_FIM = 6.25

    def __init__(self, largura=500, altura=500):
        self.W = int(largura)
        self.H = int(altura)
        self.placa_tex = criar_textura_placa_734()
        self.materiais = self._criar_materiais()
        self.primitivas = self._construir_cena_bloco734()
        self.presets = self._definir_presets()
        self.pick_buffer = None

        # Cache das posições 3D da placa para o _avaliar_textura
        self._placa_z_ini = (
            self.ARCO_CZ
            - self.ARCO_RAIO_PLACA * math.sin(self.ARCO_ANG_PLACA)
            - self.PLACA_MEIA_LARGURA
        )
        self._placa_z_fim = self._placa_z_ini + 2.0 * self.PLACA_MEIA_LARGURA
        self._placa_z_span = max(self._placa_z_fim - self._placa_z_ini, 1e-9)
        self._placa_y_span = max(self.PLACA_Y_FIM - self.PLACA_Y_INI, 1e-9)

    def _criar_materiais(self):
        return {
            "asfalto": Material(
                "Asfalto CBUQ & Faixa de Pedestres (Textura Procedural)",
                kd=(0.16, 0.17, 0.20), ke=(0.12, 0.12, 0.14),
                shininess=12.0, textura="asfalto",
            ),
            "calcada": Material(
                "Calçada Intertravada (Textura Procedural)",
                kd=(0.68, 0.67, 0.64), ke=(0.15, 0.15, 0.15),
                shininess=10.0, textura="calcada",
            ),
           
