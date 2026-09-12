import numpy as np

# Parametros do veiculo (FSCar do FSDS), iguais aos do pure pursuit
WHEELBASE = 1.55             # m
MAX_STEER_DEG = 25.0         # graus de estercamento quando steering = 1.0

# Horizonte de predicao
HORIZON = 14                 # passos
DT = 0.12                    # s por passo  -> ~1.7 s de horizonte
MIN_PLAN_SPEED = 1.5         # m/s usado na discretizacao com o carro lento
                             # (senao o horizonte espacial degenera)

# Pesos do custo quadratico
Q_LAT = 4.0                  # erro lateral [1/m^2]
Q_HEAD = 10.0                # erro de heading [1/rad^2]
TERMINAL_FACTOR = 3.0        # multiplica os pesos no ultimo passo
R_STEER = 2.0                # desvio do steering em relacao ao feedforward
R_RATE = 12.0                # variacao de steering entre passos

MAX_STEER_RATE = np.deg2rad(120.0)  # rad/s nas rodas (limite fisico imposto)


class ModelPredictiveController:
    """MPC lateral por linearizacao no caminho (modelo bicicleta cinematico).

    Mesma interface do PurePursuit: computeSteering(path, speed) com o
    caminho no referencial do carro como [lateral, frente] e retorno do
    comando de steering do FSDS em [-1, 1] (+1 = direita).

    Funcionamento por ciclo:
      1. Reamostra o caminho a frente em N passos de comprimento v*DT e
         extrai heading, curvatura e o steering de feedforward
         (delta_ff = atan(L * kappa)) de cada passo.
      2. Lineariza a dinamica do erro (lateral e_y, heading e_psi) em
         torno dessa referencia -- modelo LTV exato o bastante para o
         horizonte curto.
      3. Resolve o problema quadratico em lote por minimos quadrados
         (numpy, sem solver externo) e aplica apenas o primeiro comando,
         saturado em amplitude e em taxa.

    Em relacao ao pure pursuit, o MPC antecipa as curvas pelo horizonte
    inteiro (entra na curva no ponto certo em vez de reagir tarde) e o
    custo de variacao do steering rejeita o ruido frame a frame da
    percepcao sem precisar de lookahead longo.
    """

    def __init__(self, wheelbase=WHEELBASE, max_steer_deg=MAX_STEER_DEG,
                 horizon=HORIZON, dt=DT):
        self.wheelbase = wheelbase
        self.max_steer_rad = np.deg2rad(max_steer_deg)
        self.horizon = horizon
        self.dt = dt
        self.last_delta = 0.0

    def reset(self):
        self.last_delta = 0.0

    def computeSteering(self, path, speed, dt=None):
        """Comando de steering em [-1, 1]; None se o caminho nao da para usar."""
        ref = self._buildReference(path, speed)
        if ref is None:
            return None
        pts, theta, delta_ff, ds = ref

        n = len(delta_ff)                      # passos de controle
        e_y0, e_psi0 = self._initialError(pts, theta)

        # dinamica do erro discretizada: x = [e_y, e_psi], u = delta - delta_ff
        #   e_y[k+1]   = e_y[k] + ds * e_psi[k]
        #   e_psi[k+1] = e_psi[k] + (ds / L) * u[k]
        A = np.array([[1.0, ds], [0.0, 1.0]])
        B = np.array([[0.0], [ds / self.wheelbase]])

        # predicao em lote: X = Phi @ x0 + Gam @ U  (X empilha x_1..x_n)
        Phi = np.zeros((2 * n, 2))
        Gam = np.zeros((2 * n, n))
        Ak = np.eye(2)
        powers = [np.eye(2)]
        for k in range(n):
            Ak = A @ Ak
            Phi[2 * k:2 * k + 2] = Ak
            powers.append(Ak)
        for k in range(n):          # bloco (k, j) = A^(k-j) @ B para j <= k
            for j in range(k + 1):
                Gam[2 * k:2 * k + 2, j] = (powers[k - j] @ B)[:, 0]

        # pesos dos estados (terminal reforcado)
        w = np.tile([np.sqrt(Q_LAT), np.sqrt(Q_HEAD)], n)
        w[-2:] *= np.sqrt(TERMINAL_FACTOR)

        x0 = np.array([e_y0, e_psi0])

        # custo empilhado como minimos quadrados ||C @ U - b||^2
        rows = []
        rhs = []

        rows.append(w[:, None] * Gam)                  # estados
        rhs.append(-w * (Phi @ x0))

        rows.append(np.sqrt(R_STEER) * np.eye(n))      # desvio do feedforward
        rhs.append(np.zeros(n))

        D = np.eye(n) - np.eye(n, k=-1)                # taxa de variacao
        target = np.zeros(n)
        target[0] = self.last_delta                    # continuidade do comando
        rows.append(np.sqrt(R_RATE) * D)
        rhs.append(np.sqrt(R_RATE) * (target - D @ delta_ff))

        C = np.vstack(rows)
        b = np.concatenate(rhs)
        u, *_ = np.linalg.lstsq(C, b, rcond=None)

        delta = float(u[0] + delta_ff[0])

        # saturacao de taxa (usa o dt real do ciclo quando disponivel)
        step = MAX_STEER_RATE * (dt if dt and dt > 0.0 else self.dt)
        delta = np.clip(delta, self.last_delta - step, self.last_delta + step)
        delta = float(np.clip(delta, -self.max_steer_rad, self.max_steer_rad))
        self.last_delta = delta

        return delta / self.max_steer_rad

    # ---------- construcao da referencia ----------

    def _buildReference(self, path, speed):
        """Reamostra o caminho a frente do carro em passos de v*DT.

        Retorna (pontos (fwd, lat), heading, delta feedforward, ds) ou
        None se o caminho for curto/degenerado demais.
        """
        path = np.asarray(path, dtype=np.float64)
        if path.ndim != 2 or len(path) < 3:
            return None

        # [lateral, frente] -> (frente, lateral); descarta o ponto do carro
        pts = np.column_stack([path[:, 1], path[:, 0]])
        if np.linalg.norm(pts[0]) < 1e-6:
            pts = pts[1:]
        if len(pts) < 2:
            return None

        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = np.concatenate([[True], seg > 1e-6])
        pts = pts[keep]
        if len(pts) < 2:
            return None
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])

        # projeta o carro (origem) na polilinha -> abscissa s0
        s0 = 0.0
        best = np.inf
        for i in range(len(pts) - 1):
            a, d = pts[i], pts[i + 1] - pts[i]
            dd = d @ d
            t = np.clip(-(a @ d) / dd, 0.0, 1.0)
            p = a + t * d
            d2 = p @ p
            if d2 < best:
                best = d2
                s0 = s[i] + t * np.sqrt(dd)

        v = max(abs(speed), MIN_PLAN_SPEED)
        ds = v * self.dt
        si = s0 + ds * np.arange(self.horizon + 1)
        si = si[si <= s[-1]]
        if len(si) < 4:                      # horizonte curto demais
            return None

        fwd = np.interp(si, s, pts[:, 0])
        lat = np.interp(si, s, pts[:, 1])
        rpts = np.column_stack([fwd, lat])

        d = np.diff(rpts, axis=0)
        theta = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))   # + = para a direita

        kappa = np.zeros(len(theta))
        if len(theta) > 1:
            kappa[:-1] = np.diff(theta) / ds
            kappa[-1] = kappa[-2]
        delta_ff = np.clip(np.arctan(self.wheelbase * kappa),
                           -self.max_steer_rad, self.max_steer_rad)

        return rpts, theta, delta_ff, ds

    def _initialError(self, pts, theta):
        """Erro lateral (+ = carro a direita da rota) e de heading em s0."""
        p0, th0 = pts[0], theta[0]
        # normal da rota apontando para +lateral (direita)
        e_y = p0[0] * np.sin(th0) - p0[1] * np.cos(th0)
        e_psi = -np.arctan2(np.sin(th0), np.cos(th0))
        return e_y, e_psi


def _demo():
    """Segue uma pista sintetica em S com o modelo bicicleta (sem o sim).

    Valida sinal do steering, estabilidade e erro de tracking do MPC.
    """
    rng = np.random.default_rng(2)

    # linha central global: S largo, percorrido a 5 m/s
    s = np.linspace(0.0, 120.0, 400)
    cx = s
    cy = 6.0 * np.sin(s / 14.0)
    center = np.column_stack([cx, cy])

    mpc = ModelPredictiveController()
    x, y, yaw, v = 0.0, 1.0, 0.0, 5.0      # comeca 1 m fora da linha
    dt = 1.0 / 60.0
    max_err = 0.0

    for step in range(int(25.0 / dt)):
        # janela local da rota no referencial do carro: [lateral, frente]
        c, sn = np.cos(yaw), np.sin(yaw)
        dx, dy = center[:, 0] - x, center[:, 1] - y
        fwd = dx * c + dy * sn
        lat = dx * sn - dy * c             # + = direita (plano fwd/lat do MPC)
        ahead = (fwd > -2.0) & (fwd < 25.0)
        if not ahead.any():
            break
        path = np.column_stack([lat[ahead], fwd[ahead]])
        path = path[np.argsort(path[:, 1])]
        path = np.vstack([[0.0, 0.0], path])
        path[1:, 0] += rng.normal(0.0, 0.05, len(path) - 1)  # ruido de percepcao

        cmd = mpc.computeSteering(path, v, dt)
        if cmd is None:
            break
        delta = cmd * mpc.max_steer_rad

        # integra a bicicleta cinematica (yaw + = direita, como no MPC)
        yaw -= v / mpc.wheelbase * np.tan(delta) * dt
        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt

        if step * dt > 3.0:                # ignora a convergencia inicial
            err = np.min(np.hypot(center[:, 0] - x, center[:, 1] - y))
            max_err = max(max_err, err)

    print(f"erro lateral maximo apos convergir: {max_err:.2f} m")
    assert max_err < 0.35, "MPC nao esta seguindo a linha"
    print("demo MPC OK")


if __name__ == "__main__":
    _demo()
