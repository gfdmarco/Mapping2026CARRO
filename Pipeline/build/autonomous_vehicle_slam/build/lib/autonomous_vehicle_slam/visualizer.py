import numpy as np
import matplotlib.pyplot as plt

from .pathPlanning import BLUE_ID, YELLOW_ID


class MapVisualizer:
    """Plot ao vivo do mundo: mapa do SLAM, pose do carro, trajetorias.

    Vista de cima no referencial do mundo (x norte, y leste do FSDS).
    Plota com (y, x) para o plot ficar com a orientacao usual de mapa.
    """

    def __init__(self):
        plt.ion()
        # Impede a janela de roubar o foco a cada plt.pause() (backend Qt)
        plt.rcParams['figure.raise_window'] = False
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.history = []

    def update(self, pose, mapper, trajectory=None, local_path=None, mode="",
               particles=None):
        ax = self.ax
        ax.clear()

        self.history.append([pose[0], pose[1]])
        hist = np.array(self.history)
        ax.plot(hist[:, 1], hist[:, 0], color='lightsteelblue', lw=1,
                label='percorrido')

        # nuvem de particulas do FastSLAM (espalhamento = incerteza da pose)
        if particles is not None and len(particles):
            ax.scatter(particles[:, 1], particles[:, 0], c='gray', s=4,
                       alpha=0.5, label='particulas')

        cones = mapper.confirmed()
        if len(cones):
            ids = cones[:, 2].astype(int)
            blue, yellow = ids == BLUE_ID, ids == YELLOW_ID
            other = ~(blue | yellow)
            ax.scatter(cones[blue, 1], cones[blue, 0], c='blue', s=30)
            ax.scatter(cones[yellow, 1], cones[yellow, 0], c='gold', s=30,
                       edgecolors='k', linewidths=0.4)
            if other.any():
                ax.scatter(cones[other, 1], cones[other, 0], c='orange', s=30)

        if trajectory is not None:
            pts, spd = trajectory.points, trajectory.speeds
            sc = ax.scatter(pts[:, 1], pts[:, 0], c=spd, cmap='RdYlGn',
                            s=8, label='trajetoria global')
            if not getattr(self, '_cbar', None):
                self._cbar = self.fig.colorbar(sc, ax=ax, shrink=0.7,
                                               label='v alvo [m/s]')

        # rota local (volta 1), levada do referencial do carro ao mundo
        if local_path is not None and len(local_path):
            x, y, yaw = pose
            c, s = np.cos(yaw), np.sin(yaw)
            lp = np.asarray(local_path)
            wx = x + lp[:, 1] * c - lp[:, 0] * s
            wy = y + lp[:, 1] * s + lp[:, 0] * c
            ax.plot(wy, wx, '-', color='red', lw=1.5, label='rota local')

        # carro: seta na direcao do heading
        ax.scatter([pose[1]], [pose[0]], c='black', marker='o', s=50,
                   zorder=5, label='carro')
        ax.arrow(pose[1], pose[0], 2.5 * np.sin(pose[2]), 2.5 * np.cos(pose[2]),
                 head_width=0.8, color='black', zorder=5)

        ax.set_xlabel('leste [m]')
        ax.set_ylabel('norte [m]')
        ax.set_title(f'modo: {mode}   cones no mapa: {len(cones)}')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)
