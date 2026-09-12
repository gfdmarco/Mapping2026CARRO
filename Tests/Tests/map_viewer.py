#!/usr/bin/env python3
"""
map_viewer.py — Visualizador em tempo real via Matplotlib

Mostra:
  - Histórico de posição do carro (trajetória)
  - Cones detectados acumulados (azul / amarelo / laranja)
  - Nuvem de pontos LiDAR (sem chão)

Uso:
    python3 map_viewer.py
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import PointCloud2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import tf2_ros
from tf2_ros import TransformException

# ── Parâmetros ──────────────────────────────────────────────────────────────
GROUND_Z_MIN = -0.20
GROUND_Z_MAX =  0.15
CONE_MERGE_DIST = 1.5
MAX_PATH_POINTS = 2000   # limita memória do histórico de posição
LIDAR_SUBSAMPLE = 5      # pega 1 a cada N pontos LiDAR (performance)
UPDATE_INTERVAL_MS = 50 # refresh do plot em ms


# ── Nó ROS 2 (roda em thread separada) ──────────────────────────────────────
class MapNode(Node):
    def __init__(self):
        super().__init__('map_viewer')

        # Dados compartilhados (protegidos por lock)
        self.lock = threading.Lock()
        self.path_x   = deque(maxlen=MAX_PATH_POINTS)
        self.path_y   = deque(maxlen=MAX_PATH_POINTS)
        self.cones: list[dict] = []   # [{x, y, class_id}]
        self.lidar_x  = np.array([])
        self.lidar_y  = np.array([])

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(Odometry,    '/fsdsOdometry', self._cb_odom,  10)
        self.create_subscription(PoseArray,   '/fsdsCones',    self._cb_cones, 10)
        self.create_subscription(PointCloud2, '/fsdsLidar3D',  self._cb_lidar, 10)

        self.get_logger().info('map_viewer iniciado — aguardando dados...')

    # ── Odometria → trajetória ───────────────────────────────────────────────
    def _cb_odom(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        with self.lock:
            if self.path_x and abs(x - self.path_x[-1]) < 0.05 and abs(y - self.path_y[-1]) < 0.05:
                return
            self.path_x.append(x)
            self.path_y.append(y)

    # ── Cones → mapa acumulado ───────────────────────────────────────────────
    def _cb_cones(self, msg: PoseArray):
        # Tenta pegar TF base_link → odom - CORRIGIDO PARA MAP por Gabriel(o frame de referência da odometria)
        try:
            tf = self.tf_buffer.lookup_transform(
                'odom', 'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
            tx = tf.transform.translation.x
            ty = tf.transform.translation.y
            q  = tf.transform.rotation
            yaw = 2.0 * np.arctan2(q.z, q.w)
        except TransformException:
            # Fallback: usa posição atual da odometria
            with self.lock:
                if not self.path_x:
                    return
                tx, ty = self.path_x[-1], self.path_y[-1]
            # talvez essa linha esteja atrapalhando o set do referencial: yaw = 0.0

        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        with self.lock:
            for pose in msg.poses:
                lx = pose.position.x
                ly = pose.position.y
                mx = tx + cos_y * lx - sin_y * ly
                my = ty + sin_y * lx + cos_y * ly
                self._merge_cone(mx, my)

    def _merge_cone(self, x, y, class_id=-1):
        for c in self.cones:
            if np.hypot(x - c['x'], y - c['y']) < CONE_MERGE_DIST:
                n = c['n']
                c['x'] = (c['x'] * n + x) / (n + 1)
                c['y'] = (c['y'] * n + y) / (n + 1)
                c['n'] += 1
                return
        self.cones.append({'x': x, 'y': y, 'class_id': class_id, 'n': 1})

    # ── LiDAR → nuvem filtrada ───────────────────────────────────────────────
    def _cb_lidar(self, msg: PointCloud2):
        data = np.frombuffer(msg.data, dtype=np.float32)
        if data.size < 3:
            return
        pts = data.reshape(-1, 3)
        mask = ~((pts[:, 2] >= GROUND_Z_MIN) & (pts[:, 2] <= GROUND_Z_MAX))
        pts = pts[mask][::LIDAR_SUBSAMPLE]
        with self.lock:
            self.lidar_x = pts[:, 0].copy()
            self.lidar_y = pts[:, 1].copy()


# ── Plot em tempo real ───────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = MapNode()

    # Roda o spin do ROS em thread separada
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Configura o matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor('#16213e')

    line_path,   = ax.plot([], [], '-', color='#e94560',  lw=3.0,  label='Trajetória', zorder=3)
    scat_lidar   = ax.scatter([], [], s=1,  c='#4fc3f7',  alpha=0.5, label='LiDAR',    zorder=2)
    scat_cones   = ax.scatter([], [], s=80, marker='^',   zorder=4, label='Cones')
    dot_car,     = ax.plot([], [], 'o', color='white', ms=8, zorder=5, label='Carro')

    ax.set_xlabel('X (m)', color='white')
    ax.set_ylabel('Y (m)', color='white')
    ax.set_title('FSDS SLAM — Mapa em Tempo Real', color='white', fontsize=13)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')
    ax.grid(True, color='#333', linewidth=0.5)
    legend = ax.legend(loc='upper left', facecolor='#1a1a2e', labelcolor='white', fontsize=9)
    ax.set_aspect('equal')

    def update(_):
        with node.lock:
            px = list(node.path_x)
            py = list(node.path_y)
            cones = list(node.cones)
            lx = node.lidar_x.copy()
            ly = node.lidar_y.copy()

        # Trajetória
        line_path.set_data(px, py)

        # Ponto atual do carro
        if px:
            dot_car.set_data([px[-1]], [py[-1]])

        # LiDAR — transforma de base_link para odom (translação simples)
        if lx.size > 0 and px:
            cx, cy = px[-1], py[-1]
            scat_lidar.set_offsets(np.c_[lx + cx, ly + cy])
        else:
            scat_lidar.set_offsets(np.empty((0, 2)))

        # Cones
        if cones:
            cx_arr = np.array([c['x'] for c in cones])
            cy_arr = np.array([c['y'] for c in cones])
            # Cor por contagem (proxy para confiança): azul se poucas vezes, amarelo se muitas
            # Como class_id = -1 por ora, usa contagem como heurística de cor
            counts = np.array([c['n'] for c in cones], dtype=float)
            colors = ['#2979ff' if n < 3 else '#ffd600' for n in counts]
            scat_cones.set_offsets(np.c_[cx_arr, cy_arr])
            scat_cones.set_color(colors)
        else:
            scat_cones.set_offsets(np.empty((0, 2)))

        # Ajusta zoom automaticamente
        all_x = px + (lx.tolist() if lx.size > 0 and px else [])
        all_y = py + (ly.tolist() if ly.size > 0 and px else [])
        if all_x:
            pad = 5
            ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
            ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

        return line_path, scat_lidar, scat_cones, dot_car

    ani = animation.FuncAnimation(
        fig, update,
        interval=UPDATE_INTERVAL_MS,
        blit=False,
        cache_frame_data=False
    )

    plt.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
