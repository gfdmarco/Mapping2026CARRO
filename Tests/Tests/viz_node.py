#!/usr/bin/env python3
"""
viz_node.py — Nó de visualização para FSDS SLAM

Faz três coisas:
  1. Acumula histórico de cones e publica como MarkerArray colorido
  2. Publica o path (trajetória) do carro no frame 'map'
  3. Filtra o chão da nuvem de pontos LiDAR e republica

Tópicos consumidos:
  /fsdsCones      (geometry_msgs/PoseArray)   — cones detectados no frame base_link
  /fsdsLidar3D    (sensor_msgs/PointCloud2)   — nuvem bruta do LiDAR
  /fsdsOdometry   (nav_msgs/Odometry)         — odometria para construir o path

Tópicos publicados:
  /cone_markers           (visualization_msgs/MarkerArray)
  /trajectory             (nav_msgs/Path)
  /fsdsLidar3D_filtered   (sensor_msgs/PointCloud2)
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseArray, PoseStamped
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import struct
import tf2_ros
from tf2_ros import TransformException

# --- Parâmetros de filtragem do chão ---
GROUND_Z_MIN = -0.20   # pontos abaixo disso (em base_link) são chão → ignorar
GROUND_Z_MAX = 0.15    # pontos até aqui também são chão → ignorar

# --- Parâmetros de fusão de cones ---
CONE_MERGE_DIST = 1.5  # metros — cones mais próximos que isso são fundidos

# Cores por classe (RGB 0-1): 0=azul, 1=amarelo, outros=laranja
CONE_COLORS = {
    0: (0.1, 0.4, 1.0),   # azul
    1: (1.0, 0.9, 0.0),   # amarelo
    -1: (1.0, 0.5, 0.0),  # indefinido / laranja
}


class VizNode(Node):
    def __init__(self):
        super().__init__('viz_node')

        # --- Estado interno ---
        self.cone_map: list[dict] = []   # [{x, y, class_id, count}]
        self.path_poses: list[PoseStamped] = []

        # --- Publishers ---
        self.pub_markers = self.create_publisher(MarkerArray, '/cone_markers', 10)
        self.pub_path    = self.create_publisher(Path, '/trajectory', 10)
        self.pub_lidar   = self.create_publisher(PointCloud2, '/fsdsLidar3D_filtered', 10)

        # --- Subscribers ---
        self.create_subscription(PoseArray,   '/fsdsCones',    self.cb_cones,  10)
        self.create_subscription(Odometry,    '/fsdsOdometry', self.cb_odom,   10)
        self.create_subscription(PointCloud2, '/fsdsLidar3D',  self.cb_lidar,  10)

        # --- TF para transformar cones de base_link → map ---
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info('viz_node iniciado.')

    # ------------------------------------------------------------------ #
    #  CONES                                                               #
    # ------------------------------------------------------------------ #
    def cb_cones(self, msg: PoseArray):
        stamp = msg.header.stamp

        # Tenta obter transform base_link → map
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_link', stamp,
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
        except TransformException:
            try:
                tf = self.tf_buffer.lookup_transform(
                    'map', 'base_link',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.05)
                )
            except TransformException as e:
                self.get_logger().warn(f'TF base_link→map indisponível: {e}')
                return

        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        q  = tf.transform.rotation
        yaw = 2.0 * np.arctan2(q.z, q.w)  # ângulo 2D do robô no mapa

        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        for pose in msg.poses:
            # cone em base_link
            lx = pose.position.x
            ly = pose.position.y

            # rotaciona + translada para map
            mx = tx + cos_y * lx - sin_y * ly
            my = ty + sin_y * lx + cos_y * ly

            # class_id embutido na orientação.z (atalho usado em main.py)?
            # Como PoseArray não carrega o class_id diretamente, usamos -1 (indefinido)
            # Para ter cor correta, adaptar main.py para publicar como MarkerArray
            # ou usar tópico separado com cor. Por ora: inferir pela posição relativa.
            class_id = -1

            self._merge_cone(mx, my, class_id)

        self._publish_markers(stamp)

    def _merge_cone(self, x: float, y: float, class_id: int):
        for cone in self.cone_map:
            if np.hypot(x - cone['x'], y - cone['y']) < CONE_MERGE_DIST:
                # Média ponderada da posição
                n = cone['count']
                cone['x'] = (cone['x'] * n + x) / (n + 1)
                cone['y'] = (cone['y'] * n + y) / (n + 1)
                cone['count'] += 1
                if class_id != -1:
                    cone['class_id'] = class_id
                return
        self.cone_map.append({'x': x, 'y': y, 'class_id': class_id, 'count': 1})

    def _publish_markers(self, stamp):
        ma = MarkerArray()

        # Apaga marcadores antigos primeiro
        del_marker = Marker()
        del_marker.action = Marker.DELETEALL
        ma.markers.append(del_marker)

        for i, cone in enumerate(self.cone_map):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = 'cones'
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD

            m.pose.position.x = cone['x']
            m.pose.position.y = cone['y']
            m.pose.position.z = 0.15   # altura do cone
            m.pose.orientation.w = 1.0

            m.scale.x = 0.3
            m.scale.y = 0.3
            m.scale.z = 0.30

            r, g, b = CONE_COLORS.get(cone['class_id'], CONE_COLORS[-1])
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 0.85

            m.lifetime.sec = 0  # persiste para sempre até DELETEALL

            ma.markers.append(m)

        self.pub_markers.publish(ma)

    # ------------------------------------------------------------------ #
    #  TRAJETÓRIA                                                          #
    # ------------------------------------------------------------------ #
    def cb_odom(self, msg: Odometry):
        stamp = msg.header.stamp

        # Tenta obter transform base_link → map
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_link', stamp,
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
        except TransformException:
            try:
                tf = self.tf_buffer.lookup_transform(
                    'map', 'base_link',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.05)
                )
            except TransformException as e:
                self.get_logger().warn(f'TF base_link→map indisponível: {e}')
                return

        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = 'map'

         # 4. Copia os dados de Posição (Translação)
        ps.pose.position.x = tf.transform.translation.x
        ps.pose.position.y = tf.transform.translation.y
        ps.pose.position.z = tf.transform.translation.z
        
        # 5. Copia os dados de Orientação (Rotação em Quaternion)
        ps.pose.orientation.x = tf.transform.rotation.x
        ps.pose.orientation.y = tf.transform.rotation.y
        ps.pose.orientation.z = tf.transform.rotation.z
        ps.pose.orientation.w = tf.transform.rotation.w


        # Evita adicionar poses duplicadas (sem movimento)
        if self.path_poses:
            last = self.path_poses[-1].pose.position
            dx = ps.pose.position.x - last.x
            dy = ps.pose.position.y - last.y
            if np.hypot(dx, dy) < 0.05:
                return

        self.path_poses.append(ps)

        path = Path()
        path.header.stamp    = msg.header.stamp
        path.header.frame_id = 'map'
        path.poses = self.path_poses

        self.pub_path.publish(path)

    # ------------------------------------------------------------------ #
    #  FILTRAGEM DO CHÃO                                                   #
    # ------------------------------------------------------------------ #
    def cb_lidar(self, msg: PointCloud2):
        # Decodifica a nuvem manualmente (sem sensor_msgs_py para evitar deps extra)
        point_step = msg.point_step
        data = msg.data
        n_points = msg.width * msg.height

        xs = np.frombuffer(data, dtype=np.float32)[0::3]
        ys = np.frombuffer(data, dtype=np.float32)[1::3]
        zs = np.frombuffer(data, dtype=np.float32)[2::3]

        # Máscara: mantém pontos FORA da faixa de chão
        mask = ~((zs >= GROUND_Z_MIN) & (zs <= GROUND_Z_MAX))

        xs_f = xs[mask]
        ys_f = ys[mask]
        zs_f = zs[mask]

        if len(xs_f) == 0:
            return

        out = np.column_stack([xs_f, ys_f, zs_f]).astype(np.float32)

        out_msg = PointCloud2()
        out_msg.header = msg.header
        out_msg.height = 1
        out_msg.width  = len(xs_f)
        out_msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        out_msg.is_bigendian = False
        out_msg.point_step   = 12
        out_msg.row_step     = 12 * len(xs_f)
        out_msg.is_dense     = True
        out_msg.data         = out.tobytes()

        self.pub_lidar.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
