import os
import sys
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, PoseArray, Pose
from sensor_msgs.msg import PointCloud2, PointField, Imu
from cartographer_ros_msgs.msg import LandmarkList, LandmarkEntry
import numpy as np
import tf2_ros
import cv2
from ultralytics import YOLO
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import struct

# --- Configuração do FSDS ---
fsds_lib_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
sys.path.insert(0, fsds_lib_path)
import fsds

# --- Constantes ---
ODOM_FREQUENCY = 60
PERCEPTION_FREQUENCY = 10 

FX = 448.14
CX = 640.0
CY = 360.0
CONF_THRESHOLD = 0.85
ANGLE_DEGREE = 2.5
MODEL_PATH = "/home/felipe_capovilla/Documents/E-Racing/Perception/Modelos/16_01.pt"

# ==========================================
# CLASSE DE PERCEPÇÃO
# ==========================================
class Perception:
    def __init__(self, client):
        try:
            self.model = YOLO(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"YOLO failed to start.\n{e}")

        # Cliente exclusivo da percepção
        self.client = client
    
    def lidarMatch(self, points, u, v):
        ray2d = np.array([1.0, (u - CX) / FX])
        ray2d /= np.linalg.norm(ray2d)

        points2d = points[:, :2] 
        norms = np.linalg.norm(points2d, axis=1)
        
        validMask = norms > 0.1
        validPoints = points[validMask]
        validPoints2D = points2d[validMask]
        validNorms = norms[validMask]

        if validPoints.size == 0: return None

        dirs2D = validPoints2D / validNorms[:, None]
        cossAngles = dirs2D @ ray2d
        mask = cossAngles > np.cos(np.deg2rad(ANGLE_DEGREE))
        candidates = validPoints[mask]

        if candidates.size == 0: return None

        return candidates[np.argmin(np.linalg.norm(candidates[:, :2], axis=1))]
    
    def detectCones(self, img):
        # Transforma os dados de imagem extraindo uma cópia segura
        img_array = np.frombuffer(img.image_data_uint8, dtype=np.uint8).copy()
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None: 
            return np.array([]), np.array([])

        # Puxa o Lidar usando o cliente exclusivo da percepção
        lidarData = self.client.getLidarData(lidar_name='Lidar1')
        points = np.array(lidarData.point_cloud, dtype=np.float32).reshape((-1, 3))

        # Inferência
        results = self.model(image, verbose=False, conf=CONF_THRESHOLD)
        rawDetection = []

        for box in results[0].boxes:
            u, v = float(box.xywh[0][0]), float(box.xywh[0][1])
            classId = int(box.cls[0])
            hit = self.lidarMatch(points, u, v)

            if hit is not None:
                x_frente = float(hit[0])
                y_lateral = (u - CX) * x_frente / FX

                if 0.5 < x_frente < 35.0:
                    rawDetection.append([y_lateral, x_frente, classId])
        
        finalCones = []
        for cone in rawDetection:
            if not any(np.linalg.norm(np.array(cone[:2]) - np.array(f[:2])) < 1.2 for f in finalCones):
                finalCones.append(cone)

        return np.array(finalCones), points

# ==========================================
# NÓ PRINCIPAL ROS 2
# ==========================================
class fsdsNode(Node):
    def __init__(self):
        super().__init__('fsdsNode')

        try:
            # 1. Cria cliente exclusivo para ODOMETRIA e IMU (Rápido)
            self.client_odom = fsds.FSDSClient()
            self.client_odom.confirmConnection()
            
            # 2. Cria cliente exclusivo para PERCEPÇÃO e LIDAR3D (Lento)
            self.client_perc = fsds.FSDSClient()
            self.client_perc.confirmConnection()
            
            self.get_logger().info('Conectado ao FSDS com sucesso (Múltiplos Clientes)!')
        except Exception as e:
            raise RuntimeError(f'Unable to communicate with simulator {e}')
        
        # Inicializa a Percepção injetando o cliente correto
        self.get_logger().info('Carregando modelo YOLO...')
        self.perception = Perception(self.client_perc)
        self.get_logger().info('Modelo YOLO carregado!')

        # Publishers Existentes
        self.odom_publisher = self.create_publisher(Odometry, 'fsdsOdometry', 10)
        self.cones_publisher = self.create_publisher(PoseArray, 'fsdsCones', 10)
        
        # NOVOS Publishers para o Cartographer
        self.lidar_publisher = self.create_publisher(PointCloud2, 'fsdsLidar3D', 10)
        self.imu_publisher = self.create_publisher(Imu, 'fsdsImu', 10)
        self.landmark_publisher = self.create_publisher(LandmarkList, 'fsdsLandmarks', 10)

        # Callback Groups para MultiThreading seguro
        self.odom_cb_group = MutuallyExclusiveCallbackGroup()
        self.perc_cb_group = MutuallyExclusiveCallbackGroup()

        # Timers
        self.create_timer(1.0 / ODOM_FREQUENCY, self.odometry_process, callback_group=self.odom_cb_group)
        self.create_timer(1.0 / PERCEPTION_FREQUENCY, self.perception_process, callback_group=self.perc_cb_group)

    # ---------------------------------------------------------
    # PROCESSO DA ODOMETRIA E IMU (Usa self.client_odom)
    # ---------------------------------------------------------
    def odometry_process(self):
        try:
            car_state = self.client_odom.getCarState()
            kinematics = car_state.kinematics_estimated
        except Exception as e:
            self.get_logger().warn(f'Falha ao ler telemetria do simulador: {e}')
            return

        tempo_atual = self.get_clock().now().to_msg()

        # --- 1. PROCESSAMENTO DA ODOMETRIA ---
        odom = Odometry()
        odom.header.stamp = tempo_atual
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'fsds/FSCar'

        sigma_x, sigma_y, sigma_yaw = 0.05, 0.05, 0.02
        sigma_vx, sigma_wz = 0.01, 0.005

        noise_x = np.random.normal(0, sigma_x)
        noise_y = np.random.normal(0, sigma_y)

        odom.pose.pose.position.x = float(kinematics.position.x_val) + noise_x
        odom.pose.pose.position.y = float(-kinematics.position.y_val) + noise_y
        odom.pose.pose.position.z = float(-kinematics.position.z_val)

        odom.pose.pose.orientation.x = float(kinematics.orientation.x_val)
        odom.pose.pose.orientation.y = float(-kinematics.orientation.y_val)
        odom.pose.pose.orientation.z = float(-kinematics.orientation.z_val)
        odom.pose.pose.orientation.w = float(kinematics.orientation.w_val)

        odom.twist.twist.linear.x = float(kinematics.linear_velocity.x_val) + np.random.normal(0, sigma_vx)
        odom.twist.twist.linear.y = float(-kinematics.linear_velocity.y_val)
        odom.twist.twist.linear.z = float(-kinematics.linear_velocity.z_val)

        odom.twist.twist.angular.x = float(kinematics.angular_velocity.x_val)
        odom.twist.twist.angular.y = float(-kinematics.angular_velocity.y_val)
        odom.twist.twist.angular.z = float(-kinematics.angular_velocity.z_val) + np.random.normal(0, sigma_wz)

        pose_cov = np.zeros((6, 6))
        pose_cov[0, 0], pose_cov[1, 1], pose_cov[5, 5] = sigma_x**2, sigma_y**2, sigma_yaw**2
        odom.pose.covariance = pose_cov.flatten().tolist()

        twist_cov = np.zeros((6, 6))
        twist_cov[0, 0], twist_cov[5, 5] = sigma_vx**2, sigma_wz**2
        odom.twist.covariance = twist_cov.flatten().tolist()

        self.odom_publisher.publish(odom)

        # --- 2. PROCESSAMENTO DA IMU ---
        imu = Imu()
        imu.header.stamp = tempo_atual
        imu.header.frame_id = 'fsds/FSCar' # Vinculada diretamente ao centro físico do robô

        # Velocidade angular (Giroscópio) com sinal invertido para o padrão ROS do FSDS
        imu.angular_velocity.x = float(kinematics.angular_velocity.x_val)
        imu.angular_velocity.y = float(-kinematics.angular_velocity.y_val)
        imu.angular_velocity.z = float(-kinematics.angular_velocity.z_val)

        # Aceleração linear (Acelerômetro)
        imu.linear_acceleration.x = float(kinematics.linear_acceleration.x_val)
        imu.linear_acceleration.y = float(-kinematics.linear_acceleration.y_val)
        imu.linear_acceleration.z = float(-kinematics.linear_acceleration.z_val)

        # Orientação vinda do filtro interno do FSDS
        imu.orientation = odom.pose.pose.orientation

        self.imu_publisher.publish(imu)


    # ---------------------------------------------------------
    # PROCESSO DA PERCEPÇÃO E LIDAR 3D (Usa self.client_perc)
    # ---------------------------------------------------------
    def perception_process(self):
        try:
            responses = self.client_perc.simGetImages([
                fsds.ImageRequest("ZED_RGB", fsds.ImageType.Scene, False, True)
            ])
            if not responses:
                return
            
            img_response = responses[0]
            # Modificado para retornar os cones e a nuvem de pontos bruta lida no mesmo instante
            cones, raw_points = self.perception.detectCones(img_response)

            tempo_atual = self.get_clock().now().to_msg()

            # --- 1. PUBLICAÇÃO DA NUVEM DE PONTOS (LiDAR 3D) ---
            if raw_points.size > 0:
                self.publish_pc2(raw_points, tempo_atual)

            # --- 2. PUBLICAÇÃO DOS CONES (PoseArray & Landmarks) ---
            if cones.size == 0:
                return 

            msg_cones = PoseArray()
            msg_cones.header.stamp = tempo_atual
            msg_cones.header.frame_id = 'fsds/FSCar' 

            landmark_list = LandmarkList()
            landmark_list.header.stamp = tempo_atual
            landmark_list.header.frame_id = 'fsds/FSCar'

            for idx, cone in enumerate(cones):
                y_lateral, x_frente, class_id = cone[0], cone[1], int(cone[2])
                
                # Inversão de coordenadas do FSDS para o padrão ROS
                x_ros = float(x_frente)
                y_ros = float(-y_lateral)

                # Estrutura clássica PoseArray
                pose = Pose()
                pose.position.x = x_ros
                pose.position.y = y_ros
                pose.position.z = 0.0
                pose.orientation.w = 1.0
                msg_cones.poses.append(pose)

                # Estrutura específica do Google Cartographer (Landmarks)
                landmark = LandmarkEntry()
                # Cria uma string única usando o ID da classe (Cor) e índice
                # Ex: classe 0 = azul_0, azul_1; classe 1 = amarelo_0, etc.
                cor = "azul" if class_id == 0 else "amarelo"
                landmark.id = f"{cor}_{idx}"
                landmark.tracking_from_landmark_transform.position.x = x_ros
                landmark.tracking_from_landmark_transform.position.y = y_ros
                landmark.tracking_from_landmark_transform.position.z = 0.0
                landmark.tracking_from_landmark_transform.orientation.w = 1.0
                landmark.translation_weight = 15.0 # Peso/Confiança que o Grafo dará a este cone
                landmark.rotation_weight = 0.0     # Cones são pontos, rotação não importa
                
                landmark_list.landmarks.append(landmark)

            self.cones_publisher.publish(msg_cones)
            self.landmark_publisher.publish(landmark_list)

        except Exception as e:
            self.get_logger().error(f'Erro na percepção ou LiDAR 3D: {e}')

    # ---------------------------------------------------------
    # FUNÇÃO AUXILIAR: SERIALIZAR POINTCLOUD2 EM ALTA VELOCIDADE
    # ---------------------------------------------------------
    def publish_pc2(self, points, timestamp):
        msg = PointCloud2()
        msg.header.stamp = timestamp
        msg.header.frame_id = 'fsds/Lidar1' # Referencial onde o sensor está fisicamente montado

        msg.height = 1
        msg.width = len(points)

        # Define os campos estruturais x, y, z (32-bit float)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        msg.is_bigendian = False
        msg.point_step = 12  # 3 campos * 4 bytes cada
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True

        # Inversão de coordenadas do FSDS diretamente via vetorização NumPy para não estragar a taxa de quadros
        ros_points = np.zeros_like(points)
        ros_points[:, 0] = points[:, 0]   # X_ros = X_fsds (Frente)
        ros_points[:, 1] = -points[:, 1]  # Y_ros = -Y_fsds (Esquerda)
        ros_points[:, 2] = -points[:, 2]  # Z_ros = -Z_fsds (Cima)

        # Serializa os bytes brutos eficientemente
        msg.data = ros_points.astype(np.float32).tobytes()
        self.lidar_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = fsdsNode()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()