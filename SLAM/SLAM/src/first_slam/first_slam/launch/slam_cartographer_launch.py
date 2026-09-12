import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Caminho do pacote atual (Substitua pelo nome real do seu pacote ROS 2)
    package_name = 'nome_do_seu_pacote' 
    pkg_share = get_package_share_directory(package_name)
    
    config_dir = os.path.join(pkg_share, 'config')
    config_basename = 'cartographer_config.lua'

    # 1. Nó do seu Simulador (O script Python que ajustamos)
    fsds_node = Node(
        package=package_name,
        executable='fsdsNode', # O nome do executável definido no seu setup.py
        name='fsds_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # 2. TF Estática: Conecta o carro (base_link) ao LiDAR (lidar_link)
    # Valores baseados no seu settings.json: X=0.45, Z=0.55
    static_tf_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0.45', '0.0', '0.55', '0', '0', '0', 'base_link', 'lidar_link'],
        parameters=[{'use_sim_time': True}]
    )

    # 3. Nó do Google Cartographer (com Remapeamento de Tópicos)
    cartographer_node = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=['-configuration_directory', config_dir,
                   '-configuration_basename', config_basename],
        remappings=[
            ('odom', 'fsdsOdometry'),
            ('points2', 'fsdsLidar3D'),
            ('imu', 'fsdsImu'),
            ('landmarks', 'fsdsLandmarks')
        ]
    )

    # 4. Nó do Occupancy Grid (Gera o mapa visual em preto e branco no tópico /map)
    occupancy_grid_node = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='occupancy_grid_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=['-resolution', '0.05'] # Resolução do mapa: 5cm por pixel
    )

    return LaunchDescription([
        fsds_node,
        static_tf_lidar,
        cartographer_node,
        occupancy_grid_node
    ])