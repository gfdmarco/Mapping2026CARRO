#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/../../SLAM/SLAM/src/first_slam/first_slam/config"

echo "=== Iniciando TF base_link -> lidar_link ==="
ros2 run tf2_ros static_transform_publisher \
    --x 0 \
    --y 0 \
    --z 0 \
    --roll 0 \
    --pitch 0 \
    --yaw 0 \
    --frame-id base_link \
    --child-frame-id lidar_link &
LIDAR_TF_PID=$!

echo "=== Iniciando Cartographer ==="
ros2 run cartographer_ros cartographer_node \
    --configuration_directory "$CONFIG_DIR" \
    --configuration_basename fsds_cartographer.lua \
    --ros-args \
    --remap odom:=fsdsOdometry \
    --remap points2:=fsdsLidar3D \
    --remap imu:=fsdsImu \
    --remap landmarks:=fsdsLandmarks &
CARTO_PID=$!

echo "=== Iniciando viz_node ==="
python3 "$SCRIPT_DIR/viz_node.py" &
VIZ_PID=$!

echo "=== Iniciando map_viewer (matplotlib) ==="
python3 "$SCRIPT_DIR/map_viewer.py"

echo "Viewer fechado. Encerrando tudo..."
kill $CARTO_PID $VIZ_PID $LIDAR_TF_PID 2>/dev/null
