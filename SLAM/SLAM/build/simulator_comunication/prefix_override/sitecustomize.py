import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/felipe_capovilla/Documents/E-Racing/Mapping/SLAM/install/simulator_comunication'
