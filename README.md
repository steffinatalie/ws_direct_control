# Persiapan OS

## Fresh install ubuntu 22.04
[Download Ubuntu 22.04 Image](https://releases.ubuntu.com/jammy/)

## Vmware Workstation Player Pro 17
serial key ambil dari [repo github](https://gist.github.com/PurpleVibe32/30a802c3c8ec902e1487024cdea26251)
```
--checked
MC60H-DWHD5-H80U9-6V85M-8280D
4A4RR-813DK-M81A9-4U35H-06KND
--unchecked
NZ4RR-FTK5H-H81C1-Q30QH-1V2LA
JU090-6039P-08409-8J0QH-2YR7F
4Y09U-AJK97-089Z0-A3054-83KLA
4C21U-2KK9Q-M8130-4V2QH-CF810
```

## Setting koneksi internet VM
VM -> Removable Devices -> Network Adapter -> Settings
- Ubah Network Connection menjadi Bridge 
- Enable Replicate physical network connection state
- Klik OK
- Turn off lalu turn on wired connection

## Pilih repository terdekat, untuk saat ini server singapore tercepat
Buka aplikasi Software & Update
Ubah 'download from' ke server singapore: mirror.aktkn.sg

## Update system ubuntunya
```
sudo apt update
sudo apt upgrade
```

## Install dependency utama
```
sudo apt install python3-pip
sudo apt install git
```

# Clone repository ini
```
cd ~/
git clone https://github.com/rizkyrivaldi/ws_direct_control.git --recursive
git submodule update --init --recursive --remote
git submodule sync --recursive
git submodule update --init --recursive
```

# Instalasi Aplikasi Yang Dibutuhkan

## Install ROS Humble (LTS version untuk ros 2 saat ini)
Source: [Ros documentation](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)

```
locale  # check for UTF-8
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
locale  # verify settings
```

```
sudo apt install software-properties-common
sudo add-apt-repository universe
```

```
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

```
sudo apt update
sudo apt upgrade
```

```
sudo apt install ros-humble-desktop-full
pip install --user -U empy pyros-genmsg setuptools
```

Menambahkan ros ke dalam bashrc

```
nano ~/.bashrc
```
Masukkan potongan program berikut ke line terakhir bashrc
```
source /opt/ros/humble/setup.bash
```

## Setup Micro XRCE-DDS Agent&Client
Source : [ros2_comm documentation](https://docs.px4.io/main/en/ros/ros2_comm.html)

```
cd ~/
git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent
mkdir build
cd build
cmake ..
make
sudo make install
```

## Install dependency PX4-Autopilot dari githubnya
```
cd ~/
git clone https://github.com/PX4/PX4-Autopilot.git
cd PX4-Autopilot
git checkout release/1.14
git submodule update --init --recursive --remote
git submodule sync --recursive
git submodule update --init --recursive
cd ..
bash ./PX4-Autopilot/Tools/setup/ubuntu.sh
```

## Install gazebo classic, timpa gazebo fortress yang udah keinstall dari ros2
Source: [px4 gazebo classic documentation](https://docs.px4.io/main/en/sim_gazebo_classic/)

```
sudo apt install aptitude
sudo aptitude install gazebo libgazebo11 libgazebo-dev
```

## Memasukkan topic yang diperlukan untuk input dan output control mixer
Mengubah isi dari file .yaml berikut:
```
nano ~/PX4-Autopilot/src/modules/uxrce_dds_client/dds_topics.yaml
```

Menambahkan isi berupa topik uORB yang ingin dipublikasi ke ROS 2 dengan cara menambahkan program berikut ke dalam bagian publication:
```
  - topic: /fmu/out/manual_control_setpoint
    type: px4_msgs::msg::ManualControlSetpoint

  - topic: /fmu/out/actuator_controls_status_0
    type: px4_msgs::msg::ActuatorControlsStatus

  - topic: /fmu/out/actuator_outputs_sim
    type: px4_msgs::msg::ActuatorOutputs

  - topic: /fmu/out/actuator_outputs # PWM output 0 - 2000
    type: px4_msgs::msg::ActuatorOutputs

  - topic: /fmu/out/vehicle_torque_setpoint # control output u1 u2 u3 (roll, pitch, yaw)
    type: px4_msgs::msg::VehicleTorqueSetpoint

  - topic: /fmu/out/vehicle_thrust_setpoint # thrust control output u4
    type: px4_msgs::msg::VehicleThrustSetpoint

  - topic: /fmu/out/actuator_motors
    type: px4_msgs::msg::ActuatorMotors
```

## Compile PX4 Autopilot
```
cd ~/PX4-Autopilot
pip uninstall em
pip uninstall empy
sudo apt-get install libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-bad gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-ugly -y
DONT_RUN=1 make px4_sitl_default gazebo-classic
```

## Menonaktifkan Lockstep Simulation
```
cd ~/PX4-Autopilot
make px4_sitl_default boardconfig
```
Toolchain --> Force Disable Lockstep Simulation

## Install Colcon (seperti cmake)
Source: [colcon documentation](https://colcon.readthedocs.io/en/released/user/installation.html)

```
sudo sh -c 'echo "deb [arch=amd64,arm64] http://repo.ros2.org/ubuntu/main `lsb_release -cs` main" > /etc/apt/sources.list.d/ros2-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt install python3-colcon-common-extensions
```

## Compile ws_direct_control
```
cd ~/ws_direct_control
colcon build
```

## Setup contoh listener ke topic uORB
```
mkdir -p ~/ws_sensor_combined/src/
cd ~/ws_sensor_combined/src/
git clone https://github.com/PX4/px4_msgs.git
git clone https://github.com/PX4/px4_ros_com.git
cd ..
source /opt/ros/humble/setup.bash
colcon build
```

## Setup QGroundControl (untuk manual control)
Source: [qgroundcontrol documentation](https://docs.qgroundcontrol.com/master/en/getting_started/download_and_install.html)

```
cd ~/
mkdir ./QGroundControl/
cd ./QGroundControl/
sudo apt-get remove modemmanager -y
sudo apt install gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl -y
sudo apt install libqt5gui5 -y
sudo apt install libfuse2 -y
wget https://d176tv9ibo4jno.cloudfront.net/latest/QGroundControl.AppImage
chmod +x ./QGroundControl.AppImage
```

# Menjalankan Simulasi tanpa menggunakan BPNN
### Terminal Pertama
```
sudo ldconfig /usr/local/lib/
MicroXRCEAgent udp4 -p 8888 # Start agent
```

### Terminal Kedua
```
cd ~/PX4-Autopilot
make px4_sitl_default gazebo-classic
```

### Terminal Ketiga
```
cd ~/QgroundControl
./QGroundControl.AppImage
```
Pastikan bahwa aktuator menggunakan motor 1, motor 2, motor 3, dan motor 4
Simulasi sudah dapat digerakkan menggunakan joystick

### Terminal Keempat (Apabila ingin menggunakan pergerakan helix)
```
cd ~/ws_direct_control
source ./install/setup.bash
cd src/direct_control/direct_control/
python3 orbit.py
```

# Menjalankan Simulasi dengan BPNN
### Terminal Pertama
```
sudo ldconfig /usr/local/lib/
MicroXRCEAgent udp4 -p 8888 # Start agent
```

### Terminal Kedua
```
cd ~/PX4-Autopilot
make px4_sitl_default gazebo-classic
```

### Terminal Ketiga
```
cd ~/QgroundControl
./QGroundControl.AppImage
```
Pastikan bahwa aktuator menggunakan Offboard Actuator Set 1, Offboard Actuator Set 2, Offboard Actuator Set 3, dan Offboard Actuator Set 4

### Terminal Keempat
```
cd ~/ws_direct_control
source ./install/setup.bash
cd src/direct_control/direct_control/
python3 actuator_mixer_NN_v4.py
```
Simulasi sudah dapat digerakkan menggunakan joystick

### Terminal Kelima (Apabila ingin menggunakan pergerakan helix)
```
cd ~/ws_direct_control
source ./install/setup.bash
cd src/direct_control/direct_control/
python3 orbit.py
```

# Menjalankan simulasi dengan listen ke topic sensor combined
### TERMINAL PERTAMA

Menjalankan MicroXRCEAgent sebagai jembatan antara px4 dengan ros2
```
sudo ldconfig /usr/local/lib/
MicroXRCEAgent udp4 -p 8888 # Start agent
```

### TERMINAL KEDUA

Menjalankan simulasi px4
```
cd ~/PX4-Autopilot
make px4_sitl_default gazebo-classic
```
Alternatif
```
make px4_sitl_default gazebo-classic_iris__baylands
```

### TERMINAL KETIGA

Pengujian node listener
```
cd ~/ws_sensor_combined/
source /opt/ros/humble/setup.bash
source install/local_setup.bash
ros2 launch px4_ros_com sensor_combined_listener.launch.py
```

### TERMINAL KEEMPAT (Opsional, untuk buka QGroundControl)

Menjalankan QGroundControl
```
cd ~/QgroundControl
./QGroundControl.AppImage
```

# MISC
Path ke dds topics
```
PX4-Autopilot/src/modules/uxrce_dds_client/dds_topics.yaml
```

Topic yang ditambahkan
```
  - topic: /fmu/out/manual_control_setpoint
    type: px4_msgs::msg::ManualControlSetpoint

  - topic: /fmu/out/actuator_controls_status_0
    type: px4_msgs::msg::ActuatorControlsStatus

  - topic: /fmu/out/actuator_outputs_sim
    type: px4_msgs::msg::ActuatorOutputs

  - topic: /fmu/out/actuator_outputs # PWM output 0 - 2000
    type: px4_msgs::msg::ActuatorOutputs

  - topic: /fmu/out/vehicle_torque_setpoint # control output u1 u2 u3 (roll, pitch, yaw)
    type: px4_msgs::msg::VehicleTorqueSetpoint

  - topic: /fmu/out/vehicle_thrust_setpoint # thrust control output u4
    type: px4_msgs::msg::VehicleThrustSetpoint

  - topic: /fmu/out/actuator_motors
    type: px4_msgs::msg::ActuatorMotors
```

