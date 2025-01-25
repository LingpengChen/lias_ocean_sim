# lias_ocean_sim
roslaunch lias_ocean_sim world_setup.launch
roslaunch lias_ocean_sim launch_rexrov.launch	
roslaunch uuv_control_cascaded_pid joy_velocity.launch uuv_name:=rexrov model_name:=rexrov joy_id:=0
roslaunch uuv_control_cascaded_pid key_board_velocity.launch uuv_name:=rexrov model_name:=rexrov 


roslaunch uuv_control_cascaded_pid predefined_velocity.launch  uuv_name:=rexrov model_name:=rexrov 
rosrun uuv_control_cascaded_pid run_trajectory.py 
rosrun uuv_control_cascaded_pid run_trajectory_stop.py

pip3 install opencv-contrib-python==3.4.18.65

/home/clp/catkin_ws/src/uuv_simulator/uuv_sensor_plugins/uuv_sensor_ros_plugins/urdf/dvl_snippets.xacro 

---
header:
  stamp:
    sec: 1733038908
    nanosec: 494526342
  frame_id: ''
time: 103.62171936035156
velocity:
  x: 0.017932971939444542
  y: -0.022317176684737206
  z: -0.03312775120139122
fom: 0.01066591776907444
covariance:
- 3.794834628934041e-05
- 3.7816455005668104e-05
- 3.787489185924642e-05
- 3.7816455005668104e-05
- 3.79431621695403e-05
- 3.7856058042962104e-05
- 3.787489185924642e-05
- 3.7856058042962104e-05
- 3.787028981605545e-05
altitude: 0.4210711717605591
beams:
- id: 0
  velocity: -0.04092848300933838
  distance: 0.44840002059936523
  rssi: -31.260032653808594
  nsd: -87.8836669921875
  valid: true
- id: 1
  velocity: -0.029279420152306557
  distance: 0.4366000294685364
  rssi: -34.737876892089844
  nsd: -89.63419342041016
  valid: true
- id: 2
  velocity: 0.02118644490838051
  distance: 0.5074000358581543
  rssi: -25.473129272460938
  nsd: -88.89904022216797
  valid: true
- id: 3
  velocity: -0.03160589560866356
  distance: 0.49560001492500305
  rssi: -40.38396072387695
  nsd: -87.63865661621094
  valid: true
velocity_valid: true
status: 0
time_of_validity: 1671710718731439.0
time_of_transmission: 1671710718868895.0
form: json_v3.1
type: velocity
---


header:
  stamp:
    sec: 1733038908
    nanosec: 582777103
  frame_id: ''
time: 105.90362548828125
velocity:
  x: 0.009230274707078934
  y: -0.02523154206573963
  z: -0.021309489384293556
fom: 0.008343380875885487
covariance:
- 2.3224389224196784e-05
- 2.3142034478951246e-05
- 2.3162019715528004e-05
- 2.3142034478951246e-05
- 2.322346335859038e-05
- 2.3161126591730863e-05
- 2.3162019715528004e-05
- 2.3161126591730863e-05
- 2.316414611414075e-05
altitude: 0.41834571957588196
beams:
- id: 0
  velocity: -0.029018986970186234
  distance: 0.44840002059936523
  rssi: -30.28619384765625
  nsd: -84.67668151855469
  valid: true
- id: 1
  velocity: -0.01532764732837677
  distance: 0.4366000294685364
  rssi: -38.279319763183594
  nsd: -85.50286865234375
  valid: true
- id: 2
  velocity: -0.010608713142573833
  distance: 0.4838000237941742
  rssi: -29.180313110351562
  nsd: -85.00135803222656
  valid: true
- id: 3
  velocity: -0.02383679524064064
  distance: 0.5074000358581543
  rssi: -39.09894943237305
  nsd: -83.90379333496094
  valid: true
velocity_valid: true
status: 0
time_of_validity: 1671710718835720.0
time_of_transmission: 1671710718956338.0
form: json_v3.1
type: velocity


---
header:
  stamp:
    sec: 1733038908
    nanosec: 684962964
  frame_id: ''
time: 110.47744750976562
velocity:
  x: 0.012754406780004501
  y: -0.03565404564142227
  z: -0.02363988570868969
fom: 0.011032556183636189
covariance:
- 4.0775557863526046e-05
- 3.958481829613447e-05
- 4.013371290056966e-05
- 3.958481829613447e-05
- 4.0782349969958887e-05
- 4.0164781239582226e-05
- 4.013371290056966e-05
- 4.0164781239582226e-05
- 4.015938611701131e-05
altitude: 0.4128948152065277
beams:
- id: 0
  velocity: -0.022473393008112907
  distance: 0.4602000117301941
  rssi: -28.620174407958984
  nsd: -87.28787994384766
  valid: true
- id: 1
  velocity: -0.015797313302755356
  distance: 0.4012000262737274
  rssi: -43.518028259277344
  nsd: -88.98853302001953
  valid: true
- id: 2
  velocity: -0.007335664238780737
  distance: 0.4838000237941742
  rssi: -29.58641815185547
  nsd: -88.018310546875
  valid: true
- id: 3
  velocity: -0.028277799487113953
  distance: 0.5074000358581543
  rssi: -35.880043029785156
  nsd: -87.17474365234375
  valid: true
velocity_valid: true
status: 0
time_of_validity: 1671710718941619.0
time_of_transmission: 1671710719059135.0
form: json_v3.1
type: velocity
CLP
