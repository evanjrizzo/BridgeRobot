# BridgeRobot
RIT capstone project. Integrates FANUC LR Mate 200iC and external computer (raspi 5 16gb) for custom automated target creation and command management.

The Pi/runops directory contains all files needed for controller-computer interfacing and standard control loop. Pi/devtools contains some tools that were used for april tag tracking and homography calculations and other one-time tests.

RobotController contains a full working copy of the r30iA Mate controller's firmware. This can be loaded onto a compactflash cart and then installed through the controller's "load backup" feature. This will load all necessary configurations and programs on the robot controller. "autonew4.PC" is the KAREL server which allows for socket communication, and "bridmain.TP" is the teach pendant program which reads flags set by the server and commands robot motion.

For any questions on robot or computer configuration, contact Evan Rizzo.
