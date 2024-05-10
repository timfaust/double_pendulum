Protocol For Running Experiments in the given Slot: 

    - The Double Pendubum Acrobot/Pedubot is prepared at DFKI RIC, Bremen such that the teams can access the robot via a local control PC running Ubuntu.  
    - The experiments on the real robot will be carried out remotely using VPN+SSH.  
    - A video stream via Microsoft Teams call and video file post-experiment runs will be provided. 
    - First, a VPN must be connected to enter the private network setup for the experiments. For this, each team will be provided with a VPN config file.  
    - We use/support the wireguard VPN on Ubuntu. For installing the VPN, the teams have to install the following packages via apt: wireguard-tools, wireguard, and resolvconf. This can be done via the command: sudo apt-get install wireguard-tools wireguard resolvconf 
    - After installing, you can go to the folder containing the provided VPN config file and run the following to start the VPN: wg-quick up wg-client.conf (Hint: Sometimes one has to provide the full path of wg-client.conf) 
    - To exit the VPN, run: wg-quick down wg-client.conf (Hint: Sometimes one has to provide the full path of wg-client.conf) 
    - Once you are within the VPN, you can SSH to the control computer whose IP address will be provided at the start of each experiment session.  
    - For SSH, a username and password will be provided to each team. For SSH, the following command can be used: ssh <username>@<IP Address>. (Hint: ssh –Y <username>@<IP Address> can be used to view the plots after experiments without copying the data. This can sometimes cause issues though.) 
    - Once in the control PC via SSH, teams can execute scripts remotely and copy data in/out from the PC. The data can be foundTools such as scp/git are suggested to be used for transferring code/data. (Hint: A tutorial on scp to copy data: https://linuxize.com/post/how-to-use-scp-command-to-securely-transfer-files/) 
    - The double pendulum repo library along with motor drivers are installed on the control PC at the root. Hence, they should be available for all teams/users. 

  

Some rules and information for the hardware experiments regarding experiment duration and safety limits: 

    - Each attempt must not exceed a total time duration of 60 seconds (swing-up + stabilization) 
    - Friction compensation on both joints is allowed in both pendubot and acrobot configurations. The teams are free to choose a friction compensation model of their choice but the utilized torque on the passive joint must not exceed 0.5 Nm. 
    - The controller must inherit from the AbstractController class provided in the project repository. 
    - The following hardware restriction must be respected by the controller: 
    - Control Loop Frequency: 500Hz Max. Usually around 400Hz. 
    - Torque Limit: 6Nm 
    - Velocity Limit: 20 rad/s 
    - Position Limits: +- 360 degrees for both joints 
    - When the motors exceed these limits, the controller is (usually) automatically switched off and a damper is applied to bring the system to zero velocity. Once zero velocity is achieved, experiments can start again. 
    - When the motors are initially enabled, they set the “zero position”. This happens every time they are enabled. 
    - For the hardware experiments, the Acrobot Pendubot system parameters are the same but different from the ones in the simulation. We have done the basic system identification and the teams can re-train their controllers using the following system parameters for the hardware: https://github.com/dfki-ric-underactuated-lab/double_pendulum/blob/main/data/system_identification/identified_parameters/design_C.1/model_1.0/model_parameters.yml
    - A person will be watching the experiments and will have access to an Emergency Stop. 
