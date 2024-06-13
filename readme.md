# Gameplay filter
Due to the rerouting problem of the ROS infrastructure within the Go2, accessing wirelessly the ROS from local PC is currently not possible.
Gameplay filter is a dedicated socket-based server running on local PC instead.
Instruction to run Gameplay filter server:
1. Check the IP of local PC
2. Run the `gameplay_server.py` file on local PC with the correct IP set for `HOST`
3. Run `test_gameplay_shielding.py` on robot, check that the IP for `HOST` is the local PC's IP