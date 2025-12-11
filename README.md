# Final Project – Advanced AI and RL for Engineers
# Reinforcement Learning for Autonomous Drone Control in a Target-Acquisition Task
**Student:** Kazi Sifatul Islam  
**Instructor:** Dr. Damian Valles  
**Course:** Advanced Artificial Intelligence and Reinforcement Learning for Engineers  

---

## 🏗️ Project Overview
This project implements and compares two reinforcement learning (RL) algorithms—**Advantage Actor–Critic (A2C)** and **Deep Q-Network (DQN)**—to control a simulated 2D drone.  
A custom environment was built from scratch with Newtonian physics, thrust dynamics, rotational action, and target-reaching behavior.  
Both algorithms are trained under identical conditions to evaluate which approach yields more stable and efficient navigation.

---

## Reprocubility Prcedure

## 📦 Dependencies (Install Manually)

Clone the repository.

Creat a virtual environment (Optional but recommended)

The project requires the following libraries:

| Library | Purpose |
|--------|---------|
| **Python 3.8+** | Programming environment |
| **NumPy** | Mathematical operations |
| **Matplotlib** | Plotting training curves |
| **Stable-Baselines3** | A2C and DQN implementation |
| **TensorBoard** | For Data monitoring |
| **Gym** | RL environment interface |
| **Pygame** | Rendering 2D drone environment |

Install each dependency manually:

pip install numpy
pip install matplotlib
pip install stable-baselines3
pip install torch
pip install gym
pip install pygame

All python files are the helper of the main notebook(Drone_Training.ipynb)

Run the Drone_Training.ipynb in Jupyter Lab(Which need to install in the venv), and run the cells sequentially. It will train and save the model for both A2C and DQN.

After that model saved and DroneLog, and assets also saved, then easily can run the trained model.

All training can be done by CPU, but CUDA enable device will be train faster. 
