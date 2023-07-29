import gym
from gym import spaces
import numpy as np
import pennylane as qml

class Qenv(gym.Env):
    def __init__(self):
        super(Qenv, self).__init__()

        # action space
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        
        # obs space
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.complex64)

        # quantum device and circuit
        self.device = qml.device('default.qubit', wires=1)
        self.state = self.reset()

    def quantum_circuit(self, params):
        qml.RY(params[0], wires=0)
        return qml.state()

    def step(self, action):
        # action
        circuit = qml.QNode(self.quantum_circuit, self.device)
        self.state = circuit(action)

        # cal uncertainties
        position_uncertainty = np.std(np.abs(self.state.real)) 
        momentum_uncertainty = np.std(np.abs(self.state.imag))

        # reward
        reward = -position_uncertainty * momentum_uncertainty

        # statevector
        observation = self.state
        return observation, reward, False, {}

    def reset(self):
        # reset
        self.state = np.array([1+0j, 0+0j])
        return self.state


env = Qenv()
print(env.reset())
for _ in range(10):
    action = env.action_space.sample()  # random action
    obs, reward, done, info = env.step(action)
    print(f"Observation: {obs}, Reward: {reward}, Done: {done}")