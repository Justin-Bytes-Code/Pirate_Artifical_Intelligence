### Pirate_Artifical_Intelligence

Pirate AI is an interactive maze game where a pirate AI uses Deep Q-Learning to find treasure. Built with Python, TensorFlow, and Keras, it features a visual maze, experience replay, and customizable AI training parameters.

The work sheet can be found here. 
pirate-intelligent-agent/TreasureHuntGame_starterCode.ipynb

---

## Features

- **Deep Q-Learning AI** 
  - The pirate agent learns the optimal path to the treasure using reinforcement learning.  
- **Interactive Maze Visualization**
  - Displays the maze, pirate, visited cells, and treasure for easy tracking.  
- **Experience Replay**
  - Stores episodes to improve training efficiency and stability.  
- **Customizable Training Parameters** 
  - Adjust epochs, memory size, batch size, and exploration rate.  
- **Reward System**
  - Encourages efficient navigation with positive rewards for reaching the treasure and penalties for each move.  
- **Flexible Neural Network** 
  - Built with Keras and TensorFlow; supports multiple layers and adjustable activations.  


---

**Tools Used**

- **Python**   
- **TensorFlow & Keras** 
  - For building and training the neural network.  
- **NumPy**  
  - For numerical operations and maze representation.  
- **Matplotlib** 
  - To visualize the maze and agent movements.  
- **Jupyter Notebook** 
  - Interactive environment for running and testing the game.  

---

## How to Customize

You can tweak the AI’s behavior and training by adjusting the following parameters in the `qtrain()` function:

- **n_epoch** 
  - Number of training epochs 
    - Increase for better learning.  
- **max_memory**
  - Maximum size of the experience replay memory. 
    - Higher values allow the AI to remember more past episodes.  
- **data_size**
  - Number of experiences sampled for each training step.  
- **epsilon**
  - Exploration rate 
    - Higher values make the AI explore more random paths.  
- **epsilon_min** 
  - Minimum exploration rate 
    - Prevents the AI from always exploiting learned paths.  
- **epsilon_decay** 
  - Rate at which exploration decreases over time.  

You can also modify the neural network architecture in the `build_model()` function:

- **Number of layers** 
  - Add or remove `Dense` layers to increase or decrease model complexity.  
- **Activation functions** 
  - Adjust activations like `PReLU` to change how the network learns.  
- **Optimizer and loss** 
  - Swap `Adam` optimizer or `MSE` loss for alternatives to experiment with learning behavior.  

Experimenting with these settings allows you to control how fast the AI learns, how much it explores the maze, and how well it generalizes to find the treasure.


---
**Ways To Expand**

- Creating a larger maze or more complex maze will be a good test to see how this AI can hold up under more stress. This will be the easiest one to make. 
- Introduce more pirates or competitors to simulate multi-agent reinforcement learning. 

---

**Rubric**

CS 370 Project Two Guidelines and Rubric
Competencies
In this project, you will demonstrate your mastery of the following competencies:

Explain the basic concepts and techniques that pertain to artificial intelligence and intelligent systems
Analyze how algorithms are used in artificial intelligence to solve a variety of complex problems
Scenario
You are working as an AI developer for a gaming company. The company is developing a treasure hunt game where the player needs to find the treasure before the pirates find it. As an AI developer, you have been asked to design an intelligent agent of the game for an NPC (non-player character) to represent the pirate. The pirate will need to navigate the game world, which consists of different pathways and obstacles, in order to find the treasure. The pirate agent’s goal is to find the treasure before the human player. This is commonly called a pathfinding problem, as the agent you create will need to find a path towards its goal.

You have been provided with some starter code and a sample environment where your pirate agent will be placed. You will need to create a deep Q-learning algorithm to train your pirate agent. Finally, you have also been asked to write a design defense that demonstrates your understanding of the fundamental AI concepts involved in creating and training your intelligent agent.

Directions
Pirate Intelligent Agent
As part of your project, you will create a pirate intelligent agent to meet the specifications that you have been given. You will be using the same notebook as the Project Two Milestone. Be sure to review any feedback that you received on your Project Two Milestone and update the code before submitting the final version of your intelligent agent. Follow these steps to complete your intelligent agent:

Before creating your pirate intelligent agent, be sure to review the Pirate Intelligent Agent Specifications document, located in the Supporting Materials section. This document provides details about the code that you have been given, and what aspects you will need to create.
Access the Virtual Lab (Codio) by using the 7-3 Project Two link in  Module Seven.  This will open the notebook containing the code you used in the Project Two Milestone. 
Be sure to review the starter code that you have been given. Watch the Project Two Walkthrough video located in the Supporting Materials section to help you understand this code in more detail.

IMPORTANT: Do not modify any of the .py files that you have been given.

Note: Developing these algorithms can take a long period of time, similar to developing models in the real world. You can download the codebase by navigating to the Project tab and clicking Export as a Zip if you wish to run the code locally.
Complete the code for the Q-Training Algorithm section in your Jupyter Notebook titled TreasureHuntGame.ipynb. In order to successfully complete the code, you must do the following:
Develop code that meets the given specifications:
Complete the program for the intelligent agent so that it achieves its goal: The pirate should get the treasure.
Apply a deep Q-learning algorithm to solve a pathfinding problem.
Create functional code that runs without error.
Use industry standard best practices such as in-line comments to enhance readability and maintainability.
After you have finished creating the code for your notebook, save your work. Make sure that your notebook contains your name in the filename (such as, “Doe_Jane_ProjectTwo.ipynb”). This will help your instructor access and grade your work easily. Be sure to download a copy of your notebook as an HTML file for your submission.
Design Defense
As a part of your project, you will also submit a design defense. This design defense will demonstrate the approach you took in solving this problem, explain how the intelligent agent works, and evaluate the algorithm you chose to use. In order to adequately defend your designs, you will need to support your ideas with research from your readings. You must include citations for sources that you used.

Analyze the differences between human and machine approaches to solving problems.
Describe the steps a human being would take to solve this maze.
Describe the steps your intelligent agent is taking to solve this pathfinding problem.
What are the similarities and differences between these two approaches?
Assess the purpose of the intelligent agent in pathfinding.
What is the difference between exploitation and exploration? What is the ideal proportion of exploitation and exploration for this pathfinding problem? Explain your reasoning.
How can reinforcement learning help to determine the path to the goal (the treasure) by the agent (the pirate)?
Evaluate the use of algorithms to solve complex problems.
How did you implement deep Q-learning using neural networks for this game?
What to Submit
To complete this project, you must submit the following:

Pirate Intelligent Agent (TreasureHuntGame.ipynb)
When you have completed this Jupyter Notebook, download it as an HTML file and submit the HTML file containing the code for your pirate intelligent agent.

Design Defense
Your submission should be a 2– to 3–page Word document with 12-point Times New Roman font, double spacing, and one-inch margins. Sources should be cited according to APA style.

Supporting Materials
The following resource(s) may help support your work on the project:

Reading: CS 370 Pirate Intelligent Agent Specifications
This specifications document provides important details about the code that has already been created for the pirate intelligent agent and what you will need to create.

Starter Code: CS 370 Project Two Notebook
The starter code for this project is located in the Pirate Intelligent Agent Notebook in the Codio environment, accessible through the Codio link in Module Seven. Included are two Python files: TreasureMaze.py, which represents the environment and includes a maze object defined as a matrix; and GameExperience.py, which stores the episodes. You have also been given starter code in your Jupyter Notebook, TreasureHuntGame.ipynb. You will need to complete two code blocks in the Jupyter Notebook to complete your pirate intelligent agent.

Video: Project Two Walkthrough
This video will help you understand the starter code that you have been given for Project Two, as well as showing you what areas of the code will need to be completed. A video transcript is available: Transcript for Project Two Walkthrough.

Reading: TreasureHuntGame Sample Output
This document displays sample output for the TreasureHuntGame.ipynb file after the algorithm has been completed and the notebook has been run. Your actual output may vary in some places but should follow the same basic pattern of outputs.

Reading: Jupyter Notebook in Codio Tutorial
This tutorial will help you navigate the technology you will be using in this course. You will learn how to get into the Jupyter Notebook via Codio, as well as how to complete, save, and download your work.

---

**Questions**

(You don't have to read this. This is only for a homework assignment with this repository) 

Briefly explain the work that you did on this project: What code were you given? What code did you create yourself?

- In this project I was given a pre established game and told to create an AI for this game. The code I was given can be found in the TreasureMaze.py. I was also given a parital start code worksheet where I did all my code which can be found in TreasureHuntGame_starterCode.ipynb. 

What do computer scientists do and why does it matter?

- Computer Scientist design and implement algorithms, systems, and AI to solve real world problems that affect the every day man. It allows them to automate task or to create AI that could change the entire world. 

How do I approach a problem as a computer scientist?

- Regardless of if a computer scientist or not you should always break any problem down to smaller components. You should have a firm understanding of what needs to be done then slowly build a solution that can always do more then what is needed with a stable base. Once the problem is completed having good testing is also important to ensure that your problem is solved to the best of your abilties. 

What are my ethical responsibilities to the end user and the organization?

- Your ethical responsibilities as a developer are also required by law. You must protect user data, avoid bias in AI systems, and create solutions that do not harm users or collect more data than necessary. Following GDPR guidelines is considered best practice, as it is the most up-to-date regulation for AI and data protection.

---
