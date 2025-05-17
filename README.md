# Modeling Optimal Firm Investment Decisions with Frictions: A Computational Approach

This repository contains the computational implementation of a dynamic investment model with adjustment costs, developed as part of the **Computational Assignment for the course ENECO 630 - Macroeconomics I** in the Master in Economics program at the University of Chile (Fall 2025), taught by Professor Eduardo Engel.

![sigma](https://github.com/cridonoso/bellman-firm/blob/main/figuras/p7/sigma_move.gif?raw=true)

## Model Description

The model represents a firm that rents capital in a competitive market and decides whether to adjust its capital stock in response to productivity shocks. The firm's optimal decision depends on:

- **When the new capital becomes operational**: either with a construction delay or immediately.
- **The form of the adjustment cost**: either a fixed cost \( F \) or a proportional cost \( P \) based on profits.

The firm maximizes the present discounted value of net profits by solving a Bellman equation that varies across four model configurations.

## Repository Structure

- `main.m`: Main script that runs experiments based on configuration files stored in the `config/` folder.
- `utils/`: Auxiliary MATLAB functions for solving the Bellman equation, saving results, and managing iterations.
- `config/`: `.json` files with model parameters for each experiment (`p3`, `p7`, `p8`, etc.).
- `backup/`: Output directory for experiment results in `.mat` format.
- `visualize.ipynb`: Jupyter Notebook (Python) to generate figures (optimal policy and inaction bands).
- `informe_donoso.pdf`: Report with theoretical analysis and economic interpretation of the results.

## How to Run

### Requirements

- **MATLAB** for solving the model numerically.
- **Python (optional)** with `numpy`, `scipy`, `matplotlib`, and `seaborn` for visualizing results.

### Instructions

1. Clone this repository and open MATLAB in the project root directory.
2. Make sure the `utils/` folder and configuration `.json` files are accessible.
3. Edit the `main.m` file to select the experiment you want to run (e.g., `params_p3.json`).
4. Run the main script
5. Results will be saved in `./backup/[experiment_name]/`.

### Visualization in Python

Open `visualize.ipynb` and run the cells to generate figures. By default, plots will be saved to the `./figures/` folder.

### Report
Theoretical background, Bellman equations, numerical results, and visualizations are documented in [informe_donoso.pdf](./informe_donoso.pdf)

### Author
Cristóbal Donoso Oliva
University of Chile

