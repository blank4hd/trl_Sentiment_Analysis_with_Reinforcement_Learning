# Airline Sentiment Analysis with Reinforcement Learning (TRL)

## ✈️ Project Overview

This project implements a sentiment analysis system that adapts over time using **Reinforcement Learning (RL)** and **Active Learning**. Unlike traditional static classifiers, this model optimizes its accuracy based on feedback loops, moving from a standard supervised baseline to a preference-tuned model aligned with human judgment.

## 🏗️ System Architecture

The following diagram illustrates the multi-stage training pipeline, specifically highlighting the data splitting strategy and the active learning loop.

```mermaid
graph TD
    %% Data Ingestion Layer
    Raw[Raw Data: Twitter US Airline Sentiment] --> Split{Data Splitting}

    %% Data Splits
    Split -->|40% - Balanced via Oversampling| TrainSFT[SFT Training Set]
    Split -->|40% - No Oversampling| TrainDPO[DPO Training Set]
    Split -->|10%| Val[Validation Set]
    Split -->|10%| Test[Test Set]

    %% Phase 1: SFT
    subgraph Phase 1: Supervised Fine-Tuning
        GPT2[Pre-trained GPT-2] --> TrainerSFT[SFT Training Loop]
        TrainSFT --> TrainerSFT
        TrainerSFT --> ModelSFT[SFT Baseline Model]
    end

    %% Phase 2: Active Learning & DPO
    subgraph Phase 2: Active Learning Loop
        ModelSFT -->|Predict on Uncertain Data| Uncertainty[Uncertainty Sampling]
        Uncertainty -->|Request Feedback| Human[Human Annotation]
        Human -->|Identify Chosen/Rejected| Pairs[Preference Pairs]

        %% DPO Training
        Pairs --> TrainerDPO[DPO Training Loop]
        TrainDPO --> TrainerDPO
        ModelSFT -->|Reference Model| TrainerDPO
        TrainerDPO --> ModelDPO[DPO Aligned Model]
    end

    %% Phase 3: Eval
    subgraph Phase 3: Evaluation
        Test --> Eval[Compare Models]
        ModelSFT --> Eval
        ModelDPO --> Eval
        Eval --> Report[Classification Report]
    end
```

---

## 🎯 Design Decisions & Intent

### 1\. Model Choice: Why GPT-2 over BERT?

While BERT is a standard encoder for classification, we chose **GPT-2** for this project.

- **Reasoning**: TRL (Transformer Reinforcement Learning) is optimized for generative, autoregressive models. BERT's masked modeling objective does not naturally align with TRL's preference optimization abstractions.
- **Implementation**: We treat sentiment analysis as a generative task (Prompt: `Tweet: X`, Output: `Sentiment: Positive`). This unifies SFT and RL into a single generative framework.

### 2\. RL Algorithm: Why DPO over PPO?

We selected **Direct Preference Optimization (DPO)** instead of Proximal Policy Optimization (PPO).

- **Reasoning**: PPO requires a separate reward model and complex hyperparameter tuning, which causes instability in small-scale setups. DPO is strictly more stable as it optimizes preferences directly without an explicit reward function.
- **Implementation**: The "reward" is implicit—derived from the log-probability divergence between the trained policy and the frozen SFT reference model.

### 3\. Data Strategy: Hybrid Splitting

To prevent data leakage and bias, we use a stratified splitting strategy:

- **SFT Split (40%)**: We **oversample** minority classes here to ensure the baseline learns all classes equally.
- **DPO Split (40%)**: We **do not oversample** here. Repeating examples in preference training can destabilize the policy by artificially inflating confidence.

---

## 💾 Data Schema

The project expects a dataset file named `Tweets.csv` with the following structure:

| Column              | Type   | Description                                                            |
| :------------------ | :----- | :--------------------------------------------------------------------- |
| `text`              | String | The raw tweet content (cleaned internally by `trl_utils.clean_tweet`). |
| `airline_sentiment` | String | Target label. Must be one of: `negative`, `neutral`, `positive`.       |

---

## 🐳 Docker Setup

This project includes a containerized environment to ensure reproducibility.

### To Build the Image

```bash
docker-compose build
```

### To Run the Container

```bash
docker-compose up
```

**Expected Terminal Output:**

- The container will launch and mount the local volume.
- Jupyter Lab will start on port **8888**.
- Streamlit will be exposed on port **8501**.

**Accessing the Services:**

- **Jupyter Lab**: `http://localhost:8888` (Token: `airline`)
- **Streamlit App**:
  1.  Keep the container running.
  2.  Open a new terminal.
  3.  Run: `docker exec -it trl_airline_lab streamlit run trl_app.py`
  4.  Visit `http://localhost:8501`

---

## 📂 Project Structure

| File                | Intent                                                                                                       |
| :------------------ | :----------------------------------------------------------------------------------------------------------- |
| `trl_utils.py`      | Contains reusable logic for data loading, cleaning, and the `SentimentModel` wrapper. Keeps notebooks clean. |
| `trl.API.ipynb`     | Documents the API interface. Demonstrates how to initialize models and run predictions.                      |
| `trl.example.ipynb` | The executable "proof of concept" notebook. Runs the full pipeline from data loading to evaluation.          |
| `trl_app.py`        | Interactive dashboard for human-in-the-loop feedback and testing.                                            |

---

## 🚀 Usage Guide

### 1\. Running the Benchmark (Notebook)

Open `trl.example.ipynb` in Jupyter Lab.

- **Step 1**: Run all cells to load the `Tweets.csv` data.
- **Step 2**: The notebook will initialize the 4 model variants (BERT, GPT-2 Baseline, Improved SFT, DPO).
- **Step 3**: It will perform inference on the held-out test set and generate a classification report.

### 2\. Active Learning Playground (Streamlit)

Launch the Streamlit app to test the feedback loop.

- **Input**: Type a custom airline tweet.
- **Feedback**: Correct the model's prediction if it is wrong.
- **RL Update**: In a full production loop, this feedback is saved as a preference pair to retrain the DPO model.

<!-- end list -->
