# Monopoly AI Training and Simulation Grounds

## Installation

Before running any scripts, make sure you have the required dependencies installed.

### Python Dependencies

From the project root, install Python dependencies:

```bash
pip install -r requirements.txt
```

### Frontend Dependencies

Navigate to the frontend folder and install Node.js packages:

```bash
cd path/to/monopoly-frontend
npm install
```

---

## Training Agents

To train agents, run:

```bash
python play_agents.py
```

* Press **R** during execution to toggle board rendering on/off.

---

## Watching Pre-Trained Agents

To watch pre-trained agents, run:

```bash
python watch_agents.py
```

---

## Frontend for Watching Pre-Trained Agents

A better frontend is available in the **`monopoly-frontend/`** folder.

1. Open a terminal and navigate to the frontend folder:

   ```bash
   cd path/to/monopoly-frontend
   ```

2. Start the frontend:

   ```bash
   npm run dev
   ```

   * The frontend will be available at [http://localhost:8080](http://localhost:8080) in your browser.

3. In another terminal, navigate to the backend folder:

   ```bash
   cd path/to/monopoly-backend
   ```

   and start the server:

   ```bash
   python server.py
   ```
