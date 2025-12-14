# FastSLAM 1.0 Minimal Implementation

A particle filter-based implementation of the FastSLAM algorithm for simultaneous localization and mapping (SLAM). Each particle maintains its own pose estimate and occupancy grid map for robust real-time mapping.

<img src="docs/fig1_demo.png"/>

## Features

- **Particle Filter Approach**: Uses multiple particles to represent pose uncertainty
- **Occupancy Grid Mapping**: Each particle maintains a log-odds occupancy grid
- **Motion Model**: Noisy motion prediction with customizable noise parameters
- **Measurement Update**: Likelihood-based weight updates from sensor observations
- **Adaptive Resampling**: Systematic resampling based on effective sample size
- **Log-Odds Fusion**: Stable map updates with measurement confidence filtering
- **Optimized Performance**: Vectorized operations for real-time performance (>10 Hz)

## TODO
- Add support for different sensor models
- Add configuration file support (define all parameters in a config file)
- Implement map saving/loading functionality

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ilyes-Origamist/FastSLAM-Tests/
   ```

2. **Install dependencies:**
    ```bash
    pip install numpy scipy matplotlib
    ```

3. **Run the FastSLAM algorithm:**
   ```bash
   python main.py
   ```

4. **Configuration:**
In `FastSLAM_main.py`:
   - Choose motion strategy: `teleop`, `random_nav` or `predefined`.
   - Adjust particle filter and noise parameters.

## Documentation

For detailed algorithm information, see [docs/FastSLAM1_summary.md](docs/FastSLAM1_summary.md).