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
    pip install numpy scipy matplotlib pyyaml
    ```

3. **Configure parameters:**
   Edit `config.yaml` to adjust:
   - Motion strategy (`teleop`, `random_nav`, or `predefined`)
   - Number of particles
   - Noise parameters
   - Map dimensions
   - Other algorithm parameters

4. **Run the FastSLAM algorithm:**
   ```bash
   python FastSLAM_main.py
   ```

**Note:** All parameters are now centralized in `config.yaml`. No need to modify the source code for common adjustments.


## Running Tests
Test scripts for checking each test unit for debugging.
```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_particle

# Run with verbose output
python -m unittest discover tests -v
```

## Documentation

For detailed algorithm information, see [docs/FastSLAM1_summary.md](docs/FastSLAM1_summary.md).