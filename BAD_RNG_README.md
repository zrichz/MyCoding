# Bad RNG Visualization Project

This project demonstrates why the Linear Congruential Generator (LCG) with the formula `x[i + 1] = (5 * x[i] + 1) mod 256` is a terrible random number generator.

## Files in this Project

### Main Scripts
- **`bad_rng_3d_polar_visualization.py`** - Main visualization script with 3D polar plots
- **`test_bad_rng.py`** - Simple analysis script without graphics dependencies
- **`run_bad_rng_viz.bat`** - Windows batch file to run the main visualization
- **`run_test_bad_rng.bat`** - Windows batch file to run the simple test

## What Makes This RNG Terrible?

### 1. **Complete Predictability**
Once you know the formula and any single value, you can predict the entire sequence:
```
x[i + 1] = (5 * x[i] + 1) mod 256
```

### 2. **Strong Linear Correlations** 
Successive values are linearly related, creating patterns instead of randomness.

### 3. **Limited Period**
The sequence repeats every 256 values, which is very short for most applications.

### 4. **Poor Statistical Properties**
- Non-uniform distribution in multi-dimensional space
- Visible geometric patterns in 3D plots
- Fails most standard randomness tests

## Visualizations Included

### 3D Polar Plots
- **Bad RNG**: Shows clustering and patterns in 3D space
- **Good RNG**: Shows uniform distribution for comparison
- **2D Projections**: Reveals linear correlations and geometric patterns

### Statistical Analysis
- Sequence plots showing predictable patterns
- Histogram comparisons with proper RNGs
- Period detection and cycle analysis
- Distribution analysis by quarters

## How to Run

### Option 1: Full 3D Visualization (Requires matplotlib)
```bash
# Windows
run_bad_rng_viz.bat

# Or manually:
python bad_rng_3d_polar_visualization.py
```

### Option 2: Simple Text Analysis (No graphics)
```bash
# Windows  
run_test_bad_rng.bat

# Or manually:
python test_bad_rng.py
```

## Dependencies

### For 3D Visualization:
- matplotlib
- numpy  
- scipy

### For Simple Test:
- No external dependencies (uses built-in Python only)

## Key Insights from the Visualization

### What You'll See:
1. **3D Clustering**: Instead of filling 3D space uniformly, points cluster in predictable patterns
2. **Geometric Patterns**: Regular geometric structures instead of random distribution
3. **Linear Correlations**: Clear relationships between successive values
4. **Poor Space-Filling**: Large empty regions in the 3D space

### Why This Matters:
This type of poor RNG can cause serious problems in:
- **Cryptography**: Predictable "random" keys
- **Scientific Simulations**: Biased results due to patterns
- **Monte Carlo Methods**: Incorrect probability estimates  
- **Gaming**: Predictable behavior that can be exploited

## Educational Value

This project demonstrates:
- How to visualize randomness quality using 3D polar coordinates
- Why proper RNG design is crucial for applications
- The difference between pseudo-random and truly random sequences
- Statistical tests for randomness quality

## Comparison with Good RNGs

The visualization includes comparisons with Python's built-in random generator to show:
- Uniform distribution vs. clustering
- Random patterns vs. geometric structures
- Proper space-filling vs. empty regions
- Statistical uniformity vs. bias

## Mathematical Background

### Linear Congruential Generator Formula:
```
x[n+1] = (a * x[n] + c) mod m
```

### Our Bad Parameters:
- a = 5 (multiplier)
- c = 1 (increment)  
- m = 256 (modulus)

### Why These Parameters Are Bad:
- Small modulus (256) limits period
- Poor choice of multiplier (5) creates patterns
- Fails spectral test for randomness
- Creates hyperplane structures in multi-dimensional space

## Further Reading

For better understanding of RNG quality:
- Knuth's "The Art of Computer Programming, Volume 2"
- NIST Statistical Test Suite for Random Number Generators
- "Numerical Recipes" chapter on random numbers
- TestU01 statistical test library

## License

This is an educational project demonstrating poor RNG properties.
Use this code to understand what NOT to do in random number generation!