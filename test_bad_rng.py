"""
Simple Test of Bad RNG without Graphics

This script tests the Bad RNG logic without requiring matplotlib,
so we can verify the mathematical properties work correctly.
"""

class BadRNG:
    """Simple Linear Congruential Generator with poor parameters"""
    
    def __init__(self, seed=1):
        self.x = seed % 256
        self.sequence = [self.x]
    
    def next(self):
        """Generate next value using: x[i + 1] = (5 * x[i] + 1) mod 256"""
        self.x = (5 * self.x + 1) % 256
        self.sequence.append(self.x)
        return self.x
    
    def generate_sequence(self, n):
        """Generate n random numbers"""
        values = []
        for _ in range(n):
            values.append(self.next())
        return values
    
    def reset(self, seed=1):
        """Reset the generator with a new seed"""
        self.x = seed % 256
        self.sequence = [self.x]

def analyze_rng_patterns(rng, n_values=100):
    """Analyze patterns in the RNG without visualization"""
    print("=== Bad RNG Pattern Analysis ===")
    
    # Generate values
    values = rng.generate_sequence(n_values)
    
    # Basic statistics
    print(f"Generated {len(values)} values")
    print(f"Range: {min(values)} to {max(values)}")
    print(f"Mean: {sum(values)/len(values):.2f} (expected for uniform: 127.5)")
    
    # Look for obvious patterns
    print(f"First 20 values: {values[:20]}")
    
    # Check for correlations - adjacent differences
    diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
    print(f"First 10 adjacent differences: {diffs[:10]}")
    
    # Look for period
    full_rng = BadRNG(seed=1)
    states = [full_rng.x]
    
    for i in range(300):
        current = full_rng.next()
        if current == states[0]:
            print(f"Period detected: {i + 1}")
            break
        states.append(current)
    
    # Check distribution in quarters
    q1 = sum(1 for x in values if 0 <= x < 64)
    q2 = sum(1 for x in values if 64 <= x < 128)
    q3 = sum(1 for x in values if 128 <= x < 192)
    q4 = sum(1 for x in values if 192 <= x < 256)
    
    print(f"Distribution by quarters:")
    print(f"  0-63:   {q1:3d} ({q1/len(values)*100:5.1f}%)")
    print(f"  64-127: {q2:3d} ({q2/len(values)*100:5.1f}%)")
    print(f"  128-191:{q3:3d} ({q3/len(values)*100:5.1f}%)")
    print(f"  192-255:{q4:3d} ({q4/len(values)*100:5.1f}%)")
    print("  (Good RNG should be ~25% each)")
    
    return values

def demonstrate_bad_properties():
    """Demonstrate why this RNG is bad"""
    print("Demonstrating Bad RNG Properties")
    print("=" * 40)
    
    rng = BadRNG(seed=1)
    values = analyze_rng_patterns(rng, 200)
    
    print("\n=== Predictability Test ===")
    # Show how predictable it is
    test_rng = BadRNG(seed=1)
    predicted = test_rng.generate_sequence(10)
    print(f"If you know the formula and seed, next 10 values: {predicted}")
    
    print("\n=== Linear Correlation Test ===")
    # Show linear relationships
    pairs = [(values[i], values[i+1]) for i in range(min(20, len(values)-1))]
    print("First 20 (current, next) pairs:")
    for i, (curr, next_val) in enumerate(pairs):
        predicted_next = (5 * curr + 1) % 256
        print(f"  {curr:3d} -> {next_val:3d} (formula predicts: {predicted_next:3d}) {'✓' if next_val == predicted_next else '✗'}")
    
    print(f"\nThis demonstrates that the RNG is:")
    print("1. Completely predictable with the formula")
    print("2. Has strong linear correlations between successive values")
    print("3. Has a relatively short period (256)")
    print("4. Not suitable for any serious random number applications")

if __name__ == "__main__":
    demonstrate_bad_properties()