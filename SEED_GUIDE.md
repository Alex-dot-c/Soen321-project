# Should You Use --seed?

## Quick Answer

**Most of the time: NO** - Don't use `--seed`

**For specific cases: YES** - Use `--seed` when you need reproducibility

---

## When NOT to Use --seed (Recommended)

### ✅ General Testing and Exploration
```bash
# Run multiple times to see variety of adversarial examples
python step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3
python step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3
python step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3
```

**Why?**
- Each run shows **different adversarial examples**
- Helps you understand the attack better
- See more variety in how the model fails
- More interesting visualizations

### ✅ Exploring Different Attack Scenarios
```bash
# Without seed, you'll see different examples each time
python step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3
# → Shows examples from batch 7
# Run again...
python step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3
# → Shows examples from batch 13 (different images!)
```

---

## When TO Use --seed

### 1. **Creating Reports or Presentations**
```bash
# Always show the same examples for consistency
python step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3 --seed 42
```

**Use case:**
- You found good visual examples
- Want to recreate them for a presentation
- Need consistent screenshots for documentation

### 2. **Comparing Different Models**
```bash
# Test model A
python step3_attack.py --model outputs/vgg_shallow_k3.pt --depth shallow --kernel 3 --seed 42

# Test model B with SAME examples
python step3_attack.py --model outputs/vgg_deep_k3.pt --depth deep --kernel 3 --seed 42
```

**Use case:**
- Fair comparison between models
- Both models tested on exact same adversarial examples
- Can directly compare which model is more robust

### 3. **Debugging or Reproducing Issues**
```bash
# Share exact results with teammates
python step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3 --seed 12345
```

**Use case:**
- Found an interesting failure case
- Want others to see the exact same results
- Debugging why certain images fail

### 4. **Academic/Research Work**
```bash
# Reproducible experiments for papers
python step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3 --seed 2024
```

**Use case:**
- Publishing results
- Need reproducibility for peer review
- Documenting experimental methodology

---

## Comparison

| Scenario | Command | Result |
|----------|---------|--------|
| **Exploring** | No `--seed` | Different examples each run ✨ |
| **Presentation** | `--seed 42` | Same examples every time 📊 |
| **Comparison** | `--seed 100` on both | Fair model comparison ⚖️ |

---

## Real-World Example

### Without --seed (Recommended for learning)
```bash
# Run 1
python step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3
# Visualization shows: airplane, cat, frog, dog, ship

# Run 2 (different examples!)
python step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3
# Visualization shows: truck, bird, horse, deer, automobile

# Run 3 (different again!)
python step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3
# Visualization shows: ship, cat, airplane, truck, frog
```
**Benefit:** You see 15 different examples across 3 runs!

### With --seed (For reproducibility)
```bash
# Run 1
python step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3 --seed 42
# Visualization shows: airplane, cat, frog, dog, ship

# Run 2 (identical!)
python step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3 --seed 42
# Visualization shows: airplane, cat, frog, dog, ship

# Run 3 (still identical!)
python step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3 --seed 42
# Visualization shows: airplane, cat, frog, dog, ship
```
**Benefit:** Exact reproducibility for presentations or comparisons.

---

## My Recommendation

### For Learning and Experimentation
```bash
# Just run it without --seed
python step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3

# Or use --shuffle for even more variety
python step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3 --shuffle
```

### For Final Reports/Presentations
```bash
# Run once, find good examples, then use seed
python step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3
# ^ Found interesting examples? Note the images shown

# Now lock it in for your report
python step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3 --seed 42
# ^ Will always generate same visualizations
```

---

## Summary

**Default behavior (no --seed):**
- ✅ Shows different examples each run
- ✅ Better for exploration
- ✅ More interesting
- ✅ **Recommended for most users**

**With --seed:**
- ✅ Reproducible results
- ✅ Good for comparisons
- ✅ Good for presentations
- ⚠️ Less variety

**Bottom line:** Unless you need reproducibility, skip the `--seed` flag and enjoy seeing different adversarial examples each time! 🎯
