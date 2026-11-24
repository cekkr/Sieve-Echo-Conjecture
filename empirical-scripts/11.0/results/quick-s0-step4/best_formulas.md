# Best formulas (selected on validation)

Inputs are z-scored with train statistics:

- `d$0` = log n
- `d$1` = row.v8.logL.mean
- `d$2` = row.v8.H_paper.mean
- `d$3` = row.v8.H_nodc.mean
- `d$4` = row.v8.H_paper.std
- `d$5` = row.v8.H_paper.min
- `d$6` = row.v8.Hn_nodc.mean
- `d$7` = row.v8.mean.mean
- `d$8` = log lcm(L_b)
- `d$9` = log n - log lcm
- `d$10` = lcm divides n-1
- `d$11` = share of L_b dividing n-1

## omega

- guide `policy`, variant `knowledge`, seed 999, 4 effective instructions, selected at N = 7000
- validation NMSE 0.4217, test NMSE 0.4246, test accuracy 0.6561

```
(2 + ((sin(-1) % [lcm divides n-1]) % [log n - log lcm]))
```

Instructions:

```
✓ d$5 = SIN(-1.0)
✓ d$4 = MOD(d$5, d$10)
✓ d$12 = MOD(d$4, d$9)
✓ d$12 = ADD(2.0, d$12)
```

## is_prime

- guide `policy`, variant `adaptive`, seed 0, 3 effective instructions, selected at N = 7366
- validation NMSE 0.0058, test NMSE 0.0000, test accuracy 1.0000

```
(1 % (log([lcm divides n-1]) ^ [log n - log lcm]))
```

Instructions:

```
✓ d$2 = LOG(d$10)
✓ d$12 = POW(d$2, d$9)
✓ d$12 = MOD(1.0, d$12)
```
