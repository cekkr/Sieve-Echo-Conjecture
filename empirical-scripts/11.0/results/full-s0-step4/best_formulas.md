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

- guide `policy`, variant `warm`, seed 4, 5 effective instructions, selected at N = 95000
- validation NMSE 0.2754, test NMSE 0.2763, test accuracy 0.6550

```
(([log n - log lcm] * sqrt(sqrt(abs([lcm divides n-1])))) + 2.71828)
```

Instructions:

```
✓ d$2 = ABS(d$10)
✓ d$2 = SQRT(d$2)
✓ d$2 = SQRT(d$2)
✓ d$1 = MUL(d$9, d$2)
✓ d$12 = ADD(d$1, 2.71828)
```

## is_prime

- guide `none`, variant `adaptive`, seed 3, 3 effective instructions, selected at N = 100000
- validation NMSE 0.0012, test NMSE 0.0015, test accuracy 0.9999

```
([lcm divides n-1] ^ (2 % ([row.v8.mean.mean] / 10)))
```

Instructions:

```
✓ d$12 = DIV(d$7, 10.0)
✓ d$12 = MOD(2.0, d$12)
✓ d$12 = POW(d$10, d$12)
```
