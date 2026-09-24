# Previously selected qualitative cases (images excluded)

EXP-041; VOC FIT out-of-fold post-hoc interpretation. These are six categories, not six independent images. Q2/Q3/Q5 share one saved state. No cases were selected or replaced for this export. The numerical case summary is qualitative_cases_source.csv.

Figure 1 and Q1–Q6 photographs, masks and plate files are excluded because display/redistribution rights remain unconfirmed. This omission does not mean the original experiments or plates are missing. The saved selected/oracle annotations below do not resolve the unverified action pair in the original Error change layer; no action-specific pixel-improvement claim is made from that layer.

```latex
Q1 (cat): agreement on STOP. R1 selects STOP. STOP and A3 are tied oracle actions; the plate displays STOP.
```

```latex
Q2 (train): good residual with a missed repair. R1 selects STOP; A6 is the sole oracle action. This is the same saved state as Q3 and Q5.
```

```latex
Q3 (train): false STOP. R1 selects STOP; A6 is the sole oracle action. This category reuses the Q2/Q5 state and supplies no additional independent example.
```

```latex
Q4 (bicycle): harmful intervention. R1 selects A3; STOP is the sole oracle action. The unfavorable intervention remains part of the original case selection.
```

```latex
Q5 (train): global/local disagreement. R1 selects STOP and \texttt{R0\_CM\_P} selects A3; A6 is the sole oracle action. The saved global predictions are $\widehat r(\mathrm{STOP})=0.012096804566681385$ and $\widehat r(\mathrm{A3})=0.012084698614974817$. Both display as 0.0121 at four decimals, while the saved predictions distinguish the actions. This is the same state as Q2 and Q3.
```

```latex
Q6 (person): historical-to-prospective disagreement. R1 selects STOP and \texttt{RETRO\_R1\_FIT\_OOF} selects A3. Both STOP and A3 are oracle actions; the plate displays STOP. This comparison changes several pipeline components and does not isolate runtime features.
```
