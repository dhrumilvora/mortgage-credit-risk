# Mortgage Credit Risk Cheat Sheet

## Core Credit Risk Terminology

| Acronym | Full Term | Meaning |
|---------|-----------|---------|
| PD | Probability of Default | Probability borrower defaults |
| LGD | Loss Given Default | Loss if default occurs |
| EAD | Exposure at Default | Exposure when default occurs |
| UPB | Unpaid Principal Balance | Remaining mortgage principal |
| DPD | Days Past Due | Number of days payment is overdue |
| REO | Real Estate Owned | Property acquired through foreclosure |
| RPL | Reperforming Loan | Previously delinquent loan now performing |
| ZBC | Zero Balance Code | Reason loan terminated |

### Expected Loss

Expected Loss = PD × LGD × EAD

## Delinquency and Loan Termination

### Current Loan Delinquency Status

| Code | Interpretation | Credit Risk Meaning |
|---|---|---|
| 00 | Current / less than 30 Days Past Due (DPD) | Performing |
| 01 | 30–59 DPD | Early delinquency |
| 02 | 60–89 DPD | Elevated delinquency |
| 03 | 90–119 DPD | Serious delinquency |
| 04+ | 120+ DPD | Severe delinquency |
| RA | Real Estate Owned (REO) Acquisition | Severe credit event |
| XX | Not available | Unknown |

For this project, serious delinquency begins at 90+ Days Past Due (DPD),
corresponding to numeric delinquency status >= 03.

---

### Zero Balance Code (ZBC)

Zero Balance Code describes why a mortgage terminates or leaves active
performance reporting.

| Code | Meaning | Project Interpretation |
|---|---|---|
| 01 | Prepaid or matured | Non-credit termination |
| 02 | Third-party sale | Adverse credit termination |
| 03 | Short sale / charge-off | Adverse credit termination |
| 09 | Real Estate Owned (REO) disposition | Severe adverse termination |
| 15 | Whole-loan sale | Special termination |
| 16 | Reperforming Loan (RPL) securitization | Special termination / prior distress |
| 96 | Defect-related removal | Special termination |

### Delinquency Status vs. Zero Balance Code

These variables describe different parts of the mortgage lifecycle:

- **Current Loan Delinquency Status** describes the loan's credit condition
  during a reporting month.
- **Zero Balance Code (ZBC)** describes the reason the loan terminates or
  leaves active reporting.

A loan may therefore progress through delinquency states before eventually
receiving a Zero Balance Code.
