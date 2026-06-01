# GDPR Notes

CGMPy is a **tool**, not a service. The General Data Protection
Regulation (GDPR) applies to **data controllers** and **data
processors** that determine the purposes and means of processing
personal data. CGMPy is none of these — it is software that runs
locally on your data.

That said, if you use CGMPy to process personal data (including
health data, which is a **special category** under GDPR Art. 9), you
are likely a data controller and have obligations.

This page summarizes the **library's posture**, not legal advice.

## Posture of the CGMPy library

- **CGMPy does not collect any data.** It is a pure-Python library
  that runs locally.
- **CGMPy does not transmit any data.** No telemetry, no analytics, no
  network calls.
- **CGMPy ships with anonymization helpers** (`scripts/anonymize_cgm.py`)
  to help you comply with the data-minimization principle (Art. 5(1)(c)).
- **CGMPy explicitly forbids committing real patient data** to the
  repository (see [SECURITY.md](https://github.com/javierpa95/cgmpy/blob/main/SECURITY.md)).

## What you (the user / controller) must do

If you use CGMPy on personal data:

1. **Have a lawful basis** for processing (Art. 6).
2. **For health data**, satisfy Art. 9 (special categories), typically
   through explicit consent or another Art. 9(2) basis.
3. **Inform the data subject** (Art. 13/14) about how their data is
   used.
4. **Implement appropriate technical and organizational measures**
   (Art. 32). CGMPy is one of those measures — it does not exempt you
   from the others.
5. **Honor data subject rights** (access, rectification, erasure,
   restriction, portability — Arts. 15-20).
6. **Document your processing** (Art. 30) and run a DPIA if high-risk
   (Art. 35).

## PHI / special category data

CGM data is **health data** and therefore a special category under
Art. 9. Treat it as such:

- **Anonymize** before sharing for bug reports or examples.
- **Encrypt** at rest and in transit.
- **Restrict access** to those who need it.
- **Audit** access logs.
- **Plan for deletion** when the lawful basis expires.

CGMPy provides a basic anonymization script at
`scripts/anonymize_cgm.py`. Use it on any real dataset before
including it in a bug report.

## Anonymization vs. pseudonymization

CGMPy's helper produces **pseudonymization** (patient IDs are replaced,
but timestamps are only shifted, not removed). True anonymization
requires irreversible transformations, which depend on the use case.

If you need stronger guarantees, consider:

- Removing all timestamps (or binning to weeks/months).
- Aggregating to daily summaries.
- Using k-anonymity or differential-privacy techniques for cohort
  analyses.

## International transfers

CGMPy is hosted on GitHub. If you are in the EU and use GitHub, your
data may be transferred to the US. See
[GitHub's DPA](https://github.com/security/platform-compliance) for
their commitments under the EU-US Data Privacy Framework.

## Right to erasure

If a data subject requests erasure (Art. 17):

- Erase the data from your local storage.
- Erase any derived datasets (aggregates, models).
- If you have published CGMPy results, retract or correct the
  publication.

CGMPy does not retain any of your data, so there is nothing to erase
on the library side.

## See also

- [Privacy](privacy.md) — general privacy posture.
- [Security policy](https://github.com/javierpa95/cgmpy/blob/main/SECURITY.md).
- [GDPR full text](https://gdpr-info.eu/).
- [EDPB guidelines on health data](https://edpb.europa.eu/our-work-tools/our-documents/guidelines/guidelines-032020-data-protection-by-design-and-default_en).
