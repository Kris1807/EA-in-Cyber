## This repository is used and written by:

- Kristian Pitshugin
- Tyler Corse
- Varshini Bonagiri
- Jess Hill
  For CSCI4560/6560 research project assignment.

  **This project and repo will later become Kristian's MS Thesis, with respect and credit to given to the teammates.**

## Our research question is:

"Can coevolutionary algorithms generate realistic adversarial behaviors and adaptive defense mechanisms that improve overall cyber resilience?"

## Abstract:

Cybersecurity is a constant arms race between attackers developing new malware and defenders building detection mechanisms to stop them. Traditional malware detection systems often struggle to stay effective against evolving threats. In this project, we explore how coevolutionary algorithms can be used to simulate and improve this dynamic interaction.
Instead of using real malicious code, we use a publicly available static malware dataset such as EMBER, which contains feature representations of malware and benign files. In our framework, two populations evolve together: an attacker population that mutates malware feature vectors to evade detection, and a defender population that learns to identify and block these evolving threats.
The attacker’s fitness is determined by how many defenders its malware can bypass, while the defender’s fitness depends on how accurately it can classify files rewarding successful detection and optionally penalizing false positives. Through repeated evolutionary cycles, both attackers and defenders continuously adapt to each other, creating an environment that mimics real-world adversarial evolution. By analyzing these interactions, we aim to understand how adaptive defenses emerge and how coevolution can strengthen malware detection systems. Our results are expected to show that coevolutionary training can produce defenders that are more robust, adaptable, and resilient against novel attacks compared to traditional static models. This project ultimately demonstrates how evolutionary computation can serve as a powerful experimental tool for enhancing cyber resilience in malware detection.
