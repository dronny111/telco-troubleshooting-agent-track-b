# [Track B] Clarification on Output Content and Format + Network Device List + Phase 1 Ground Truth

*9 May 2026, 11:35 · 17 replies*

## Official Output Rules

1. **Topology link questions** — Port numbers in the answer must follow the interface name in the output of the `display current-configuration` command. If `display current-configuration` is unavailable, use the interface name from `display interface brief`. If the interface information is followed by other information such as interface bandwidth or rate, **do not retain** that information.

2. **Path questions** — All nodes (including nodes on L2 paths) on the paths must be output in the answer. If there are multiple paths, each path is output on a separate line; output multiple lines in total.

3. **Fault-cause questions** — You must select the **most specific and closest** fault cause.

---

## Discussion

### AntonioDeDomenico — 9 May 2026, 12:25 · ⬆ 3

To help participants better answer the questions in Track B, here is the list of all device names involved in the questions:

```
AGG_SW_01                AGG_SW_02                AGG_SW_03                AGG_SW_04
BJHQ_CSR1000V_GW_01      BaiduWebServer01         ChinaUnicom_SW
Core_SW_01               Core_SW_02
EMPLOYEE_WIFI_CLIENT01   EMPLOYEE_WIFI_CLIENT02   EMPLOYEE_WIFI_CLIENT03
FW_01                    FW_02
GUEST_WIFI_CLIENT01      GUEST_WIFI_CLIENT02      GUEST_WIFI_CLIENT03
GoogleWebServer01        HQ-DHCP-Server           HQ_DNS_Server_01
HQ_FIN_Client01          HQ_FIN_PC01              HQ_FTP_Server_01
HQ_HR_AP01               HQ_HR_PC01               HQ_HTTP_Server_01
HQ_MKT_AP01              HQ_MKT_Client01          HQ_MKT_PC01
HQ_PROC_AP01             HQ_PROC_PC01             Internet_PC01
Outside_FTP_Client01     PE1                      PE2                      PE3
SH_AR                    SH_Core                  SH_FAC_PC01              SH_SAL_PC01
SH_STO_PC01              SW-DMZ-ACC-01            SZ_AR                    SZ_Core
SZ_Server_Cluster1       SZ_Server_Cluster2       SZ_Server_Cluster3
```

#### Sutee82 — 10 May 2026, 07:41 · ⬆ 4

> Hi Antonio, my agent managed to query these devices as well and got valid outputs as far as I can tell:
>
> ```
> DEV-BL-01           DEV-PE-01           DEV-SP-02
> DEV-SP-01           DEV-BL-02           DEV-FW-01
> DEV-SL-01           DEV-SL-02           DEV-PE-02
> DEV-FW-02           DEV-PE-03           DEV-PC-01
> DEV-PC-02           DEV-CUS-FW-01
> DEV-CUS-{BL-01, SW-01, SW-02, SL-01, PC-01}
> ```
>
> Are these aliases or am I missing something?

---

### Cisco_ — 9 May 2026, 12:34

This is a very important message. Every participant should know it. This can help us better solve the questions.

---

### netadmin — 9 May 2026, 16:46 · ⬆ 1

> @AntonioDeDomenico — If there are multiple lines, does their sorting order (e.g. compared to the ground truth) affect correctness? If so, what sorting strategy does the ground truth use?

#### AntonioDeDomenico — 11 May 2026, 12:22

No, the order does not count.

---

### Juliuss (Freelance) — 9 May 2026, 17:38

For questions phrased *"please provide a minimal set of fault root causes"* where the simulator has injected multiple independent root causes (each of which would block the connection alone), should the answer:

- **(a)** include all injected faults on separate lines (and does the order matter)?
- **(b)** include only the most specific one?
- **(c)** accept any one valid root cause?

---

### ssifisafi — 9 May 2026, 17:45

**Réponse complète et finale pour le format multi-chemins :**

**Règle générale**
- Un chemin valide = une ligne
- Plusieurs chemins valides = plusieurs lignes distinctes
- Format strict : `NodeA->NodeB->...->Destination`
- Pas d'espaces superflus au début, à la fin, ou entre les flèches
- Chaque ligne doit être autonome et représenter un chemin complet

**Application à la question 36**

Deux sauts possibles depuis `Demeter-Node-01` vers `10.1.1.10` :
- via `182.158.2.9`
- via `182.158.2.1`

La réponse doit contenir deux lignes :

```
Hermes-Node-01->...->Demeter-Node-01->182.158.2.9->2.1.5.1->10.1.1.10
Hermes-Node-01->...->Demeter-Node-01->182.158.2.1->2.1.5.1->10.1.1.10
```

**Points clés**
- Ne jamais fusionner les chemins en une seule ligne
- Répéter la partie commune (`Hermes-Node-01->...->Demeter-Node-01`) pour chaque chemin
- Varier uniquement la portion qui change (ici l'adresse du saut)
- Pour 3 ou 4 chemins, écrire 3 ou 4 lignes dans le même format

---

### gbengapelumi (Prairie View A&M University) — 10 May 2026, 00:35 · ⬆ 3

Could you kindly give a few examples of what the output should look like for the different scenarios?

---

### netadmin — 10 May 2026, 14:52 · ⬆ 2

+1 — Please show several ground truth examples. Some of the questions are marked with all 0, but with manual inspection they are correct. @AntonioDeDomenico

---

### Ammart90 — 11 May 2026, 12:04

@AntonioDeDomenico — I was out of the submissions limit before this post; I didn't have the chance to align.

---

### AntonioDeDomenico — 11 May 2026, 12:23

I have uploaded the **ground truth of Phase 1 for Track B** here:

🔗 https://huggingface.co/datasets/netop/Telco-Troubleshooting-Agentic-Challenge/blob/main/Track%20B/data/Phase_1/gt_phase1.csv

#### Namotechno — 11 May 2026, 12:59 · ⬆ 3

What about those of us who had already exhausted our submissions before your post was published? I don't think that's quite fair.

#### soul0101 — 11 May 2026, 17:17

Yeah, I agree — this changes things quite a bit. I personally joined the challenge fairly late, so with only 3 training examples it was difficult to build and validate a strong solution in time. It would be greatly appreciated if we could get a few additional submissions 🙏 @AntonioDeDomenico

#### vaderyang — 12 May 2026, 09:48

I really hope this would have been published earlier — it changed the competition strategy completely. This created an obvious unfair advantage for users not submitting yet, especially those who have dummy accounts. Anyway, thanks for one more submission limit increase.

#### Juliuss (Freelance) — 12 May 2026, 10:18

Thanks Antonio. In Phase 1 we saw a lot of "games" in the public leaderboard. I am okay with current limits on submission.

---

### jiang_janice — 16 May 2026, 14:37 · ⬆ 2

Hi @AntonioDeDomenico — I'd like to ask if a device list will be provided for Phase 3? In Phase 2, most challenges use Linux hosts as the starting/ending point, and without a device list it's impossible to jump to neighboring nodes via commands. If Phase 3 also has a similar issue, I believe our current agent will be unable to generalize to other scenarios.

---

### mayi

Hi @AntonioDeDomenico —

I would like to raise a doubt regarding the interpretation of *"minimal root cause"* and *"the most specific and closest fault cause."*

In our understanding, if an upstream node is already identified as faulty, and that fault can explain the downstream impact, then the upstream fault should be considered the closest necessary root cause to report. Based on that interpretation, we initially stopped at the upstream faulty node and submitted only that cause.

However, the answer was judged incorrect. We then continued tracing further downstream and found another faulty node. After submitting both the upstream and downstream faults, the answer was accepted.

This makes us unsure about the intended meaning of *"minimal"* and *"most specific/closest fault cause."* If a downstream fault also needs to be reported even when an upstream fault already exists, does that mean the expected answer may include multiple fault causes along the upstream/downstream path? If so, how should we distinguish between a fault that is merely affected by an upstream issue and a fault that must be reported as an additional root cause?

Could you clarify whether *"minimal root causes"* means **all independently faulty nodes** that contribute to the final symptom, even if some are downstream of another reported faulty node?
