# ATT&CK retrieval (Task 6.1)

Finds candidate MITRE ATT&CK techniques for an incident from the authentication behaviour it shows. The candidates go to the investigation summaries (6.2), which use them as context, and to verification (6.3), which decides whether the evidence supports each one. **A candidate is not a finding:** it is retrieved because its text matches a behaviour.

## Snapshot

| | |
| --- | --- |
| Source | Enterprise ATT&CK **19.2** (collection modified 2026-08-05), STIX 2.1 bundle from [mitre-attack/attack-stix-data](https://github.com/mitre-attack/attack-stix-data) |
| Retrieved | 2026-10-06 |
| Bundle SHA-256 | `dc1639caa5501d720e280cf1cbd8fbe009884a0c9b3e6e9ed9d0c25166c3d8f4` |
| Kept | 697 active techniques (222 techniques + 475 sub-techniques); revoked and deprecated ones are dropped |
| Catalogue | `data/reference/attack/enterprise_techniques.json` (1.4 MB, committed) |

Each technique keeps its ID, name, description (citation markers removed), tactics, platforms, parent, `attack.mitre.org` URL, and the **data components and log sources** that ATT&CK's detection strategies list for it. For example, Pass the Hash lists *Logon Session Creation* and *WinEventLog:Security EventCode=4624, 4648*. 6.3 can use these to tell whether a technique is visible in logon records at all.

Rebuild or move to a new release:

```powershell
python scripts/fetch_attack_snapshot.py               # 19.2; refuses if the download's hash differs
python scripts/fetch_attack_snapshot.py --version 19.3
```

The 54 MB bundle is kept in `data/raw/attack/` and is not committed. ATT&CK is © The MITRE Corporation, used under its [Terms of Use](https://attack.mitre.org/resources/legal-and-branding/terms-of-use/).

## How retrieval works

- **Index:** TF-IDF over each technique's name (counted three times), description and data components, ranked by cosine similarity (`src/dualscope/attack/retrieval.py`). It needs no model download or GPU, and the same query always returns the same ranking.
- **Cut-off:** results scoring below **0.10** are dropped, so a weak query returns nothing rather than a forced match. The value was chosen on the sample queries below: the strongest weak query scored 0.083, and the weakest wanted match (Valid Accounts) scored 0.107.
- **Logon-visible only:** template searches skip techniques whose data components include nothing an authentication log can show. Those components are *Active Directory Credential Request*, *Logon Session Creation*, *Logon Session Metadata* and *User Account Authentication*. Without this filter, "Network Logon Script" (T1037.003) outranks Remote Services for a lateral-movement query.

## Query templates (behaviours)

`src/dualscope/attack/queries.py` checks an incident's individual events (from the alert handoff's `events.parquet`) against four rules. A behaviour counts when **at least one event** matches. It carries the number of matching events and up to 20 `auth.txt` references, so 6.2 can cite them and 6.3 can check them.

| Behaviour | An event matches when | Search phrase retrieves |
| --- | --- | --- |
| `failed_logons` | result is `Fail` | T1110.001 Password Guessing, T1110.003 Password Spraying, T1110 Brute Force |
| `ntlm_logon_to_new_destination` | NTLM, network logon, to a destination the user (or the source computer) never reached before | T1550.002 Pass the Hash, plus false candidates (see below) |
| `network_logon_to_new_destination` | network logon to a never-reached destination | T1021.002 SMB/Windows Admin Shares, T1021 Remote Services |
| `logon_from_new_source` | the user's first event from that source computer | T1078 Valid Accounts |

"Never before" means since day 1 of the data (see [lanl_features.md](lanl_features.md)). Log-off events are ignored by the three novelty rules. A log-off is recorded on the machine being left, often with source = destination, so it does not show a new logon path; counting log-offs made `logon_from_new_source` fire on 438 incidents instead of 232.

There is no Kerberos-ticket template. Ticket requests (TGS/TGT) are about 16% of events in alerted hours and have no novelty flag that sets them apart.

If no rule matches, the incident gets no candidates and `insufficient_evidence: true`. That is a valid outcome, which 6.2 should report as "no supported mapping".

### Why event-level rules

Rules on hourly counts ("any NTLM in the hour") fired on every one of the 497 alerted incidents, so Pass the Hash would have been suggested everywhere. Requiring a single event that combines the signals (for example an NTLM network logon *to a new host*) narrows it to 190, and every behaviour comes with the events that triggered it.

## Manually checked samples

`python scripts/check_attack_retrieval.py` reruns these and fails if any expected top result changes. It writes `outputs/attack_retrieval/sample_checks.json`. Scores are cosine similarities.

| Query | Returned (score) | Review |
| --- | --- | --- |
| failed_logons | T1110.001 Password Guessing (0.233), T1110.003 Password Spraying (0.199), T1110 Brute Force (0.162) | All relevant. Spraying needs failures across many accounts, which a single-user incident cannot show. |
| ntlm_logon_to_new_destination | T1550.002 Pass the Hash (0.319), T1550.003 Pass the Ticket (0.230), T1556.004 Network Device Authentication (0.115), T1110.002 Password Cracking (0.106) | Pass the Hash is relevant: an NTLM logon to a new host is *consistent with* it but does not prove a hash was used. The other three are **false candidates**: Pass the Ticket is Kerberos, not NTLM; Network Device Authentication is about altering network devices; Password Cracking happens offline and leaves no logon record. 6.3 must reject them. |
| network_logon_to_new_destination | T1021.002 SMB/Windows Admin Shares (0.167), T1021 Remote Services (0.120) | Relevant. Logon records cannot show *which* service (SMB, RDP…) was used, so the parent T1021 is the safer mapping. |
| logon_from_new_source | T1078 Valid Accounts (0.107) | Relevant; consistent with stolen credentials, never proof. |
| kerberos_tickets (free text) | T1558 Steal or Forge Kerberos Tickets (0.384), T1558.002 Silver Ticket, T1558.001 Golden Ticket, T1550.003 Pass the Ticket, T1558.003 Kerberoasting | Relevant; shows free-text search works for 6.2's own queries. |
| "user logged on during the working day" | nothing | Correct: routine activity, no technique. |
| "computer account authenticating to many hosts" | nothing | Correct: normal machine-account behaviour. |
| "quarterly marketing budget spreadsheet" | nothing | Correct: unrelated text. |

### On the days 17–30 alert incidents

Run over all 497 incidents in the alert handoff ([alert_handoff.md](alert_handoff.md)):

| Behaviour | Incidents |
| --- | --- |
| `network_logon_to_new_destination` | 279 |
| `logon_from_new_source` | 232 |
| `ntlm_logon_to_new_destination` | 190 |
| `failed_logons` | 68 |
| none (insufficient evidence) | 107 |

The one red-team incident (`INC-TEST-D28-U737_DOM1-003`) fires `ntlm_logon_to_new_destination` and `network_logon_to_new_destination`:

- `auth.txt:463603187`: U737 logs on from C19497 to C1973 with NTLM, a first for both user and host pair.
- `auth.txt:463665849`: U737 logs on from C19497 to C24793 with Kerberos, also a first.

Its top candidates are Pass the Hash and SMB/Windows Admin Shares. It does not fire `logon_from_new_source`: its three new-source events are all log-offs.

## Using it from 6.2

The investigation pipeline that uses this is described in [genai_investigation.md](genai_investigation.md).

```python
from dualscope.attack.retrieval import TechniqueRetriever
from dualscope.attack.queries import retrieve_candidates

retriever = TechniqueRetriever.from_file()            # committed 19.2 catalogue
result = retrieve_candidates(incident_events, retriever)  # rows of events.parquet for one incident
# result["behaviours"]: behaviour, description, event_count, evidence_references
# result["candidates"]: technique_id, name, score, url, description, data_components, retrieved_by
# result["insufficient_evidence"], result["snapshot"] (ATT&CK version and bundle hash, for logging)
retriever.search("free-text query", k=5)               # 6.2's own queries
```

`TechniqueCatalog.load(...)` gives 6.3 an ID check (`"T1550.002" in catalog`) and each technique's data components.
