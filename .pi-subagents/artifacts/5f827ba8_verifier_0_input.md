# Task for verifier

[Read from: C:\Users\pre\Documents\Projects\trail\outputs\quantized-llm-decode-kernels.draft.md, C:\Users\pre\Documents\Projects\trail\notes\quantized-llm-decode-kernels-research-summary.md]

Verify cited draft at outputs/quantized-llm-decode-kernels.draft.md for URL reachability, inline citation/source consistency, and unsupported source mappings. Do not rewrite the whole draft unless necessary. Return FATAL/MAJOR/MINOR findings and suggested fixes. If you create a cited version, save as outputs/quantized-llm-decode-kernels.verified.md.

---
Update progress at: C:\Users\pre\Documents\Projects\trail\.pi-subagents\artifacts\progress\5f827ba8\progress.md

---
**Output:**
Write your findings to exactly this path: C:\Users\pre\Documents\Projects\trail\.pi-subagents\artifacts\outputs\5f827ba8\notes\quantized-llm-decode-kernels-verifier.md
This path is authoritative for this run.
Ignore any other output filename or output path mentioned elsewhere, including output destinations in the base agent prompt, system prompt, or task instructions.

## Acceptance Contract
Acceptance level: checked
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Return a concise result and residual risks when applicable

Required evidence: changed-files, tests-added, commands-run, residual-risks, no-staged-files

Finish with a fenced JSON block tagged `acceptance-report` in this shape:
Use empty arrays when no items apply; array fields contain strings unless object entries are shown.
`criteriaSatisfied[].status` must be exactly one of: satisfied, not-satisfied, not-applicable.
`commandsRun[].result` must be exactly one of: passed, failed, not-run.
`manualNotes` and `notes` are optional strings; an empty string means no note and does not satisfy `manual-notes` evidence.
```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "specific proof"
    }
  ],
  "changedFiles": [
    "src/file.ts"
  ],
  "testsAddedOrUpdated": [
    "test/file.test.ts"
  ],
  "commandsRun": [
    {
      "command": "command",
      "result": "passed",
      "summary": "short result"
    }
  ],
  "validationOutput": [
    "validation output or concise summary"
  ],
  "residualRisks": [
    "none"
  ],
  "noStagedFiles": true,
  "diffSummary": "short description of the diff",
  "reviewFindings": [
    "blocker: file.ts:12 - issue found, or no blockers"
  ],
  "manualNotes": "anything else the parent should know"
}
```