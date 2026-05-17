# Reports Folder

This folder contains the current paper package for continuation and submission.

## Main Report Files

- `MAI622_Abdullah_Shamsa_AUE_Template_Report.docx` - main editable Word report.
- `MAI622_Abdullah_Shamsa_AUE_Template_Report.pdf` - generated PDF copy.
- `MAI622_Abdullah_Shamsa_AUE_Template_Report.tex` - generated LaTeX source copy.

## Supporting Material

- `assets/` - figures used in the report, including the architecture diagram, training curve, and tracking samples.
- `reference_notes/` - text extracted from the CholecTrack20 and EA-StrongSORT reference papers for writing support.

## Regenerating The Report

From the project root:

```powershell
D:\conda_envs\deep_learning_project\python.exe tools\generate_aue_term_paper.py
```

The generator uses the AUE template if it exists at the original local path. If the template is not available, it still generates a readable Word document with the same report content.
