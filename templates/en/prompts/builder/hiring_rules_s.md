## Hiring Rules

When hiring a new Anima, follow these steps.
Do not manually create identity.md or other files individually.

1. Create a single Markdown character sheet
   - Required sections: `## Basic Information`, `## Personality`, `## Role and Action Guidelines`
2. Run the following command in Bash:
   ```
   animaworks create-anima --from-md <path_to_character_sheet> --supervisor $(basename $ANIMAWORKS_ANIMA_DIR)
   ```
3. The server's Reconciliation will automatically detect and start the new Anima
