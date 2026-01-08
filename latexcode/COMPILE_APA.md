# Compiling LaTeX with APA Style Citations

Your document is now configured to use **APA style** citations via `natbib` with the `apalike` bibliography style.

## Compilation Steps

1. **First compilation:**
   ```bash
   pdflatex main.tex
   ```

2. **Process bibliography with bibtex:**
   ```bash
   bibtex main
   ```

3. **Second compilation:**
   ```bash
   pdflatex main.tex
   ```

4. **Third compilation (to resolve all references):**
   ```bash
   pdflatex main.tex
   ```

## Using latexmk (Recommended)

You can also use `latexmk` which automatically handles the compilation sequence:

```bash
latexmk -pdf -pvc main.tex
```

The `-pvc` flag enables continuous compilation (auto-recompiles on save).

## APA Citation Format

With this setup, your citations will automatically appear in APA style:
- In-text: (Author, Year) format
- Bibliography: Full APA format with proper author-year formatting
- Multiple authors: (Author1, Author2, \& Author3, Year)
- Multiple citations: (Author1, Year1; Author2, Year2)

## Citation Commands

- `\cite{key}` - Produces (Author, Year)
- `\citep{key}` - Same as \cite (parenthetical)
- `\citet{key}` - Produces Author (Year) - textual citation

## Troubleshooting

If citations don't appear:
- Make sure you've run `bibtex main` after the first pdflatex
- Check that `references.bib` is in the same directory
- Verify citation keys match between `.tex` and `.bib` files
- Run pdflatex multiple times until all references resolve

