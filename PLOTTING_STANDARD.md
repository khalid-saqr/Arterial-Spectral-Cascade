# Physics of Fluids / AIP plotting standard

The publication-figure layer is locked to the following mechanical requirements used by AIP Publishing for *Physics of Fluids* submissions.

- Maximum single-column width: 3.37 in (8.5 cm).
- Maximum two-column width: 6.69 in (17 cm).
- Maximum depth: 8.25 in (21.1 cm).
- Minimum text size at final reproduction: 8 pt.
- Minimum reproduced line width: 0.5 pt.
- Multipart figures use `(a)`, `(b)`, ... panel labels.
- Figures are generated at final publication dimensions.
- PDF and SVG outputs retain vector line art and embedded/retained text.
- PNG output is written at 600 dpi for line/combination graphics.
- Curves must not rely on color alone for identification; line style and/or markers are also used.
- Figure titles are not embedded inside scientific panels; captions belong in the manuscript.
- Alt-text sidecars are generated with the publication figures.

The implementation is in `src/arterial_spectral_cascade/plotting.py`.

Official author-instruction reference:

https://publishing.aip.org/resources/researchers/author-instructions/
