(TeX-add-style-hook "Man"
 (function
  (lambda ()
    (LaTeX-add-labels
     "samples"
     "sec:gausspdf"
     "like"
     "sub:apriori"
     "gaussmod"
     "discr"
     "iso"
     "unsup"
     "algos"
     "eq:dist")
    (TeX-add-symbols
     '("PBS" 1)
     '("com" 1)
     '("mat" 1)
     "pdfgraphics"
     "tab"
     "temp"
     "RR"
     "RL"
     "CC")
    (TeX-run-style-hooks
     "a4wide"
     "color"
     "html"
     "array"
     "amsmath"
     "amssymb"
     "rotating"
     "graphicx"
     "latex2e"
     "art10"
     "article"
     "a4paper"
     "titlepage"))))

