L2HUTILS=../../../l2h-utils

NAVBUTTONS=$(L2HUTILS)/contents.gif $(L2HUTILS)/next.gif $(L2HUTILS)/prev.gif \
	   $(L2HUTILS)/contents.png $(L2HUTILS)/next.png $(L2HUTILS)/prev.png

POSTL2H = $(L2HUTILS)/post_l2h

TEX=$(shell find . \( -name "*.tex" \) -a \( ! -name "imag*" \) -exec grep -q "begin{document}" {} \; -print | sed -e s+"./"++)
PDF=$(patsubst %.tex,%.pdf,$(TEX))
DVI=$(patsubst %.tex,%.dvi,$(TEX))
HTML=$(patsubst %.tex,%.html,$(TEX))
PSGZ=$(patsubst %.tex,%.ps.gz,$(TEX))

EPSFIG=$(wildcard *.eps)
PDFFIG=$(patsubst %.eps,%.pdf,$(EPSFIG))

default: $(PDF) $(DVI) $(PSGZ) $(PDFFIG)  $(HTML) 

html: $(HTML)

pdf: $(PDF)

$(DVI): %.dvi: %.tex
	@( echo "generating $@ ..." ; \
	   log=`basename $< .tex`.log ; \
	   latex $< > /dev/null 2>&1 ; \
	   latex_count=5 ; \
	   while egrep -s 'Rerun (LaTeX|to get cross-references right)' $$log && [ $$latex_count -gt 0 ] ; do \
	     echo "Rerunning latex...." ; \
	     latex $< > /dev/null 2>&1 ; \
	     latex_count=`expr $$latex_count - 1` ;\
	   done \
	)

$(PSGZ): %.ps.gz: %.dvi
	@echo "generating $@ ..." ;
	@dvips $< -o  > /dev/null 2>&1 ;
	@gzip -f `basename $< .dvi`.ps

$(PDFFIG): %.pdf: %.eps
	@echo "generating $@ ..." ;
	@epstopdf $< ;

$(PDF): %.pdf: %.tex
	@( echo "generating $@ ..." ; \
	   aux=`basename $< .tex`.aux ; \
	   rm -rf $$aux ; \
	   log=`basename $< .tex`.log ; \
	   pdflatex $< > /dev/null 2>&1 ; \
	   latex_count=5 ; \
	   while egrep -s 'Rerun (LaTeX|to get cross-references right)' $$log && [ $$latex_count -gt 0 ] ; do \
	     echo "Rerunning latex...." ; \
	     pdflatex $< > /dev/null 2>&1 ; \
	     latex_count=`expr $$latex_count - 1` ;\
	   done \
	)

$(HTML): %.html: %.tex $(DVI)
	@echo running latex2html ...
	@latex2html -init_file ../latex2html-init -math -style ../../ci.css $< ;	
	@echo post processing latex2html output
	@(for f in *.html; do \
            $(POSTL2H) $$f > /tmp/f.$$$$; \
            mv /tmp/f.$$$$ $$f; \
          done; )
	@cp $(NAVBUTTONS) .

clean:
	@rm -rf *.out *.toc *.log *.aux *.bbl *.blg 
	@rm -rf $(PDF) $(PSGZ) $(DVI) $(HTML) *.html
	@rm -rf *.pl im*.gif *.png *.gif *.old WARNINGS images.*
