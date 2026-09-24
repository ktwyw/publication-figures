PY ?= python3
FIGDIR := figures
SCRIPTS := $(sort $(wildcard $(FIGDIR)/fig*.py))

.PHONY: all figures guide gallery check clean

all: figures guide gallery

figures:
	@set -e; for s in $(SCRIPTS); do \
	  echo "== $$s"; MPLBACKEND=Agg $(PY) $$s; \
	done

check: figures
	@n=$$(ls $(FIGDIR)/fig[0-9]*.png | wc -l); \
	 echo "rendered PNGs: $$n"; test $$n -ge 100
	@n=$$(ls $(FIGDIR)/fig[0-9]*.pdf | wc -l); \
	 echo "rendered PDFs: $$n"; test $$n -ge 100

guide:
	cd $(FIGDIR) && $(PY) build_guide.py

gallery:
	$(PY) tools/make_gallery.py

clean:
	rm -f $(FIGDIR)/fig[0-9]*.png $(FIGDIR)/fig[0-9]*.pdf
	rm -rf $(FIGDIR)/thumbs
	rm -f $(FIGDIR)/figure_guide.aux $(FIGDIR)/figure_guide.log \
	      $(FIGDIR)/figure_guide.out $(FIGDIR)/figure_guide.toc
