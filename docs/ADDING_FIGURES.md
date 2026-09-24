# Adding figure 101 and beyond

The library is designed to grow. Each new figure is one standalone
script plus one index line; everything else (guide, gallery, CI) follows
automatically.

1. **Scaffold it**

   ```
   python tools/new_figure.py 101 my_short_slug "One-line description"
   ```

   This writes `figures/fig101_my_short_slug.py` from the house template
   and prints the README bullet to paste.

2. **Honour the contract** (the template enforces it):
   - one-line docstring stating what the figure demonstrates;
   - `HERE = Path(__file__).resolve().parent` and
     `plt.style.use(str(HERE / "publication.mplstyle"))`;
   - every random draw through a seeded `np.random.default_rng(...)`;
   - curves **derived from governing equations**, never sketched;
   - a built-in self-check: an `assert` on a landmark the physics fixes
     (a peak on a locus, a tangency, an area identity), plus a printed
     confirmation with the key numbers;
   - `fig.savefig(HERE / f"fig101_my_short_slug.{ext}")` for both
     `png` and `pdf`.

3. **Index it**: add exactly one line to `figures/README.md`, matching
   the existing format:

   ```
   - `fig101_my_short_slug.py` — one-line description
   ```

   `build_guide.py` parses that line (it accepts >= 100 entries), makes
   the thumbnail, and typesets the new entry into Part II of the guide.
   Entries 101+ fall under section H unless you add a new section in
   `SECTIONS`; an optional domain tag goes in `DOMTAG`.

4. **Verify like the first hundred**: `make figures` (your assert must
   pass), *look at the pixels*, then `make guide gallery`. CI repeats
   all of it on push.
