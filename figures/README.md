# Publication-figure starter kit (matplotlib)

One hundred and fifty standalone scripts covering the most common publication figure types, all sharing one style sheet so your whole paper looks consistent.

**Companion guide:** `figure_guide.pdf` (built by `build_guide.py` from this README) — Part I distils the craft, Part II is the illustrated catalog of all 150 examples, Part III the failure-mode catalog, Part IV working checklists.

## Requirements

```
pip install numpy scipy matplotlib
```

## Usage

Keep `publication.mplstyle` next to the scripts and run any of them directly, e.g. `python fig001_curve_fit.py`. Each script writes a 600-dpi PNG (for previewing) and a vector PDF (what you should actually submit) beside itself.

- `fig001_curve_fit.py` — data + model fit with error bars (curve_fit, parameter annotation)
- `fig002_multipanel.py` — 2×2 composite with bold a–d panel labels, s.e.m. bands, colorbar, distributions, box + raw points
- `fig003_regression_ci.py` — scatter + linear regression with an analytically computed 95% confidence band and R²/P/n annotation
- `fig004_bars_stats.py` — bars with individual data points, s.e.m. error bars, and significance brackets from t-tests
- `fig005_heatmap.py` — annotated correlation heatmap with a diverging colormap pinned to [−1, 1]
- `fig006_raincloud.py` — raincloud plots: half-violin distribution + box summary + jittered raw data
- `fig007_kaplan_meier.py` — Kaplan–Meier survival curves with censor marks, median guides, log-rank test (scipy ≥ 1.11), and a number-at-risk table
- `fig008_volcano.py` — volcano plot with thresholds, per-side labeled top hits, and rasterized points for small PDFs
- `fig009_spectrum_inset.py` — annotated spectrum with a zoom inset via `inset_axes` + `indicate_inset_zoom`
- `fig010_loglog_powerlaw.py` — log-log power-law fit with a residual panel (`height_ratios` + `sharex`)
- `fig011_image_scalebar.py` — microscopy-style image: scale bar, percentile contrast stretch, calibrated colorbar, no axis ticks
- `fig012_forest_plot.py` — meta-analysis forest plot: weighted squares, pooled diamond, aligned OR/weight text columns, heterogeneity stats
- `fig013_roc_curves.py` — ROC curves with AUC computed from scratch in numpy and a marked Youden optimum
- `fig014_joint_marginals.py` — scatter with marginal histograms/KDEs via GridSpec (a "joint plot" without seaborn)
- `fig015_clustered_heatmap.py` — clustered heatmap with scipy dendrograms aligned to the reordered matrix (a "clustermap" from scratch)
- `fig016_paired_slope.py` — paired before/after slope chart with per-subject lines, group means ± 95% CI, and paired t-test
- `fig017_contour_map.py` — 2-D response surface: filled + labelled contours, design points, marked optimum
- `fig018_bland_altman.py` — Bland–Altman method comparison: bias, limits of agreement, CI bands, right-edge value labels
- `fig019_ridgeline.py` — ridgeline (joy) plot: overlapping KDEs with occlusion and fills colour-mapped to each row's mean
- `fig020_coefficients.py` — regression coefficient dot-and-whisker plot comparing adjusted vs unadjusted models
- `fig021_stacked_composition.py` — normalized stacked-area composition with semantic colours and direct right-edge labels (no legend)
- `fig022_streamplot.py` — vector field as speed-coloured streamlines with a masked body and an inset colorbar matched to equal-aspect axes
- `fig023_wind_rose.py` — wind rose: stacked speed classes on a compass-oriented polar axis, radial percent ticks
- `fig024_qq_plot.py` — normal Q–Q plot with quartile reference line, Beta-distribution 95% envelope, and Shapiro–Wilk test
- `fig025_calibration.py` — calibration curve with Wilson CIs, Brier/ECE, and a prediction-histogram panel
- `fig026_sankey.py` — Sankey/alluvial flow diagram from scratch: smoothstep ribbons via fill_between, widths ∝ flow
- `fig027_pca_biplot.py` — PCA via SVD with 95% confidence ellipses per group and loading arrows
- `fig028_waterfall.py` — waterfall chart: floating signed contributions between start and end totals with connectors
- `fig029_time_series_events.py` — dated time series: ConciseDateFormatter axis, rolling mean, shaded event periods, annotated peak
- `fig030_fan_chart.py` — forecast fan chart: nested 50/80/90% quantile bands from an ensemble with proxy-patch legend
- `fig031_upset.py` — UpSet plot of set intersections: bars + dot matrix + set sizes, GridSpec with shared axes
- `fig032_dose_response.py` — dose–response curves: 4-parameter logistic fits in log space, EC50 drops and labels
- `fig033_confusion_matrix.py` — row-normalized confusion matrix with counts, percentages, and adaptive text colour
- `fig034_training_curves.py` — ML training curves: mean ± s.d. across seeds, train/val line styles, best-epoch marker
- `fig035_manhattan.py` — GWAS Manhattan plot: alternating chromosome colours, significance thresholds, rasterized points, labelled loci
- `fig036_ternary.py` — ternary composition diagram from scratch: grid, edge ticks, rotated axis titles, inset colorbar
- `fig037_raster_psth.py` — spike raster (eventplot) + PSTH with Gaussian-smoothed rate and shaded stimulus window
- `fig038_small_multiples.py` — small-multiples grid: shared axes, per-panel trend fits coloured by sign, sorted panels, supxlabel/supylabel
- `fig039_ecdf_comparison.py` — ECDF comparison via Axes.ecdf with two-sample KS tests and the maximum-gap marker
- `fig040_carpet_plot.py` — hour-of-day × day-of-year carpet plot with pcolormesh cell edges and month-centred ticks
- `fig041_dumbbell.py` — dumbbell chart of change between two time points, sorted, with Δ labels at the right edge
- `fig042_split_violin.py` — split violins: two conditions per category sharing a baseline, white median bars
- `fig043_broken_axis.py` — bar chart with a broken y-axis: stacked axes, hidden joint, diagonal break marks
- `fig044_hexbin_density.py` — hexbin density with LogNorm for 10^5+ points, polygon gate with enclosed fraction, stroked labels
- `fig045_funnel_plot.py` — contour-enhanced funnel plot: significance shading, pseudo-95% funnel, Egger's test
- `fig046_bubble_chart.py` — bubble chart: log axis with plain ticks, colour by group, hand-built size legend (diameter = √area)
- `fig047_superplot.py` — SuperPlot: from-scratch beeswarm coloured by replicate, replicate means overlaid, test on n = replicates
- `fig048_tornado.py` — tornado diagram for one-way sensitivity analysis, sorted by swing, currency tick formatter
- `fig049_energy_levels.py` — Jablonski energy-level diagram drawn in code: states, vibrational sublevels, radiative/non-radiative arrows
- `fig050_bode_plot.py` — Bode magnitude/phase plot from scipy.signal with −3 dB marker, slope guide, log grid
- `fig051_xrd.py` — stacked powder XRD with reference stick pattern and rotated (hkl) indices
- `fig052_tga_dsc.py` — TGA + DSC on a colour-coded twin axis with mass-loss arrows and "exo up" marker
- `fig053_stress_strain.py` — engineering stress–strain: 0.2% offset yield construction, UTS/fracture markers, elastic-region inset
- `fig054_cyclic_voltammetry.py` — CVs at five scan rates with a Randles–Ševčík (i_p vs √v) inset
- `fig055_nyquist_eis.py` — Nyquist impedance plot with Randles-model fit, frequency tags, and a code-drawn equivalent-circuit inset
- `fig056_isotherm_psd.py` — type-IV N₂ isotherm with H1 hysteresis (IUPAC open/closed symbols) + BJH pore-size panel
- `fig057_arrhenius.py` — Arrhenius plot with weighted fit, E_a, and an exact secondary °C axis via `secondary_xaxis`
- `fig058_phase_diagram.py` — parameterised binary eutectic phase diagram: labelled fields, eutectic point, tie line
- `fig059_ftir_stack.py` — stacked FTIR on a reversed wavenumber axis with dotted band guides and rotated assignments
- `fig060_tauc_bandgap.py` — Tauc plot: linear-region fits extrapolated to the abscissa for optical band gaps
- `fig061_carreau_flowcurve.py` — journal-style theory figure: Carreau slit-flow curves (Cu vs Q) computed from the wall-shear integral, with small-Cu/power-law/large-Cu asymptotes, slope triangles, marker-triple legends, boxed inward-tick axes, and STIX serif math
- `journal_style.py` — shared module for the theory figures: physics-journal rc overrides (STIX serif, boxed inward-tick axes), slope-triangle helper, panel labels
- `fig062_moody.py` — Moody diagram computed from 64/Re and the Colebrook equation (vectorised fixed-point iteration), transition band, direct roughness labels
- `fig063_falkner_skan.py` — Falkner–Skan boundary layers by shooting (solve_ivp + brentq with blow-up events): profiles across the pressure-gradient range and wall shear f''(0) vs β with separation/Blasius/stagnation anchors
- `fig064_thiele.py` — effectiveness factor vs generalised Thiele modulus for slab/cylinder/sphere (tanh, Bessel I₁/I₀, coth), asymptotes and slope triangle
- `fig065_rtd_tanks.py` — tanks-in-series residence-time distributions with the σ²=1/N scaling inset
- `fig066_vdw_maxwell.py` — van der Waals isotherms with the Maxwell equal-area construction solved by brentq (analytic antiderivative), binodal dome, analytic spinodal, shaded equal-area lobes
- `fig067_transient_conduction.py` — slab-conduction eigenseries (ζ tan ζ = Bi solved per branch), one-term Heisler approximation with validity band, and ζ₁ vs Bi with both asymptotes
- `fig068_mccabe_thiele.py` — McCabe–Thiele construction with the stage-stepping algorithm coded (equilibrium/operating lines, sloped q-line, numbered stages, automatic feed-stage detection)
- `fig069_tafel.py` — Tafel plot from Butler–Volmer kinetics with a cathodic mass-transfer limit, dashed extrapolations meeting at i₀, slope labels matched to the axes aspect
- `fig070_breakthrough.py` — fixed-bed breakthrough curves (Thomas-type logistic), breakthrough/exhaustion markers, shaded mass-transfer zone
- `fig071_ellingham.py` — Ellingham diagram from ΔH − TΔS with the Mg boiling-point kink, the exact 1000 K carbothermic Fe crossing, along-line labels plus nudged right-margin labels with leaders
- `fig072_ttt.py` — TTT diagram whose C-curves emerge from ln t = a/T + b/(T(Tm−T)²), with A₁/Ms/Mf lines and two cooling paths (quench → martensite, slow → pearlite)
- `fig073_common_tangent.py` — common-tangent construction on regular-solution G curves, solved robustly by parameterising the tangent by slope (brentq), with chemical-potential intercepts and coexisting compositions
- `fig074_pourbaix.py` — simplified Fe Pourbaix (E–pH) diagram: every boundary a Nernst line, self-consistent triple points, water-stability window, filled immunity/corrosion/passivation regions
- `fig075_ashby.py` — Ashby modulus–density selection chart: material-family ellipses drawn parametrically in log space, minimum-mass guide lines E/ρ, E^{1/2}/ρ, E^{1/3}/ρ with aspect-matched labels
- `fig076_residue_curves.py` — ternary residue curve map integrated from dx/dξ = x − y (constant α), node topology labelled, direction arrows anchored mid-descent to dodge the near-node crawl
- `fig077_hallpetch_weibull.py` — Hall–Petch fit with a nonlinear secondary grain-size axis, plus Weibull median-rank analysis with the double-log linearisation and a secondary failure-probability axis
- `fig078_evans.py` — Evans mixed-potential diagram: anodic dissolution vs activation- and diffusion-controlled cathodics; i_corr = i_L when the anodic line meets the vertical limit
- `fig079_cstr_multiplicity.py` — CSTR van Heerden diagram: sigmoid heat generation vs removal lines, three steady states (stable/unstable), ignition and extinction tangencies found by brentq on x′(T) = κ
- `ashby.py` — shared module for Ashby-style selection charts: log-space family bubbles (major axis along the tilt), decade axes, and guide lines whose label rotations are computed from the rendered axes aspect
- `fig080_ashby_strength_density.py` — Ashby strength–density chart on the module, with minimum-mass guide lines for ties (σ/ρ), beams (σ^{2/3}/ρ) and panels (σ^{1/2}/ρ)
- `fig081_ashby_toughness_modulus.py` — Ashby K_IC–E chart with toughness contours G_c = K²/E (slope ½), making the square in the toughness relation visible
- `fig082_smith_chart.py` — Smith chart generated from the bilinear map Γ = (z−1)/(z+1), with a worked two-step match: a series line (ℓ ≈ 0.24λ, clockwise at constant |Γ|) onto r = 1, then a series capacitor to the centre
- `fig083_ber_waterfall.py` — BER waterfalls from Q-function expressions for BPSK/QPSK, 16-QAM and 64-QAM, an illustrative 6 dB coding-gain shift, and the Shannon limit at −1.59 dB
- `fig084_buckley_leverett.py` — Buckley–Leverett waterflood: Welge tangent from (S_wc, 0) fixing the shock saturation and breakthrough average, plus rarefaction-and-shock saturation profiles at two times
- `fig085_mohr_coulomb.py` — Mohr–Coulomb failure circles tangent to τ = c + σ tanφ by construction (centre-to-line distance equals radius analytically), with the friction angle and cohesion annotated at equal aspect
- `fig086_streeter_phelps.py` — Streeter–Phelps dissolved-oxygen sag for three reaeration rates, critical points from the analytic t_c and D_c, and the 5 mg/L aquatic-life threshold
- `fig087_antenna_patterns.py` — polar dB patterns computed from array theory: half-wave dipole element factor and the N = 8, d = λ/2 array factor, broadside and steered to 60° (β = −π cos 60°), with the −13 dB first side lobe
- `fig088_roofline_amdahl.py` — roofline model (P = min(P_peak, BW·I), ridge point, no-vectorisation ceiling, kernels under their roofs; reuses ashby.setup/slope_angle) beside Amdahl saturation vs Gustafson scaling
- `fig089_arps_decline.py` — Arps decline family q = q_i/(1+bD_it)^{1/b} on semilog time (b = 0 straight) and as rate–cumulative with analytic N_p(q): EUR at the economic limit spans 715 → 2856 · 10³ bbl from b = 0 to 1
- `fig090_influence_lines.py` — influence lines for R_A, V_C and M_C of a simply supported beam from statics, with beam sketch, positive/negative shading, the unit shear jump at C and the a(L−a)/L moment peak
- `fig091_response_spectrum.py` — elastic response spectrum computed honestly: seeded band-passed, enveloped ground motion (PGA 0.30 g) integrated per-period with a vectorised central-difference SDOF sweep at ζ = 2, 5, 10 %; S_a → PGA at short period
- `fig092_activated_sludge.py` — chemostat washout diagram from Monod kinetics: S*, X*, and productivity D·X vs dilution rate, washout at D_c = µ_mS_0/(K_s+S_0), analytic D_opt asserted against the numerical maximum
- `fig093_planck_blackbody.py` — Planck spectra at four temperatures with the Wien displacement locus drawn through the peaks (a built-in check), the Rayleigh–Jeans divergence, and the visible band
- `fig094_qho_wavefunctions.py` — quantum harmonic oscillator: Hermite–Gaussian eigenfunctions riding their levels E_n = (n+½)ℏω inside V = ξ²/2, classical turning points dotted, zero-point energy bracketed at the well bottom
- `fig095_titration_curves.py` — titration curves solved exactly from the proton condition with brentq (dilution included): strong + four weak acids, half-equivalence circles landing at pH = pKa by construction
- `fig096_eye_diagram.py` — eye diagrams from folded raised-cosine traces (β = 0.35, singularities handled analytically): the Nyquist pinch at ±1 at the transmitter, then ISI + noise closing the eye after a band-limited channel
- `fig097_pz_material_balance.py` — gas material balance: noisy synthetic surveys on the exact p/Z line recover OGIP by extrapolation (G_hat = 99.8 vs true 100), with the water-drive bend and its misleading early-time tangent (apparent G ≈ 154)
- `fig098_terzaghi_consolidation.py` — Terzaghi consolidation from the diffusion series: pore-pressure isochrones for a doubly drained layer and U(T_v) with the √(4T_v/π) approximation and the T_50/T_90 milestones
- `fig099_double_slit.py` — double-slit interference under its single-slit envelope, with a rendered screen strip: d = 4a makes the m = ±4 orders coincide with envelope nulls and vanish — the missing orders are the built-in check
- `fig100_carnot_cycle.py` — the Carnot cycle in both planes: computed p–V branches enclosing the net work beside the T–s rectangle, with the loop integral asserted equal to (T_h − T_c)ΔS — 1729 J either way, because work between the same states through the same reversible cycle cannot care which plane you drew it in
- `fig101_grouped_bars_planned.py` — two-factor (genotype × treatment) bars with a second label row, every replicate over mean ± s.d., and exact Holm-adjusted Welch P values on brackets placed from the data maximum; the Holm step is asserted against its definition
- `fig102_tumour_growth_arms.py` — per-animal longitudinal simulation summarised as mean ± s.d. bands, arms labelled at the right edge in their own colour instead of a legend, a dosing-period bar above the axes; pre-dosing arm means asserted to agree
- `fig103_volcano_bh.py` — volcano on Benjamini–Hochberg adjusted P from per-gene standard errors, the BH step coded and asserted equal to scipy's, 6,000 genes as a rasterized layer under vector axes, class counts in a header row
- `fig104_clustermap_annotated.py` — clustermap with correlation-distance average linkage on both axes, a condition bar between the column dendrogram and the matrix, colorbar and legend placed in mm; a two-cluster cut is asserted to reproduce the conditions
- `fig105_raincloud_kruskal.py` — raincloud with KDE clouds trimmed where the density is negligible, a bimodal group its box alone would hide, and a Kruskal–Wallis header with n; KDE areas and mode counts asserted
- `fig106_regression_marginals.py` — OLS line with an analytic 95% band for the mean response plus marginal KDEs on shared axes from a millimetre GridSpec; slope ± s.e., r, P and n in-panel; residual orthogonality and the band's waist at the mean of x asserted
- `fig107_km_confidence_bands.py` — Kaplan–Meier curves with 95% Greenwood log-log confidence bands, censor ticks and a number-at-risk table on the tick positions; the hand-coded log-rank test is asserted to reproduce scipy's
- `fig108_subgroup_forest.py` — trial subgroup forest: grouped rows with banding, events/patients text columns set in mm, weight-sized squares, overall diamond, plain-tick log hazard-ratio axis; arm totals asserted identical for every factor
- `fig109_roc_pr_bootstrap.py` — ROC beside precision–recall on equal square panels, AUC with 95% bootstrap intervals, direct labels in each empty corner; the rank-sum AUC is asserted equal to the trapezoid area under the plotted curve
- `fig110_dose_response_ci.py` — 4-parameter logistic fitted to all replicates with log IC50 as the parameter and a 95% CI from the covariance matrix; mean ± s.d. with distinct markers, plain decimal ticks; true IC50 asserted inside each interval
- `fig111_embedding_dotplot.py` — single-cell style pair: a 6,000-point simulated embedding (rasterized layer, direct labels) beside a marker dot plot (area = % expressing, colour = scaled mean) with a hand-built size legend and mm-placed colorbar
- `fig112_composition_diversity.py` — per-sample stacked bars for two groups with group brackets and a stack-ordered legend, beside a Shannon-diversity box-plus-points panel with a Mann–Whitney test; every bar asserted to close at 100%
- `fig113_response_surface_fit.py` — quadratic response surface fitted by least squares to a 13-run central composite design: optimum solved from the Hessian (asserted negative definite), R², contour levels echoed on the colorbar
- `fig114_manhattan_loci.py` — Manhattan plot of about 90,000 variants on real autosome proportions: rasterized points, thresholds named in the right margin, loci labelled at their lead variants; genomic inflation of the null asserted ≈ 1
- `fig115_paired_estimation.py` — paired slope chart coloured by direction of change, beside an estimation panel of the paired differences with their mean and 95% t interval; scipy's paired t asserted equal to the hand formula
- `fig116_model_benchmark.py` — four-panel ML benchmark as one argument on a mm grid: training curves, seed-level ablation dots with Δ versus the full model, reliability diagram with ECE, confusion matrix; mean ± s.d. over 5 seeds throughout
- `fig117_xrd_annealing.py` — annealing series of pseudo-Voigt diffraction patterns sharpening out of an amorphous hump: sequential blues, right-margin trace labels, Miller indices at local maxima; peak finding asserted to recover all six reflections
- `fig118_raster_psth_sem.py` — two-condition spike raster over a binned PSTH with mean ± s.e.m. across trials as a band, rotated block labels, stimulus bar, shared time axis; binning asserted to conserve every spike
- `fig119_image_plate_quant.py` — 2 × 3 dark image plate (nuclei, marker, merge) in cyan/magenta with shared linear display limits and one calibrated scale bar, beside field-level quantification measured from the same simulated images
- `fig120_hero_composite.py` — asymmetric composite: a hero time course spanning two rows beside two stacked supporting panels, one role per panel, mean ± s.d. (n = 4) throughout, direct labels; hero edges asserted to match the stack within 1.5 pt
- `fig121_radar_benchmark.py` — radar chart of three methods on six tasks built on a bare polar axes: every spoke carries its own worst-to-best scale so outward is always better, proposed method filled, baselines outlined; shoelace polygon area asserted equal to its closed form and largest for the proposed method
- `fig122_critical_difference.py` — Demšar critical-difference diagram for six classifiers on 24 datasets: average ranks on a reversed axis, Nemenyi CD ruler, thick bars joining methods that do not differ, names on elbow leaders; hand-coded Friedman χ² asserted equal to scipy's
- `fig123_pareto_front.py` — accuracy against inference cost (log axis, plain ticks) for 40 models in three families: Pareto frontier as a staircase, frontier models coloured and labelled directly, dominated models grey; sorted-sweep frontier asserted equal to the brute-force non-dominated set
- `fig124_parallel_coordinates.py` — parallel coordinates of a 150-run hyperparameter sweep over five axes in real units (log, categorical, power-of-two, linear), lines coloured by validation accuracy, top five runs thick, tick labels in clear lanes; every axis map asserted to round-trip and the labelled best run to be the argmax
- `fig125_shap_beeswarm.py` — SHAP summary beeswarm for a linear model, where the Shapley values are exact: 400 samples × 8 features sorted by mean absolute contribution, from-scratch swarm layout, dots coloured by feature value; additivity (contributions sum to prediction minus mean prediction) asserted for every sample
- `fig126_gp_regression.py` — Gaussian-process regression coded from the kernel equations (RBF kernel, Cholesky solve): posterior mean, 95% band and sample paths over noisy observations, beside the log marginal likelihood against length scale with its maximiser; Cholesky mean asserted equal to the direct solve
- `fig127_genome_tracks.py` — genome-browser panel for a 200-kb window: three coverage tracks piled up from simulated fragments on one printed y scale, peak calls, gene models with strand chevrons, a shaded enhancer, Mb axis and scale bar; each coverage integral asserted equal to reads × fragment length
- `fig128_sequence_logo.py` — sequence logo of a 12-position motif from 200 simulated sites: letters are TextPath outlines stretched to exact boxes, stacked by information content with the small-sample correction, colour-blind-safe bases; glyph heights asserted to sum to 2 − H − e(n) bits
- `fig129_oncoprint.py` — oncoprint of 12 genes × 60 samples: copy-number events fill the cell and mutations sit as an inner band so two can coexist, memo-sorted columns give the mutual-exclusivity staircase, with subtype row and marginal bars; printed percentages asserted equal to the row means
- `fig130_lollipop_mutations.py` — protein lollipop plot: head area proportional to recurrence, colour by mutation class, over a domain bar labelled inside where the name fits; labelled hotspots asserted to be exactly the residues above a Poisson background threshold
- `fig131_hic_contact_triangle.py` — simulated Hi-C map (power-law decay × domain enrichment) as a 45°-rotated rasterized triangle on a log colour scale, over an insulation-score track on the same genomic axis; every planted boundary asserted to have an insulation minimum within two bins
- `fig132_tree_trait_heatmap.py` — UPGMA tree of 24 taxa drawn from the linkage matrix with branches coloured by clade, aligned tip labels and a scale bar, beside a row-matched trait heatmap; tip order, ultrametric root-to-tip lengths and a cophenetic correlation above 0.8 asserted
- `fig133_swimmer_plot.py` — swimmer plot of time on treatment for 26 patients: bars shaded by dose cohort and sorted by duration, marker shapes for response, progression and death, arrow heads for ongoing treatment; every event asserted to lie on its own bar
- `fig134_response_waterfall.py` — waterfall of best tumour change for 40 patients with threshold lines, response categories derived from them, cut bars flagged, a biomarker strip, and the response rate with its exact 95% CI; Clopper–Pearson limits from beta quantiles asserted equal to scipy's
- `fig135_consort_flow.py` — CONSORT participant-flow diagram drawn in code: counts tallied from a simulated enrolment table, boxes sized from rendered text extents, right-angle connectors placed in millimetres; counts asserted to conserve at every node and every text to lie inside its box
- `fig136_competing_risks_cif.py` — hand-coded Aalen–Johansen cumulative incidence of two competing events for two groups with a number-at-risk table, plus the naive 1 − Kaplan–Meier curve that overstates relapse; the two incidences and overall survival asserted to sum to 1
- `fig137_decision_curve.py` — decision curves of net benefit against threshold probability for two risk models versus treat-all and treat-none, the winning range shaded, a second axis in false positives per true positive; the treat-all curve asserted equal to its analytic line
- `fig138_specification_curve.py` — specification curve of 72 OLS fits (five analytic choices): ranked estimates with 95% CIs, median and main specification marked, above a dot matrix of the choices; hand-coded slope and s.e. asserted equal to linregress and lstsq
- `fig139_chord_diagram.py` — chord diagram of a 6 × 6 directed flow matrix from scratch: arcs spanning each group's total, source-coloured Bézier ribbons, the three largest flows saturated, group labels clear of the tick ring; ribbon ends asserted to tile every arc and the circle to close at 2π
- `fig140_network_communities.py` — 60-node stochastic block model drawn with a numpy Fruchterman–Reingold layout, coloured by spectral modularity clustering, node area by degree, within-community edges darker than bridges, hubs labelled; modularity above 0.3 and recovery of the planted blocks asserted
- `fig141_euler_proportional.py` — three area-proportional two-set Euler diagrams on one scale: radii from set sizes, centre distance solved by brentq so the lens equals the intersection, counts placed in each region; lens area asserted equal to the intersection
- `fig142_treemap_squarified.py` — squarified treemap (Bruls–Huizing–van Wijk, coded from scratch) of 30 leaves in 5 groups: group-colour tints, white gutters, labels only where the rendered text fits; one area scale, no overlap and a worst aspect ratio below slice-and-dice asserted
- `fig143_streamgraph.py` — streamgraph of 7 categories on the Byron–Wattenberg wiggle baseline, layers ordered inside-out, stroke-free fills with one saturated story layer, labels where each layer is thickest, a scale bar in place of the y axis; layer thickness, column totals and the baseline formula asserted
- `fig144_bump_chart.py` — bump chart of 9 items over 8 periods: ranks joined by smoothstep connectors, three story items in colour with rank numerals in their markers, the rest grey, names at both ends with net change; ranks asserted to be permutations equal to scipy's rankdata
- `fig145_spectrogram_chirp.py` — waveform of a quadratic chirp plus tone burst in noise over its Hann-window STFT spectrogram in dB (rasterized, aligned colorbar), analytic instantaneous frequency overlaid; the spectrogram ridge asserted to follow f(t) within one frequency bin plus window smearing
- `fig146_corner_posterior.py` — corner plot of a three-parameter posterior from a hand-coded Metropolis–Hastings sampler (4 chains): step marginals with median and 16th/84th percentiles, 39.3% and 86.5% mass contours, truth crosshairs; R-hat below 1.05, contour mass and truth coverage asserted
- `fig147_phase_portrait_fhn.py` — FitzHugh–Nagumo phase plane (nullclines, direction field, trajectories onto the limit cycle, unstable focus with its eigenvalues) beside v(t) and w(t) with the period bracketed; fixed point, instability and two independent period estimates asserted to agree
- `fig148_hh_traces_scalebars.py` — Hodgkin–Huxley responses to five current steps as axis-free stacked traces with an L-shaped scale bar and the step protocol beneath, beside the f–I curve with matching markers; resting fixed point and non-decreasing spike counts asserted
- `fig149_image_zoom_profile.py` — simulated bead-and-filament fluorescence field with scale bar and ROI, the ROI magnified pixel-for-pixel with its own bar and profile path, and the line profile with a Gaussian fit and FWHM bracket; fitted FWHM asserted to match the PSF within 5%
- `fig150_energy_landscape_3d.py` — Müller–Brown potential as a rasterized 3-D surface over floor contours beside a map with numerically located minima and saddles, their energies and the steepest-descent minimum-energy path; gradients, Hessian signatures and literature minimum energies asserted
- `manuscript.py` — shared module for the manuscript-panel figures (101–150): exact-size canvas in millimetres with no tight crop, mm margins and gutters, fixed-offset panel letters, exact-P brackets, and asserts for panel alignment and the 5 pt type floor

## Adapting to your data

Each script has a clearly marked `DATA` block with simulated values — replace it with your own arrays and everything downstream (fits, stats, annotations) updates automatically.

## What the style sheet encodes (edit once, applies everywhere)

- Okabe–Ito colour-blind-safe colour cycle
- Arial/Helvetica with DejaVu Sans fallback, 7–8 pt type sized for print
- Despined axes, thin lines, outward ticks, frameless legends
- `constrained_layout` + tight bounding box, 600 dpi PNG export
- Figure widths follow journal conventions: ~3.5 in (89 mm) single column, ~7.1 in (183 mm) double column
- Figures 101–150 go one step further through `manuscript.py`: the canvas is set in millimetres and saved without a tight crop, so the PDF page is exactly the journal width

## Tips

- Submit the PDF versions: vector graphics stay sharp at any zoom and keep text selectable/editable.
- If your journal wants serif figures, change `font.family` to `serif` in the style sheet.
- To match LaTeX manuscripts exactly, add `text.usetex: True` (requires a local LaTeX installation).
- Prefer reporting exact P values in captions; adjust the star thresholds in `fig4` to your journal's convention.
