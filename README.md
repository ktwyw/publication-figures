# One Hundred and Fifty Publication-Quality Figures

[![verify](https://github.com/ktwyw/publication-figures/actions/workflows/verify.yml/badge.svg)](https://github.com/ktwyw/publication-figures/actions/workflows/verify.yml)
[![Code: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)
[![Figures: CC BY 4.0](https://img.shields.io/badge/figures-CC%20BY%204.0-lightgrey.svg)](LICENSE-FIGURES.md)

**150 standalone matplotlib scripts** for the figure types scientists and
engineers actually publish -- statistics, materials characterization,
transport phenomena, electrochemistry, Ashby charts, classics from
across engineering, and manuscript panels for genomics, clinical
research, machine learning, networks and imaging. Every curve is **derived from governing equations**
(never sketched), every random draw is **seeded**, and wherever the
physics fixes a landmark the script **checks itself** with an `assert`
-- the Wien locus threads every Planck peak, Mohr circles touch the
envelope exactly, the Carnot loop integral equals its T-s rectangle.
One shared style sheet keeps the whole set consistent; each script
writes a 600-dpi PNG and a submission-ready vector PDF.

**The companion guide** ([`figures/figure_guide.pdf`](figures/figure_guide.pdf))
distils the craft: Part I the eight habits these scripts embody, Part II
the illustrated catalog of all 150, Part III sixteen named failure
patterns (each harvested from a real bug), Part IV working checklists.

## Quick start

```bash
git clone https://github.com/ktwyw/publication-figures.git
cd publication-figures
pip install -r requirements.txt
python figures/fig001_curve_fit.py      # any script runs standalone
make figures                            # or render all 150
make guide gallery                      # rebuild the PDF guide + gallery
```

## The catalog

One line per figure lives in [`figures/README.md`](figures/README.md)
(the single source of truth the guide is typeset from). Sections:
**A** statistical & general data graphics (1-50) - **B** chemistry &
materials characterization (51-60) - **C** case study: reproducing a
journal theory figure (61) - **D** transport phenomena & reactors
(62-67) - **E** chem-eng & materials classics (68-73) -
**F** electrochemistry, stability & selection (74-79) - **G** Ashby
charts on the shared module (80-81) - **H** across science &
engineering (82-100) - **I** manuscript panels at exact printed size
(101-120) - **J** beyond the standard chart: evidence panels across
fields, on the same exact-size module (121-150).

## Gallery

Click any thumbnail to open its script.

<table>
<tr><td align="center"><a href="figures/fig001_curve_fit.py"><img src="gallery/thumbs/fig001_curve_fit.png" width="150" alt="fig001_curve_fit"></a></td><td align="center"><a href="figures/fig002_multipanel.py"><img src="gallery/thumbs/fig002_multipanel.png" width="150" alt="fig002_multipanel"></a></td><td align="center"><a href="figures/fig003_regression_ci.py"><img src="gallery/thumbs/fig003_regression_ci.png" width="150" alt="fig003_regression_ci"></a></td><td align="center"><a href="figures/fig004_bars_stats.py"><img src="gallery/thumbs/fig004_bars_stats.png" width="150" alt="fig004_bars_stats"></a></td><td align="center"><a href="figures/fig005_heatmap.py"><img src="gallery/thumbs/fig005_heatmap.png" width="150" alt="fig005_heatmap"></a></td></tr>
<tr><td align="center"><a href="figures/fig006_raincloud.py"><img src="gallery/thumbs/fig006_raincloud.png" width="150" alt="fig006_raincloud"></a></td><td align="center"><a href="figures/fig007_kaplan_meier.py"><img src="gallery/thumbs/fig007_kaplan_meier.png" width="150" alt="fig007_kaplan_meier"></a></td><td align="center"><a href="figures/fig008_volcano.py"><img src="gallery/thumbs/fig008_volcano.png" width="150" alt="fig008_volcano"></a></td><td align="center"><a href="figures/fig009_spectrum_inset.py"><img src="gallery/thumbs/fig009_spectrum_inset.png" width="150" alt="fig009_spectrum_inset"></a></td><td align="center"><a href="figures/fig010_loglog_powerlaw.py"><img src="gallery/thumbs/fig010_loglog_powerlaw.png" width="150" alt="fig010_loglog_powerlaw"></a></td></tr>
<tr><td align="center"><a href="figures/fig011_image_scalebar.py"><img src="gallery/thumbs/fig011_image_scalebar.png" width="150" alt="fig011_image_scalebar"></a></td><td align="center"><a href="figures/fig012_forest_plot.py"><img src="gallery/thumbs/fig012_forest_plot.png" width="150" alt="fig012_forest_plot"></a></td><td align="center"><a href="figures/fig013_roc_curves.py"><img src="gallery/thumbs/fig013_roc_curves.png" width="150" alt="fig013_roc_curves"></a></td><td align="center"><a href="figures/fig014_joint_marginals.py"><img src="gallery/thumbs/fig014_joint_marginals.png" width="150" alt="fig014_joint_marginals"></a></td><td align="center"><a href="figures/fig015_clustered_heatmap.py"><img src="gallery/thumbs/fig015_clustered_heatmap.png" width="150" alt="fig015_clustered_heatmap"></a></td></tr>
<tr><td align="center"><a href="figures/fig016_paired_slope.py"><img src="gallery/thumbs/fig016_paired_slope.png" width="150" alt="fig016_paired_slope"></a></td><td align="center"><a href="figures/fig017_contour_map.py"><img src="gallery/thumbs/fig017_contour_map.png" width="150" alt="fig017_contour_map"></a></td><td align="center"><a href="figures/fig018_bland_altman.py"><img src="gallery/thumbs/fig018_bland_altman.png" width="150" alt="fig018_bland_altman"></a></td><td align="center"><a href="figures/fig019_ridgeline.py"><img src="gallery/thumbs/fig019_ridgeline.png" width="150" alt="fig019_ridgeline"></a></td><td align="center"><a href="figures/fig020_coefficients.py"><img src="gallery/thumbs/fig020_coefficients.png" width="150" alt="fig020_coefficients"></a></td></tr>
<tr><td align="center"><a href="figures/fig021_stacked_composition.py"><img src="gallery/thumbs/fig021_stacked_composition.png" width="150" alt="fig021_stacked_composition"></a></td><td align="center"><a href="figures/fig022_streamplot.py"><img src="gallery/thumbs/fig022_streamplot.png" width="150" alt="fig022_streamplot"></a></td><td align="center"><a href="figures/fig023_wind_rose.py"><img src="gallery/thumbs/fig023_wind_rose.png" width="150" alt="fig023_wind_rose"></a></td><td align="center"><a href="figures/fig024_qq_plot.py"><img src="gallery/thumbs/fig024_qq_plot.png" width="150" alt="fig024_qq_plot"></a></td><td align="center"><a href="figures/fig025_calibration.py"><img src="gallery/thumbs/fig025_calibration.png" width="150" alt="fig025_calibration"></a></td></tr>
<tr><td align="center"><a href="figures/fig026_sankey.py"><img src="gallery/thumbs/fig026_sankey.png" width="150" alt="fig026_sankey"></a></td><td align="center"><a href="figures/fig027_pca_biplot.py"><img src="gallery/thumbs/fig027_pca_biplot.png" width="150" alt="fig027_pca_biplot"></a></td><td align="center"><a href="figures/fig028_waterfall.py"><img src="gallery/thumbs/fig028_waterfall.png" width="150" alt="fig028_waterfall"></a></td><td align="center"><a href="figures/fig029_time_series_events.py"><img src="gallery/thumbs/fig029_time_series_events.png" width="150" alt="fig029_time_series_events"></a></td><td align="center"><a href="figures/fig030_fan_chart.py"><img src="gallery/thumbs/fig030_fan_chart.png" width="150" alt="fig030_fan_chart"></a></td></tr>
<tr><td align="center"><a href="figures/fig031_upset.py"><img src="gallery/thumbs/fig031_upset.png" width="150" alt="fig031_upset"></a></td><td align="center"><a href="figures/fig032_dose_response.py"><img src="gallery/thumbs/fig032_dose_response.png" width="150" alt="fig032_dose_response"></a></td><td align="center"><a href="figures/fig033_confusion_matrix.py"><img src="gallery/thumbs/fig033_confusion_matrix.png" width="150" alt="fig033_confusion_matrix"></a></td><td align="center"><a href="figures/fig034_training_curves.py"><img src="gallery/thumbs/fig034_training_curves.png" width="150" alt="fig034_training_curves"></a></td><td align="center"><a href="figures/fig035_manhattan.py"><img src="gallery/thumbs/fig035_manhattan.png" width="150" alt="fig035_manhattan"></a></td></tr>
<tr><td align="center"><a href="figures/fig036_ternary.py"><img src="gallery/thumbs/fig036_ternary.png" width="150" alt="fig036_ternary"></a></td><td align="center"><a href="figures/fig037_raster_psth.py"><img src="gallery/thumbs/fig037_raster_psth.png" width="150" alt="fig037_raster_psth"></a></td><td align="center"><a href="figures/fig038_small_multiples.py"><img src="gallery/thumbs/fig038_small_multiples.png" width="150" alt="fig038_small_multiples"></a></td><td align="center"><a href="figures/fig039_ecdf_comparison.py"><img src="gallery/thumbs/fig039_ecdf_comparison.png" width="150" alt="fig039_ecdf_comparison"></a></td><td align="center"><a href="figures/fig040_carpet_plot.py"><img src="gallery/thumbs/fig040_carpet_plot.png" width="150" alt="fig040_carpet_plot"></a></td></tr>
<tr><td align="center"><a href="figures/fig041_dumbbell.py"><img src="gallery/thumbs/fig041_dumbbell.png" width="150" alt="fig041_dumbbell"></a></td><td align="center"><a href="figures/fig042_split_violin.py"><img src="gallery/thumbs/fig042_split_violin.png" width="150" alt="fig042_split_violin"></a></td><td align="center"><a href="figures/fig043_broken_axis.py"><img src="gallery/thumbs/fig043_broken_axis.png" width="150" alt="fig043_broken_axis"></a></td><td align="center"><a href="figures/fig044_hexbin_density.py"><img src="gallery/thumbs/fig044_hexbin_density.png" width="150" alt="fig044_hexbin_density"></a></td><td align="center"><a href="figures/fig045_funnel_plot.py"><img src="gallery/thumbs/fig045_funnel_plot.png" width="150" alt="fig045_funnel_plot"></a></td></tr>
<tr><td align="center"><a href="figures/fig046_bubble_chart.py"><img src="gallery/thumbs/fig046_bubble_chart.png" width="150" alt="fig046_bubble_chart"></a></td><td align="center"><a href="figures/fig047_superplot.py"><img src="gallery/thumbs/fig047_superplot.png" width="150" alt="fig047_superplot"></a></td><td align="center"><a href="figures/fig048_tornado.py"><img src="gallery/thumbs/fig048_tornado.png" width="150" alt="fig048_tornado"></a></td><td align="center"><a href="figures/fig049_energy_levels.py"><img src="gallery/thumbs/fig049_energy_levels.png" width="150" alt="fig049_energy_levels"></a></td><td align="center"><a href="figures/fig050_bode_plot.py"><img src="gallery/thumbs/fig050_bode_plot.png" width="150" alt="fig050_bode_plot"></a></td></tr>
<tr><td align="center"><a href="figures/fig051_xrd.py"><img src="gallery/thumbs/fig051_xrd.png" width="150" alt="fig051_xrd"></a></td><td align="center"><a href="figures/fig052_tga_dsc.py"><img src="gallery/thumbs/fig052_tga_dsc.png" width="150" alt="fig052_tga_dsc"></a></td><td align="center"><a href="figures/fig053_stress_strain.py"><img src="gallery/thumbs/fig053_stress_strain.png" width="150" alt="fig053_stress_strain"></a></td><td align="center"><a href="figures/fig054_cyclic_voltammetry.py"><img src="gallery/thumbs/fig054_cyclic_voltammetry.png" width="150" alt="fig054_cyclic_voltammetry"></a></td><td align="center"><a href="figures/fig055_nyquist_eis.py"><img src="gallery/thumbs/fig055_nyquist_eis.png" width="150" alt="fig055_nyquist_eis"></a></td></tr>
<tr><td align="center"><a href="figures/fig056_isotherm_psd.py"><img src="gallery/thumbs/fig056_isotherm_psd.png" width="150" alt="fig056_isotherm_psd"></a></td><td align="center"><a href="figures/fig057_arrhenius.py"><img src="gallery/thumbs/fig057_arrhenius.png" width="150" alt="fig057_arrhenius"></a></td><td align="center"><a href="figures/fig058_phase_diagram.py"><img src="gallery/thumbs/fig058_phase_diagram.png" width="150" alt="fig058_phase_diagram"></a></td><td align="center"><a href="figures/fig059_ftir_stack.py"><img src="gallery/thumbs/fig059_ftir_stack.png" width="150" alt="fig059_ftir_stack"></a></td><td align="center"><a href="figures/fig060_tauc_bandgap.py"><img src="gallery/thumbs/fig060_tauc_bandgap.png" width="150" alt="fig060_tauc_bandgap"></a></td></tr>
<tr><td align="center"><a href="figures/fig061_carreau_flowcurve.py"><img src="gallery/thumbs/fig061_carreau_flowcurve.png" width="150" alt="fig061_carreau_flowcurve"></a></td><td align="center"><a href="figures/fig062_moody.py"><img src="gallery/thumbs/fig062_moody.png" width="150" alt="fig062_moody"></a></td><td align="center"><a href="figures/fig063_falkner_skan.py"><img src="gallery/thumbs/fig063_falkner_skan.png" width="150" alt="fig063_falkner_skan"></a></td><td align="center"><a href="figures/fig064_thiele.py"><img src="gallery/thumbs/fig064_thiele.png" width="150" alt="fig064_thiele"></a></td><td align="center"><a href="figures/fig065_rtd_tanks.py"><img src="gallery/thumbs/fig065_rtd_tanks.png" width="150" alt="fig065_rtd_tanks"></a></td></tr>
<tr><td align="center"><a href="figures/fig066_vdw_maxwell.py"><img src="gallery/thumbs/fig066_vdw_maxwell.png" width="150" alt="fig066_vdw_maxwell"></a></td><td align="center"><a href="figures/fig067_transient_conduction.py"><img src="gallery/thumbs/fig067_transient_conduction.png" width="150" alt="fig067_transient_conduction"></a></td><td align="center"><a href="figures/fig068_mccabe_thiele.py"><img src="gallery/thumbs/fig068_mccabe_thiele.png" width="150" alt="fig068_mccabe_thiele"></a></td><td align="center"><a href="figures/fig069_tafel.py"><img src="gallery/thumbs/fig069_tafel.png" width="150" alt="fig069_tafel"></a></td><td align="center"><a href="figures/fig070_breakthrough.py"><img src="gallery/thumbs/fig070_breakthrough.png" width="150" alt="fig070_breakthrough"></a></td></tr>
<tr><td align="center"><a href="figures/fig071_ellingham.py"><img src="gallery/thumbs/fig071_ellingham.png" width="150" alt="fig071_ellingham"></a></td><td align="center"><a href="figures/fig072_ttt.py"><img src="gallery/thumbs/fig072_ttt.png" width="150" alt="fig072_ttt"></a></td><td align="center"><a href="figures/fig073_common_tangent.py"><img src="gallery/thumbs/fig073_common_tangent.png" width="150" alt="fig073_common_tangent"></a></td><td align="center"><a href="figures/fig074_pourbaix.py"><img src="gallery/thumbs/fig074_pourbaix.png" width="150" alt="fig074_pourbaix"></a></td><td align="center"><a href="figures/fig075_ashby.py"><img src="gallery/thumbs/fig075_ashby.png" width="150" alt="fig075_ashby"></a></td></tr>
<tr><td align="center"><a href="figures/fig076_residue_curves.py"><img src="gallery/thumbs/fig076_residue_curves.png" width="150" alt="fig076_residue_curves"></a></td><td align="center"><a href="figures/fig077_hallpetch_weibull.py"><img src="gallery/thumbs/fig077_hallpetch_weibull.png" width="150" alt="fig077_hallpetch_weibull"></a></td><td align="center"><a href="figures/fig078_evans.py"><img src="gallery/thumbs/fig078_evans.png" width="150" alt="fig078_evans"></a></td><td align="center"><a href="figures/fig079_cstr_multiplicity.py"><img src="gallery/thumbs/fig079_cstr_multiplicity.png" width="150" alt="fig079_cstr_multiplicity"></a></td><td align="center"><a href="figures/fig080_ashby_strength_density.py"><img src="gallery/thumbs/fig080_ashby_strength_density.png" width="150" alt="fig080_ashby_strength_density"></a></td></tr>
<tr><td align="center"><a href="figures/fig081_ashby_toughness_modulus.py"><img src="gallery/thumbs/fig081_ashby_toughness_modulus.png" width="150" alt="fig081_ashby_toughness_modulus"></a></td><td align="center"><a href="figures/fig082_smith_chart.py"><img src="gallery/thumbs/fig082_smith_chart.png" width="150" alt="fig082_smith_chart"></a></td><td align="center"><a href="figures/fig083_ber_waterfall.py"><img src="gallery/thumbs/fig083_ber_waterfall.png" width="150" alt="fig083_ber_waterfall"></a></td><td align="center"><a href="figures/fig084_buckley_leverett.py"><img src="gallery/thumbs/fig084_buckley_leverett.png" width="150" alt="fig084_buckley_leverett"></a></td><td align="center"><a href="figures/fig085_mohr_coulomb.py"><img src="gallery/thumbs/fig085_mohr_coulomb.png" width="150" alt="fig085_mohr_coulomb"></a></td></tr>
<tr><td align="center"><a href="figures/fig086_streeter_phelps.py"><img src="gallery/thumbs/fig086_streeter_phelps.png" width="150" alt="fig086_streeter_phelps"></a></td><td align="center"><a href="figures/fig087_antenna_patterns.py"><img src="gallery/thumbs/fig087_antenna_patterns.png" width="150" alt="fig087_antenna_patterns"></a></td><td align="center"><a href="figures/fig088_roofline_amdahl.py"><img src="gallery/thumbs/fig088_roofline_amdahl.png" width="150" alt="fig088_roofline_amdahl"></a></td><td align="center"><a href="figures/fig089_arps_decline.py"><img src="gallery/thumbs/fig089_arps_decline.png" width="150" alt="fig089_arps_decline"></a></td><td align="center"><a href="figures/fig090_influence_lines.py"><img src="gallery/thumbs/fig090_influence_lines.png" width="150" alt="fig090_influence_lines"></a></td></tr>
<tr><td align="center"><a href="figures/fig091_response_spectrum.py"><img src="gallery/thumbs/fig091_response_spectrum.png" width="150" alt="fig091_response_spectrum"></a></td><td align="center"><a href="figures/fig092_activated_sludge.py"><img src="gallery/thumbs/fig092_activated_sludge.png" width="150" alt="fig092_activated_sludge"></a></td><td align="center"><a href="figures/fig093_planck_blackbody.py"><img src="gallery/thumbs/fig093_planck_blackbody.png" width="150" alt="fig093_planck_blackbody"></a></td><td align="center"><a href="figures/fig094_qho_wavefunctions.py"><img src="gallery/thumbs/fig094_qho_wavefunctions.png" width="150" alt="fig094_qho_wavefunctions"></a></td><td align="center"><a href="figures/fig095_titration_curves.py"><img src="gallery/thumbs/fig095_titration_curves.png" width="150" alt="fig095_titration_curves"></a></td></tr>
<tr><td align="center"><a href="figures/fig096_eye_diagram.py"><img src="gallery/thumbs/fig096_eye_diagram.png" width="150" alt="fig096_eye_diagram"></a></td><td align="center"><a href="figures/fig097_pz_material_balance.py"><img src="gallery/thumbs/fig097_pz_material_balance.png" width="150" alt="fig097_pz_material_balance"></a></td><td align="center"><a href="figures/fig098_terzaghi_consolidation.py"><img src="gallery/thumbs/fig098_terzaghi_consolidation.png" width="150" alt="fig098_terzaghi_consolidation"></a></td><td align="center"><a href="figures/fig099_double_slit.py"><img src="gallery/thumbs/fig099_double_slit.png" width="150" alt="fig099_double_slit"></a></td><td align="center"><a href="figures/fig100_carnot_cycle.py"><img src="gallery/thumbs/fig100_carnot_cycle.png" width="150" alt="fig100_carnot_cycle"></a></td></tr>
<tr><td align="center"><a href="figures/fig101_grouped_bars_planned.py"><img src="gallery/thumbs/fig101_grouped_bars_planned.png" width="150" alt="fig101_grouped_bars_planned"></a></td><td align="center"><a href="figures/fig102_tumour_growth_arms.py"><img src="gallery/thumbs/fig102_tumour_growth_arms.png" width="150" alt="fig102_tumour_growth_arms"></a></td><td align="center"><a href="figures/fig103_volcano_bh.py"><img src="gallery/thumbs/fig103_volcano_bh.png" width="150" alt="fig103_volcano_bh"></a></td><td align="center"><a href="figures/fig104_clustermap_annotated.py"><img src="gallery/thumbs/fig104_clustermap_annotated.png" width="150" alt="fig104_clustermap_annotated"></a></td><td align="center"><a href="figures/fig105_raincloud_kruskal.py"><img src="gallery/thumbs/fig105_raincloud_kruskal.png" width="150" alt="fig105_raincloud_kruskal"></a></td></tr>
<tr><td align="center"><a href="figures/fig106_regression_marginals.py"><img src="gallery/thumbs/fig106_regression_marginals.png" width="150" alt="fig106_regression_marginals"></a></td><td align="center"><a href="figures/fig107_km_confidence_bands.py"><img src="gallery/thumbs/fig107_km_confidence_bands.png" width="150" alt="fig107_km_confidence_bands"></a></td><td align="center"><a href="figures/fig108_subgroup_forest.py"><img src="gallery/thumbs/fig108_subgroup_forest.png" width="150" alt="fig108_subgroup_forest"></a></td><td align="center"><a href="figures/fig109_roc_pr_bootstrap.py"><img src="gallery/thumbs/fig109_roc_pr_bootstrap.png" width="150" alt="fig109_roc_pr_bootstrap"></a></td><td align="center"><a href="figures/fig110_dose_response_ci.py"><img src="gallery/thumbs/fig110_dose_response_ci.png" width="150" alt="fig110_dose_response_ci"></a></td></tr>
<tr><td align="center"><a href="figures/fig111_embedding_dotplot.py"><img src="gallery/thumbs/fig111_embedding_dotplot.png" width="150" alt="fig111_embedding_dotplot"></a></td><td align="center"><a href="figures/fig112_composition_diversity.py"><img src="gallery/thumbs/fig112_composition_diversity.png" width="150" alt="fig112_composition_diversity"></a></td><td align="center"><a href="figures/fig113_response_surface_fit.py"><img src="gallery/thumbs/fig113_response_surface_fit.png" width="150" alt="fig113_response_surface_fit"></a></td><td align="center"><a href="figures/fig114_manhattan_loci.py"><img src="gallery/thumbs/fig114_manhattan_loci.png" width="150" alt="fig114_manhattan_loci"></a></td><td align="center"><a href="figures/fig115_paired_estimation.py"><img src="gallery/thumbs/fig115_paired_estimation.png" width="150" alt="fig115_paired_estimation"></a></td></tr>
<tr><td align="center"><a href="figures/fig116_model_benchmark.py"><img src="gallery/thumbs/fig116_model_benchmark.png" width="150" alt="fig116_model_benchmark"></a></td><td align="center"><a href="figures/fig117_xrd_annealing.py"><img src="gallery/thumbs/fig117_xrd_annealing.png" width="150" alt="fig117_xrd_annealing"></a></td><td align="center"><a href="figures/fig118_raster_psth_sem.py"><img src="gallery/thumbs/fig118_raster_psth_sem.png" width="150" alt="fig118_raster_psth_sem"></a></td><td align="center"><a href="figures/fig119_image_plate_quant.py"><img src="gallery/thumbs/fig119_image_plate_quant.png" width="150" alt="fig119_image_plate_quant"></a></td><td align="center"><a href="figures/fig120_hero_composite.py"><img src="gallery/thumbs/fig120_hero_composite.png" width="150" alt="fig120_hero_composite"></a></td></tr>
<tr><td align="center"><a href="figures/fig121_radar_benchmark.py"><img src="gallery/thumbs/fig121_radar_benchmark.png" width="150" alt="fig121_radar_benchmark"></a></td><td align="center"><a href="figures/fig122_critical_difference.py"><img src="gallery/thumbs/fig122_critical_difference.png" width="150" alt="fig122_critical_difference"></a></td><td align="center"><a href="figures/fig123_pareto_front.py"><img src="gallery/thumbs/fig123_pareto_front.png" width="150" alt="fig123_pareto_front"></a></td><td align="center"><a href="figures/fig124_parallel_coordinates.py"><img src="gallery/thumbs/fig124_parallel_coordinates.png" width="150" alt="fig124_parallel_coordinates"></a></td><td align="center"><a href="figures/fig125_shap_beeswarm.py"><img src="gallery/thumbs/fig125_shap_beeswarm.png" width="150" alt="fig125_shap_beeswarm"></a></td></tr>
<tr><td align="center"><a href="figures/fig126_gp_regression.py"><img src="gallery/thumbs/fig126_gp_regression.png" width="150" alt="fig126_gp_regression"></a></td><td align="center"><a href="figures/fig127_genome_tracks.py"><img src="gallery/thumbs/fig127_genome_tracks.png" width="150" alt="fig127_genome_tracks"></a></td><td align="center"><a href="figures/fig128_sequence_logo.py"><img src="gallery/thumbs/fig128_sequence_logo.png" width="150" alt="fig128_sequence_logo"></a></td><td align="center"><a href="figures/fig129_oncoprint.py"><img src="gallery/thumbs/fig129_oncoprint.png" width="150" alt="fig129_oncoprint"></a></td><td align="center"><a href="figures/fig130_lollipop_mutations.py"><img src="gallery/thumbs/fig130_lollipop_mutations.png" width="150" alt="fig130_lollipop_mutations"></a></td></tr>
<tr><td align="center"><a href="figures/fig131_hic_contact_triangle.py"><img src="gallery/thumbs/fig131_hic_contact_triangle.png" width="150" alt="fig131_hic_contact_triangle"></a></td><td align="center"><a href="figures/fig132_tree_trait_heatmap.py"><img src="gallery/thumbs/fig132_tree_trait_heatmap.png" width="150" alt="fig132_tree_trait_heatmap"></a></td><td align="center"><a href="figures/fig133_swimmer_plot.py"><img src="gallery/thumbs/fig133_swimmer_plot.png" width="150" alt="fig133_swimmer_plot"></a></td><td align="center"><a href="figures/fig134_response_waterfall.py"><img src="gallery/thumbs/fig134_response_waterfall.png" width="150" alt="fig134_response_waterfall"></a></td><td align="center"><a href="figures/fig135_consort_flow.py"><img src="gallery/thumbs/fig135_consort_flow.png" width="150" alt="fig135_consort_flow"></a></td></tr>
<tr><td align="center"><a href="figures/fig136_competing_risks_cif.py"><img src="gallery/thumbs/fig136_competing_risks_cif.png" width="150" alt="fig136_competing_risks_cif"></a></td><td align="center"><a href="figures/fig137_decision_curve.py"><img src="gallery/thumbs/fig137_decision_curve.png" width="150" alt="fig137_decision_curve"></a></td><td align="center"><a href="figures/fig138_specification_curve.py"><img src="gallery/thumbs/fig138_specification_curve.png" width="150" alt="fig138_specification_curve"></a></td><td align="center"><a href="figures/fig139_chord_diagram.py"><img src="gallery/thumbs/fig139_chord_diagram.png" width="150" alt="fig139_chord_diagram"></a></td><td align="center"><a href="figures/fig140_network_communities.py"><img src="gallery/thumbs/fig140_network_communities.png" width="150" alt="fig140_network_communities"></a></td></tr>
<tr><td align="center"><a href="figures/fig141_euler_proportional.py"><img src="gallery/thumbs/fig141_euler_proportional.png" width="150" alt="fig141_euler_proportional"></a></td><td align="center"><a href="figures/fig142_treemap_squarified.py"><img src="gallery/thumbs/fig142_treemap_squarified.png" width="150" alt="fig142_treemap_squarified"></a></td><td align="center"><a href="figures/fig143_streamgraph.py"><img src="gallery/thumbs/fig143_streamgraph.png" width="150" alt="fig143_streamgraph"></a></td><td align="center"><a href="figures/fig144_bump_chart.py"><img src="gallery/thumbs/fig144_bump_chart.png" width="150" alt="fig144_bump_chart"></a></td><td align="center"><a href="figures/fig145_spectrogram_chirp.py"><img src="gallery/thumbs/fig145_spectrogram_chirp.png" width="150" alt="fig145_spectrogram_chirp"></a></td></tr>
<tr><td align="center"><a href="figures/fig146_corner_posterior.py"><img src="gallery/thumbs/fig146_corner_posterior.png" width="150" alt="fig146_corner_posterior"></a></td><td align="center"><a href="figures/fig147_phase_portrait_fhn.py"><img src="gallery/thumbs/fig147_phase_portrait_fhn.png" width="150" alt="fig147_phase_portrait_fhn"></a></td><td align="center"><a href="figures/fig148_hh_traces_scalebars.py"><img src="gallery/thumbs/fig148_hh_traces_scalebars.png" width="150" alt="fig148_hh_traces_scalebars"></a></td><td align="center"><a href="figures/fig149_image_zoom_profile.py"><img src="gallery/thumbs/fig149_image_zoom_profile.png" width="150" alt="fig149_image_zoom_profile"></a></td><td align="center"><a href="figures/fig150_energy_landscape_3d.py"><img src="gallery/thumbs/fig150_energy_landscape_3d.png" width="150" alt="fig150_energy_landscape_3d"></a></td></tr>
</table>

## Figures 101-120: manuscript panels at exact size

The first hundred let `constrained_layout` and a tight bounding box
choose the final size. The twenty in section **I** do what a journal
asks instead, through the shared module
[`figures/manuscript.py`](figures/manuscript.py): the canvas is set in
millimetres (89, 120 or 183 mm wide) and saved without a crop, margins
and gutters are in millimetres, type is 7/6 pt with a 5 pt floor, panel
letters sit at a fixed point offset, and multi-panel scripts assert
that their plot areas line up within 1.5 pt. Most revisit a figure type
from sections A-B and add what a reviewer asks for next -- confidence
bands and a cross-checked log-rank test on the Kaplan-Meier curves,
bootstrap intervals and a precision-recall panel beside the ROC curves,
Holm-adjusted exact P values on the bar chart. Each carries its own
built-in check, and all of their data are simulated.

## Figures 121-150: beyond the standard chart

Section **J** adds thirty figure types the first 120 did not cover,
all on `manuscript.py` at exact printed size and all with simulated
data:

- **Model evaluation (121-126)** - radar benchmark, critical-difference
  diagram, Pareto frontier, parallel coordinates, SHAP beeswarm,
  Gaussian-process regression.
- **Genomics (127-132)** - genome-browser tracks, sequence logo,
  oncoprint, mutation lollipop, Hi-C contact triangle, tree with trait
  heatmap.
- **Clinical research (133-138)** - swimmer plot, response waterfall,
  CONSORT flow diagram, competing-risks cumulative incidence,
  decision curve, specification curve.
- **Networks, flows, sets and rankings (139-144)** - chord diagram,
  community graph, area-proportional Euler diagrams, squarified
  treemap, streamgraph, bump chart.
- **Signals, inference, dynamics and imaging (145-150)** - chirp
  spectrogram, corner plot from a hand-coded sampler, phase portrait,
  Hodgkin-Huxley traces with scale bars, image zoom with line profile,
  3-D energy landscape.

Where a library would normally do the layout (chord ribbons, treemap,
streamgraph baseline, force-directed graph, logo glyphs, Aalen-Johansen
estimator, Metropolis-Hastings sampler) the script codes it from
scratch with numpy and scipy, and asserts a landmark the mathematics
fixes: ribbon ends tile every arc, rectangle areas match their values,
the two incidences and survival sum to one.

## Growing the library

The set is built to keep growing -- see
[`docs/ADDING_FIGURES.md`](docs/ADDING_FIGURES.md).
`python tools/new_figure.py 151 my_slug "description"` scaffolds a new
figure on the house contract (style sheet, seeded RNG, derived curves,
built-in self-check, twin outputs); one index line in
`figures/README.md` flows it into the guide and this gallery
automatically, and CI re-verifies everything on push.

## Verification

Every push renders every figure on GitHub Actions; the scripts'
built-in asserts are the test suite. The guide PDF is rebuilt
and uploaded as an artifact.

## Licences & citation

Code: [MIT](LICENSE). Rendered figures & guide:
[CC BY 4.0](LICENSE-FIGURES.md). To cite, see
[`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button
uses it).
