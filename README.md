# MR Tracking Tests

Below are instructions for generating figures pertaining to the static and dynamic tracking tests. 

The static tracking test figures include tip tracking error heatmaps for data acquired with the SRI_Original and FH512_noDither_gradSpoiled tracking sequences.

The dynamic tracking test figures include error line plots and box plots for data acquired with the SRI_Original2 and FH512_noDither_gradSpoiled2 tracking sequences.

These plots can be found in Arjun Gupta’s MSc thesis, titled “Active catheter tracking error characterization for MR-guided cardiac interventions”.

## Static Tracking Tests

There are 3 static tracking analysis notebooks called “static_tracking_heatmaps_Y0.ipynb”, “static_tracking_heatmaps_Y1.ipynb”, & “static_tracking_heatmaps_Y2.ipynb”, for the data acquired at Y=45mm, Y=20mm, & Y=-5mm, respectively. To ensure these notebooks run with no errors, you must do the following:

1. Input the data paths according to where the data is located on your system. The lines that you must change are Lines 2-4 under the “Constants and Paths” heading.

2. Input the plotting paths according to where you would like the heatmaps to be saved on your system. The line that you must change is Line 9 under the “Constants and Paths” heading.


## Dynamic Tracking Tests

There is 1 dynamic tracking analysis notebook, called “dynamic_tracking_analysis.ipynb”. To ensure it runs with no errors, you must do the following:

1. Input the data paths according to where the data is located on your system. The lines you must change are Line 2 & Lines 5-9, under the “Constants and Paths” heading.

2. Input the plotting paths according to where you would like the line and box plots to be saved on your system. The lines that you must change are Line 12 & Line 15, under the “Constants and Paths” heading.

3. Depending on the type of box plots you want to generate, you must uncomment specific lines. 

    a. If you want to generate coil error box plots organized by both coils and sequences, you must uncomment Lines 204, Lines 229-231, & Lines 246-250, under the “Plotting Functions” heading. Leave them commented otherwise.

    b. If you want to generate tip tracking box plots organized by sequences only, you must uncomment Lines 207-208, Lines 234-236, & Lines 253-257, under the “Plotting Functions” heading. Leave them commented otherwise.
