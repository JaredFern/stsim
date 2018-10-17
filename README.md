# STSIM-C Texture Similarity

## Experimental Procedure
0. STSIM vector extraction
1. Compute **covariance** matrix
2. Approximate Cross Validation
	* Compute cluster center or set example for retrieval 
	* Measure Precision@1 and RoC 

## Experimental Settings: 
* Global or Intraclass covariance matrix calculation
* Color or Grayscale features

## Datasets:
* CuReT: Fewer classes, many examples per class
	* Represent class with many exemplars
	* Tests under variation of lighting and viewing conditions 
* Identical: Same texture, perceptually identical 
* ViSiProg: Different classes; Perceptually similar
* Distortions: Single image with applied distortion translation, rotations, scaling, etc.

## Differences from previous work:
* Previous work would have made assumptions among feature correlation
* Similarity is order invariant
* 1-,2-Color improves accuracym; Minimal gains for 4-color

