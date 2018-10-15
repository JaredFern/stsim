for i in {1..10} 
do
	echo "$i Exemplars"
	python stsim-metric.py \
		--normalize z-norm \
		--distance_metric cov \
		--scope intraclass \
		--exemplar cluster_center \
		--cluster_cnt 1 \
       		--cluster_method kmeans \
		--fold_cnt 5 \
		--aca_color_cnt 1 \
		--aca_color_ordering luminance;
		
done;
