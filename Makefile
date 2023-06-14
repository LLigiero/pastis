default :
	echo "Makefile ready"

# FOLDER MANAGEMENT

reinstall_sample :
	rm -rf raw_data/PASTIS-R-sample
	gsutil -m cp -r gs://pastis-raw-data/PASTIS-R-sample raw_data/

reinstall_requirements:
	pip freeze | xargs pip uninstall -y
	pip install -r requirements.txt

run_unet_baseline_train:
	python -c 'from pastis.interface.main import train_baseline; train_baseline()'

run_unet_eval:
	python -c 'from pastis.interface.main import evaluate_unet; evaluate_unet()'

run_unet_convlstm_train:
	python -c 'from pastis.interface.main import train_unet_clstm; train_unet_clstm()'

run_unet_convlstm_eval:
	python -c 'from pastis.interface.main import evaluate_unet_clstm; evaluate_unet_clstm()'

run_unet_convlstm_radar_train:
	python -c 'from pastis.interface.main import train_unet_clstm_radar; train_unet_clstm_radar()'


reset_local_files :
	rm -rf raw_data
	mkdir raw_data
	mkdir raw_data/PASTIS-R
	mkdir raw_data/PASTIS-R/ANNOTATIONS
	mkdir raw_data/PASTIS-R/DATA_S2

download_unet_data : reset_local_files
	gsutil -m cp -r gs://$(BUCKET_NAME)/PASTIS-R/ANNOTATIONS/*.npy raw_data/PASTIS-R/ANNOTATIONS/
	gsutil cp gs://$(BUCKET_NAME)/PASTIS-R/metadata.geojson raw_data/PASTIS-R/
	gsutil -m cp -r gs://$(BUCKET_NAME)/PASTIS-R/DATA_S2/*.npy raw_data/PASTIS-R/DATA_S2/
	gsutil cp gs://$(BUCKET_NAME)/raw_data/data/ISIC_2019_Training_Metadata.csv  raw_data/

test :
	echo $(BUCKET_NAME)
