default :
	echo "Makefile ready"


reset_sample :
	rm -rf raw_data/PASTIS-R-sample
	gsutil -m cp -r gs://pastis-raw-data/PASTIS-R-sample raw_data/

reinstall_requirements:
	pip freeze | xargs pip uninstall -y
	pip install -r requirements.txt
