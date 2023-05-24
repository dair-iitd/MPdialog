# Place the ComSet unzipped in the same directory
conda env create -f environment_droplet.yml
conda activate segmentation
python -m spacy download en_core_web_sm
# Download FRCNN weights
python generate_dataset.py --base_dir data --urls ComSet --model_dir ./FCRNN
rm -r visual
mv segmented visual
rm data/scraped_transcripts.json
rm problems_panel_segmentation.txt
