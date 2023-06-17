import json



#print(average_length)

from deeponto.align.bertmap import BERTMapPipeline, DEFAULT_CONFIG_FILE
from deeponto.onto import Ontology

config = BERTMapPipeline.load_bertmap_config(DEFAULT_CONFIG_FILE)
src_onto_file = "/home/ellen/LakerMAP/onto/snomed.body.owl"  
tgt_onto_file = "/home/ellen/LakerMAP/onto/ncit.pharm.owl"

src_onto = Ontology(src_onto_file)
tgt_onto = Ontology(tgt_onto_file)

bertmap = BERTMapPipeline(src_onto, tgt_onto, config)
