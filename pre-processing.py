from deeponto.onto import Ontology
import json
from tqdm import tqdm
import time

from deeponto.align.bertmap import BERTMapPipeline, DEFAULT_CONFIG_FILE

config = BERTMapPipeline.load_bertmap_config(DEFAULT_CONFIG_FILE)
src_onto_file = "/home/ellen/LakerMAP/onto/doid.owl"  
tgt_onto_file = "/home/ellen/LakerMAP/onto/ncit.owl"

src_onto = Ontology(src_onto_file)
tgt_onto = Ontology(tgt_onto_file)

bertmap = BERTMapPipeline(src_onto, tgt_onto, config)

src_class_annotations = bertmap.src_annotation_index['http://purl.obolibrary.org/obo/DOID_8239']

doid = Ontology("/home/ellen/LakerMAP/onto/doid.owl")
fmabody = Ontology("/home/ellen/LakerMAP/onto/fma.body.owl")
neoplas = Ontology("/home/ellen/LakerMAP/onto/ncit.neoplas.owl")
ncit = Ontology("")

def get_id(ontology):
    id_list = {}
    for key in ontology.owl_classes.keys():
        id = key.split("/")[-1]
        id_list[id] = key
    
    return id_list

def get_name(ontology):
    concept_list = {}
    for key in ontology.owl_classes.keys():
        
        id = key.split("/")[-1]

        concept = ontology.get_owl_object_annotations(

                    ontology.get_owl_object_from_iri(key),
                    annotation_property_iri='http://www.w3.org/2000/01/rdf-schema#label',
                    annotation_language_tag=None,
                    apply_lowercasing=False,
                    normalise_identifiers=False
                    )

        concept_list[id] = next(iter(concept))
    
    return concept_list

def get_relations(ontology):
    super_lists = {}
    for key in tqdm(ontology.owl_classes.keys()):
        doid_class = ontology.get_owl_object_from_iri(key)
        superlists = ontology.reasoner.get_inferred_super_entities(doid_class, direct=True) 

        #for superlist in superlists:

        super_lists[key] = superlists
    
    return super_lists


id_list = get_id(doid)
with open("entities_id.json", "w") as json_file:
    json.dump(id_list, json_file)

concept_list = get_name(doid)
with open("entities_name.json", "w") as json_file:
    json.dump(concept_list, json_file)

triplets_list = get_relations(doid)
result = [(key, value) for key, value_list in triplets_list.items() for value in value_list]

# Convert the list of tuples to a list of dictionaries
data_as_dicts = [{k: v} for k, v in result]

with open("triplets.json", "w") as json_file:
    json.dump(data_as_dicts, json_file, indent=4)

concept_list = get_name(fmabody)
with open("fmabody_entities_name.json", "w") as json_file:
    json.dump(concept_list, json_file)

dict1 = get_id(ncit1)
dict2 = get_id(ncit2)

keys1 = set(dict1.keys())
keys2 = set(dict2.keys())

common_keys = keys1.intersection(keys2)
unique_keys = keys1 - keys2.union(common_keys) | keys2 - keys1.union(common_keys)

doid = Ontology("/home/ellen/LakerMAP/onto/doid.owl")
ncit = Ontology("/home/ellen/LakerMAP/onto/ncit.owl")
omim= Ontology("/home/ellen/LakerMAP/onto/omim.owl")
ordo= Ontology("/home/ellen/LakerMAP/onto/ordo.owl")
ncit1 = Ontology("/home/ellen/LakerMAP/onto/ncit.neoplas.owl")
ncit2 = Ontology("/home/ellen/LakerMAP/onto/ncit.pharm.owl")
fma = Ontology("/home/ellen/LakerMAP/onto/fma.owl")
snomed1 = Ontology("/home/ellen/LakerMAP/onto/snomed.body.owl")
snomed2 = Ontology("/home/ellen/LakerMAP/onto/snomed.neoplas.owl")
snomed3 = Ontology("/home/ellen/LakerMAP/onto/snomed.pharm.owl")


print(fma.get_owl_object_from_iri('http://purl.org/sig/ont/fma/fma9531'))
doid_class = fma.get_owl_object_from_iri('http://purl.org/sig/ont/fma/fma9531')
superlists = fma.reasoner.get_inferred_super_entities(doid_class, direct=True) 
doid_relation = get_relations(fma)
result = [(key, value) for key, value_list in doid_relation.items() for value in value_list]

# Convert the list of tuples to a list of dictionaries
data_as_dicts = [{k: v} for k, v in result]

with open("/home/ellen/LakerMAP/onto/preprocessed/fma_triplets_direct.json", "w") as json_file:
    json.dump(data_as_dicts, json_file, indent=4)

with open('/home/ellen/LakerMAP/onto/preprocessed/snomed3_triplets_direct.json') as f:
    data = json.load(f)
# Create an empty dictionary to store the result
result_dict = {}

# Iterate through the list of dictionaries
for d in data:
    # Iterate through the key-value pairs in each dictionary
    for k, v in d.items():
        # Add the key-value pair to the result dictionary
        result_dict[k] = v

# Find the start and end keys for each path
start_keys = set(result_dict.keys())
end_keys = set(result_dict.values())
path_keys = start_keys - end_keys

# Construct the paths from start to end
paths = []
for key in path_keys:
    path = [key]
    while path[-1] in result_dict:
        path.append(result_dict[path[-1]])
    paths.append(path)

#print(len(paths))
total_length = sum(len(elem) for elem in paths)
num_elements = len(paths)
average_length = total_length / num_elements

