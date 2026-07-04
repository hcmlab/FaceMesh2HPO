from src.hpo_tree.hpo_term import HumanPhenotypeTerm


def build_modified_hpo_tree(data_dir: str, download: bool = False):
    hpo = HumanPhenotypeTerm.load_ontology(data_dir, download=download)
    hp_abnorm_face = hpo.find_successor('HP:0000271')  # Abnormality of the face
    hp_abnorm_eye = hpo.find_successor('HP:0000478')  # Abnormality of the eye
    hp_abnorm_eyebrow = hpo.find_successor('HP:0000534')  # Abnormal eyebrow morphology
    hp_abnorm_face.add_successor(hp_abnorm_eye)  # Move eye to face
    hp_abnorm_face.add_successor(hp_abnorm_eyebrow)  # Move eyebrow to face
    hp_abnorm_face.define_as_root()
    return hp_abnorm_face
